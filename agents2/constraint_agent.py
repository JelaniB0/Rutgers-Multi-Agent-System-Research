"""
constraint_agent.py

- Uses LLM to determine if a student meets prequisites for a course. 
- Uses credit to determine if a student's current standing suits a course's level. I.E. 15 credit freshmen shouldn't be recommended 300/400 level courses.
- Uses penalties (0 - 0.6) so PlanningAgent can down-rank courses because of credit standing but, still recommend them. 
- Still works without missing transcript information, but can give better recommendations with it (clarifying question posed by orchestrator can collect information). 

Credit Standings for Rutgers:
- Freshman: 0-29 credits -> should be recommended mostly 100/200 level courses, with some 300/400 if they have AP/transfer credits or are doing well.
- Sophomore: 30-59 credits -> can be recommended more 200/300 level courses, with stronger emphasis toward level 200 courses. 
- Junior: 60-89 credits -> can be recommended 300 level courses, with some 400 if they have strong performance and relevant completed courses.
- Senior: 90+ credits -> can be recommended any of the four levels, with stronger emphasis on 300/400 level courses.

Penalty Score Explanation(Subject to change or tuning): 
- 0.0: No penalty, fully meets prerequisites and is appropriate for standing
- 0.15: One tier below recommended stadnding. 
- 0.35: Two tiers below recommended standing.
- 0.6: Three or more tiers below recommended standing. 
"""

import os
import re
import json
import traceback
from typing import Dict, Any, Optional, Tuple, List
from agent_framework import ChatAgent
from agent_framework.openai import OpenAIResponsesClient
from .shared_types import AgentResponse, ConversationState, ConstraintViolation
from .academic_profile import allowed_course, course_level, graduate_check
from .dag_builder import batch_check_eligibility
from .paths import DAG_FILE

# Configurations

LEVEL_STANDING: Dict[str, Tuple[int, str]] = {
    1: (0,  "open to all students"),
    2: (30, "Sophomore standing (30+ credits)"),
    3: (60, "Junior standing (60+ credits)"),
    4: (90, "Senior standing (90+ credits)"),
}

STANDING_PENALTY: Dict[int, float] = {
    0: 0.0,   # Meets or exceeds standing
    1: 0.15,  # One tier below
    2: 0.35,  # Two tiers below
    3: 0.6,   # Three or more tiers below
}

class ConstraintAgent(ChatAgent):
    """
    Validates prequisite codes and credit standing for course recommendation. 
    """
 
    SYSTEM_PROMPT = """You are an academic constraint validator for Rutgers University CS courses.
 
    Your job is to determine whether a student satisfies the prerequisites for a list of courses.
    
    You will receive:
    - A numbered list of courses, each with a code, title, and full description
    - The student's completed course codes
    - The student's in-progress course codes
    
    Rules for each course:
    - A prerequisite is MET if the course code appears in completed OR in-progress courses.
    - AND logic: ALL listed prerequisites must be met.
    - OR logic: ANY ONE of the listed prerequisites is sufficient.
    - "Permission of instructor" or "by arrangement" -> treat as met (cannot be verified automatically).
    - No prerequisite mentioned -> always eligible.
    - Co-requisites (courses that can be taken simultaneously) -> treat as met if in-progress.
    
    Return ONLY a valid JSON object keyed by course code — no explanation, no markdown:
    {
    "01:198:XXX": {
        "eligible": true or false,
        "met_prerequisites": ["prereq codes satisfied"],
        "unmet_prerequisites": ["prereq codes NOT satisfied"],
        "reasoning": "one sentence explanation",
        "pathway_suggestion": "what to take first to become eligible, or null if already eligible"
    },
    "01:198:YYY": { ... }
    }
    
    Include every course code from the input. Do not skip any.
    """


    def __init__(self, client: OpenAIResponsesClient, model: str):
        """
        Initialize constraint agent

        Args:
            client: OpenAIResponsesClient instance shared with other agents
            model: Model ID string to use
        """

        super().__init__(
            chat_client=client,
            default_options={"model_id": model},
            instructions=self.SYSTEM_PROMPT
        )

        self.model = model
        self.dag = self._load_or_build_dag()
        # print(f"[ConstraintAgent] Initialized with model: {model}, using LLM-based constraint validation")

    def _load_or_build_dag(self) -> Dict:
        dag_path = DAG_FILE
        if os.path.exists(dag_path):
            with open(dag_path) as f:
                return json.load(f)
        return {}  # build_dag in driver ensures it exists before agents initialize

    async def validate_courses(
            self, 
            courses: List[Dict], 
            state: ConversationState,

    ) -> AgentResponse:
        """
        Validate a list of recommended courses against the student's transcript.
 
        Called by the Orchestrator after PlanningAgent produces its ranked list.
        Each output course is annotated with:
            constraint_check.eligible          (bool)   — prereqs met?
            constraint_check.standing_penalty  (float)  — 0.0–0.6 down-rank weight
            constraint_check.standing_note     (str)    — human-readable context
 
        PlanningAgent should subtract standing_penalty from match_score when
        re-ranking or presenting results to the student.
 
        Args:
            courses: List of course dicts from DataAgent / PlanningAgent.
            state:   ConversationState — transcript_data is read from here.
 
        Returns:
            AgentResponse with data = {
                "eligible_courses":   [...],
                "ineligible_courses": [...],
                "validation_summary": str,
                "violations":         [ConstraintViolation.to_dict(), ...]
            }
        """
        try:
            completed, in_progress, completed_credits, in_progress_credits = (
                self._get_student_data(state)
            )
            has_transcript = state.transcript_data is not None

            eligible: List[Dict] = []
            ineligible: List[Dict] = []
            violations: List[ConstraintViolation] = []

            regular = [c for c in courses if allowed_course(c, state) and course_level(c) != "graduate"]
            prereq_results = await self._batch_check_prereqs(regular, completed, in_progress) if regular else {}
            unverified = []

            for course in courses:
                if not allowed_course(course, state) or course_level(course) == "graduate":
                    result = graduate_check(course, state)
                    annotated = dict(course, constraint_check=result)
                    (ineligible if result["eligible"] is False else unverified).append(annotated)
                    continue
                prereq_result = prereq_results.get(course.get("code", ""), self._safe_default())
                # print(f"[DEBUG constraint] {course.get('code')} eligible={prereq_result['eligible']} unmet={prereq_result['unmet_prerequisites']}")
                standing_result = self._check_standing(
                    course, completed_credits, in_progress_credits, has_transcript
                )

                annotated = course.copy()
                annotated["constraint_check"] = {
                    **prereq_result, # interesting
                    "standing_penalty":  standing_result["penalty"],
                    "standing_note":     standing_result["note"],
                    "standing_eligible": standing_result["meets_standing"],
                }

                if prereq_result["eligible"]: # if prereqs met, check standing but still consider for recommendation with penalty if standing isn't ideal. 
                    eligible.append(annotated)

                    if standing_result["penalty"] > 0:
                        violations.append(
                            ConstraintViolation(
                                constraint_type="credit_standing",
                                severity=(
                                    "medium" if standing_result["penalty"] <= 0.15 else "high"
                                ),
                                message=(
                                    f"{course.get('code', 'Unknown')}: "
                                    f"{standing_result['note']}"
                                ),
                                affected_courses=[course.get("code", "")],
                                suggestion=standing_result.get("suggestion")
                            )
                        )   
                else:
                    ineligible.append(annotated)
                    violations.append(
                        ConstraintViolation(
                            constraint_type="prerequisite",
                            severity="high",
                            message=(
                                f"{course.get('code', 'Unknown')} requires: "
                                f"{', '.join(prereq_result['unmet_prerequisites'])}"
                            ),
                            affected_courses=[course.get("code", "")],
                            suggestion=prereq_result.get("pathway_suggestion")
                        )
                    )

            # builds summary of findings. 

            summary = self._build_summary(eligible, ineligible, violations)
            if unverified:
                summary = f"{len(eligible)} verified, {len(ineligible)} blocked, {len(unverified)} require graduate eligibility verification."
            # print(
            #     f"[ConstraintAgent] Validated {len(courses)} courses — "
            #     f"{len(eligible)} prereq-eligible, {len(ineligible)} ineligible"
            # )

            return AgentResponse(
                success=True,
                data={
                    "eligible_courses": eligible, 
                    "ineligible_courses": ineligible,
                    "unverified_courses": unverified,
                    "validation_summary": summary,
                    "violations": [v.to_dict() for v in violations],
                },
                metadata={
                   "model_used":               self.model,
                    "total_validated":          len(courses),
                    "eligible_count":           len(eligible),
                    "ineligible_count":         len(ineligible),
                    "completed_credits_used":   completed_credits,
                    "in_progress_credits_used": in_progress_credits,
                    "effective_credits":        completed_credits + in_progress_credits, 
                }
            )
        
        except Exception as e:
            # print(f"[ConstraintAgent] ERROR during validation: {e}")
            traceback.print_exc()
            return AgentResponse(
                success=False,
                data=None,
                errors=[f"Constraint validation error: {str(e)}"]
            )
        
    async def check_single_course( 
        self, 
        course: Dict,
        state: ConversationState,
    ) -> AgentResponse:
        """
        Validate a single course using Orchestrator's _handle_course_info method when student ask about specific course. 

        Return AgentResponse with data = full constraint_check dict including standing penalty and standing_note. 
        """

        try:
            if not allowed_course(course, state) or course_level(course) == "graduate":
                return AgentResponse(success=True, data=graduate_check(course, state))
            completed, in_progress, completed_credits, in_progress_credits = (
                self._get_student_data(state)
            )
            has_transcript = state.transcript_data is not None

            prereq_results  = await self._batch_check_prereqs([course], completed, in_progress)
            prereq_result   = prereq_results.get(course.get("code", ""), self._safe_default())
            standing_result = self._check_standing(
                course, completed_credits, in_progress_credits, has_transcript
            )

            return AgentResponse(
                success=True,
                data={
                    **prereq_result,
                    "standing_penalty":  standing_result["penalty"],
                    "standing_note":     standing_result["note"],
                    "standing_eligible": standing_result["meets_standing"],
                },
                metadata={
                    "model_used": self.model,
                    "courses_checked": course.get("code")
                }
            )
        
        except Exception as e:
            # print(f"[ConstraintAgent] ERROR checking single courses: {e}")
            return AgentResponse(
                success=False,
                data=None,
                errors=[f"Single-course check error: {str(e)}"]
            )
        
    # FULL LLM PREREQ CHECK

    async def _batch_check_prereqs(
        self,
        courses: List[Dict],
        completed: set,
        in_progress: set
    ) -> Dict:
        # Step 1: DAG gives us ground truth eligibility
        dag_results = batch_check_eligibility(courses, self.dag, completed, in_progress)

        # Step 2: LLM enriches with reasoning and pathway suggestions, but CANNOT change eligibility
        course_lines = ""
        for i, course in enumerate(courses, 1):
            code = course.get("code", "")
            dag = dag_results.get(code, {})
            course_lines += (
                f"{i}. [{code}] {course.get('title', '')}\n"
                f"   Description: {course.get('description', 'No description available.')}\n"
                f"   DAG Eligibility (FIXED, do not change): eligible={dag.get('eligible')}, "
                f"   unmet={dag.get('unmet_prerequisites', [])}, "
                f"   met={dag.get('met_prerequisites', [])}\n\n"
            )

        prompt = f"""You are reviewing prerequisite results for a Rutgers CS student.
        
    The eligibility and unmet/met prerequisites have already been determined by a 
    prerequisite graph and are FIXED. You cannot change eligible, met_prerequisites, 
    or unmet_prerequisites. Your only job is to write a helpful pathway_suggestion 
    for each course.

    STUDENT PROFILE:
    - Completed: {sorted(completed) or ['none']}
    - In-progress: {sorted(in_progress) or ['none']}

    COURSES WITH FIXED ELIGIBILITY:
    {course_lines}

    Return ONLY valid JSON keyed by course code:
    {{
        "01:198:XXX": {{
            "eligible": <copy exactly from DAG result>,
            "met_prerequisites": <copy exactly from DAG result>,
            "unmet_prerequisites": <copy exactly from DAG result>,
            "pathway_suggestion": "your helpful 1-2 sentence suggestion, or null if eligible"
        }}
    }}
    """

        try:
            response = await self.run(prompt)
            response_text = response.messages[-1].contents[0].text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if not json_match:
                return dag_results  # fall back to pure DAG if LLM fails

            llm_results = json.loads(json_match.group())

            # Step 3: Merge — DAG values are authoritative, LLM only contributes pathway_suggestion
            final_results = {}
            for course in courses:
                code = course.get("code", "")
                dag = dag_results.get(code, self._safe_default())
                llm = llm_results.get(code, {})

                final_results[code] = {
                    "eligible":            dag["eligible"],              # DAG, always
                    "met_prerequisites":   dag["met_prerequisites"],     # DAG, always
                    "unmet_prerequisites": dag["unmet_prerequisites"],   # DAG, always
                    "pathway_suggestion":  llm.get("pathway_suggestion") # LLM only
                                        or dag.get("pathway_suggestion")
                }

            return final_results

        except Exception as e:
            print(f"[ConstraintAgent] LLM enrichment failed, using DAG only: {e}")
            return dag_results  # DAG is always the fallback
 
    def _safe_default(self) -> Dict:
        """
        Optimistic fallback if a course result is missing or unparseable.
        Treats the course as eligible so we never silently block a student.
        """
        return {
            "eligible": True,
            "met_prerequisites": [],
            "unmet_prerequisites": [],
            "reasoning": "Could not verify prerequisites — treated as eligible. Please confirm manually.",
            "pathway_suggestion": None
        }
    
    # Using arithmetic for standing, instead of LLM. -> Should be pointless, not natural language. -> look over further

    def _check_standing(
        self,
        course: Dict,
        completed_credits: float,
        in_progress_credits: float,
        has_transcript: bool
    ) -> Dict:
        """
        Assess whether the student's credit standing suits this course's level.
        In-progress credits count toward effective standing.
        """
        course_code  = course.get("code", "")
        course_level = self._extract_course_level(course_code)
 
        if course_level is None:
            return {"meets_standing": True, "penalty": 0.0, "note": "", "suggestion": None}
 
        expected_credits, standing_label = LEVEL_STANDING.get(
            course_level, (0, "open to all students")
        )
 
        if expected_credits == 0:
            return {"meets_standing": True, "penalty": 0.0, "note": "", "suggestion": None}
 
        if not has_transcript:
            return {
                "meets_standing": True,
                "penalty": 0.0,
                "note": (
                    f"This is a {course_level * 100}-level course typically suited for "
                    f"students with {standing_label}. "
                    f"Upload your transcript so I can verify your standing."
                ),
                "suggestion": None
            }
 
        effective_credits = completed_credits + in_progress_credits
        shortfall = max(0.0, expected_credits - effective_credits)
 
        if shortfall == 0:
            return {"meets_standing": True, "penalty": 0.0, "note": "", "suggestion": None}
 
        tiers_below = self._tiers_below(effective_credits, expected_credits)
        penalty = STANDING_PENALTY.get(min(tiers_below, 3), 0.60)
 
        note = (
            f"This {course_level * 100}-level course is typically taken by students with "
            f"{standing_label}. You currently have ~{effective_credits:.0f} credits "
            f"(including in-progress), which is about {shortfall:.0f} credits short of "
            f"the usual threshold — so it's ranked a bit lower for you."
        )
        suggestion = (
            f"You'll be a stronger candidate after gaining roughly "
            f"{shortfall:.0f} more credits to reach {standing_label}."
        )
 
        return {
            "meets_standing": False,
            "penalty": penalty,
            "note": note,
            "suggestion": suggestion
        }
 
    def _tiers_below(self, effective_credits: float, expected_credits: int) -> int:
        shortfall = max(0.0, expected_credits - effective_credits)
        return int(shortfall // 30) + (1 if shortfall % 30 > 0 else 0)
 
    def _extract_course_level(self, course_code: str) -> Optional[int]:
        """Extract hundreds digit from course code e.g. "01:198:314" → 3"""
        match = re.search(r':(\d{3})$', course_code)
        if match:
            return int(match.group(1)) // 100
        return None

    # Extract student data

    def _get_student_data(
        self, state: ConversationState
    ) -> Tuple[set, set, float, float]:
        """Pull completed/in-progress codes and credit counts from ConversationState."""
        if not state.transcript_data:
            return set(), set(), 0.0, 0.0
 
        td = state.transcript_data
        in_progress_courses = td.get("in_progress_courses", [])
 
        completed = (
            {c["code"] for c in td.get("completed_courses", []) if c.get("code")}
            | {c["code"] for c in td.get("transfer_courses", []) if c.get("code")}
            | {c["code"] for c in td.get("ap_credits", []) if c.get("code")}
        )
        in_progress = {c["code"] for c in in_progress_courses if c.get("code")}

        # print(f"[Debug] Transfer courses: {state.transcript_data.get('transfer_courses', [])}")
        # print(f"[Debug] Completed codes: {sorted(completed)}")
 
        completed_credits = float(td.get("total_degree_credits") or 0.0)
        in_progress_credits = sum(float(c["credits"]) for c in in_progress_courses)
 
        return completed, in_progress, completed_credits, in_progress_credits
    
    # helper functions below utilized in main methods above

    def _build_summary(
        self,
        eligible: List[Dict],
        ineligible: List[Dict],
        violations: List[ConstraintViolation]
    ) -> str:
        standing_warnings = [v for v in violations if v.constraint_type == "credit_standing"]
 
        if not ineligible and not standing_warnings:
            total = len(eligible) + len(ineligible)
            return f"All {total} recommended course(s) are a great fit for your current standing."
 
        parts = []
        if eligible:
            parts.append(
                f"{len(eligible)} course(s) available: "
                f"{', '.join(c.get('code', '') for c in eligible)}."
            )
        if ineligible:
            parts.append(
                f"{len(ineligible)} course(s) with unmet prerequisites: "
                f"{', '.join(c.get('code', '') for c in ineligible)}."
            )
        if standing_warnings:
            flagged = [v.affected_courses[0] for v in standing_warnings if v.affected_courses]
            parts.append(
                f"{len(standing_warnings)} course(s) ranked lower due to credit standing: "
                f"{', '.join(flagged)}."
            )
        return " ".join(parts)
    
    @staticmethod
    def summarize_for_prompt(validation_data: Dict) -> str:
        """
        Compact string for injecting constraint results into Orchestrator prompts.
        Mirrors TranscriptAgent.summarize_for_prompt() style.
        """
        eligible   = validation_data.get("eligible_courses", [])
        ineligible = validation_data.get("ineligible_courses", [])
        violations = validation_data.get("violations", [])
 
        prereq_violations = [v for v in violations if v["constraint_type"] == "prerequisite"]
 
        eligible_str   = ", ".join(c.get("code", "") for c in eligible)   or "None"
        unverified_str = "; ".join(c.get("code", "") + ": " + c.get("constraint_check", {}).get("standing_note", "Requires verification")
                                   for c in validation_data.get("unverified_courses", [])) or "None"
        ineligible_str = ", ".join(c.get("code", "") for c in ineligible) or "None"
 
        penalty_lines = ""
        for course in eligible:
            chk     = course.get("constraint_check", {})
            penalty = chk.get("standing_penalty", 0.0)
            note    = chk.get("standing_note", "")
            if penalty > 0 and note:
                penalty_lines += f"  - {course.get('code')}: {note}\n"
 
        violation_lines = "\n".join(
            f"  - {v['message']}"
            + (f" → {v['suggestion']}" if v.get("suggestion") else "")
            for v in prereq_violations
        )
 
        return (
            f"PREREQUISITE & STANDING VALIDATION:\n"
            f"  Prereq-Eligible:  {eligible_str}\n"
            f"  Eligibility UNKNOWN: {unverified_str}\n"
            f"  Missing Prereqs:  {ineligible_str}\n"
            f"  Summary:          {validation_data.get('validation_summary', '')}\n"
            + (f"  Prereq Violations:\n{violation_lines}\n" if violation_lines else "")
            + (
                f"  Credit Standing Notes (these courses are ranked lower):\n{penalty_lines}"
                if penalty_lines else ""
            )
        )

    # Notes for writing: 

    # Design decision made where we inject all courses of interest into 1 prompt and validate all courses with 1 LLM call instead of 10 LLM calls -> decreases cost, especially for low resource
    # project. 
    # the note above is for prereq checker.
