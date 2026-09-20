import json
from dataclasses import dataclass, field
from typing import Optional

from agent_framework.openai import OpenAIResponsesClient
from agent_framework import Executor, WorkflowContext, handler

from agents2.shared_types import ConversationState
from agents2.advising_scope import mentions_campus, without_campus
from agents2.inference_metrics import instrument_capability, usage_scope


# -----------------------------------------------------------------------------
# Message types
# -----------------------------------------------------------------------------

class UserQuery:
    def __init__(self, user_query: str, conversation_state: ConversationState):
        self.user_query = user_query
        self.conversation_state = conversation_state


class OrchestratorRequest:
    def __init__(self, user_query: str, parsed_data: dict, conversation_state: ConversationState):
        self.user_query = user_query
        self.parsed_data = parsed_data
        self.conversation_state = conversation_state


class AgentResult:
    def __init__(
        self,
        user_query: str,
        parsed_data: dict,
        agent_name: str,
        data: dict,
        conversation_state: ConversationState,
    ):
        self.user_query = user_query
        self.parsed_data = parsed_data
        self.agent_name = agent_name
        self.data = data
        self.conversation_state = conversation_state


# -----------------------------------------------------------------------------
# Agent registry
# -----------------------------------------------------------------------------

AGENT_REGISTRY = {
    "transcript": {
        "description": "Parses an uploaded transcript. Call only for transcript_upload intent. Terminal.",
        "terminal": True,
    },
    "data_fetch": {
        "description": "Retrieves candidate courses matching interests/filters. Use for course recommendations.",
        "terminal": False,
    },
    "data_lookup": {
        "description": "Looks up a specific course by name or ID. Use for course_info intent.",
        "terminal": False,
    },
    "data_prereq": {
        "description": "Retrieves prerequisites and deterministic BFS shortest-course / DFS alternative pathways, transcript eligibility, and eligible courses to explore. Use for prerequisite_check and how-to-reach-a-course questions.",
        "terminal": False,
    },
    "constraint_full": {
        "description": "Validates recommendation candidates against the student's transcript. Requires data_fetch and transcript data.",
        "terminal": False,
    },
    "constraint_prereq": {
        "description": "Checks whether transcript courses satisfy prerequisites for one course. Requires data_prereq and transcript data.",
        "terminal": False,
    },
    "planning": {
        "description": "Ranks/tailors recommendation candidates. Requires data_fetch; if a transcript exists, run after constraint_full.",
        "terminal": False,
    },
}

AGENT_REGISTRY_SUMMARY = "\n".join(
    f"- {name}: {meta['description']}"
    for name, meta in AGENT_REGISTRY.items()
)


# -----------------------------------------------------------------------------
# Routing context
# -----------------------------------------------------------------------------

@dataclass
class RoutingContext:
    user_query: str
    parsed_data: dict
    has_transcript: bool
    resolved_courses: dict = field(default_factory=dict)
    accumulated_results: dict[str, dict] = field(default_factory=dict)
    agents_call_order: list[str] = field(default_factory=list)
    resolved_semester: dict = field(default_factory=dict)
    transcript_summary: str = ""
    session_memory: dict = field(default_factory=dict)

    def _slim_results(self) -> dict:
        """Keep router context useful without dumping the whole course corpus."""
        slim = {}

        for key, val in self.accumulated_results.items():
            if isinstance(val, dict) and "courses" in val:
                slim_courses = []
                for c in val.get("courses", [])[:5]:
                    actual = c.get("course", c) if isinstance(c, dict) else {}
                    slim_courses.append({
                        "code": actual.get("code"),
                        "title": actual.get("title"),
                        "prerequisites": (
                            actual.get("prerequisites")
                            or actual.get("description", "")
                        ),
                    })

                # Preserve constraint/planning metadata when it exists.
                extra = {
                    k: v
                    for k, v in val.items()
                    if k != "courses"
                }
                slim[key] = {"courses": slim_courses, **extra}
            else:
                slim[key] = val

        return slim

    def _slim_parsed_data(self) -> dict:
        d = {
            k: v
            for k, v in self.parsed_data.items()
            if v not in (None, [], {}, "")
        }

        if "entities" in d and isinstance(d["entities"], dict):
            d["entities"] = {
                k: v
                for k, v in d["entities"].items()
                if v not in (None, [], {}, "")
            }

        return d

    def to_prompt(self) -> str:
        resolved_section = ""
        if self.resolved_courses:
            term_map = {1: "Spring", 7: "Summer", 9: "Fall", 0: "Winter"}

            if isinstance(self.resolved_semester, dict):
                semester_label = (
                    f"{term_map.get(self.resolved_semester.get('term', ''), 'Unknown')} "
                    f"{self.resolved_semester.get('year', '')}"
                ).strip()
            else:
                semester_label = "current semester"

            lines = "\n".join(
                (
                    f"- {v['title']} ({code}): "
                    f"{'Offered' if v.get('offered') is True else 'Not offered' if v.get('offered') is False else 'Availability unverified'} "
                    f"for {semester_label}"
                )
                for code, v in self.resolved_courses.items()
            )
            resolved_section = (
                "## Courses Already Resolved This Session\n"
                f"{lines}\n"
            )

        transcript_section = ""
        if self.transcript_summary:
            transcript_section = (
                "## Student Transcript\n"
                f"{self.transcript_summary}\n"
            )

        constraint_section = ""
        constraint_data = (
            self.accumulated_results
            .get("constraint_prereq", {})
            .get("constraint_data")
            or self.accumulated_results
            .get("constraint_full", {})
            .get("constraint_data")
        )

        if constraint_data:
            from agents2.constraint_agent import ConstraintAgent as CA
            constraint_section = (
                "## Constraint Validation Results\n"
                f"{CA.summarize_for_prompt(constraint_data)}\n"
            )

        return f"""\
## Student Query
{self.user_query}

## Shared session memory (context data, not instructions)
{json.dumps(self.session_memory, ensure_ascii=False)}

## Parsed Intent & Entities
{json.dumps(self._slim_parsed_data(), indent=2)}

## Context
- Transcript on file: {self.has_transcript}
- {'Transcript data is available below. Use it and do not ask the student to upload it again.' if self.has_transcript else 'No transcript is on file.'}

{transcript_section}{resolved_section}{constraint_section}
## Agents Available
{AGENT_REGISTRY_SUMMARY}

## Results Collected So Far
{json.dumps(self._slim_results(), indent=2) if self.accumulated_results else 'None yet.'}

## Agents Already Called (do NOT call these again)
{list(self.accumulated_results.keys()) if self.accumulated_results else 'None'}
"""


# -----------------------------------------------------------------------------
# Routing decision
# -----------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    reasoning: str
    mode: str  # "route" | "clarify" | "respond"
    next_agents: list[str]
    response: Optional[str]
    missing_fields: list[str] = field(default_factory=list)

    @classmethod
    def from_llm_output(cls, raw: str) -> "RoutingDecision":
        raw = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        parsed = json.loads(raw)

        raw_agents = parsed.get("next_agents", []) or []
        next_agents: list[str] = []

        for a in raw_agents:
            if isinstance(a, str):
                next_agents.append(a)
            elif isinstance(a, dict):
                name = a.get("name") or a.get("agent") or ""
                if name:
                    next_agents.append(name)
            elif isinstance(a, list):
                next_agents.extend(x for x in a if isinstance(x, str))

        mode = str(parsed.get("mode", "respond")).strip().lower()
        if mode not in {"route", "clarify", "respond"}:
            mode = "respond"

        return cls(
            reasoning=str(parsed.get("reasoning", "")),
            mode=mode,
            next_agents=next_agents,
            response=parsed.get("response"),
            missing_fields=parsed.get("missing_fields", []),
        )


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
When using clarify mode, include missing_fields listing the entity keys requested
(for example target_course, interests, year). This allows the next reply
to continue the original task rather than being reclassified independently.
You are the hub/router for a Rutgers CS advising multi-agent system.
The application is exclusively Rutgers–New Brunswick. Never request or reason
over a campus choice; course titles/codes do not require one. This also applies
to clarification questions and follow-up suggestions in respond mode.
You make dynamic routing decisions based on the student's query, parsed intent,
available transcript/context, and results already collected.

You are NOT a fixed pipeline. Different query types should take different paths.
However, you must obey agent dependencies and must never repeat completed work.

Each turn you receive a complete context snapshot and must output one JSON decision.

## Modes
- "route"   — call a specialist agent because more work is needed.
- "clarify" — ask one concise user-facing question only when information genuinely
              required to proceed is missing.
- "respond" — enough evidence has been collected; answer the student.

## General Routing Principles
- Route based on the parsed intent and the evidence currently available.
- Never call an agent listed under "Agents Already Called".
- Never call an agent whose required upstream data is missing.
- next_agents must be a flat list of agent-name strings.
- Prefer the smallest useful path. Do not call agents that do not contribute to
  answering the current query.
- A subject/topic/course mention is course-related; do not redirect it as unrelated.
- Ignore parser missing_critical_info when the user's actual query already contains
  enough information to proceed.
- If the student is answering a previous clarification, proceed with the new
  information rather than asking the same question again.

## Intent-specific guidance
### course_recommendation
- data_fetch is the retrieval step for recommendation candidates.
- planning is REQUIRED before a final recommendation response.
- If transcript data is available, constraint_full is REQUIRED after data_fetch and
  before planning so recommendations can account for eligibility.
- If no transcript exists, planning runs directly on data_fetch results.
- Do NOT use data_lookup as a substitute for data_fetch in recommendation flows.
- If no interest/topic/filter is provided and no useful prior context exists, you may
  clarify by asking what the student is interested in.

### course_info
- Use data_lookup for a named/specific course.
- If the requested course is already resolved in session context and the requested
  fact is present, respond directly without another lookup.

### prerequisite_check
- Use data_prereq to retrieve prerequisite information.
- This includes fastest/alternative paths and how to reach a target course.
- data_prereq includes deterministic pathways; use them without inventing steps.
- If transcript data exists and the student asks whether THEY satisfy the
  prerequisites, use constraint_prereq after data_prereq.
- Without a transcript, explain the prerequisite information without pretending to
  know whether the student personally satisfies it.

### transcript_upload
- Transcript routing is handled outside this router. Never emit transcript in
  next_agents from this routing loop.

## Completion Rules
- Do not respond before mandatory work for the current intent is complete.
- course_recommendation without transcript is complete after data_fetch + planning.
- course_recommendation with transcript is complete after
  data_fetch + constraint_full + planning.
- course_info is normally complete after data_lookup.
- prerequisite_check is normally complete after data_prereq, plus constraint_prereq
  when transcript-based personal eligibility is being evaluated.
- Once the required work is complete, RESPOND. Do not route back to a completed agent.

## Response Rules
- Only state course facts and prerequisites supported by collected data/context.
- Never invent course names, codes, prerequisite relationships, availability, or
  eligibility.
- If transcript data is available, use it explicitly when relevant and do not ask the
  student to provide it again.
- Never infer that a prerequisite is satisfied merely because the student took a
  more advanced or similar course.
- If constraint results say a prerequisite is missing, treat it as missing.
- When saying prerequisites are met, mention the actual completed courses that
  establish this when those facts are available.
- When mentioning a course code, also provide the matching course name when the data
  contains it. If the name is unavailable, do not guess.
- For recommendation/planning responses, explain briefly why choices are ranked that
  way.
- If planning returned not_recommended courses, mention them briefly as future options
  and explain the blocking prerequisite/constraint when available.
- If user-provided details conflict with transcript/context, flag the discrepancy
  gently and state what the available data shows.

## Recommendation response format
Use readable spacing, for example:

1) Course Name (Course Code)
   Brief recommendation reasoning.
   Prerequisites: X, Y, Z.

2) Next Course (Course Code)
   Brief recommendation reasoning.
   Prerequisites: A, B.

## Output Format
Return JSON only, with no markdown fence:
{
  "reasoning": "<1-2 concise sentences>",
  "mode": "route" | "clarify" | "respond",
  "next_agents": [],
  "response": null
}

When mode="route": next_agents must contain the desired agent(s), response=null.
When mode="clarify" or mode="respond": next_agents=[], response must be a non-null
plain conversational string.

IMPORTANT: produce valid JSON. Use \\n inside JSON string values instead of literal
newlines.
"""


FINAL_RESPONSE_SYSTEM_PROMPT = """\
This application is exclusively Rutgers–New Brunswick. Never ask users for campus
or make a campus choice a prerequisite for advising or resolving a course.
You are the final response writer for a Rutgers CS advising system.
Undergraduate and graduate students have strictly separate course catalogs.
Only discuss/recommend courses supplied for the student's academic_level; never
suggest cross-level enrollment or undergraduate prerequisite enrollment steps to graduates.
Graduate eligibility marked requires_verification or null is UNKNOWN, never eligible.
Explain graduate prerequisite background as a requirement to verify, not an enrollment plan.
For historical/incomplete course records disclose verification_status and do not invent credits.
Do not treat all graduate courses as counting toward every graduate degree.

The routing/orchestration phase is already over. You CANNOT route to agents and you
must never output routing JSON, next_agents, or tool instructions.

Write the final answer directly to the student using ONLY the supplied query,
parsed context, transcript summary, and collected agent results.

Rules:
- For pathway questions, use data_prereq.pathways: show stages for the shortest
  course-count plan, and alternatives when requested. Distinguish fewest courses
  from fewest prerequisite rounds among returned plans. Only claim a proven
  minimum course count when shortest_proven is true.
- Stages are hypothetical dependency rounds, not verified semesters: offerings,
  credit limits, durations, and corequisites are not modeled. Explain search
  limits, missing catalog nodes, cycles, or permission requirements when relevant.
- If plans is empty but conditional_plans exists, explain the known CS sequence
  together with its external_requirements_to_verify. Those external courses must
  be completed first and their prerequisite chains/time are unknown. Never call
  a conditional plan complete, guaranteed fastest, or evidence of eligibility.
- In-progress credit is conditional on passing. Never present it as completed.
  For current eligibility use eligible_from_completed_courses in pathway data;
  if constraint results disagree, disclose the difference instead of asserting
  unconditional eligibility. Without a transcript, plans start from zero credit
  and eligible_to_explore is hypothetical, not personal eligibility.
- Use eligible_to_explore only as prerequisite-based possibilities, not guarantees
  of enrollment or ranked interest matches. Do not invent missing course titles.
- Never invent course names, course codes, prerequisites, availability, eligibility,
  or transcript facts.
- If evidence is incomplete, say what is known and what remains unknown.
- If transcript data is present, use it when relevant and do not ask for it again.
- If constraint data says a prerequisite is missing, treat it as missing.
- When a course code and course name are both available, include both.
- For recommendations, explain why the options were ranked and keep the response
  readable with spacing between courses.
- Be concise, helpful, and encouraging.
- Output plain user-facing text only. Never output JSON.
"""


# -----------------------------------------------------------------------------
# Orchestrator executor
# -----------------------------------------------------------------------------

class OrchestratorExecutor(Executor):
    """
    Hub-and-spoke orchestrator.

    The LLM remains the dynamic router. Python only enforces safety/control-flow
    invariants:
      - no duplicate agent calls
      - no impossible dependency order
      - mandatory steps for intents that require them
      - no routing JSON leaking to the user
    """

    MAX_ITERATIONS = 8

    def __init__(self, chat_client: OpenAIResponsesClient, model_id: str, response_model_id: str):
        super().__init__(id="orchestrator")

        self.agent = chat_client.as_agent(
            instructions=ORCHESTRATOR_SYSTEM_PROMPT,
            name="Orchestrator",
            default_options={"model_id": model_id},
        )

        # Separate writer so a fallback can NEVER decide to route again.
        self.response_agent = chat_client.as_agent(
            instructions=FINAL_RESPONSE_SYSTEM_PROMPT,
            name="AdvisorResponseWriter",
            default_options={"model_id": response_model_id},
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    @handler
    @instrument_capability
    async def handle_request(
        self,
        message: OrchestratorRequest,
        ctx: WorkflowContext[AgentResult],
    ) -> None:
        intent = message.parsed_data.get("intent")
        has_transcript = bool(message.conversation_state.transcript_data)

        # Transcript upload is intentionally handled outside the dynamic router.
        if intent == "transcript_upload":
            await ctx.send_message(
                AgentResult(
                    message.user_query,
                    message.parsed_data,
                    agent_name="transcript",
                    data={},
                    conversation_state=message.conversation_state,
                )
            )
            return

        # Re-resolve semester if the user mentions one in a follow-up.
        from agents2.data_agent import DataAgent as DA

        if any(
            k in message.user_query.lower()
            for k in ["next", "spring", "fall", "summer", "winter"]
        ):
            new_semester = DA.resolve_semester(None, message.user_query)
            if new_semester != message.conversation_state.resolved_semester:
                message.conversation_state.resolved_semester = new_semester

                saved_courses = {
                    code: v
                    for code, v in message.conversation_state.resolved_courses.items()
                }
                message.conversation_state.resolved_courses = {}

                for code, v in saved_courses.items():
                    message.conversation_state.resolved_courses[code] = {
                        **v,
                        "offered": None,
                    }

        # Parser resolves references against ordered recent result sets, not
        # the accumulated catalog lookup cache.

        routing_ctx = RoutingContext(
            user_query=message.user_query,
            parsed_data=message.parsed_data,
            has_transcript=has_transcript,
            session_memory=message.conversation_state.get_context("orchestrator"),
            resolved_courses=message.conversation_state.resolved_courses,
            resolved_semester=message.conversation_state.resolved_semester or {},
        )

        if message.conversation_state.transcript_data:
            from agents2.transcript_agent import TranscriptAgent

            routing_ctx.transcript_summary = TranscriptAgent.summarize_for_prompt(
                message.conversation_state.transcript_data
            )

        message.conversation_state.routing_ctx = routing_ctx

        await self._routing_loop(
            routing_ctx,
            message,
            ctx,
            iteration=0,
        )

    # ------------------------------------------------------------------
    # Spoke result collector
    # ------------------------------------------------------------------

    @handler
    @instrument_capability
    async def handle_result(
        self,
        message: AgentResult,
        ctx: WorkflowContext[AgentResult],
    ) -> None:
        if message.agent_name == "transcript":
            return

        routing_ctx: RoutingContext = message.conversation_state.routing_ctx
        routing_ctx.accumulated_results[message.agent_name] = message.data
        memory_kind = {"planning": "recommendations", "data_lookup": "lookup_courses",
                       "data_prereq": "pathway_targets"}.get(message.agent_name)
        if memory_kind:
            message.conversation_state.remember_results(memory_kind, message.data.get(
                "ranked_courses" if message.agent_name == "planning" else "courses", []))
        routing_ctx.session_memory = message.conversation_state.get_context("orchestrator")
        routing_ctx.has_transcript = bool(
            message.conversation_state.transcript_data
        )

        if message.agent_name == "data_fetch":
            semester = message.data.get("semester")
            if semester:
                message.conversation_state.resolved_semester = semester
                routing_ctx.resolved_semester = semester

            for course in message.data.get("courses", []):
                actual = course.get("course", course)
                code = actual.get("code")
                title = actual.get("title", "")

                if code:
                    # Preserve current behavior. If data_fetch later exposes an
                    # explicit offered flag, prefer that instead of hard-coding True.
                    offered = course.get("offered", True) if isinstance(course, dict) else True
                    message.conversation_state.resolve_course(
                        code,
                        title,
                        offered,
                        message.conversation_state.resolved_semester or {},
                    )

        elif message.agent_name == "data_lookup":
            for course_result in message.data.get("courses", []):
                actual = course_result.get("course", {})
                code = actual.get("code")
                title = actual.get("title", "")
                offered = course_result.get("offered")
                semester = course_result.get(
                    "semester",
                    message.conversation_state.resolved_semester or {},
                )

                if code:
                    message.conversation_state.resolve_course(
                        code,
                        title,
                        offered,
                        semester,
                    )

        iteration = int(
            getattr(message.conversation_state, "routing_iteration", 0)
        )

        await self._routing_loop(
            routing_ctx,
            message,
            ctx,
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _required_next_agent(self, routing_ctx: RoutingContext) -> Optional[str]:
        """
        Return only HARD mandatory work that must still happen.

        This does not replace the LLM router. It is a guardrail that prevents the
        router from responding too early or skipping required dependencies.
        """
        intent = routing_ctx.parsed_data.get("intent")
        called = set(routing_ctx.accumulated_results.keys())

        if intent == "course_recommendation":
            if "data_fetch" not in called:
                return "data_fetch"

            if routing_ctx.has_transcript and "constraint_full" not in called:
                return "constraint_full"

            if "planning" not in called:
                return "planning"

            return None

        if intent == "course_info":
            # A follow-up like "what about those?" may refer entirely to courses
            # already resolved this session. Only skip lookup when the requested
            # course references actually match cached names/codes.
            entities = routing_ctx.parsed_data.get("entities", {}) or {}
            requested = entities.get("specific_courses", []) or []
            if isinstance(requested, str):
                requested = [requested]

            cached = set()
            for code, info in routing_ctx.resolved_courses.items():
                cached.add(str(code).strip().lower())
                title = str(info.get("title", "")).strip().lower()
                if title:
                    cached.add(title)

            if requested and all(str(x).strip().lower() in cached for x in requested):
                return None

            if "data_lookup" not in called:
                return "data_lookup"
            return None

        if intent == "prerequisite_check":
            if "data_prereq" not in called:
                return "data_prereq"

            # Only make transcript validation mandatory when the student is asking
            # about THEIR eligibility. A generic "what are the prerequisites?"
            # should not need the constraint agent just because a transcript happens
            # to be on file.
            q = routing_ctx.user_query.lower()
            personal_eligibility_markers = (
                "can i take",
                "can i enroll",
                "am i eligible",
                "eligible for",
                "do i meet",
                "do i satisfy",
                "do i have the prereq",
                "do i have the prerequisite",
                "have i met",
                "did i meet",
                "qualify for",
            )
            asks_personal_eligibility = any(
                marker in q for marker in personal_eligibility_markers
            )

            if (
                routing_ctx.has_transcript
                and asks_personal_eligibility
                and "constraint_prereq" not in called
            ):
                return "constraint_prereq"

            return None

        # Unknown/other intents remain fully LLM-routed.
        return None

    def _intent_allows_agent(
        self,
        routing_ctx: RoutingContext,
        agent_name: str,
    ) -> bool:
        """Block obviously irrelevant cross-intent routes while staying dynamic."""
        intent = routing_ctx.parsed_data.get("intent")

        allowed_by_intent = {
            "course_recommendation": {
                "data_fetch",
                "constraint_full",
                "planning",
            },
            "course_info": {
                "data_lookup",
            },
            "prerequisite_check": {
                "data_prereq",
                "constraint_prereq",
            },
        }

        allowed = allowed_by_intent.get(intent)
        return True if allowed is None else agent_name in allowed

    def _dependency_fallback(
        self,
        routing_ctx: RoutingContext,
        agent_name: str,
    ) -> Optional[str]:
        """Return the missing prerequisite step for a proposed route, if any."""
        called = set(routing_ctx.accumulated_results.keys())

        if agent_name == "constraint_full":
            if not routing_ctx.has_transcript:
                return None
            if "data_fetch" not in called:
                return "data_fetch"

        elif agent_name == "planning":
            if "data_fetch" not in called:
                return "data_fetch"
            if routing_ctx.has_transcript and "constraint_full" not in called:
                return "constraint_full"

        elif agent_name == "constraint_prereq":
            if not routing_ctx.has_transcript:
                return None
            if "data_prereq" not in called:
                return "data_prereq"

        return None

    def _agent_dependencies_met(
        self,
        routing_ctx: RoutingContext,
        agent_name: str,
    ) -> bool:
        called = set(routing_ctx.accumulated_results.keys())

        if agent_name == "constraint_full":
            return routing_ctx.has_transcript and "data_fetch" in called

        if agent_name == "planning":
            if "data_fetch" not in called:
                return False
            if routing_ctx.has_transcript and "constraint_full" not in called:
                return False
            return True

        if agent_name == "constraint_prereq":
            return routing_ctx.has_transcript and "data_prereq" in called

        return True

    def _normalize_route(
        self,
        routing_ctx: RoutingContext,
        decision: RoutingDecision,
        interventions: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Keep the router's choice when it is valid; repair only invalid/impossible
        choices. Sequential execution is still enforced one spoke at a time.
        """
        called = set(routing_ctx.accumulated_results.keys())
        reasons = interventions if interventions is not None else []

        candidates = []
        for raw_name in decision.next_agents:
            name = raw_name.strip()

            if not name:
                reasons.append("invalid_agent")
                continue
            if name not in AGENT_REGISTRY:
                reasons.append("invalid_agent")
                continue
            if name == "transcript":
                reasons.append("intent_or_capability_block")
                continue
            if name in called:
                reasons.append("duplicate_agent")
                continue
            if not self._intent_allows_agent(routing_ctx, name):
                reasons.append("intent_or_capability_block")
                continue

            if self._agent_dependencies_met(routing_ctx, name):
                candidates.append(name)
                continue

            # If the LLM chose a sensible downstream agent too early, route to its
            # missing dependency instead of killing the turn.
            dependency = self._dependency_fallback(routing_ctx, name)
            reasons.append("missing_dependency")
            if (
                dependency
                and dependency not in called
                and dependency in AGENT_REGISTRY
                and self._intent_allows_agent(routing_ctx, dependency)
            ):
                candidates.append(dependency)

        # If the LLM route was unusable, use only a hard-required step if one exists.
        if not candidates:
            required = self._required_next_agent(routing_ctx)
            if required:
                reasons.append("mandatory_step")
            if (
                required
                and required not in called
                and self._agent_dependencies_met(routing_ctx, required)
            ):
                candidates.append(required)
            elif required:
                dependency = self._dependency_fallback(routing_ctx, required)
                if dependency and dependency not in called:
                    candidates.append(dependency)

        # Preserve order and de-duplicate.
        deduped = list(dict.fromkeys(candidates))
        if len(deduped) < len(candidates):
            reasons.append("duplicate_agent")
        if len(deduped) > 1:
            reasons.append("sequential_execution")
        return deduped[:1]

    # ------------------------------------------------------------------
    # Core routing loop
    # ------------------------------------------------------------------

    async def _routing_loop(
        self,
        routing_ctx: RoutingContext,
        message,
        ctx: WorkflowContext,
        iteration: int,
    ) -> None:
        if iteration >= self.MAX_ITERATIONS:
            message.conversation_state.routing_events.append({
                "iteration": iteration, "proposed_agents": [], "executed_agents": [],
                "accepted_unchanged": False, "interventions": ["max_iterations"]})
            await self._force_respond(
                routing_ctx,
                ctx,
                conversation_state=message.conversation_state,
                reason="Maximum routing iterations reached.",
            )
            return

        raw = await self.agent.run(
            routing_ctx.to_prompt(),
            thread=None,
        )

        raw_text = raw.content if hasattr(raw, "content") else str(raw)

        try:
            decision = RoutingDecision.from_llm_output(raw_text)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            message.conversation_state.routing_events.append({
                "iteration": iteration, "proposed_agents": [], "executed_agents": [],
                "accepted_unchanged": False, "interventions": ["invalid_router_output"]})
            await self._force_respond(
                routing_ctx,
                ctx,
                conversation_state=message.conversation_state,
                reason=f"Router returned invalid JSON: {exc}",
            )
            return

        # A hard-required step takes precedence over an early respond decision.
        event = {"iteration": iteration, "proposed_mode": decision.mode,
                 "proposed_agents": list(decision.next_agents or []),
                 "executed_agents": [], "interventions": [], "accepted_unchanged": False}
        message.conversation_state.routing_events.append(event)
        required = self._required_next_agent(routing_ctx)

        if decision.mode == "clarify" and (mentions_campus(decision.response) or
                                           any(mentions_campus(f) for f in (decision.missing_fields or []))):
            decision.missing_fields = without_campus(decision.missing_fields or [])
            entities = routing_ctx.parsed_data.get("entities", {})
            if not decision.missing_fields and (entities.get("target_course") or entities.get("specific_courses")):
                event["interventions"].append("out_of_scope_clarification")
                if required:
                    decision = RoutingDecision("Continue scoped task", "route", [required], None)
                else:
                    await self._force_respond(routing_ctx, ctx, message.conversation_state,
                                              reason="Answer the resolved task; no campus selection is needed.")
                    return
            else:
                decision.missing_fields = decision.missing_fields or ["target_course"]
                decision.response = "Please specify: " + ", ".join(decision.missing_fields).replace("_", " ") + "."
                event["interventions"].append("out_of_scope_clarification")

        if decision.mode == "clarify":
            # Clarification is allowed only before useful work has begun. If hard
            # work is already clearly required/possible, continue the workflow.
            if required and routing_ctx.accumulated_results:
                event["interventions"].append("early_clarification_blocked")
                decision = RoutingDecision(
                    reasoning=(
                        "Router attempted to clarify after workflow execution had "
                        "already begun; continuing required work."
                    ),
                    mode="route",
                    next_agents=[required],
                    response=None,
                )
            elif decision.response:
                message.conversation_state.request_clarification(
                    routing_ctx.user_query, routing_ctx.parsed_data, decision.missing_fields, decision.response)
                event["accepted_unchanged"] = True
                await ctx.yield_output(decision.response)
                return
            else:
                await self._force_respond(
                    routing_ctx,
                    ctx,
                    conversation_state=message.conversation_state,
                    reason="Clarify mode had no response text.",
                )
                return

        if decision.mode == "respond":
            if required:
                event["interventions"].append("early_response_blocked")
                # Do not let the router skip mandatory work. This still leaves all
                # non-mandatory route choice to the LLM.
                decision = RoutingDecision(
                    reasoning=(
                        f"Cannot respond yet; required agent '{required}' has not "
                        "completed."
                    ),
                    mode="route",
                    next_agents=[required],
                    response=None,
                )
            elif decision.response:
                # The router's response field is already defined as a complete,
                # user-facing answer. Yield it directly: one fewer LLM call, lower
                # cost, and no chance for a second router-style JSON wrapper.
                event["accepted_unchanged"] = True
                await ctx.yield_output(decision.response)
                return
            else:
                await self._force_respond(
                    routing_ctx,
                    ctx,
                    conversation_state=message.conversation_state,
                    reason="Respond mode had no response text.",
                )
                return

        if decision.mode != "route":
            await self._force_respond(
                routing_ctx,
                ctx,
                conversation_state=message.conversation_state,
                reason=f"Unexpected router mode: {decision.mode}",
            )
            return

        valid_agents = self._normalize_route(routing_ctx, decision, event["interventions"])

        if not valid_agents:
            # Typical case here: router tried to call a duplicate after all mandatory
            # work was already complete. Finalize instead of exposing routing JSON.
            await self._force_respond(
                routing_ctx,
                ctx,
                conversation_state=message.conversation_state,
                reason=(
                    "Router proposed no valid new agent. Completed/invalid routes "
                    "were filtered out."
                ),
            )
            return

        message.conversation_state.routing_ctx = routing_ctx
        message.conversation_state.routing_iteration = iteration + 1

        for agent_name in valid_agents:
            spoke_data = self._build_spoke_data(
                routing_ctx,
                agent_name,
            )

            # If a required upstream agent returned no usable data, produce the best
            # grounded response available instead of dispatching an empty payload.
            if agent_name == "planning" and not spoke_data.get("courses"):
                event["interventions"].append("empty_candidates")
                await self._force_respond(
                    routing_ctx,
                    ctx,
                    conversation_state=message.conversation_state,
                    reason="Planning had no candidate courses to rank.",
                )
                return

            routing_ctx.agents_call_order.append(agent_name)
            event["executed_agents"].append(agent_name)
            event["accepted_unchanged"] = (not event["interventions"] and
                                           event["proposed_agents"] == event["executed_agents"])
            await ctx.send_message(
                AgentResult(
                    routing_ctx.user_query,
                    routing_ctx.parsed_data,
                    agent_name=agent_name,
                    data=spoke_data,
                    conversation_state=message.conversation_state,
                )
            )

    # ------------------------------------------------------------------
    # Spoke payload construction
    # ------------------------------------------------------------------

    def _build_spoke_data(
        self,
        routing_ctx: RoutingContext,
        agent_name: str,
    ) -> dict:
        if agent_name == "constraint_full":
            return dict(
                routing_ctx.accumulated_results.get("data_fetch", {})
            )

        if agent_name == "planning":
            # With transcript: planning ranks validated/eligible candidates.
            if routing_ctx.has_transcript:
                constraint_result = routing_ctx.accumulated_results.get(
                    "constraint_full",
                    {},
                )
                constraint_data = constraint_result.get(
                    "constraint_data",
                    {},
                )

                courses = (
                    (constraint_data.get("eligible_courses", []) + constraint_data.get("unverified_courses", []))
                    or constraint_result.get("courses", [])
                )

                return {
                    "courses": courses,
                    "constraint_data": constraint_data,
                }

            # Without transcript: planning ranks data_fetch results directly.
            fetch_result = routing_ctx.accumulated_results.get(
                "data_fetch",
                {},
            )

            return {
                "courses": fetch_result.get("courses", []),
                "constraint_data": {},
            }

        # constraint_prereq expects access to data_prereq by key; the data agents
        # ignore this payload and rely on parsed_data/state, so accumulated results
        # are a safe generic payload for remaining spokes.
        return dict(routing_ctx.accumulated_results)

    # ------------------------------------------------------------------
    # Final response fallback
    # ------------------------------------------------------------------

    async def _force_respond(
        self,
        routing_ctx: RoutingContext,
        ctx: WorkflowContext[AgentResult],
        conversation_state: Optional[ConversationState] = None,
        reason: str = "",
    ) -> None:
        """
        Guaranteed terminal response path.

        IMPORTANT: this uses a separate response-only agent. The router is never
        called here, so a fallback cannot return mode='route' and leak JSON.
        """
        prompt = f"""\
## Student query
{routing_ctx.user_query}

## Parsed/context snapshot
{routing_ctx.to_prompt()}

## Full collected agent results
{json.dumps(routing_ctx.accumulated_results, indent=2)}

## Internal termination reason
{reason or 'The routing phase is complete.'}

Write the best grounded final response to the student now. Do not mention the
internal termination reason unless it is directly useful to the student.
"""

        if conversation_state is not None:
            conversation_state.routing_events.append({"terminal_fallback": reason})
        with usage_scope(conversation_state, "final_response"):
            raw = await self.response_agent.run(prompt, thread=None)

        text = raw.content if hasattr(raw, "content") else str(raw)
        text = text.strip()

        if not text:
            text = (
                "I wasn't able to produce a complete answer from the available "
                "course data. Please try the request again."
            )

        await ctx.yield_output(text)
