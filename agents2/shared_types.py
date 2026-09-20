"""
shared_types.py
Shared data structures used across multiple agents to avoid circular imports

These classes are imported by all agents to ensure consistent data structures
without creating circular dependencies between agent modules.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from copy import deepcopy
from .advising_scope import clean_parsed_scope, without_campus, mentions_campus
from .inference_metrics import InferenceMetrics
from .academic_profile import normalize_level, student_level


class ConversationState:
    """
    Tracks the state of an ongoing conversation with the user
    
    Attributes:
        user_query: The current user query being processed
        conversation_history: List of previous messages in the conversation
        extracted_entities: Entities extracted from the conversation so far
        current_intent: The detected intent of the current query
        user_profile: Information about the user (year, major, etc.)
        recommendations: Course recommendations made so far
        clarification_needed: Fields that need clarification from user
        session_id: Unique identifier for this conversation session
    """
    
    def __init__(
        self,
        user_query: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        transcript_data: Optional[Dict[str, Any]] = None
    ):
        self.user_query = user_query
        self.conversation_history = conversation_history or []
        self.session_id = session_id
        self.transcript_data = transcript_data
        self.last_intent = ""
        self.resolved_semester = None
        self.resolved_courses: Dict[str, Dict] = {}
        self.MAX_HISTORY = 12
        self.input_tokens = 0
        self.output_tokens = 0
        self.awaiting_transcript: bool = False
        self.inference_metrics = InferenceMetrics()
        self.routing_ctx = None
        self.routing_iteration = 0
        self.routing_events = []
        self.extracted_entities = {}
        self.interests = []
        self.goals = []
        self.preferences = {}
        self.last_recommendations = []
        self.last_lookup_courses = []
        self.last_pathway_targets = []
        self.latest_result_kind = None
        self.pending_clarification = None
        self.clarification_events = []

    def request_clarification(self, query, parsed, missing_fields=None, question=""):
        """Remember an unresolved task only when a clarification is actually shown."""
        clean_parsed_scope(parsed)
        entities = deepcopy(parsed.get("entities", {}))
        fields = missing_fields or parsed.get("missing_critical_info", [])
        allowed = {"target_course", "specific_courses", "interests", "year", "career_path",
                   "credit_hours", "difficulty_preference", "time_constraints", "file_path",
                   "academic_level", "graduate_program"}
        fields = [f for f in fields if isinstance(f, str) and f in allowed] if isinstance(fields, list) else []
        if not fields:
            fields = ["target_course"] if parsed.get("intent") in ("prerequisite_check", "course_info") else ["interests"]
        if mentions_campus(question):
            question = "Please specify: " + ", ".join(fields).replace("_", " ") + "."
        self.pending_clarification = {"original_query": query, "intent": parsed.get("intent"),
                                      "known_entities": entities, "missing_fields": fields,
                                      "question": question}
        self.clarification_events.append({"action": "requested", "original_intent": parsed.get("intent"),
                                          "missing_fields": list(fields)})

    def resume_clarification(self, parsed):
        """Preserve task identity; only nonempty reply entities replace known values."""
        clean_parsed_scope(parsed)
        pending = self.pending_clarification
        if not pending:
            return
        pending["missing_fields"] = without_campus(pending["missing_fields"])
        pending["known_entities"] = without_campus(pending["known_entities"])
        event = {"pending_existed": True, "original_intent": pending["intent"],
                 "missing_fields": list(pending["missing_fields"]), "merged": False, "cleared": False,
                 "explicit_task_change": parsed.get("clarification_action") == "new_task"}
        self.clarification_events.append(event)
        if event["explicit_task_change"]:
            self.pending_clarification = None
            event["cleared"] = True
            return
        supplied = {key: value for key, value in parsed.get("entities", {}).items()
                    if value not in (None, "", [])}
        if not supplied.get("target_course") and supplied.get("specific_courses"):
            supplied["target_course"] = supplied["specific_courses"][0]
        merged = dict(pending["known_entities"], **supplied)
        if supplied.get("target_course") and "target_course" in pending["missing_fields"]:
            merged["specific_courses"] = supplied.get("specific_courses") or [supplied["target_course"]]
        parsed["intent"] = pending["intent"]
        parsed["entities"] = merged
        parsed["effective_query"] = pending["original_query"]
        parsed["clarification_resumed"] = True
        remaining = [key for key in pending["missing_fields"] if key not in supplied]
        if parsed.get("clarification_action") == "incomplete" and not remaining:
            remaining = list(pending["missing_fields"])
        event["merged"] = bool(supplied)
        event["missing_fields"] = remaining
        if remaining:
            pending.update(known_entities=merged, missing_fields=remaining)
            parsed["clarification_question"] = "Please clarify: " + ", ".join(remaining).replace("_", " ") + "."
        else:
            self.pending_clarification = None
            parsed["needs_clarification"] = False
            parsed["missing_critical_info"] = []
            event["cleared"] = True

    def apply_memory_updates(self, updates):
        """Apply explicit user facts only; lists support add/remove/replace."""
        if not isinstance(updates, dict):
            return
        updates = without_campus(updates)
        for key in ("interests", "goals"):
            change = updates.get(key)
            if isinstance(change, list):
                change = {"add": change}
            if not isinstance(change, dict):
                continue
            clean = lambda values: [v.strip()[:120] for v in values
                                    if isinstance(v, str) and v.strip()] if isinstance(values, list) else []
            current = clean(change["replace"]) if isinstance(change.get("replace"), list) else list(getattr(self, key))
            removed = {v.casefold() for v in clean(change.get("remove", []))}
            current = [v for v in current if v.casefold() not in removed]
            current += clean(change.get("add", []))
            unique = {}
            for value in current:
                unique.setdefault(value.casefold(), value)
            setattr(self, key, list(unique.values())[-12:])
        preferences = updates.get("preferences", {})
        if isinstance(preferences, dict):
            for key in ("difficulty_preference", "gpa_priority", "credit_hours", "time_constraints", "year", "academic_level", "graduate_program"):
                if key not in preferences:
                    continue
                value = preferences[key]
                if value is None:
                    self.preferences.pop(key, None)
                elif isinstance(value, (str, int, float, bool)):
                    self.preferences[key] = value[:160] if isinstance(value, str) else value
            explicit = normalize_level(preferences.get("academic_level")) or normalize_level(preferences.get("year"))
            if explicit:
                self.preferences["academic_level"] = explicit
                if explicit == "undergraduate":
                    self.preferences.pop("graduate_program", None)

    def remember_results(self, kind, courses):
        """Bounded, ordered references, not cached eligibility verdicts."""
        if kind not in ("recommendations", "lookup_courses", "pathway_targets"):
            return
        references = []
        for item in courses:
            course = item.get("course", item)
            code = course.get("code") or course.get("course_code")
            if code and code not in {r["code"] for r in references}:
                references.append({"code": code, "title": course.get("title") or course.get("course_name", "")})
        if references:
            setattr(self, "last_" + kind, references[:7])
            self.latest_result_kind = kind

    def get_context(self, capability):
        """Project bounded memory only; transcript/constraints use existing paths."""
        if capability == "constraint":
            return {}  # No interests, chat history, or stale eligibility assertions.
        context = {"interests": self.interests, "goals": self.goals, "preferences": self.preferences}
        if capability != "data":
            context.update(last_recommendations=self.last_recommendations,
                           last_lookup_courses=self.last_lookup_courses,
                           last_pathway_targets=self.last_pathway_targets,
                           latest_result_kind=self.latest_result_kind)
        if capability == "parser":
            context["pending_clarification"] = self.pending_clarification
            context["recent_messages"] = [{"role": m["role"], "content": m["content"][:1600]}
                                          for m in self.conversation_history[-4:]]
        return deepcopy(context)

    def enrich_parsed_query(self, parsed):
        """Apply updates before inheriting defaults; resolve explicit parser references."""
        old_level = student_level(self)
        old_program = self.preferences.get("graduate_program")
        self.apply_memory_updates(parsed.get("memory_updates"))
        entities = parsed.setdefault("entities", {})
        explicit_level = normalize_level(entities.get("academic_level")) or normalize_level(entities.get("year"))
        if explicit_level:
            self.preferences["academic_level"] = explicit_level
        if explicit_level != "undergraduate" and entities.get("graduate_program") in ("masters", "phd", "msds"):
            self.preferences["graduate_program"] = entities["graduate_program"]
            self.preferences["academic_level"] = "graduate"
        level = student_level(self)
        if level != old_level or old_program != self.preferences.get("graduate_program"):
            self.last_recommendations = []
            self.last_lookup_courses = []
            self.last_pathway_targets = []
            self.latest_result_kind = None
            self.resolved_courses.clear()
            self.extracted_entities.clear()
            if normalize_level(self.preferences.get("year")) != level:
                self.preferences.pop("year", None)
            if level != "graduate":
                self.preferences.pop("graduate_program", None)
        if level:
            self.preferences["academic_level"] = level
            entities["academic_level"] = level
            if normalize_level(entities.get("year")) == level:
                self.preferences["year"] = entities["year"]
            if level == "undergraduate":
                entities.pop("graduate_program", None)
        defaults = dict(self.preferences, interests=self.interests, career_path="; ".join(self.goals))
        for key, value in defaults.items():
            if entities.get(key) in (None, "", []):
                entities[key] = deepcopy(value)
        reference = parsed.get("course_reference")
        if not isinstance(reference, dict):
            return
        source = reference.get("source", "latest")
        kind = self.latest_result_kind if source == "latest" else source
        courses = getattr(self, "last_" + kind, []) if kind in ("recommendations", "lookup_courses", "pathway_targets") else []
        indices = reference.get("indices")
        if indices is not None:
            if not isinstance(indices, list) or not indices or any(type(i) is not int or not 1 <= i <= len(courses) for i in indices):
                courses = []
            else:
                courses = [courses[i - 1] for i in indices]
        if not courses:
            parsed["reference_error"] = "Which courses do you mean? Please name them so I can use the right list."
            return
        parsed["reference_courses"] = deepcopy(courses)
        entities["specific_courses"] = [c["code"] for c in courses]
        entities["target_course"] = courses[0]["code"]
        entities["related_courses"] = [c["code"] for c in courses[1:]]


    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def reset_usage(self) -> None:
        self.clarification_events = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.inference_metrics.reset_query()
        self.routing_ctx = None
        self.routing_iteration = 0
        self.routing_events = []
        self.last_intent = ""
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        # Keep only last N messages to avoid token overflow
        if len(self.conversation_history) > self.MAX_HISTORY:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY:]
    
    def update_entities(self, new_entities: Dict[str, Any]):
        """Update extracted entities with new values, only if they are not None/empty"""
        for key, value in new_entities.items():
            if value is not None and value != [] and value != "":
                self.extracted_entities[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "user_query": self.user_query,
            "conversation_history": self.conversation_history,
            "session_id": self.session_id,
            "academic_profile": {"academic_level": student_level(self),
                                 "graduate_program": self.preferences.get("graduate_program")},
            "transcript_data": self.transcript_data
        }
    
    def resolve_course(self, code: str, title: str, offered: bool, semester: str):
        """Cache a resolved course — minimal data only to save tokens"""
        self.resolved_courses[code] = {
            "title": title,
            "offered": offered,
            "semester": semester
        }

    def get_resolved_course(self, code: str) -> Optional[Dict]:
        return self.resolved_courses.get(code)

    def is_course_resolved(self, code: str) -> bool:
        return code in self.resolved_courses


@dataclass
class AgentResponse:
    """
    Standard response structure returned by all agents
    
    Attributes:
        success: Whether the agent operation succeeded
        data: The main data payload from the agent (parsed info, recommendations, etc.)
        errors: List of error messages if operation failed
        metadata: Additional metadata about the response (model used, timestamp, etc.)
        next_action: Suggested next action for the orchestrator
        requires_user_input: Whether user input is needed before proceeding
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    next_action: Optional[str] = None
    requires_user_input: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "metadata": self.metadata,
            "next_action": self.next_action,
            "requires_user_input": self.requires_user_input
        }
    
    def __str__(self) -> str:
        """String representation for logging"""
        if self.success:
            return f"AgentResponse(success=True, next_action={self.next_action})"
        else:
            return f"AgentResponse(success=False, errors={self.errors})"


class AgentType(Enum):
    """
    Enumeration of different agent types in the system
    """
    ORCHESTRATOR = "orchestrator"
    PARSER = "parser"
    DATA = "data"
    CONSTRAINT = "constraint"
    PLANNING = "planning"


class IntentType(Enum):
    """
    Enumeration of possible query intents
    """
    COURSE_RECOMMENDATION = "course_recommendation"
    CLARIFICATION = "clarification"
    GENERAL_QUESTION = "general_question"
    PREREQUISITE_CHECK = "prerequisite_check"
    SCHEDULE_PLANNING = "schedule_planning"
    OFF_TOPIC = "off_topic"
    TRANSCRIPT_UPLOAD = "transcript_upload"
    UNKNOWN = "unknown"


@dataclass
class CourseRecommendation:
    """
    Structure for a single course recommendation
    
    Attributes:
        course_code: The course identifier (e.g., "CS 101")
        course_name: Full name of the course
        credits: Number of credit hours
        reason: Why this course is recommended
        confidence: Confidence score for this recommendation (0.0 to 1.0)
        prerequisites: List of prerequisite courses
        difficulty: Estimated difficulty level
        relevance_score: How relevant to user's interests (0.0 to 1.0)
    """
    course_code: str
    course_name: str
    credits: int
    reason: str
    confidence: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    difficulty: str = "moderate"
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "course_code": self.course_code,
            "course_name": self.course_name,
            "credits": self.credits,
            "reason": self.reason,
            "confidence": self.confidence,
            "prerequisites": self.prerequisites,
            "difficulty": self.difficulty,
            "relevance_score": self.relevance_score
        }


@dataclass
class ConstraintViolation:
    """
    Structure for constraint violations
    
    Attributes:
        constraint_type: Type of constraint violated
        severity: How severe the violation is (high/medium/low)
        message: Description of the violation
        affected_courses: List of courses involved in the violation
        suggestion: Suggested fix for the violation
    """
    constraint_type: str
    severity: str
    message: str
    affected_courses: List[str] = field(default_factory=list)
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "constraint_type": self.constraint_type,
            "severity": self.severity,
            "message": self.message,
            "affected_courses": self.affected_courses,
            "suggestion": self.suggestion
        }
