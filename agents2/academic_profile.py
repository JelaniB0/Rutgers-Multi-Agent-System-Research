"""Deterministic academic-level boundaries shared by every advising capability."""
import json
from .paths import GRADUATE_COURSES_FILE


def normalize_level(value):
    value = str(value or "").strip().lower()
    if value in {"undergraduate", "undergrad", "freshman", "sophomore", "junior", "senior"}:
        return "undergraduate"
    if value in {"graduate", "grad", "masters", "master's", "ms", "phd", "ph.d.", "doctoral"}:
        return "graduate"
    return None


def student_level(state, entities=None):
    entities = entities or {}
    transcript = state.transcript_data or {}
    for source in (state.preferences, entities, transcript):
        level = normalize_level(source.get("academic_level"))
        if not level and source.get("graduate_program") in {"masters", "phd", "msds"}:
            level = "graduate"
        level = level or normalize_level(source.get("year"))
        if level:
            return level
    return normalize_level(transcript.get("year_standing"))


def course_level(course):
    code = course.get("code") or course.get("course_code", "")
    # Codes are authoritative even if imported metadata is incorrect.
    if code.startswith("16:198:"):
        return "graduate"
    if code.startswith("01:"):
        return "undergraduate"
    return course.get("academic_level")


def allowed_course(course, state, *, recommendations=False, entities=None):
    level = student_level(state, entities)
    if not level or course_level(course) != level:
        return False
    if recommendations and (course.get("recommendable") is False or course.get("cs_degree_credit") is False):
        return False
    program = (state.preferences.get("graduate_program") or (entities or {}).get("graduate_program")
               or (state.transcript_data or {}).get("graduate_program"))
    programs = course.get("allowed_programs", [])
    if recommendations and programs and program not in programs and set(programs) != {"masters", "phd"}:
        return False
    return True


def load_graduate_courses():
    with GRADUATE_COURSES_FILE.open(encoding="utf-8") as stream:
        return json.load(stream)


def graduate_check(course, state):
    """Do not infer graduate admission background or permission from empty lists."""
    program = state.preferences.get("graduate_program") or (state.transcript_data or {}).get("graduate_program")
    allowed = course.get("allowed_programs", ["masters", "phd"])
    blocked = (not allowed_course(course, state) or course.get("cs_degree_credit") is False
               or (program is not None and program not in allowed))
    transcript = state.transcript_data or {}
    completed = {c.get("code") for field in ("completed_courses", "transfer_courses", "ap_credits")
                 for c in transcript.get(field, [])
                 if str(c.get("grade") or "").upper() in {"", "A", "A-", "B+", "B", "B-", "C+", "C", "P", "PA", "S", "CR"}}
    rule = course.get("prerequisite_rule", {})
    if course.get("code") == "16:198:513" and program == "phd":
        rule = {}
    met = [c for c in rule.get("and", []) if c in completed]
    unmet = [c for c in rule.get("and", []) if c not in completed]
    for group in rule.get("or_groups", []):
        matches = [c for c in group if c in completed]
        if matches:
            met.extend(matches)
        else:
            unmet.append("one of " + ", ".join(group))
    note = ("This course is outside your academic level or graduate program's degree-credit scope."
            if blocked else "Graduate prerequisites, equivalent background, program restrictions, and any permission must be verified with the department. "
            + str(course.get("prerequisites") or "No complete prerequisite rule is recorded."))
    return {"eligible": False if blocked else None, "eligibility_status": "scope_blocked" if blocked else "requires_verification",
            "met_prerequisites": met, "unmet_prerequisites": unmet if transcript else [], "pathway_suggestion": note,
            "standing_penalty": 0.0, "standing_eligible": None, "standing_note": note}
