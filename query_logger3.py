"""Append-only evaluation CSV and compact per-turn model-usage JSONL."""
import csv
import json
from datetime import datetime
from pathlib import Path
from agents2.paths import QUERY_LOG_FILE

CSV_LOG_FILE = str(QUERY_LOG_FILE)
LEGACY_COLUMNS = [
    "session_id", "timestamp", "response_time_sec", "query", "response",
    "plan_steps", "agents_invoked", "sources_and_tools", "input_tokens",
    "output_tokens", "satisfied", "feedback", "hallucinated",
    "hallucination_type", "hallucination_notes",
]
CSV_COLUMNS = LEGACY_COLUMNS + [
    "query_id", "model_id", "topology", "llm_call_count", "model_call_count",
    "estimated_inference_cost_usd", "cumulative_session_cost_usd",
    "session_input_tokens", "session_output_tokens", "session_llm_call_count",
    "usage_complete",
]
HALLUCINATION_TYPES = ["course_invented", "prereq_wrong", "code_wrong", "ranking_wrong", "other"]


def _append_csv(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream), []) != columns:
                raise ValueError(f"Unexpected log schema in {path}; existing data was not changed.")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log_query(session_id, query, response, agents_invoked, agent_sources=None,
              plan_steps="", input_tokens=0, output_tokens=0, response_time_sec=0.0,
              satisfied="", feedback="", hallucinated="NULL", hallucination_type="",
              hallucination_notes="", filepath=CSV_LOG_FILE, *, model_id="",
              topology="star", research=None):
    if not query or not query.strip():
        return
    research = research or {}
    if "calls" in research:
        # Actual returned calls, not the shared client's default deployment.
        model_id = "|".join(sorted({c["model_id"] for c in research["calls"] if c.get("model_id")}))
    turn, session = research.get("query", {}), research.get("session", {})
    sources = agent_sources or {}
    row = dict(session_id=session_id, timestamp=datetime.now().isoformat(timespec="seconds"),
               response_time_sec=f"{response_time_sec:.2f}", query=query, response=response,
               plan_steps=plan_steps, agents_invoked="|".join(agents_invoked),
               sources_and_tools="|".join(f"{a}:{','.join(sources.get(a, ['LLM']))}" for a in agents_invoked),
               input_tokens=turn.get("input_tokens", input_tokens), output_tokens=turn.get("output_tokens", output_tokens),
               satisfied=satisfied, feedback=feedback, hallucinated=hallucinated,
               hallucination_type=hallucination_type, hallucination_notes=hallucination_notes,
               query_id=research.get("query_id"), model_id=model_id, topology=topology,
               llm_call_count=turn.get("llm_call_count"), model_call_count=turn.get("model_call_count"),
               estimated_inference_cost_usd=turn.get("estimated_inference_cost_usd"),
               cumulative_session_cost_usd=session.get("estimated_inference_cost_usd"),
               session_input_tokens=session.get("input_tokens"), session_output_tokens=session.get("output_tokens"),
               session_llm_call_count=session.get("llm_call_count"), usage_complete=turn.get("usage_complete"))
    path = Path(filepath)
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream), [])
        if header == LEGACY_COLUMNS:
            _append_csv(path.with_name(path.stem + "_metrics.csv"), row, CSV_COLUMNS)
            _append_csv(path, row, LEGACY_COLUMNS)
            return
    _append_csv(path, row, CSV_COLUMNS)


def log_turn_metrics(session_id, research, *, routing_events=None, topology="star",
                     phase="query", parsed_intent=None, model_config=None, academic_profile=None, filepath=CSV_LOG_FILE):
    """All attempted turns, including uploads/errors; no prompt/transcript content."""
    path = Path(filepath)
    path = path.with_name(path.stem + "_calls.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [dict(event, guardrail_modified=bool(event.get("interventions") or
                                                event.get("terminal_fallback")))
              for event in (routing_events or [])]
    record = dict(research, session_id=session_id, topology=topology, phase=phase,
                  parsed_intent=parsed_intent,
                  academic_profile=academic_profile or {},
                  model_config=model_config or {},
                  timestamp=datetime.now().isoformat(timespec="seconds"),
                  routing_events=events)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
