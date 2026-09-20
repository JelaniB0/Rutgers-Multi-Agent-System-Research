import os
import asyncio
import json
from datetime import datetime
import re


from agent_framework.openai import OpenAIResponsesClient
from agent_framework import WorkflowBuilder, WorkflowOutputEvent, Executor, WorkflowContext, handler
from query_logger3 import log_query                          # <-- updated import
from query_logger3 import log_turn_metrics
from agents2.inference_metrics import instrument_capability, usage_scope

from agents2.parser_agent import ParserAgent
from agents2.data_agent import DataAgent
from agents2.planning_agent import PlanningAgent
from agents2.transcript_agent import TranscriptAgent
from agents2.constraint_agent import ConstraintAgent
from agents2.shared_types import ConversationState
from agents2.dag_builder import build_dag
from agents2.graduate_graph import build_graduate_graph
from agents2.azure_openai import ModelConfig, create_agent_client, get_azure_openai_settings
from agents2.paths import COURSES_FILE, SCHEMA_FILE, DAG_FILE
from agents2.pathway_search import pathways_for_transcript
from agents2.orchestrator_agent import (
    UserQuery,
    OrchestratorRequest,
    AgentResult,
    OrchestratorExecutor,
    RoutingContext
)


# Spoke Executors, each does one job, sends AgentResult back to orchestrator

class ParserExecutor(Executor):

    def __init__(self, chat_client: OpenAIResponsesClient, model_id: str):
        super().__init__(id="parser")
        self.parser = ParserAgent(
            client=chat_client,
            model=model_id,
            schema_path=str(SCHEMA_FILE)
        )

    @handler
    @instrument_capability
    async def handle(self, message: UserQuery, ctx: WorkflowContext[AgentResult]) -> None:
        enriched_query = message.user_query

        response = await self.parser.parse(enriched_query, message.conversation_state, thread=None)

        if not response.success:
            await ctx.yield_output("I encountered an error parsing your query. Please try again.")
            return
        if response.data.get("reference_error"):
            message.conversation_state.request_clarification(
                response.data.get("effective_query", message.user_query), response.data,
                ["specific_courses"], response.data["reference_error"])
            await ctx.yield_output(response.data["reference_error"])
            return
        if response.data.get("clarification_question"):
            await ctx.yield_output(response.data["clarification_question"])
            return
        effective_query = response.data.get("effective_query", message.user_query)
        message.conversation_state.user_query = effective_query
        message.conversation_state.last_intent = response.data.get("intent")
        await ctx.send_message(
            OrchestratorRequest(effective_query, response.data, message.conversation_state)
        )


class DataExecutor(Executor):

    def __init__(self, chat_client: OpenAIResponsesClient, model_id: str):
        super().__init__(id="data")
        self.data_agent = DataAgent(
            client=chat_client, model=model_id, courses_file=str(COURSES_FILE)
        )

    @handler
    @instrument_capability
    async def handle(self, message: AgentResult, ctx: WorkflowContext[AgentResult]) -> None:
        if message.agent_name not in ("data_fetch", "data_lookup", "data_prereq"):
            return

        entities = message.parsed_data.get("entities", {})

        if message.agent_name == "data_fetch":
            response = await self.data_agent.fetch_courses(
                parsed_data=message.parsed_data, state=message.conversation_state
            )
            if not response.success:
                await ctx.yield_output("I had trouble fetching courses. Please try again.")
                return
            courses = response.data.get("courses", [])
            await ctx.send_message(AgentResult(
                message.user_query, message.parsed_data,
                agent_name="data_fetch", data=response.data,
                conversation_state=message.conversation_state
            ))

        elif message.agent_name == "data_lookup":
            specific_courses = entities.get("specific_courses", [])
            if not specific_courses:
                specific_courses = entities.get("interests", [])

            if not specific_courses:
                await ctx.yield_output("I couldn't find a course name in your query. Could you be more specific?")
                return

            results = []
            lookup_errors = []
            for course_name in specific_courses:
                response = await self.data_agent.lookup_course(course_name, message.conversation_state)
                if response.success:
                    results.append(response.data)
                else:
                    lookup_errors.extend(response.errors or [])

            if not results:
                await ctx.yield_output(" ".join(lookup_errors) or "I couldn't find that course. Please check the course name and try again.")
                return

            await ctx.send_message(AgentResult(
                message.user_query, message.parsed_data,
                agent_name="data_lookup",
                data={"courses": results},
                conversation_state=message.conversation_state
            ))

        elif message.agent_name == "data_prereq":
            targets = entities.get("specific_courses", [])
            if entities.get("target_course"):
                targets = [entities["target_course"]] + targets
            targets = list(dict.fromkeys(targets))

            if not targets:
                await ctx.yield_output("I couldn't find a course name in your query. Could you be more specific?")
                return

            results = []
            lookup_errors = []
            for target in targets:
                response = await self.data_agent.lookup_course(target, message.conversation_state)
                if response.success:
                    results.append(response.data)
                else:
                    lookup_errors.extend(response.errors or [])

            if not results:
                await ctx.yield_output(" ".join(lookup_errors) or "I couldn't find those courses. Please check the course names and try again.")
                return

            await ctx.send_message(AgentResult(
                message.user_query, message.parsed_data,
                agent_name="data_prereq",
                data={"courses": results, "pathways": self._pathways(results, message.conversation_state)},
                conversation_state=message.conversation_state
            ))

    @staticmethod
    def _pathways(results, state):
        from agents2.academic_profile import student_level, load_graduate_courses, graduate_check
        if student_level(state) == "graduate":
            catalog = {c["code"]: c for c in load_graduate_courses()}
            return [{"target": result["course"], "status": "requires_verification", "plans": [],
                     "conditional_plans": [], "eligible_to_explore": [], "shortest_proven": False,
                     "requirements": graduate_check(catalog[result["course"]["code"]], state),
                     "assumptions": ["Graduate prerequisite equivalencies and permissions need departmental verification. No automatic pathway is asserted; undergraduate courses are not offered as enrollment steps."]}
                    for result in results if result.get("course", {}).get("code") in catalog]
        with DAG_FILE.open(encoding="utf-8") as stream:
            dag = json.load(stream)
        return [pathways_for_transcript(dag, result["course"]["code"], state.transcript_data)
                for result in results if result.get("course", {}).get("code")]


class ConstraintExecutor(Executor):

    def __init__(self, chat_client: OpenAIResponsesClient, model_id: str):
        super().__init__(id="constraint")
        self.constraint_agent = ConstraintAgent(client=chat_client, model=model_id)

    @handler
    @instrument_capability
    async def handle(self, message: AgentResult, ctx: WorkflowContext[AgentResult]) -> None:
        if message.agent_name not in ("constraint_full", "constraint_prereq"):
            return

        if message.agent_name == "constraint_full":
            courses = message.data.get("courses") or message.data.get("data_fetch", {}).get("courses", [])
            constraint_data = {}
            if message.conversation_state.transcript_data:
                response = await self.constraint_agent.validate_courses(
                    courses=courses, state=message.conversation_state
                )
                if response.success:
                    constraint_data = response.data
            await ctx.send_message(AgentResult(
                message.user_query, message.parsed_data,
                agent_name="constraint_full",
                data={"courses": courses, "constraint_data": constraint_data},
                conversation_state=message.conversation_state
            ))

        elif message.agent_name == "constraint_prereq":
            prereq_results = message.data.get("data_prereq", {})
            courses_list = prereq_results.get("courses", [])
            course = None
            if courses_list:
                course = courses_list[0].get("course")

            constraint_data = {}
            if course and message.conversation_state.transcript_data:
                response = await self.constraint_agent.check_single_course(
                    course=course, state=message.conversation_state
                )
                if response.success:
                    constraint_data = response.data

            await ctx.send_message(AgentResult(
                message.user_query, message.parsed_data,
                agent_name="constraint_prereq",
                data={"courses": [{"course": course}] if course else [], "constraint_data": constraint_data},
                conversation_state=message.conversation_state
            ))


class PlanningExecutor(Executor):

    def __init__(self, chat_client: OpenAIResponsesClient, model_id: str):
        super().__init__(id="planning")
        self.planning_agent = PlanningAgent(client=chat_client, model=model_id)

    @handler
    @instrument_capability
    async def handle(self, message: AgentResult, ctx: WorkflowContext[AgentResult]) -> None:
        if message.agent_name != "planning":
            return

        print("[PlanningExecutor] Ranking courses...")
        courses = message.data.get("courses", [])
        constraint_data = message.data.get("constraint_data", {})

        constraint_context = ""
        if constraint_data:
            from agents2.constraint_agent import ConstraintAgent as CA
            constraint_context = CA.summarize_for_prompt(constraint_data)

        RANKING_FIELDS = {"code", "title", "description", "prerequisites", "credits", "topics", "constraint_check",
                          "academic_level", "recommendable", "allowed_programs", "verification_status", "prerequisite_status", "course_type"}
        courses_to_rank = [
            {k: v for k, v in c.items() if k in RANKING_FIELDS}
            for c in courses[:7]
        ]

        response = await self.planning_agent.rank_courses(
            courses=courses_to_rank,
            parsed_data=message.parsed_data,
            state=message.conversation_state,
            constraint_context=constraint_context,
            max_results=5,
            thread=None
        )

        await ctx.send_message(AgentResult(
            message.user_query, message.parsed_data,
            agent_name="planning",
            data=response.data if response.success else {},
            conversation_state=message.conversation_state
        ))


class TranscriptExecutor(Executor):

    def __init__(self, chat_client: OpenAIResponsesClient, model_id: str):
        super().__init__(id="transcript")
        self.transcript_agent = TranscriptAgent(client=chat_client, model=model_id)

    @handler
    @instrument_capability
    async def handle(self, message: AgentResult, ctx: WorkflowContext[AgentResult]) -> None:
        if message.agent_name != "transcript":
            return

        file_path = message.parsed_data.get("entities", {}).get("file_path")

        if not file_path:
            match = re.search(r'[\w./ \\-]+\.pdf', message.user_query, re.IGNORECASE)
            if match:
                file_path = match.group().strip()

        if not file_path or not os.path.exists(file_path):
            message.conversation_state.awaiting_transcript = True
            await ctx.yield_output("Sure! Go ahead and drop the path to your transcript PDF.")
            return

        message.conversation_state.awaiting_transcript = False
        response = await self.transcript_agent.parse_transcript(file_path, message.conversation_state)

        if not response.success:
            await ctx.yield_output(f"I couldn't read that transcript: {', '.join(response.errors)}")
            return

        data = response.data
        completed_cs   = [c for c in data.get("completed_courses", [])   if ":198:" in c.get("code", "")]
        in_progress_cs = [c for c in data.get("in_progress_courses", []) if ":198:" in c.get("code", "")]

        completed_str   = "\n".join(f"  - {c['code']}: {c['title']} ({c.get('grade','P')})" for c in completed_cs)  or "  - None found"
        in_progress_str = "\n".join(f"  - {c['code']}: {c['title']}"                        for c in in_progress_cs) or "  - None found"

        await ctx.yield_output(
            f"Got it! I've read your transcript. Here's what I found:\n\n"
            f"**Year:** {data.get('year_standing')}\n"
            f"**GPA:** {data.get('cumulative_gpa')}\n"
            f"**Credits Completed:** {data.get('total_degree_credits')}\n\n"
            f"**CS Courses Completed:**\n{completed_str}\n\n"
            f"**CS Courses In Progress:**\n{in_progress_str}\n\n"
            f"I'll factor all of this in when making recommendations."
        )


# Workflow assembly — hub and spoke approach where orchestrator acts as hub

def build_workflow(chat_client: OpenAIResponsesClient, model_config: ModelConfig, topology: str = "star"):
    if topology != "star":
        raise ValueError(f"Unsupported topology: {topology!r}; only 'star' is implemented.")
    parser       = ParserExecutor(chat_client, model_config.parser)
    orchestrator = OrchestratorExecutor(chat_client, model_config.orchestrator, model_config.response_writer)
    data         = DataExecutor(chat_client, model_config.data)
    constraint   = ConstraintExecutor(chat_client, model_config.constraint)
    planning     = PlanningExecutor(chat_client, model_config.planning)
    transcript   = TranscriptExecutor(chat_client, model_config.transcript)

    workflow = (
        WorkflowBuilder()
        .set_start_executor(parser)

        .add_edge(parser,       orchestrator)

        # Orchestrator -> Spokes
        .add_edge(orchestrator, data)
        .add_edge(orchestrator, constraint)
        .add_edge(orchestrator, planning)
        .add_edge(orchestrator, transcript)

        # Spokes -> Orchestrator
        .add_edge(data,         orchestrator)
        .add_edge(constraint,   orchestrator)
        .add_edge(planning,     orchestrator)
        # transcript is terminal, yields output directly

        .build()
    )

    return workflow, orchestrator


def _collect_feedback_and_hallucination() -> tuple[str, str, str, str, str]:
    """
    Prompt the user for satisfaction, hallucination, and notes.
    Returns (satisfied, feedback, hallucinated, hallucination_type, hallucination_notes).
    All values default to "NULL" / "" if skipped.
    """
    satisfied          = "NULL"
    feedback           = "NULL"
    hallucinated       = "NULL"
    hallucination_type = ""
    hallucination_notes = ""

    try:
        raw = input("Were you satisfied with that response? (y/n, or Enter to skip): ").strip().lower()
        if raw in ("y", "yes"):
            satisfied = "yes"
            feedback  = input("Any feedback? (Enter to skip): ").strip() or "NULL"
        elif raw in ("n", "no"):
            satisfied = "no"
            feedback  = input("Any feedback? (Enter to skip): ").strip() or "NULL"
    except (KeyboardInterrupt, EOFError):
        return satisfied, feedback, hallucinated, hallucination_type, hallucination_notes

    try:
        h_raw = input("Did the response hallucinate? (y/n, or Enter to skip): ").strip().lower()
        if h_raw in ("y", "yes"):
            hallucinated = "yes"
            print("Hallucination type:")
            print("  1) course_invented  — recommended a course that doesn't exist")
            print("  2) prereq_wrong     — stated incorrect prerequisite information")
            print("  3) code_wrong       — used a wrong course code")
            print("  4) ranking_wrong    — ranked ineligible course as eligible or vice versa")
            print("  5) other")
            type_raw = input("Enter number (or Enter to skip): ").strip()
            type_map = {
                "1": "course_invented",
                "2": "prereq_wrong",
                "3": "code_wrong",
                "4": "ranking_wrong",
                "5": "other",
            }
            hallucination_type  = type_map.get(type_raw, "other") if type_raw else ""
            hallucination_notes = input("Brief notes on the hallucination (Enter to skip): ").strip()
        elif h_raw in ("n", "no"):
            hallucinated = "no"
    except (KeyboardInterrupt, EOFError):
        pass

    return satisfied, feedback, hallucinated, hallucination_type, hallucination_notes


# Main

async def main():
    print("Rutgers CS Course Advisor - Hub & Spoke Multi-Agent Workflow")

    settings = get_azure_openai_settings()
    chat_client = create_agent_client(settings)
    model_config = settings.models
    conversation_state = ConversationState()
    session_id = datetime.now().isoformat(timespec="microseconds")

    # Build DAG at startup
    try:
        with usage_scope(conversation_state, "startup.dag"):
            await build_dag(courses_path=str(COURSES_FILE), output_path=str(DAG_FILE))
            build_graduate_graph()
        with usage_scope(conversation_state, "startup.index"):
            workflow, _ = build_workflow(chat_client, model_config)
    finally:
        log_turn_metrics(session_id, conversation_state.inference_metrics.snapshot(), phase="startup",
                         model_config=model_config.to_dict())

    print("Hello! I'm your Rutgers CS course advisor.")
    print("Ask me about course recommendations, prerequisites, or upload your transcript.")
    print("Type 'quit' to exit.\n")

    agent_sources = {
        "parser":       ["LLM", "query_schema.json"],
        "data":         ["rutgers_courses.json"],
        "planning":     ["LLM"],
        "transcript":   ["LLM", "transcript_pdf"],
        "constraint":   ["LLM"],
        "orchestrator": ["LLM"],
    }

    step_to_agent = {
        "transcript":        "transcript",
        "data_fetch":        "data",
        "data_lookup":       "data",
        "data_prereq":       "data",
        "constraint_full":   "constraint",
        "constraint_prereq": "constraint",
        "planning":          "planning",
        "respond":           "orchestrator",
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Thank you for using the Rutgers CS Course Advisor. Good luck!")
            break

        if not user_input:
            continue

        try:
            response_text = ""
            conversation_state.reset_usage()

            turn_start = datetime.now()

            conversation_state.add_message("user", user_input)

            # Transcript upload: PDF path detected in message
            match = re.search(r'[\w./ \\-]+\.pdf', user_input, re.IGNORECASE)
            if match:
                conversation_state.awaiting_transcript = False

                async for event in workflow.run_stream(UserQuery(user_input, conversation_state)):
                    if isinstance(event, WorkflowOutputEvent):
                        response_text = event.data
                        print(f"\nAdvisor: {response_text}\n")
                        conversation_state.add_message("assistant", response_text)

                continue

            # Normal query flow
            async for event in workflow.run_stream(UserQuery(user_input, conversation_state)):
                if isinstance(event, WorkflowOutputEvent):
                    response_text = event.data
                    print(f"\nAdvisor: {response_text}\n")
                    conversation_state.add_message("assistant", response_text)

            input_tokens = conversation_state.input_tokens
            output_tokens = conversation_state.output_tokens
            response_time_sec = (datetime.now() - turn_start).total_seconds()

            routing_ctx: RoutingContext | None = getattr(conversation_state, "routing_ctx", None)
            last_intent = getattr(conversation_state, "last_intent", None)

            agents_invoked = ["parser", "orchestrator"]
            if last_intent == "transcript_upload":
                agents_invoked.append("transcript")
            elif routing_ctx:
                for raw in routing_ctx.agents_call_order:
                    agent = step_to_agent.get(raw)
                    if agent and agent not in agents_invoked:
                        agents_invoked.append(agent)

            plan_steps = " -> ".join(routing_ctx.agents_call_order) if routing_ctx else ""
            turn_sources = {a: agent_sources.get(a, ["LLM"]) for a in agents_invoked}
            if routing_ctx and "data_prereq" in routing_ctx.agents_call_order:
                turn_sources["data"] = ["rutgers_courses.json", "prereq_dag.json", "BFS", "DFS"]

            should_log = (
                response_text and
                last_intent not in ("transcript_upload", "clarification", None)
            )

            if should_log:
                satisfied, feedback, hallucinated, hallucination_type, hallucination_notes = (
                    _collect_feedback_and_hallucination()
                )

                log_query(
                    session_id=session_id,
                    query=user_input,
                    response=response_text,
                    agents_invoked=agents_invoked,
                    agent_sources=turn_sources,
                    plan_steps=plan_steps,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time_sec=response_time_sec,
                    satisfied=satisfied,
                    feedback=feedback,
                    hallucinated=hallucinated,
                    hallucination_type=hallucination_type,
                    hallucination_notes=hallucination_notes,
                    research=conversation_state.inference_metrics.snapshot(),
                )
                snapshot = conversation_state.inference_metrics.snapshot()
                query_cost = snapshot["query"]["estimated_inference_cost_usd"]
                session_cost = snapshot["session"]["estimated_inference_cost_usd"]
                print(f"[Research] Estimated query cost (USD): {query_cost or 'unavailable'} | "
                      f"Session estimated cost (USD): {session_cost or 'unavailable'}")

        except Exception as e:
            print(f"[Workflow] Error: {e}")
            import traceback
            traceback.print_exc()

            if "tokens_limit_reached" in str(e) or "413" in str(e):
                response_time_sec = (datetime.now() - turn_start).total_seconds()
                log_query(
                    session_id=session_id,
                    query=user_input,
                    response="ERROR: tokens_limit_reached",
                    agents_invoked=["parser", "orchestrator"],
                    agent_sources={"parser": ["LLM"], "orchestrator": ["LLM"]},
                    plan_steps="CRASHED",
                    input_tokens=conversation_state.input_tokens,
                    output_tokens=conversation_state.output_tokens,
                    response_time_sec=response_time_sec,
                    satisfied="no",
                    feedback="auto-logged: token limit crash",
                    hallucinated="NULL",
                    hallucination_type="",
                    hallucination_notes="",
                    research=conversation_state.inference_metrics.snapshot(),
                )
        finally:
            log_turn_metrics(session_id, conversation_state.inference_metrics.snapshot(),
                             routing_events=conversation_state.routing_events,
                             parsed_intent=conversation_state.last_intent,
                             academic_profile={k: conversation_state.preferences.get(k)
                                               for k in ("academic_level", "graduate_program")},
                             model_config=model_config.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
