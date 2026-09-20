"""Offline memory semantics and downstream integration, not LLM extraction accuracy."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from agents2.shared_types import ConversationState
from agents2.parser_agent import ParserAgent
from agents2.data_agent import DataAgent
from agents2.orchestrator_agent import RoutingContext, OrchestratorExecutor, AgentResult


class MemoryTests(unittest.TestCase):
    def test_interest_and_goal_inherited_by_followup(self):
        state = ConversationState()
        state.enrich_parsed_query({"entities": {}, "memory_updates": {
            "interests": ["AI", "ai"], "goals": ["ML engineering"]}})
        state.reset_usage()
        parsed = {"entities": {"interests": [], "career_path": None}}
        state.enrich_parsed_query(parsed)
        self.assertEqual(parsed["entities"]["interests"], ["AI"])
        self.assertEqual(parsed["entities"]["career_path"], "ML engineering")
        self.assertEqual(state.get_context("planning")["interests"], ["AI"])
        routing = RoutingContext("Which courses?", parsed, False, session_memory=state.get_context("orchestrator"))
        self.assertIn('"AI"', routing.to_prompt())

    def test_which_two_uses_whole_latest_recommendation_set(self):
        state = ConversationState()
        state.resolved_courses = {"old": {"title": "unrelated"}}
        state.remember_results("recommendations", [{"course_code": c} for c in ["A", "B", "C"]])
        parsed = {"entities": {}, "course_reference": {"source": "latest", "indices": None}}
        state.enrich_parsed_query(parsed)
        self.assertEqual(parsed["entities"]["specific_courses"], ["A", "B", "C"])

    def test_those_and_second_follow_latest_lookup_not_recommendations(self):
        state = ConversationState()
        state.remember_results("recommendations", [{"code": "old"}])
        state.remember_results("lookup_courses", [{"course": {"code": c}} for c in ["ML", "DL"]])
        parsed = {"entities": {}, "course_reference": {"source": "latest"}}
        state.enrich_parsed_query(parsed)
        self.assertEqual(parsed["entities"]["specific_courses"], ["ML", "DL"])
        parsed["course_reference"]["indices"] = [2]
        state.enrich_parsed_query(parsed)
        self.assertEqual(parsed["entities"]["target_course"], "DL")

    def test_preference_changes_replace_and_clear(self):
        state = ConversationState()
        state.apply_memory_updates({"interests": ["AI"], "preferences": {"difficulty_preference": "challenging"}})
        state.apply_memory_updates({"interests": {"replace": ["systems"]}, "preferences": {"difficulty_preference": "light"}})
        self.assertEqual(state.preferences, {"difficulty_preference": "light"})
        self.assertEqual(state.interests, ["systems"])
        state.apply_memory_updates({"interests": {"remove": ["SYSTEMS"]}, "preferences": {"difficulty_preference": None}})
        self.assertEqual(state.interests, [])
        self.assertEqual(state.preferences, {})

    def test_session_survives_reset_but_turn_state_does_not(self):
        state = ConversationState(transcript_data={"completed_courses": []})
        state.apply_memory_updates({"interests": ["AI"], "goals": ["research"], "preferences": {"year": "senior"}})
        state.remember_results("pathway_targets", [{"code": "X"}])
        state.add_message("user", "hello")
        state.routing_ctx = object()
        state.routing_events = [{}]
        state.routing_iteration = 2
        before = state.get_context("parser")
        state.reset_usage()
        self.assertEqual(before, state.get_context("parser"))
        self.assertIsNotNone(state.transcript_data)
        self.assertIsNone(state.routing_ctx)
        self.assertEqual(state.routing_events, [])
        self.assertEqual(state.routing_iteration, 0)

    def test_projection_isolation_and_bounds(self):
        state = ConversationState()
        for _ in range(20):
            state.add_message("user", "x" * 3000)
        self.assertEqual(len(state.conversation_history), 12)
        self.assertEqual(len(state.get_context("parser")["recent_messages"]), 4)
        self.assertEqual(state.get_context("constraint"), {})
        for role in ["data", "planning", "orchestrator", "response_writer"]:
            self.assertNotIn("recent_messages", state.get_context(role))
        state.get_context("planning")["interests"].append("not real")
        self.assertEqual(state.interests, [])

    def test_invalid_reference_requests_clarification(self):
        state = ConversationState()
        parsed = {"entities": {}, "course_reference": {"source": "latest", "indices": [2]}}
        state.enrich_parsed_query(parsed)
        self.assertIn("reference_error", parsed)
        self.assertNotIn("reference_courses", parsed)


class MemoryFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_hub_captures_planning_results_for_next_turn(self):
        state = ConversationState()
        state.routing_ctx = RoutingContext("recommend", {"intent": "course_recommendation"}, False)
        hub = OrchestratorExecutor(Mock(), "router", "writer")
        hub._routing_loop = AsyncMock()
        message = AgentResult("recommend", {}, "planning", {"ranked_courses": [
            {"course_code": "A", "course_name": "First"},
            {"course_code": "B", "course_name": "Second"}]}, state)
        await hub.handle_result(message, Mock())
        self.assertEqual(state.latest_result_kind, "recommendations")
        self.assertEqual(state.last_recommendations[1]["code"], "B")
        self.assertEqual(state.routing_ctx.session_memory["last_recommendations"], state.last_recommendations)

    async def test_parser_updates_memory_without_extra_model_call(self):
        state = ConversationState()
        parser = SimpleNamespace(model="test", _llm_parse=AsyncMock(side_effect=[
            {"entities": {}, "memory_updates": {"interests": ["AI"]}},
            {"entities": {}},
        ]))
        first = await ParserAgent.parse(parser, "I'm interested in AI", state)
        state.reset_usage()
        second = await ParserAgent.parse(parser, "Which courses should I take?", state)
        self.assertTrue(first.success and second.success)
        self.assertEqual(second.data["entities"]["interests"], ["AI"])
        self.assertEqual(parser._llm_parse.await_count, 2)

    async def test_referenced_candidates_reuse_catalog_without_broad_retrieval(self):
        state = ConversationState()
        state.preferences["academic_level"] = "undergraduate"
        state.resolved_semester = {"term": "test"}
        agent = SimpleNamespace(courses_data=[{"code": c, "academic_level": "undergraduate"} for c in ["A", "B", "old"]], model="test",
                                _rag_retrieve=AsyncMock(), _fetch_soc_courses=AsyncMock(return_value=None))
        parsed = {"entities": {}, "reference_courses": [{"code": "B"}, {"code": "A"}]}
        result = await DataAgent.fetch_courses(agent, parsed, state)
        self.assertTrue(result.success)
        self.assertEqual([c["code"] for c in result.data["courses"]], ["B", "A"])
        agent._rag_retrieve.assert_not_awaited()
