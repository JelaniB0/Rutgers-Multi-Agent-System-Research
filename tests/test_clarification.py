"""Pending-task continuation without network calls or changes to routing policy."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from agents2.shared_types import ConversationState
from agents2.parser_agent import ParserAgent
from agents2.orchestrator_agent import OrchestratorExecutor, OrchestratorRequest, RoutingContext, UserQuery
from driver3 import ParserExecutor


class ClarificationTests(unittest.IsolatedAsyncioTestCase):
    def pending(self, query="How can I take machine learning", transcript=None, missing=None):
        state = ConversationState(transcript_data=transcript)
        state.preferences["academic_level"] = "undergraduate"
        state.request_clarification(query, {"intent": "prerequisite_check", "entities": {
            "interests": ["AI"], "target_course": "machine learning"}}, missing or ["target_course"])
        return state

    async def parse_reply(self, state, reply="Machine Learning Principles", action="resume", entities=None):
        parser = SimpleNamespace(model="test", _llm_parse=AsyncMock(return_value={
            "intent": "course_info", "entities": entities if entities is not None else {
                "target_course": "Machine Learning Principles", "interests": []},
            "clarification_action": action}))
        return await ParserAgent.parse(parser, reply, state)

    async def test_router_stores_original_task_when_asking(self):
        state = ConversationState()
        hub = OrchestratorExecutor(Mock(), "router", "writer")
        hub.agent = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(content=json.dumps({
            "mode": "clarify", "next_agents": [], "response": "Which course?",
            "missing_fields": ["target_course"]}))))
        parsed = {"intent": "prerequisite_check", "entities": {}}
        routing = RoutingContext("How can I take machine learning", parsed, False)
        ctx = SimpleNamespace(yield_output=AsyncMock(), send_message=AsyncMock())
        await hub._routing_loop(routing, OrchestratorRequest(routing.user_query, parsed, state), ctx, 0)
        self.assertEqual(state.pending_clarification["intent"], "prerequisite_check")
        self.assertEqual(state.pending_clarification["missing_fields"], ["target_course"])
        ctx.yield_output.assert_awaited_once_with("Which course?")

    async def test_title_resumes_prerequisite_task_not_course_info(self):
        state = self.pending()
        state.reset_usage()
        response = await self.parse_reply(state)
        self.assertTrue(response.success)
        self.assertEqual(response.data["intent"], "prerequisite_check")
        self.assertEqual(response.data["entities"]["target_course"], "Machine Learning Principles")
        self.assertEqual(response.data["entities"]["interests"], ["AI"])
        self.assertIsNone(state.pending_clarification)
        hub = OrchestratorExecutor(Mock(), "router", "writer")
        routing = RoutingContext(response.data["effective_query"], response.data, False)
        self.assertEqual(hub._required_next_agent(routing), "data_prereq")

    async def test_personal_eligibility_retains_original_query_through_driver(self):
        state = self.pending("Can I take machine learning?", {"completed_courses": []})
        response = await self.parse_reply(state)
        executor = SimpleNamespace(id="parser", parser=SimpleNamespace(parse=AsyncMock(return_value=response)))
        ctx = SimpleNamespace(send_message=AsyncMock(), yield_output=AsyncMock())
        await ParserExecutor.handle(executor, UserQuery("Machine Learning Principles", state), ctx)
        request = ctx.send_message.await_args.args[0]
        self.assertEqual(request.user_query, "Can I take machine learning?")
        routing = RoutingContext(request.user_query, request.parsed_data, True,
                                 accumulated_results={"data_prereq": {"courses": []}})
        hub = OrchestratorExecutor(Mock(), "router", "writer")
        self.assertEqual(hub._required_next_agent(routing), "constraint_prereq")

    async def test_explicit_new_task_cancels_pending(self):
        state = self.pending()
        response = await self.parse_reply(state, "Actually, just tell me what it covers", "new_task")
        self.assertEqual(response.data["intent"], "course_info")
        self.assertIsNone(state.pending_clarification)
        self.assertNotIn("effective_query", response.data)
        self.assertTrue(state.clarification_events[-1]["explicit_task_change"])

    async def test_multiple_missing_fields_update_incrementally(self):
        state = self.pending(missing=["target_course", "year"])
        first = await self.parse_reply(state)
        self.assertIn("clarification_question", first.data)
        self.assertEqual(state.pending_clarification["missing_fields"], ["year"])
        state.reset_usage()
        second = await self.parse_reply(state, "senior", entities={"year": "senior"})
        self.assertEqual(second.data["intent"], "prerequisite_check")
        self.assertEqual(second.data["entities"]["target_course"], "Machine Learning Principles")
        self.assertEqual(second.data["entities"]["year"], "senior")
        self.assertIsNone(state.pending_clarification)

    async def test_incomplete_answer_keeps_pending(self):
        state = self.pending()
        response = await self.parse_reply(state, "not sure", "incomplete", {})
        self.assertIn("clarification_question", response.data)
        self.assertIsNotNone(state.pending_clarification)

    async def test_next_unrelated_query_does_not_inherit_old_task(self):
        state = self.pending()
        await self.parse_reply(state)
        state.reset_usage()
        response = await self.parse_reply(state, "Tell me about databases", entities={"target_course": "Databases"})
        self.assertEqual(response.data["intent"], "course_info")
        self.assertNotIn("effective_query", response.data)
