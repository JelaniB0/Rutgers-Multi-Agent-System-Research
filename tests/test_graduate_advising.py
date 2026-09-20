"""Offline regressions for catalog separation and graduate profile handling."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agents2.academic_profile import (allowed_course, course_level, graduate_check,
                                     load_graduate_courses, student_level)
from agents2.shared_types import ConversationState
from agents2.parser_agent import ParserAgent
from agents2.data_agent import DataAgent
from agents2.constraint_agent import ConstraintAgent
from agents2.planning_agent import PlanningAgent
from driver3 import DataExecutor


def profile(level, program=None):
    state = ConversationState()
    state.preferences['academic_level'] = level
    if program:
        state.preferences['graduate_program'] = program
    return state


UG = {'code': '01:198:440', 'title': 'Introduction to Artificial Intelligence', 'credits': 4}
GR = {'code': '16:198:520', 'title': 'Introduction To Artificial Intelligence', 'credits': 3}


class ProfileTests(unittest.TestCase):
    def test_unknown_fails_closed_and_codes_override_bad_metadata(self):
        self.assertFalse(allowed_course(UG, ConversationState()))
        self.assertFalse(allowed_course(dict(GR, academic_level='undergraduate'), profile('undergraduate')))
        self.assertFalse(allowed_course(UG, profile('graduate')))
        self.assertTrue(allowed_course(GR, profile('graduate')))
        self.assertFalse(allowed_course(GR, profile('undergraduate'), entities={'academic_level': 'graduate'}))

    def test_profile_correction_clears_course_memory_and_survives_reset(self):
        state = profile('undergraduate')
        state.remember_results('recommendations', [UG])
        state.enrich_parsed_query({'entities': {'academic_level': 'graduate', 'graduate_program': 'masters'}})
        self.assertEqual(state.last_recommendations, [])
        state.reset_usage()
        self.assertEqual(student_level(state), 'graduate')
        parsed = {'entities': {}}
        state.enrich_parsed_query(parsed)
        self.assertEqual(parsed['entities']['graduate_program'], 'masters')
        state.enrich_parsed_query({'entities': {'year': 'senior'}})
        self.assertEqual(student_level(state), 'undergraduate')
        self.assertNotIn('graduate_program', state.preferences)

    def test_memory_only_year_correction_overrides_previous_level(self):
        state = profile('graduate', 'phd')
        state.enrich_parsed_query({'entities': {}, 'memory_updates': {'preferences': {'year': 'senior'}}})
        self.assertEqual(student_level(state), 'undergraduate')
        self.assertEqual(state.preferences['year'], 'senior')

    def test_year_persists_when_first_identifying_level(self):
        state = ConversationState()
        state.enrich_parsed_query({'entities': {'year': 'junior'}})
        parsed = {'entities': {}}
        state.enrich_parsed_query(parsed)
        self.assertEqual(parsed['entities']['year'], 'junior')

    def test_partial_vector_index_is_repaired_with_level_metadata(self):
        vector = Mock()
        vector.count.return_value = 1
        agent = SimpleNamespace(vector_db=vector, courses_data=[UG, GR],
                                _create_course_documents=lambda c: c['title'],
                                _extract_course_level=lambda c: 'advanced')
        DataAgent._index_courses(agent)
        metadata = vector.upsert.call_args.kwargs['metadatas']
        self.assertEqual([m['academic_level'] for m in metadata], ['undergraduate', 'graduate'])
        vector.count.return_value = 2
        DataAgent._index_courses(agent)
        vector.upsert.assert_called_once()

    def test_transcript_level_not_inferred_from_course_numbers_or_credits(self):
        state = ConversationState(transcript_data={'total_degree_credits': 130, 'completed_courses': [GR]})
        self.assertIsNone(student_level(state))
        state.transcript_data['academic_level'] = 'graduate'
        self.assertEqual(student_level(state), 'graduate')

    def test_catalog_provenance_credits_and_restrictions(self):
        rows = load_graduate_courses()
        self.assertEqual(len(rows), 73)
        self.assertEqual(len({c['code'] for c in rows}), 73)
        for c in rows:
            self.assertEqual(course_level(c), 'graduate')
            self.assertTrue(c['source_url'].startswith('https://www.cs.rutgers.edu/'))
            if c['recommendable']:
                self.assertEqual(c['verification_status'], 'department_listing')
                self.assertTrue(c['description'])
                self.assertIsInstance(c['credits'], int)
        catalog = {c['code']: c for c in rows}
        self.assertEqual(catalog['16:198:560']['credits'], 4)
        self.assertIsNone(catalog['16:198:503']['credits'])
        self.assertEqual(catalog['16:198:701']['credits_max'], 12)
        for code in ('16:198:503', '16:198:504', '16:198:534', '16:198:701', '16:198:800'):
            self.assertFalse(allowed_course(catalog[code], profile('graduate'), recommendations=True))

    def test_graduate_prerequisites_do_not_treat_in_progress_as_passed(self):
        course = next(c for c in load_graduate_courses() if c['code'] == '16:198:536')
        state = profile('graduate', 'masters')
        state.transcript_data = {'in_progress_courses': [GR]}
        check = graduate_check(course, state)
        self.assertIsNone(check['eligible'])
        self.assertTrue(check['unmet_prerequisites'])
        state.transcript_data = {'completed_courses': [dict(GR, grade='A')]}
        check = graduate_check(course, state)
        self.assertEqual(check['met_prerequisites'], ['16:198:520'])
        self.assertEqual(check['unmet_prerequisites'], [])
        self.assertIsNone(check['eligible'])  # Equivalencies/other conditions remain unverified.

    def test_graduate_pathways_never_fall_back_to_undergraduate_dag(self):
        course = next(c for c in load_graduate_courses() if c['code'] == '16:198:536')
        result = DataExecutor._pathways([{'course': course}], profile('graduate'))[0]
        self.assertEqual(result['status'], 'requires_verification')
        self.assertEqual(result['plans'], [])
        self.assertFalse(result['shortest_proven'])


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_level_clarification_resumes_original_request(self):
        state = ConversationState()
        parser = SimpleNamespace(model='test', _llm_parse=AsyncMock(side_effect=[
            {'intent': 'course_recommendation', 'entities': {'interests': ['AI']}},
            {'intent': 'clarification', 'entities': {'academic_level': 'graduate', 'graduate_program': 'masters'},
             'clarification_action': 'resume'},
        ]))
        first = await ParserAgent.parse(parser, 'Recommend AI courses', state)
        self.assertIn('undergraduate', first.data['clarification_question'])
        second = await ParserAgent.parse(parser, "I'm a master's student", state)
        self.assertEqual(second.data['effective_query'], 'Recommend AI courses')
        self.assertEqual(second.data['intent'], 'course_recommendation')
        self.assertEqual(second.data['entities']['interests'], ['AI'])
        self.assertIsNone(state.pending_clarification)

    async def test_request_for_grad_course_does_not_change_undergrad_status(self):
        state = profile('undergraduate')
        parser = SimpleNamespace(model='test', _llm_parse=AsyncMock(return_value={
            'intent': 'course_info', 'entities': {'specific_courses': ['16:198:536']}}))
        await ParserAgent.parse(parser, 'Can I take graduate CS 536?', state)
        self.assertEqual(student_level(state), 'undergraduate')

    def data_agent(self):
        return SimpleNamespace(courses_data=[UG, GR], model='test', vector_db=Mock(),
                               _enrich_course=lambda c: c,
                               resolve_semester=lambda q: {'term': 9, 'year': 2026},
                               _fetch_soc_courses=AsyncMock(return_value=None))

    async def test_same_title_resolves_only_within_level(self):
        agent = self.data_agent()
        for level, code in [('graduate', GR['code']), ('undergraduate', UG['code'])]:
            result = await DataAgent.lookup_course(agent, 'Introduction to Artificial Intelligence', profile(level))
            self.assertEqual(result.data['course']['code'], code)
        agent.vector_db.query.assert_not_called()

    async def test_cross_level_exact_code_is_rejected_without_semantic_substitution(self):
        agent = self.data_agent()
        for level, code in [('graduate', UG['code']), ('undergraduate', GR['code']), ('undergraduate', 'CS 520')]:
            result = await DataAgent.lookup_course(agent, code, profile(level))
            self.assertFalse(result.success)
        agent.vector_db.query.assert_not_called()

    async def test_reference_candidates_are_filtered_again(self):
        agent = self.data_agent()
        agent._rag_retrieve = AsyncMock()
        state = profile('graduate')
        result = await DataAgent.fetch_courses(agent, {'entities': {}, 'reference_courses': [UG, GR]}, state)
        self.assertEqual([c['code'] for c in result.data['courses']], [GR['code']])
        agent._rag_retrieve.assert_not_awaited()

    async def test_unknown_status_never_calls_retrieval(self):
        agent = self.data_agent()
        agent._rag_retrieve = AsyncMock()
        result = await DataAgent.fetch_courses(agent, {'entities': {}}, ConversationState())
        self.assertTrue(result.requires_user_input)
        agent._rag_retrieve.assert_not_awaited()

    async def test_rag_filters_before_search_and_checks_results_again(self):
        response = SimpleNamespace(messages=[SimpleNamespace(contents=[SimpleNamespace(text='{"keep": []}')])])
        agent = self.data_agent()
        agent.run = AsyncMock(return_value=SimpleNamespace(content='artificial intelligence'))
        agent.chat_client = SimpleNamespace(get_response=AsyncMock(return_value=response))
        agent._build_search_query = lambda entities: 'AI'
        agent.vector_db.query.return_value = {'ids': [[UG['code'], GR['code']]], 'distances': [[0.01, 0.2]]}
        result = await DataAgent._rag_retrieve(agent, {'academic_level': 'graduate'}, 'course_recommendation')
        self.assertEqual([c['code'] for c in result], [GR['code']])
        self.assertEqual(agent.vector_db.query.call_args.kwargs['where'],
                         {'$and': [{'academic_level': 'graduate'}, {'recommendable': True}]})

    async def test_graduate_validation_reports_unknown_not_verified_eligible(self):
        agent = SimpleNamespace(model='test', _get_student_data=lambda s: (set(), set(), 0, 0),
                                _batch_check_prereqs=AsyncMock(),
                                _build_summary=lambda *args: '')
        result = await ConstraintAgent.validate_courses(agent, [GR, UG], profile('graduate'))
        self.assertEqual(result.data['eligible_courses'], [])
        self.assertEqual(result.data['unverified_courses'][0]['code'], GR['code'])
        self.assertEqual(result.data['ineligible_courses'][0]['code'], UG['code'])
        agent._batch_check_prereqs.assert_not_awaited()

    async def test_soc_requests_and_caches_both_levels_separately(self):
        response = Mock()
        response.json.return_value = [
            {'subject': '198', 'courseNumber': '440', 'school': '01', 'level': 'U'},
            {'subject': '198', 'courseNumber': '520', 'school': '16', 'level': 'G'},
            {'subject': '198', 'courseNumber': '999', 'school': '56', 'level': 'G'},
        ]
        client = AsyncMock()
        client.get.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = client
        agent = SimpleNamespace(soc_cache={}, CACHE_NEXT=3600, RUTGERS_SOC_API='https://example.test')
        term = {'term': 9, 'year': 2026}
        with patch('agents2.data_agent.httpx.AsyncClient', return_value=context):
            self.assertEqual(await DataAgent._fetch_soc_courses(agent, term, 'graduate'), {'520'})
            self.assertEqual(await DataAgent._fetch_soc_courses(agent, term, 'undergraduate'), {'440'})
            self.assertEqual(await DataAgent._fetch_soc_courses(agent, term, 'graduate'), {'520'})
        self.assertEqual(client.get.await_count, 2)
        self.assertEqual(client.get.await_args_list[0].kwargs['params']['level'], 'G')

    async def test_constraints_block_cross_level_even_without_transcript(self):
        result = await ConstraintAgent.check_single_course(SimpleNamespace(), GR, profile('undergraduate'))
        self.assertFalse(result.data['eligible'])
        self.assertEqual(result.data['eligibility_status'], 'scope_blocked')

    async def test_planning_receives_only_same_level_courses(self):
        agent = SimpleNamespace(model='test', _llm_rank=AsyncMock(return_value={'ranked_courses': []}))
        result = await PlanningAgent.rank_courses(agent, [UG, GR], {'entities': {}}, profile('graduate'))
        self.assertTrue(result.success)
        self.assertEqual(agent._llm_rank.await_args.args[0], [GR])

    async def test_hallucinated_exclusions_are_removed_as_well_as_rankings(self):
        payload = {'ranked_courses': [{'course_code': GR['code']}, {'course_code': UG['code']}],
                   'not_recommended': [{'course_code': UG['code']}]}
        response = SimpleNamespace(messages=[SimpleNamespace(contents=[SimpleNamespace(text=json.dumps(payload))])])
        agent = SimpleNamespace(run=AsyncMock(return_value=response))
        result = await PlanningAgent._run_and_parse(agent, 'rank', [GR])
        self.assertEqual(result['ranked_courses'], [{'course_code': GR['code']}])
        self.assertEqual(result['not_recommended'], [])
