import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agents2.academic_profile import load_graduate_courses
from agents2.graduate_graph import graduate_nodes, relationship_graph, build_graduate_graph
from agents2.dag_builder import check_eligibility
from agents2.pathway_search import search_pathways


class GraduateGraphTests(unittest.TestCase):
    def test_graph_preserves_rules_scope_and_uncertainty(self):
        nodes = graduate_nodes(load_graduate_courses())
        graph = relationship_graph(nodes)
        self.assertEqual(len(graph), 73)
        self.assertEqual(graph.number_of_edges(), 13)
        self.assertEqual(graph.edges['16:198:512', '16:198:513']['program'], 'masters')
        self.assertEqual(graph.edges['16:198:520', '16:198:536']['kind'], 'or')
        self.assertTrue(all(n.startswith('16:198:') for n in graph))
        self.assertFalse(check_eligibility('16:198:501', nodes, set(), set())['eligible'])
        self.assertEqual(search_pathways(nodes, '16:198:501')['plans'], [])

    def test_cross_level_edges_and_cycles_are_rejected(self):
        nodes = graduate_nodes(load_graduate_courses())
        nodes['16:198:501']['and'] = ['01:198:205']
        with self.assertRaises(ValueError):
            relationship_graph(nodes)
        nodes['16:198:501']['and'] = ['16:198:501']
        with self.assertRaises(ValueError):
            relationship_graph(nodes)

    def test_render_cache_refreshes_when_catalog_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, dag, image = root / 'courses.json', root / 'dag.json', root / 'graph.png'
            courses = load_graduate_courses()
            source.write_text(json.dumps(courses), encoding='utf-8')
            with patch('agents2.graduate_graph.render_graph', side_effect=lambda nodes, path: path.write_bytes(b'graph')) as render:
                build_graduate_graph(source, dag, image)
                build_graduate_graph(source, dag, image)
                self.assertEqual(render.call_count, 1)
                courses[0]['title'] = 'Updated title'
                source.write_text(json.dumps(courses), encoding='utf-8')
                build_graduate_graph(source, dag, image)
                self.assertEqual(render.call_count, 2)
                self.assertEqual(json.loads(dag.read_text(encoding='utf-8'))[courses[0]['code']]['title'], 'Updated title')
