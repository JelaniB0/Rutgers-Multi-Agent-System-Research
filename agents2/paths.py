"""Stable locations for the current workflow's resources and outputs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSES_FILE = PROJECT_ROOT / "rutgers_courses.json"
GRADUATE_COURSES_FILE = PROJECT_ROOT / "rutgers_graduate_courses.json"
SCHEMA_FILE = PROJECT_ROOT / "agents2" / "query_schema.json"
DAG_FILE = PROJECT_ROOT / "agents2" / "prereq_dag.json"
GRAPH_FILE = PROJECT_ROOT / "agents2" / "prereq_graph.png"
GRADUATE_DAG_FILE = PROJECT_ROOT / "agents2" / "graduate_prereq_dag.json"
GRADUATE_GRAPH_FILE = PROJECT_ROOT / "agents2" / "graduate_prereq_graph.png"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
QUERY_LOG_FILE = PROJECT_ROOT / "query_log3.csv"
