# dag_builder.py
import json
import re
import asyncio
import os
from typing import Dict, List
from openai import AsyncOpenAI
import networkx as nx
import matplotlib.pyplot as plt

if __package__:
    from .inference_metrics import record_response
    from .azure_openai import create_async_openai_client, get_azure_openai_settings
    from .paths import COURSES_FILE, DAG_FILE, GRAPH_FILE
else:  # Support `python agents2/dag_builder.py` from the repo root.
    from inference_metrics import record_response
    from azure_openai import create_async_openai_client, get_azure_openai_settings
    from paths import COURSES_FILE, DAG_FILE, GRAPH_FILE

DAG_SYSTEM_PROMPT = """You are a prerequisite parser for Rutgers University CS courses.

Given a course's prerequisite text, extract the logical structure as JSON.

Rules:
- "and" list: ALL of these codes must be completed
- "or_groups" list of lists: from each inner list, AT LEAST ONE code must be completed
- Ignore grade requirements (e.g. "with a grade of C or better") — just extract the code
- Ignore non-course-code text like department names
- "permission of instructor" -> set requires_permission: true, no codes needed
- No prerequisites -> both lists empty
- Course code format: "XX:XXX:XXX" e.g. "01:198:112"

Examples:

Input: "Prerequisite: 01:198:213 and one of 01:198:314, 336, 352, or 416."
Output: {
    "and": ["01:198:213"],
    "or_groups": [["01:198:314", "01:198:336", "01:198:352", "01:198:416"]],
    "requires_permission": false
}

Input: "Prereq: (01:198:205 or 14:332:202 or 14:332:312) AND (01:640:152 or 01:640:192)"
Output: {
    "and": [],
    "or_groups": [
        ["01:198:205", "14:332:202", "14:332:312"],
        ["01:640:152", "01:640:192"]
    ],
    "requires_permission": false
}

Input: "Prerequisites: 01:198:112 and 01:640:152."
Output: {
    "and": ["01:198:112", "01:640:152"],
    "or_groups": [],
    "requires_permission": false
}

Input: "Permission of instructor."
Output: {
    "and": [],
    "or_groups": [],
    "requires_permission": true
}

Return ONLY valid JSON, no explanation.
"""

async def parse_prereqs_with_llm(
    client: AsyncOpenAI,
    prereq_text: str,
    model_id: str = "gpt-5",
) -> Dict:
    if not prereq_text.strip():
        return {"and": [], "or_groups": [], "requires_permission": False}

    try:
        response = await client.responses.create(
            model=model_id,
            instructions=DAG_SYSTEM_PROMPT,
            input=f"Parse this prerequisite text:\n{prereq_text}",
            max_output_tokens=1000,
            reasoning={"effort": "minimal"},
        )
        record_response(response, model_id)
        if getattr(response, "status", None) == "incomplete":
            raise ValueError(f"Incomplete model response: {response.incomplete_details}")

        text = response.output_text.strip()
        if not text:
            raise ValueError("Model response did not contain output text")

        text = re.sub(r'^```json|^```|```$', '', text, flags=re.MULTILINE).strip()
        return json.loads(text)
    except Exception as e:
        print(f"[DAGBuilder] Failed to parse '{prereq_text[:80]}': {e}")
        # fallback: extract whatever codes we can find
        codes = re.findall(r'\d{2}:\d{3}:\d{3}', prereq_text)
        return {"and": codes, "or_groups": [], "requires_permission": False}


async def build_dag(courses_path: str = str(COURSES_FILE), output_path: str = str(DAG_FILE)):

    if os.path.exists(output_path):
        # print(f"[DAGBuilder] DAG already exists at {output_path}, skipping build.")
        with open(output_path) as f:
            dag = json.load(f)
        visualize_dag(dag)
        return dag

    settings = get_azure_openai_settings()

    with open(courses_path) as f:
        courses = json.load(f)

    dag = {}
    async with create_async_openai_client(settings) as client:
        for course in courses:
            code = course.get("code", "")
            if not code:
                continue

            # Extract prereq text from description
            description = course.get("description", "")
            prereq_match = re.search(
                r'(?:Pre(?:re)?quisites?|Prereq)\s*:(.+?)(?:\.\s+[A-Z]|\.\s*$|\Z)',
                description, re.IGNORECASE | re.DOTALL
            )
            prereq_text = prereq_match.group(1).strip() if prereq_match else ""

            parsed = await parse_prereqs_with_llm(
                client,
                prereq_text,
                model_id=settings.models.parser,
            )

            dag[code] = {
                "title": course.get("title", ""),
                "and": parsed.get("and", []),
                "or_groups": parsed.get("or_groups", []),
                "requires_permission": parsed.get("requires_permission", False)
            }

            print(f"[DAGBuilder] {code}: and={dag[code]['and']} or_groups={dag[code]['or_groups']}")

    with open(output_path, 'w') as f:
        json.dump(dag, f, indent=2)

    # print(f"\nDAG built with {len(dag)} courses -> {output_path}")
    visualize_dag(dag)
    return dag

def check_eligibility(course_code: str, dag: Dict, completed: set, in_progress: set) -> Dict:
    available = completed | in_progress

    if course_code not in dag:
        return {
            "eligible": False,
            "eligibility_status": "unknown",
            "met_prerequisites": [],
            "unmet_prerequisites": ["Prerequisite data is unavailable"],
            "pathway_suggestion": "Obtain prerequisite information before confirming eligibility."
        }

    node = dag[course_code]
    if node.get("requires_verification"):
        return {"eligible": False, "eligibility_status": "unknown",
                "met_prerequisites": [], "unmet_prerequisites": ["Prerequisite rules require verification"],
                "pathway_suggestion": "Verify the recorded requirements and any equivalents with the department."}
    and_reqs = node.get("and", [])
    or_groups = node.get("or_groups", [])
    requires_permission = node.get("requires_permission", False)

    unmet = ["Instructor permission (not verified)"] if requires_permission else []
    met = []

    for req in and_reqs:
        if req in available:
            met.append(req)
        else:
            unmet.append(req)

    for group in or_groups:
        satisfied = [r for r in group if r in available]
        if satisfied:
            met.extend(satisfied)
        else:
            unmet.append(f"one of {', '.join(group)}")

    return {
        "eligible": len(unmet) == 0,
        "eligibility_status": "conditional_on_in_progress" if not unmet and any(
            code in in_progress and code not in completed for code in met
        ) else "eligible" if not unmet else "unmet_requirements",
        "met_prerequisites": met,
        "unmet_prerequisites": unmet,
        "pathway_suggestion": (
            f"Complete {', '.join(str(u) for u in unmet)} before taking this course."
            if unmet else None
        )
    }


def batch_check_eligibility(
    courses: List[Dict],
    dag: Dict,
    completed: set,
    in_progress: set
) -> Dict:
    return {
        course.get("code", ""): check_eligibility(
            course.get("code", ""), dag, completed, in_progress
        )
        for course in courses
    }

def visualize_dag(dag: Dict, output_path: str = str(GRAPH_FILE)):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        G = nx.DiGraph()

        # Only CS 198 courses as nodes
        valid_codes = {code for code in dag.keys() if ":198:" in code}

        for code in valid_codes:
            node = dag[code]
            short = code.split(":")[-1]
            level = int(short) // 100 if short.isdigit() else 0
            G.add_node(short, title=node.get("title", ""), level=level, full_code=code)

        # Edges — only draw if both source and target are CS nodes
        for code in valid_codes:
            node = dag[code]
            short = code.split(":")[-1]

            for req in node.get("and", []):
                req_short = req.split(":")[-1]
                if req_short in G.nodes:
                    G.add_edge(req_short, short, style="and")

            for group in node.get("or_groups", []):
                for req in group:
                    req_short = req.split(":")[-1]
                    if req_short in G.nodes:
                        G.add_edge(req_short, short, style="or")

        # Hierarchical layout — 100 at top, 400 at bottom
        level_nodes = {1: [], 2: [], 3: [], 4: []}
        for node in G.nodes():
            level = G.nodes[node].get("level", 0)
            if level in level_nodes:
                level_nodes[level].append(node)

        level_y = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}
        pos = {}
        for level, nodes in level_nodes.items():
            nodes_sorted = sorted(nodes)
            total = len(nodes_sorted)
            for i, node in enumerate(nodes_sorted):
                x = (i - total / 2) * 2.5
                y = level_y[level]
                pos[node] = (x, y)

        color_map = {
            1: "#4CAF50",
            2: "#2196F3",
            3: "#FF9800",
            4: "#F44336",
        }
        node_colors = [
            color_map.get(G.nodes[n].get("level", 1), "#9E9E9E")
            for n in G.nodes()
        ]

        and_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "and"]
        or_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "or"]

        fig, ax = plt.subplots(figsize=(40, 18))

        nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors, alpha=0.95, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=and_edges, edge_color="#222222",
                               arrows=True, arrowsize=15, width=1.5,
                               connectionstyle="arc3,rad=0.15", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=or_edges, edge_color="#BBBBBB",
                               arrows=True, arrowsize=15, width=1.0, style="dashed",
                               connectionstyle="arc3,rad=0.15", ax=ax)

        # Level labels on left
        level_labels = {1: "100-level", 2: "200-level", 3: "300-level", 4: "400-level"}
        for level, y in level_y.items():
            ax.text(-0.01, y, level_labels[level],
                    transform=ax.get_yaxis_transform(),
                    fontsize=10, va="center", ha="right",
                    color=color_map[level], fontweight="bold")

        legend_handles = [
            mpatches.Patch(color="#4CAF50", label="100-level"),
            mpatches.Patch(color="#2196F3", label="200-level"),
            mpatches.Patch(color="#FF9800", label="300-level"),
            mpatches.Patch(color="#F44336", label="400-level"),
            plt.Line2D([0], [0], color="#222222", linewidth=2, label="AND requirement"),
            plt.Line2D([0], [0], color="#BBBBBB", linewidth=1.5,
                       linestyle="dashed", label="OR requirement"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

        plt.title("Rutgers CS Prerequisite DAG", fontsize=16, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        # print(f"[DAGBuilder] Graph saved to {output_path}")

    except ImportError:
        print("[DAGBuilder] Install networkx and matplotlib: pip install networkx matplotlib")

if __name__ == "__main__":
    asyncio.run(build_dag())
