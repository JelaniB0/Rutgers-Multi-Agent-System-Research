"""Render the recorded graduate relationships without inferring missing rules.

Run: python -m agents2.graduate_graph
This is an explanatory graph, not a complete eligibility or scheduling DAG.
"""
import hashlib
import json
from pathlib import Path

from .paths import GRADUATE_COURSES_FILE, GRADUATE_DAG_FILE, GRADUATE_GRAPH_FILE


def graduate_nodes(courses):
    nodes = {}
    for course in courses:
        code = course['code']
        if not code.startswith('16:198:'):
            raise ValueError(f'Non-graduate course in graduate catalog: {code}')
        rule = course.get('prerequisite_rule', {})
        nodes[code] = {
            'title': course['title'],
            'and': list(rule.get('and', [])),
            'or_groups': [list(group) for group in rule.get('or_groups', [])],
            'applies_to_program': rule.get('applies_to_program'),
            'requires_verification': True,
            'prerequisites_text': course.get('prerequisites', ''),
            'verification_status': course.get('verification_status'),
            'recommendable': course.get('recommendable', False),
            'source_url': course.get('source_url'),
        }
    return nodes


def relationship_graph(nodes):
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for code, node in nodes.items():
        groups = [('and', node['and'])] + [('or', group) for group in node['or_groups']]
        for kind, requirements in groups:
            for prerequisite in requirements:
                if prerequisite not in nodes:
                    raise ValueError(f'Prerequisite {prerequisite} is outside the graduate graph')
                graph.add_edge(prerequisite, code, kind=kind,
                               program=node.get('applies_to_program'))
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('Recorded graduate prerequisite rules contain a cycle')
    return graph


def render_graph(nodes, output_path):
    import networkx as nx
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    graph = relationship_graph(nodes)
    connected = sorted(n for n in graph if graph.degree(n))
    isolated = sorted(set(graph) - set(connected))
    fig = Figure(figsize=(20, 18), facecolor='white')
    FigureCanvasAgg(fig)
    top, bottom = fig.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1.4]})
    subgraph = graph.subgraph(connected)
    generations = list(nx.topological_generations(subgraph))
    positions = {}
    for depth, generation in enumerate(generations):
        for index, code in enumerate(sorted(generation)):
            positions[code] = (index - (len(generation) - 1) / 2, -depth)
    labels = {c: c.rsplit(':', 1)[1] for c in connected}
    colors = ['#247b87' if nodes[c]['recommendable'] else '#9099a3' for c in connected]
    nx.draw_networkx_nodes(subgraph, positions, nodelist=connected, node_color=colors,
                           node_size=1600, ax=top)
    nx.draw_networkx_labels(subgraph, positions, labels=labels, font_color='white',
                            font_size=13, font_weight='bold', ax=top)
    for kind, style, color in [('and', 'solid', '#334155'), ('or', 'dashed', '#ba5c15')]:
        edges = [(u, v) for u, v, data in subgraph.edges(data=True) if data['kind'] == kind]
        nx.draw_networkx_edges(subgraph, positions, edgelist=edges, style=style,
                                edge_color=color, node_size=1600, arrowsize=22,
                                width=2, ax=top)
    nx.draw_networkx_edge_labels(subgraph, positions,
        edge_labels={(u, v): 'MS only' for u, v, data in subgraph.edges(data=True) if data['program'] == 'masters'},
        font_size=10, ax=top)
    top.set_title('Recorded graduate prerequisites — arrows point to the dependent course', fontsize=16, pad=20)
    top.margins(0.16)
    top.axis('off')
    top.legend(handles=[
        Line2D([0], [0], color='#334155', lw=2, label='Required course'),
        Line2D([0], [0], color='#ba5c15', lw=2, ls='--', label='One of the alternative prerequisites'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#247b87', markersize=12, label='In recommendation pool'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9099a3', markersize=12, label='Lookup only / historical data'),
    ], loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

    # Include every catalog node, without implying that an isolated node has
    # no prerequisites. Titles also provide a key for the diagram above.
    bottom.set_title('Course key — * means no structured prerequisite edges recorded (not “no prerequisites”)', fontsize=13, pad=12)
    all_codes = sorted(nodes)
    rows = (len(all_codes) + 2) // 3
    import textwrap
    for index, code in enumerate(all_codes):
        col, row = divmod(index, rows)
        marker = '*' if code in isolated else ''
        title = textwrap.fill(nodes[code]['title'], width=42)
        bottom.text(col / 3, 1 - row / rows, f"{code.rsplit(':', 1)[1]}{marker}  {title}",
                    transform=bottom.transAxes, fontsize=9, va='top', color='#334155')
    bottom.axis('off')
    fig.suptitle('Rutgers graduate CS prerequisite graph · 16:198', fontsize=23, fontweight='bold', y=0.985)
    fig.text(0.5, 0.952,
             f'{len(nodes)} catalog nodes · {graph.number_of_edges()} recorded relationships · graduate catalog only\n'
             'All eligibility requires verification. Background, equivalents and permissions are not fully represented.',
             ha='center', fontsize=12)
    fig.subplots_adjust(top=0.90, bottom=0.035, left=0.04, right=0.98, hspace=0.32)
    fig.savefig(output_path, dpi=130)


def build_graduate_graph(courses_path=GRADUATE_COURSES_FILE,
                         dag_path=GRADUATE_DAG_FILE, graph_path=GRADUATE_GRAPH_FILE):
    courses_path, dag_path, graph_path = map(Path, (courses_path, dag_path, graph_path))
    courses = json.loads(courses_path.read_text(encoding='utf-8'))
    nodes = graduate_nodes(courses)
    relationship_graph(nodes)  # Validate even when the render is cached.
    encoded = json.dumps(nodes, indent=2, ensure_ascii=False) + '\n'
    fingerprint = hashlib.sha256(('graduate-graph-v1\n' + encoded).encode()).hexdigest()
    stamp = graph_path.with_suffix('.sha256')
    if (dag_path.exists() and dag_path.read_text(encoding='utf-8') == encoded
            and graph_path.exists() and stamp.exists() and stamp.read_text() == fingerprint):
        return nodes
    dag_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    render_graph(nodes, graph_path)
    dag_path.write_text(encoded, encoding='utf-8')
    stamp.write_text(fingerprint, encoding='ascii')
    return nodes


if __name__ == '__main__':
    nodes = build_graduate_graph()
    print(f'Generated {len(nodes)} graduate nodes: {GRADUATE_GRAPH_FILE}')
