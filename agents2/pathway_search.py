"""Bounded BFS/DFS over completed-course states, respecting AND/OR prerequisites.

Edges represent completing one eligible course, not merely following a graph
edge: a conventional shortest graph path would incorrectly ignore AND branches.
"""

from collections import deque


def search_pathways(dag, target, completed=(), in_progress=(), *,
                    has_transcript=False, max_states=5000, max_paths=4):
    if max_states < 1 or max_paths < 1:
        raise ValueError("Search limits must be positive")
    completed, in_progress = set(completed), set(in_progress) - set(completed)
    # In-progress courses are conditional future credit, not already earned credit.
    baseline = completed | in_progress
    relevant, missing, visiting, cycles = set(), set(), set(), set()

    def dependencies(code):
        node = dag.get(code, {})
        return set(node.get("and", [])) | {
            item for group in node.get("or_groups", []) for item in group
        }

    # Iterative DFS for dependency discovery avoids recursion depth failures.
    stack = [(target, False)]
    while stack:
        code, exiting = stack.pop()
        if exiting:
            visiting.discard(code)
            continue
        if code in baseline:
            continue
        if code in visiting:
            cycles.add(code)
            continue
        if code in relevant:
            continue
        relevant.add(code)
        if code not in dag:
            missing.add(code)
            continue
        visiting.add(code)
        stack.append((code, True))
        stack.extend((dep, False) for dep in sorted(dependencies(code), reverse=True))

    def ready(code, done):
        if (code not in dag or dag[code].get("requires_permission", False)
                or dag[code].get("requires_verification", False)):
            return False
        node = dag[code]
        return set(node.get("and", [])) <= done and all(
            set(group) & done for group in node.get("or_groups", [])
        )

    def goal(done):
        return target in done or ready(target, done)

    candidates = sorted(relevant - baseline - {target})

    def layers(selected):
        remaining, done, result = set(selected), set(baseline), []
        while remaining:
            batch = sorted(c for c in remaining if ready(c, done))
            if not batch:
                return None
            result.append(batch)
            done.update(batch)
            remaining.difference_update(batch)
        return result if goal(done) else None

    def minimize(selected):
        selected = set(selected)
        for code in sorted(selected, reverse=True):
            if layers(selected - {code}) is not None:
                selected.remove(code)
        return frozenset(selected)

    found = []
    counts = {}
    truncated = {}
    for mode in ("bfs", "dfs"):
        frontier = deque([frozenset()])
        seen = {frozenset()}
        count = 0
        while frontier and count < max_states:
            selected = frontier.popleft() if mode == "bfs" else frontier.pop()
            count += 1
            done = baseline | selected
            if goal(done):
                minimal = minimize(selected)
                if minimal not in found:
                    found.append(minimal)
                if mode == "bfs" or len(found) >= max_paths:
                    break
                continue
            for code in candidates:
                if code not in done and ready(code, done):
                    next_state = selected | {code}
                    if next_state not in seen and len(seen) < max_states:
                        seen.add(next_state)
                        frontier.append(next_state)
        counts[mode] = count
        truncated[mode] = len(seen) >= max_states
        if mode == "bfs":
            shortest_proven = bool(found) and not truncated[mode]

    def describe(codes):
        return [{"code": c, "title": dag.get(c, {}).get("title", "Unknown course")}
                for c in codes]

    plans = []
    for selected in found:
        batches = layers(selected)
        plans.append({
            "additional_course_count": len(selected),
            "prerequisite_rounds": len(batches),
            "stages": [describe(batch) for batch in batches],
            "target": describe([target])[0],
            "target_stage": len(batches) + 1,
        })
    plans.sort(key=lambda plan: (plan["additional_course_count"], plan["prerequisite_rounds"]))
    # The CS-only catalog references math/ECE courses whose own prerequisites
    # are unavailable. Offer a clearly conditional CS plan, never free credit.
    conditional_plans = []
    if not plans and missing and target in dag:
        conditional = search_pathways(
            dag, target, completed | missing, in_progress,
            has_transcript=has_transcript, max_states=max_states, max_paths=max_paths,
        )
        for plan in conditional["plans"]:
            selected = {c["code"] for stage in plan["stages"] for c in stage}
            known = baseline | selected
            external = set()
            for code in selected | {target}:
                node = dag[code]
                external.update(set(node.get("and", [])) & missing)
                for group in node.get("or_groups", []):
                    if not set(group) & known:
                        choices = sorted(set(group) & missing)
                        if choices:
                            external.add(choices[0])
            conditional_plans.append({**plan,
                "external_requirements_to_verify": describe(sorted(external)),
                "condition": "External requirements must be completed first; their prerequisite chains and time are unknown. Counts and stages cover only the known CS portion.",
            })
    available_now = sorted(c for c in dag if c not in baseline and ready(c, completed))
    eligible = target in completed or ready(target, completed)
    return {
        "target": describe([target])[0],
        "status": ("already_completed" if target in completed else
                   "in_progress" if target in in_progress else
                   "eligible" if eligible and has_transcript else
                   "path_found" if plans else "unresolved"),
        "eligible_from_completed_courses": eligible if has_transcript else None,
        "eligible_after_in_progress": goal(baseline) if has_transcript else None,
        "in_progress_assumed_passed": describe(sorted(in_progress)),
        "has_transcript": has_transcript,
        "plans": plans,
        "conditional_plans": conditional_plans,
        "shortest_proven": shortest_proven,
        "fastest_examined_plan_index": (
            min(range(len(plans)), key=lambda i: plans[i]["prerequisite_rounds"]) if plans else None
        ),
        "eligible_to_explore": describe(available_now[:20]),
        "eligible_to_explore_total": len(available_now),
        "missing_catalog_nodes": describe(sorted(missing)),
        "cycle_nodes": sorted(cycles),
        "permission_required": describe(sorted(c for c in relevant
                                                if dag.get(c, {}).get("requires_permission"))),
        "states_examined": counts,
        "search_limited": any(truncated.values()) or len(found) >= max_paths,
        "assumptions": [
            "Based only on the cached prerequisite DAG; grades, corequisites, and other catalog restrictions may be absent.",
            "Without a transcript, this is a hypothetical plan starting with no completed courses.",
            "Stages assume all listed courses can run in parallel and finish in one round; no offering or credit-load constraints are checked.",
            "In-progress courses must be passed before the displayed future stages begin.",
            "Fastest means fewest prerequisite rounds among returned plans, not a guaranteed semester schedule.",
            "Missing nodes and permission-only prerequisites cannot be automatically satisfied.",
        ],
    }


def pathways_for_transcript(dag, target, transcript=None):
    transcript = transcript or {}
    passed = {"A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "P", "PA", "PASS", "CR", "S"}
    completed = set()
    for field in ("completed_courses", "transfer_courses", "ap_credits"):
        for course in transcript.get(field, []):
            grade = str(course.get("grade") or "").upper().strip()
            if course.get("code") and (not grade or grade in passed):
                completed.add(course["code"])
    ongoing = {c["code"] for c in transcript.get("in_progress_courses", []) if c.get("code")}
    return search_pathways(dag, target, completed, ongoing, has_transcript=bool(transcript))
