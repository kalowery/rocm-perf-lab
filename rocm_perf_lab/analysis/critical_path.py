import sqlite3
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CriticalPathResult:
    critical_path_ns: int
    critical_kernel_ids: List[int]
    critical_kernel_names: List[str]
    dominant_kernel_name: str
    dominant_kernel_duration_ns: int
    contributions: Dict[str, float]


def _get_table(conn, prefix: str) -> str:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?;",
        (f"{prefix}%",),
    )
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Table with prefix {prefix} not found")
    return row[0]


def analyze_critical_path(db_path: str) -> CriticalPathResult:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    dispatch_table = _get_table(conn, "rocpd_kernel_dispatch_")
    symbol_table = _get_table(conn, "rocpd_info_kernel_symbol_")

    # Load kernel names
    cur.execute(f"SELECT id, display_name, kernel_name FROM {symbol_table};")
    name_map = {}
    for kid, display, mangled in cur.fetchall():
        name_map[kid] = display or mangled or "unknown"

    # Load dispatches
    cur.execute(
        f"SELECT id, kernel_id, queue_id, start, end FROM {dispatch_table} ORDER BY start;"
    )

    nodes = []
    for did, kid, queue, start, end in cur.fetchall():
        nodes.append(
            {
                "id": did,
                "kernel_id": kid,
                "queue": queue,
                "start": start,
                "end": end,
                "duration": end - start,
                "name": name_map.get(kid, "unknown"),
            }
        )

    # Build adjacency and indegree
    adj = {n["id"]: [] for n in nodes}
    indegree = {n["id"]: 0 for n in nodes}

    # Serial edges (same queue)
    from collections import defaultdict

    by_queue = defaultdict(list)
    for n in nodes:
        by_queue[n["queue"]].append(n)

    for qnodes in by_queue.values():
        qnodes.sort(key=lambda x: x["start"])
        for i in range(len(qnodes) - 1):
            u = qnodes[i]["id"]
            v = qnodes[i + 1]["id"]
            adj[u].append(v)
            indegree[v] += 1

    # Cross-queue inferred deps
    threshold_ns = 5_000  # 5 microseconds
    for b in nodes:
        best = None
        best_gap = None
        for a in nodes:
            if a["queue"] == b["queue"]:
                continue
            if a["end"] <= b["start"]:
                gap = b["start"] - a["end"]
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best = a
        if best is not None and best_gap is not None and best_gap < threshold_ns:
            adj[best["id"]].append(b["id"])
            indegree[b["id"]] += 1

    # Longest path DP (topo by start time)
    nodes_sorted = sorted(nodes, key=lambda x: x["start"])
    dp = {n["id"]: n["duration"] for n in nodes}
    parent = {n["id"]: None for n in nodes}

    for n in nodes_sorted:
        u = n["id"]
        for v in adj[u]:
            if dp[u] + next(x for x in nodes if x["id"] == v)["duration"] > dp[v]:
                dp[v] = dp[u] + next(x for x in nodes if x["id"] == v)["duration"]
                parent[v] = u

    # Find max
    end_node = max(dp, key=lambda k: dp[k])
    critical_length = dp[end_node]

    # Backtrack
    path_ids = []
    cur_id = end_node
    while cur_id is not None:
        path_ids.append(cur_id)
        cur_id = parent[cur_id]
    path_ids.reverse()

    path_nodes = [next(n for n in nodes if n["id"] == pid) for pid in path_ids]
    path_names = [n["name"] for n in path_nodes]

    contributions = {}
    for n in path_nodes:
        contributions[n["name"]] = n["duration"] / critical_length

    dominant = max(path_nodes, key=lambda n: n["duration"])

    conn.close()

    return CriticalPathResult(
        critical_path_ns=critical_length,
        critical_kernel_ids=path_ids,
        critical_kernel_names=path_names,
        dominant_kernel_name=dominant["name"],
        dominant_kernel_duration_ns=dominant["duration"],
        contributions=contributions,
    )
