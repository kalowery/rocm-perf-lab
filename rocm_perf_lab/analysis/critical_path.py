import sqlite3
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CriticalPathResult:
    critical_path_ns: int
    critical_kernel_ids: List[int]
    critical_kernel_names: List[str]
    dominant_dispatch_id: int
    dominant_dispatch_duration_ns: int
    dominant_symbol_name: str
    dominant_symbol_fraction: float
    dispatch_contributions: Dict[int, float]
    symbol_contributions: Dict[str, float]


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

    # Handle trivial / empty cases
    if not nodes:
        conn.close()
        return CriticalPathResult(
            critical_path_ns=0,
            critical_kernel_ids=[],
            critical_kernel_names=[],
            dominant_dispatch_id=None,
            dominant_dispatch_duration_ns=0,
            dominant_symbol_name=None,
            dominant_symbol_fraction=0.0,
            dispatch_contributions={},
            symbol_contributions={},
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

    # Cross-queue inferred deps (scale-aware threshold)
    if nodes:
        global_start = min(n["start"] for n in nodes)
        global_end = max(n["end"] for n in nodes)
        total_runtime = global_end - global_start
    else:
        total_runtime = 0

    # Allow up to max(50us, 1% of total runtime) as dependency gap
    threshold_ns = max(50_000, int(0.01 * total_runtime))

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
        if best is not None and best_gap is not None and best_gap <= threshold_ns:
            adj[best["id"]].append(b["id"])
            indegree[b["id"]] += 1

    # Longest path DP using proper topological order (Kahn's algorithm)
    from collections import deque

    dp = {n["id"]: n["duration"] for n in nodes}
    parent = {n["id"]: None for n in nodes}

    # Build quick lookup for node durations
    duration_map = {n["id"]: n["duration"] for n in nodes}

    # Kahn topo sort
    indegree_copy = indegree.copy()
    queue = deque([nid for nid, deg in indegree_copy.items() if deg == 0])
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj[u]:
            indegree_copy[v] -= 1
            if indegree_copy[v] == 0:
                queue.append(v)

    # DP over topo order
    for u in topo_order:
        for v in adj[u]:
            if dp[u] + duration_map[v] > dp[v]:
                dp[v] = dp[u] + duration_map[v]
                parent[v] = u

    # Find max
    if not dp:
        conn.close()
        return CriticalPathResult(
            critical_path_ns=0,
            critical_kernel_ids=[],
            critical_kernel_names=[],
            dominant_dispatch_id=None,
            dominant_dispatch_duration_ns=0,
            dominant_symbol_name=None,
            dominant_symbol_fraction=0.0,
            dispatch_contributions={},
            symbol_contributions={},
        )

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

    # Per-dispatch contributions
    dispatch_contributions = {
        n["id"]: n["duration"] / critical_length for n in path_nodes
    }

    # Aggregate by kernel symbol
    symbol_totals = {}
    for n in path_nodes:
        symbol_totals.setdefault(n["name"], 0)
        symbol_totals[n["name"]] += n["duration"]

    symbol_contributions = {
        name: dur / critical_length for name, dur in symbol_totals.items()
    }

    dominant_dispatch = max(path_nodes, key=lambda n: n["duration"])
    dominant_symbol_name = max(symbol_contributions, key=lambda k: symbol_contributions[k])

    conn.close()

    return CriticalPathResult(
        critical_path_ns=critical_length,
        critical_kernel_ids=path_ids,
        critical_kernel_names=path_names,
        dominant_dispatch_id=dominant_dispatch["id"],
        dominant_dispatch_duration_ns=dominant_dispatch["duration"],
        dominant_symbol_name=dominant_symbol_name,
        dominant_symbol_fraction=symbol_contributions[dominant_symbol_name],
        dispatch_contributions=dispatch_contributions,
        symbol_contributions=symbol_contributions,
    )
