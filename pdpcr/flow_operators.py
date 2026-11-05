"""Integral alternating-cycle exchange and randomized boundary transport (v2)."""
from collections import deque
import heapq
from time import perf_counter

import numpy as np


def alternating_cycles(difference, rng, deadline=None):
    """Return conformal (alpha, [(i,j,sign), ...]) cycles and completion flag.

    Each extracted cycle is complete, even when a deadline ends decomposition.
    With no deadline, their weighted sum reconstructs the balanced difference.
    """
    h = np.asarray(difference)
    if h.ndim != 2 or not np.all(np.isfinite(h)) or not np.array_equal(h, np.rint(h)):
        raise ValueError('Difference must be a finite integer matrix')
    if np.any(h.sum(0)) or np.any(h.sum(1)):
        raise ValueError('Difference must have zero row and column sums')
    nr, nc = h.shape
    adjacency = [dict() for _ in range(nr + nc)]
    for i, j in zip(*np.nonzero(h)):
        i, j = int(i), int(j)
        u, v = (i, nr+j) if h[i, j] > 0 else (nr+j, i)
        adjacency[u][v] = abs(int(h[i, j]))
    cycles = []
    active = {u for u, edges in enumerate(adjacency) if edges}
    while active:
        if deadline is not None and perf_counter() >= deadline:
            return cycles, False
        u = rng.choice(sorted(active))
        walk, positions = [], {}
        while u not in positions:
            positions[u] = len(walk)
            walk.append(u)
            u = rng.choice(list(adjacency[u]))
        nodes = walk[positions[u]:]
        edges = list(zip(nodes, nodes[1:] + nodes[:1]))
        alpha = min(adjacency[u][v] for u, v in edges)
        signed = []
        for u, v in edges:
            signed.append((u, v-nr, 1) if u < nr else (v, u-nr, -1))
            adjacency[u][v] -= alpha
            if not adjacency[u][v]:
                del adjacency[u][v]
            if not adjacency[u]:
                active.discard(u)
        cycles.append((alpha, signed))
    return cycles, True


def is_dag(flow):
    """Support-only Kahn check; flow degrees are checked by the caller."""
    adjacent = [np.flatnonzero(row) for row in flow]
    degree = (flow > 0).sum(axis=0)
    ready = deque(np.flatnonzero(degree == 0))
    visited = 0
    while ready:
        u = ready.popleft()
        visited += 1
        for v in adjacent[u]:
            degree[v] -= 1
            if degree[v] == 0:
                ready.append(v)
    return visited == len(flow)


def exchange_cycles(first, second, rng, deadline=None):
    """One coupled exchange; accept both children atomically, without fitness bias."""
    start = perf_counter()
    cycles, complete = alternating_cycles(second-first, rng, deadline)
    stats = dict(cycles_decomposed=len(cycles), decomposition_complete=int(complete),
                 decomposition_seconds=perf_counter()-start, cycles_processed=0,
                 zero_amount=0, exchanges_attempted=0, exchanges_accepted=0,
                 rejected_first=0, rejected_second=0, rejected_both=0,
                 dag_seconds=0.0)
    a, b = first.copy(), second.copy()
    rng.shuffle(cycles)
    for alpha, signed in cycles:
        if deadline is not None and perf_counter() >= deadline:
            break
        stats['cycles_processed'] += 1
        q = rng.randint(0, alpha)
        if not q:
            stats['zero_amount'] += 1
            continue
        stats['exchanges_attempted'] += 1
        ca, cb = a.copy(), b.copy()
        for i, j, sign in signed:
            ca[i, j] += q*sign
            cb[i, j] -= q*sign
        checked = perf_counter()
        valid_a, valid_b = is_dag(ca), is_dag(cb)
        stats['dag_seconds'] += perf_counter()-checked
        if valid_a and valid_b:
            a, b = ca, cb
            stats['exchanges_accepted'] += 1
        else:
            stats['rejected_first'] += int(not valid_a)
            stats['rejected_second'] += int(not valid_b)
            stats['rejected_both'] += int(not valid_a and not valid_b)
    stats['cycles_unprocessed'] = len(cycles)-stats['cycles_processed']
    return a, b, stats


def min_cost_transport(supply, demand, allowed, costs, deadline=None):
    """Integral min-cost transport by shortest augmenting paths.

    Nonnegative integer input costs; residual reverse arcs use negative costs.
    Return None on timeout so callers can retain their feasible input.
    Capacities min(supply_i,demand_j) suffice for u_ij=min(r_i,r_j).
    """
    supply, demand = np.asarray(supply, dtype=int), np.asarray(demand, dtype=int)
    nr, nc = len(supply), len(demand)
    allowed, costs = np.asarray(allowed, dtype=bool), np.asarray(costs)
    if (allowed.shape != (nr, nc) or costs.shape != (nr, nc)
            or np.any(supply < 0) or np.any(demand < 0)
            or supply.sum() != demand.sum() or not np.all(np.isfinite(costs))
            or np.any(costs < 0) or not np.array_equal(costs, np.rint(costs))):
        raise ValueError('Invalid transport data')
    size, source, sink = nr+nc+2, nr+nc, nr+nc+1
    graph = [[] for _ in range(size)]

    def edge(u, v, cap, cost):
        forward = [v, len(graph[v]), int(cap), int(cost)]
        reverse = [u, len(graph[u]), 0, -int(cost)]
        graph[u].append(forward)
        graph[v].append(reverse)
        return forward

    refs = []
    for i in range(nr):
        edge(source, i, supply[i], 0)
        for j in range(nc):
            cap = min(supply[i], demand[j])
            if allowed[i, j] and cap:
                refs.append((i, j, int(cap), edge(i, nr+j, cap, costs[i, j])))
    for j in range(nc):
        edge(nr+j, sink, demand[j], 0)
    potential, value, total = [0]*size, 0, int(supply.sum())
    while value < total:
        if deadline is not None and perf_counter() >= deadline:
            return None
        distance, parent = [float('inf')]*size, [None]*size
        distance[source] = 0
        queue = [(0, source)]
        while queue:
            dist, u = heapq.heappop(queue)
            if dist != distance[u]:
                continue
            for k, (v, _, cap, cost) in enumerate(graph[u]):
                if cap:
                    reduced = cost + potential[u]-potential[v]
                    if reduced < 0:
                        raise AssertionError('Invalid transport potential')
                    candidate = dist+reduced
                    if candidate < distance[v]:
                        distance[v], parent[v] = candidate, (u, k)
                        heapq.heappush(queue, (candidate, v))
        if parent[sink] is None:
            raise ValueError('Infeasible transport support')
        for v in range(size):
            if distance[v] != float('inf'):
                potential[v] += distance[v]
        delta, v = total-value, sink
        while v != source:
            u, k = parent[v]
            delta, v = min(delta, graph[u][k][2]), u
        v = sink
        while v != source:
            u, k = parent[v]
            e = graph[u][k]
            e[2] -= delta
            graph[v][e[1]][2] += delta
            v = u
        value += delta
    result = np.zeros((nr, nc), dtype=int)
    for i, j, cap, e in refs:
        result[i, j] = cap-e[2]
    return result
