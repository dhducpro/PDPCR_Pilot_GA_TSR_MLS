"""Cut redistribution: paper's alternating-path method and an exact oracle."""
from collections import deque

import numpy as np

from .core import checked_flow, evaluate


def _transport(cost, initial):
    d = np.asarray(cost, dtype=float)
    a = np.asarray(initial, dtype=float)
    if d.ndim != 2 or d.shape != a.shape or not d.size:
        raise ValueError("Cost and initial transport must have matching 2D shapes")
    if not np.all(np.isfinite(d)) or np.min(d) < 0:
        raise ValueError("Costs must be finite and nonnegative")
    if (not np.all(np.isfinite(a)) or np.min(a) < 0
            or not np.allclose(a, np.rint(a), atol=1e-8, rtol=0)):
        raise ValueError("Initial robot transport must be nonnegative integer")
    return d, np.rint(a).astype(int)


def bottleneck(cost, flow):
    return float(np.max(np.asarray(cost)[np.asarray(flow) > 0], initial=0))


def alternating_redistribute(cost, initial):
    """Algorithm 3 with explicit, valid residual-path indexing.

    At threshold C: add only arcs with Delta<C, and remove any positive
    reverse arc. If a used threshold arc cannot be eliminated, no strictly
    smaller bottleneck is feasible. Deterministic ties make audit runs repeatable.
    """
    d, f = _transport(cost, initial)
    original = f.copy()
    rows, cols = f.shape
    steps = []
    while np.any(f):
        c = bottleneck(d, f)
        bad = np.argwhere((f > 0) & (d >= c))
        if not len(bad):
            break
        i0, j0 = map(int, bad[0])
        source, target = i0, rows + j0
        previous = {source: None}
        queue = deque([source])
        while queue and target not in previous:
            node = queue.popleft()
            if node < rows:
                nxt = [rows+j for j in range(cols) if d[node, j] < c]
            else:
                nxt = [i for i in range(rows) if f[i, node-rows] > 0]
            for other in nxt:
                if other not in previous:
                    previous[other] = node
                    queue.append(other)
        if target not in previous:
            break
        path = [target]
        while previous[path[-1]] is not None:
            path.append(previous[path[-1]])
        path.reverse()
        subtract = [(v, u-rows) for u, v in zip(path, path[1:]) if u >= rows]
        delta = min([int(f[i0, j0])] + [int(f[i, j]) for i, j in subtract])
        f[i0, j0] -= delta
        for u, v in zip(path, path[1:]):
            if u < rows:
                f[u, v-rows] += delta
            else:
                f[v, u-rows] -= delta
        steps.append({"threshold": c, "removed_arc": [i0, j0],
                      "delta": delta, "residual_path": path})
    if not np.array_equal(f.sum(axis=0), original.sum(axis=0)):
        raise AssertionError("Redistribute changed demand")
    if not np.array_equal(f.sum(axis=1), original.sum(axis=1)):
        raise AssertionError("Redistribute changed supply")
    return {"objective": bottleneck(d, f), "flow": f, "steps": steps}


def _threshold_flow(cost, supply, demand, threshold):
    """Edmonds-Karp max flow, independent of Algorithm 3's current flow."""
    nr, nc = cost.shape
    size = nr + nc + 2
    source, sink = size-2, size-1
    capacity = np.zeros((size, size), dtype=int)
    total = int(sum(supply))
    for i in range(nr):
        capacity[source, i] = supply[i]
        for j in range(nc):
            if cost[i, j] <= threshold:
                capacity[i, nr+j] = total
    for j in range(nc):
        capacity[nr+j, sink] = demand[j]
    residual, value = capacity.copy(), 0
    while value < total:
        previous = {source: None}
        queue = deque([source])
        while queue and sink not in previous:
            u = queue.popleft()
            for v in np.flatnonzero(residual[u] > 0):
                v = int(v)
                if v not in previous:
                    previous[v] = u
                    queue.append(v)
        if sink not in previous:
            return None
        delta, v = total, sink
        while previous[v] is not None:
            u = previous[v]
            delta = min(delta, residual[u, v])
            v = u
        v = sink
        while previous[v] is not None:
            u = previous[v]
            residual[u, v] -= delta
            residual[v, u] += delta
            v = u
        value += int(delta)
    return capacity[:nr, nr:nr+nc] - residual[:nr, nr:nr+nc]


def threshold_redistribute(cost, initial):
    d, f = _transport(cost, initial)
    if not np.any(f):
        return {"objective": 0.0, "flow": f}
    levels = sorted(set(d.ravel().tolist()))
    low, high = 0, len(levels)-1
    supply, demand = f.sum(axis=1), f.sum(axis=0)
    while low < high:
        mid = (low + high) // 2
        if _threshold_flow(d, supply, demand, levels[mid]) is None:
            low = mid+1
        else:
            high = mid
    out = _threshold_flow(d, supply, demand, levels[low])
    if out is None:
        raise AssertionError("Complete transport graph must be feasible")
    return {"objective": bottleneck(d, out), "flow": out}


def redistribute_cut(instance, flow, cut, method="alternating"):
    flow = checked_flow(instance, flow)
    schedule = evaluate(instance, flow)
    # cut counts all nodes in the left prefix, including the source.
    if not 1 <= cut < instance.size:
        raise ValueError("Cut must keep source left and sink right")
    left, right = schedule.order[:cut], schedule.order[cut:]
    d = np.array([[schedule.starts[i] + instance.processing[i]
                   + instance.travel[i][j] + schedule.tails[j]
                   for j in right] for i in left])
    old = np.asarray(flow)[np.ix_(left, right)]
    solver = {"alternating": alternating_redistribute,
              "threshold": threshold_redistribute}[method]
    result = solver(d, old)
    out = np.asarray(flow, dtype=int).copy()
    out[np.ix_(left, right)] = result["flow"]
    new = evaluate(instance, out)
    if abs(new.makespan-result["objective"]) > 1e-7:
        raise AssertionError("Frozen cut objective differs from whole-DAG makespan")
    if new.makespan > schedule.makespan+1e-7:
        raise AssertionError("Exact cut redistribution worsened makespan")
    return out, result
