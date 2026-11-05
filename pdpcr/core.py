from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
import random

import numpy as np


@dataclass
class Instance:
    name: str
    robots: int
    requirements: list[int]  # Includes source and sink, each with requirement m.
    processing: list[float]
    travel: list[list[float]]

    def __post_init__(self):
        self.requirements = list(self.requirements)
        self.processing = list(self.processing)
        self.travel = [list(row) for row in self.travel]
        size = len(self.requirements)
        if not isinstance(self.robots, int) or self.robots < 1 or size < 3:
            raise ValueError("Need a positive integer fleet and at least one task")
        if len(self.processing) != size or np.shape(self.travel) != (size, size):
            raise ValueError("Inconsistent instance dimensions")
        if any(not isinstance(r, int) or not 1 <= r <= self.robots
               for r in self.requirements):
            raise ValueError("Requirements must be integers in [1, m]")
        if self.requirements[0] != self.robots or self.requirements[-1] != self.robots:
            raise ValueError("Set r_source = r_sink = m")
        if self.processing[0] != 0 or self.processing[-1] != 0:
            raise ValueError("Depots must have zero processing time")
        if (not np.all(np.isfinite(self.processing))
                or any(c < 0 for c in self.processing[1:-1])):
            # Coincident Li-Lim pickup/delivery coordinates give c_i = 0.
            # evaluate() checks acyclicity explicitly, including zero-cost arcs.
            raise ValueError("Processing times must be finite and nonnegative")
        if not np.all(np.isfinite(self.travel)) or np.min(self.travel) < 0:
            raise ValueError("Travel times must be finite and nonnegative")

    @property
    def n(self):
        return len(self.requirements) - 2

    @property
    def sink(self):
        return self.n + 1

    @property
    def size(self):
        return self.n + 2

    def save(self, path):
        Path(path).write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def checked_flow(instance: Instance, flow):
    """Check integer robots, allowed arcs, and all row/column balances."""
    a = np.asarray(flow, dtype=float)
    if a.shape != (instance.size, instance.size) or not np.all(np.isfinite(a)):
        raise ValueError("Invalid flow dimensions/values")
    if np.any(a < -1e-7) or not np.allclose(a, np.rint(a), atol=1e-7, rtol=0):
        raise ValueError("Robot flows must be nonnegative integers")
    f = np.rint(a).astype(int)
    if np.any(np.diag(f)) or np.any(f[:, 0]) or np.any(f[-1, :]):
        raise ValueError("Self-loops, arcs into source, and arcs out of sink are forbidden")
    if not np.array_equal(f.sum(axis=1), instance.requirements[:-1] + [0]):
        raise ValueError("Invalid outgoing robot count")
    if not np.array_equal(f.sum(axis=0), [0] + instance.requirements[1:]):
        raise ValueError("Invalid incoming robot count")
    return f


@dataclass
class Schedule:
    makespan: float
    starts: np.ndarray
    tails: np.ndarray
    order: list[int]


def evaluate(instance: Instance, flow) -> Schedule:
    """Algorithms 1 and 2; ties in Kahn's topological sort use node ID."""
    f = checked_flow(instance, flow)
    adj = [list(np.flatnonzero(row)) for row in f]
    indegree = (f > 0).sum(axis=0)
    ready = [int(i) for i in np.flatnonzero(indegree == 0)]
    heapq.heapify(ready)
    order = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for j in adj[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                heapq.heappush(ready, int(j))
    if len(order) != instance.size:
        raise ValueError("Cyclic flow graph")
    starts = np.full(instance.size, -np.inf)
    starts[0] = 0.0
    for i in order:
        for j in adj[i]:
            starts[j] = max(starts[j], starts[i] + instance.processing[i]
                            + instance.travel[i][j])
    tails = np.full(instance.size, -np.inf)
    tails[instance.sink] = 0.0
    for i in reversed(order):
        for j in adj[i]:
            tails[i] = max(tails[i], instance.processing[i]
                           + instance.travel[i][j] + tails[j])
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(tails)):
        raise ValueError("Every task must lie on a source-to-sink robot path")
    return Schedule(float(starts[-1]), starts, tails, order)


def flow_from_routes(instance: Instance, routes):
    if len(routes) != instance.robots:
        raise ValueError("Need one route per robot, including unused robots")
    f = np.zeros((instance.size, instance.size), dtype=int)
    for route in routes:
        path = [0] + list(route) + [instance.sink]
        if any(not 1 <= i <= instance.n for i in route) or len(set(route)) != len(route):
            raise ValueError("Invalid or repeated task on a robot route")
        for i, j in zip(path, path[1:]):
            f[i, j] += 1
    evaluate(instance, f)
    return f


def decompose_routes(instance: Instance, flow):
    evaluate(instance, flow)
    # MILP values such as 0.999999999 must be tolerance-checked and rounded,
    # never truncated to zero by a direct integer cast.
    remaining = checked_flow(instance, flow).copy()
    routes = []
    for _ in range(instance.robots):
        i, route = 0, []
        while i != instance.sink:
            nxt = np.flatnonzero(remaining[i] > 0)
            if not len(nxt):
                raise ValueError("Incomplete robot path")
            j = int(nxt[0])
            remaining[i, j] -= 1
            if j != instance.sink:
                route.append(j)
            i = j
        routes.append(route)
    if np.any(remaining):
        raise ValueError("Flow left after decomposition")
    return routes


def replay_routes(instance: Instance, routes, starts):
    """Independent physical replay of supplied times, without the DAG evaluator."""
    if len(routes) != instance.robots or len(starts) != instance.size:
        raise ValueError("Invalid replay dimensions")
    if not np.all(np.isfinite(starts)) or abs(starts[0]) > 1e-6:
        raise ValueError("Invalid start times")
    counts = [0] * instance.size
    returns = []
    for route in routes:
        if len(set(route)) != len(route):
            raise ValueError("Robot repeats a task")
        prev, finish = 0, 0.0
        for j in route:
            if not 1 <= j <= instance.n:
                raise ValueError("Unknown task")
            arrival = finish + instance.travel[prev][j]
            if starts[j] + 1e-6 < arrival:
                raise ValueError("Robot arrives after synchronized task start")
            counts[j] += 1
            prev, finish = j, float(starts[j] + instance.processing[j])
        returns.append(finish + instance.travel[prev][instance.sink])
    if counts[1:-1] != instance.requirements[1:-1]:
        raise ValueError("Incorrect number of robots assigned to a task")
    actual = max(returns)
    if starts[-1] + 1e-6 < actual:
        raise ValueError("Reported makespan precedes a robot return")
    return {"valid": True, "actual_return_max": actual,
            "robot_returns": returns, "task_robot_counts": counts[1:-1]}


def greedy(instance: Instance, seed=0, randomize=False):
    """Explicit reconstruction choice: earliest completing task/robot allocation."""
    rng = random.Random(seed)
    pending = set(range(1, instance.n + 1))
    last = [0] * instance.robots
    finish = [0.0] * instance.robots
    routes = [[] for _ in last]
    while pending:
        options = []
        for j in sorted(pending):
            bots = sorted(range(instance.robots),
                          key=lambda b: (finish[b] + instance.travel[last[b]][j], b))
            bots = bots[:instance.requirements[j]]
            done = max(finish[b] + instance.travel[last[b]][j] for b in bots)
            done += instance.processing[j]
            options.append((done, j, bots))
        options.sort()
        choice = rng.choice(options[:min(3, len(options))]) if randomize else options[0]
        done, j, bots = choice
        for b in bots:
            last[b], finish[b] = j, done
            routes[b].append(j)
        pending.remove(j)
    return flow_from_routes(instance, routes)
