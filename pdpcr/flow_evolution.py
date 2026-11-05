"""GA flow operators adapted from ideas in manu-p-1/cvrp.

This module does NOT import its permutation representation or its distance fitness.
Every candidate stays an integral flow-DAG and is independently evaluated.
"""
from time import perf_counter
import numpy as np

from .core import evaluate
from .flow_operators import exchange_cycles
from .search import reverse_neighbors


def optimized_cycle_crossover(instance, first, second, rng, wrap, trials=4, deadline=None):
    """Best-of-K *feasible* alternating-cycle pair exchanges.

    Cycle components are from PDP-CR's bipartite flow differences, NOT CVRP
    permutation cycle crossover. Prefer best child objective, then distinct
    children, then average quality. At worst, return two unchanged parents.
    """
    if trials < 1:
        raise ValueError('cycle trials must be >= 1')
    statistics = {'optimized_trials': 0, 'optimized_distinct_pairs': 0,
                  'optimized_best_improving_pairs': 0}
    if first.signature == second.signature:
        return first, second, statistics
    parents = {first.signature, second.signature}
    best_pair = (first, second)
    best_score = (min(first.objective, second.objective), 0,
                  first.objective + second.objective)
    for _ in range(trials):
        if deadline is not None and perf_counter() >= deadline:
            break
        fa, fb, extra = exchange_cycles(first.flow, second.flow, rng, deadline)
        a, b = wrap(instance, fa), wrap(instance, fb)
        novelty = int(a.signature not in parents) + int(b.signature not in parents)
        score = (min(a.objective, b.objective), -novelty,
                 a.objective+b.objective)
        statistics['optimized_trials'] += 1
        statistics['optimized_distinct_pairs'] += int(novelty > 0)
        statistics['optimized_best_improving_pairs'] += int(
            min(a.objective, b.objective) < min(first.objective, second.objective) - 1e-9)
        for key, value in extra.items():
            statistics[key] = statistics.get(key, 0)+value
        if score < best_score:
            best_pair, best_score = (a, b), score
    return *best_pair, statistics


def order_changing_mutation(instance, parent, rng, wrap, deadline=None, attempts=48):
    """Try a precedence-reversing flow move, then unrestricted 2x2 rerouting.

    Both moves preserve in/out flow totals. Accept candidates only if a full
    flow evaluator confirms an acyclic support and a changed topological order.
    Non-improving moves are allowed for diversification. Never reads OPT.
    """
    if deadline is not None and perf_counter() >= deadline:
        return parent, 'order_timeout'
    # Reverse a positive critical arc together with its incoming/outgoing cover.
    if rng.random() < 0.8:
        moves = reverse_neighbors(instance, parent.flow, deadline=deadline,
                                  max_candidates=8, rng=rng)
        candidates = [c for c in moves if tuple(evaluate(instance, c.flow).order) != parent.order]
        if candidates:
            chosen = rng.choice(candidates)
            return wrap(instance, chosen.flow), 'reverse'
    # 2x2 swaps can now introduce arcs outside the current topo order, but a
    # DAG check (rather than a priori ordering restriction) guarantees validity.
    arcs = [(int(i),int(j)) for i,j in zip(*np.nonzero(parent.flow))]
    if len(arcs) < 2:
        return parent, 'order_unchanged'
    for _ in range(attempts):
        if deadline is not None and perf_counter() >= deadline:
            break
        (i,j),(u,v) = rng.sample(arcs, 2)
        if i == u or j == v or i == v or u == j:
            continue
        bound = min(int(parent.flow[i,j]), int(parent.flow[u,v]),
                    min(instance.requirements[i],instance.requirements[v])-int(parent.flow[i,v]),
                    min(instance.requirements[u],instance.requirements[j])-int(parent.flow[u,j]))
        if bound <= 0:
            continue
        flow = parent.flow.copy()
        delta = rng.randint(1,bound)
        flow[i,j] -= delta; flow[u,v] -= delta
        flow[i,v] += delta; flow[u,j] += delta
        try:
            new_order = tuple(evaluate(instance,flow).order)
        except ValueError:
            continue
        if new_order != parent.order:
            return wrap(instance,flow), 'free_2x2'
    return parent, 'order_unchanged'
