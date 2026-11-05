"""TSR: tabu search with reroute, reverse and cut redistribution."""
from dataclasses import dataclass
import random
from time import perf_counter

import numpy as np

from .core import evaluate, greedy
from .redistribute import redistribute_cut


@dataclass
class Candidate:
    flow: np.ndarray
    objective: float
    kind: str
    added: list[tuple[int, int]]
    removed: list[tuple[int, int]]
    cut: int | None = None


def _candidate(instance, original, changes, kind):
    out = original.copy()
    for i, j, delta in changes:
        out[i, j] += delta
    if np.array_equal(out, original):
        return None
    try:
        obj = evaluate(instance, out).makespan
    except ValueError:
        return None
    added = list(zip(*np.nonzero(out > original)))
    removed = list(zip(*np.nonzero(out < original)))
    return Candidate(out, obj, kind, added, removed)


def critical_arcs(instance, flow):
    schedule = evaluate(instance, flow)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(flow))
            if abs(schedule.starts[i]+instance.processing[i]+instance.travel[i][j]
                   +schedule.tails[j]-schedule.makespan) < 1e-7]


def reroute_neighbors(instance, flow, deadline=None, max_candidates=None, rng=None):
    arcs = [(int(i), int(j)) for i, j in zip(*np.nonzero(flow))]
    critical = critical_arcs(instance, flow)
    if not critical:
        return []
    i, j = min(critical, key=lambda a: (-instance.travel[a[0]][a[1]], a))
    out = []
    if rng is not None:
        rng.shuffle(arcs)
    for u, v in arcs:
        if deadline is not None and perf_counter() >= deadline:
            break
        if i == u or j == v:
            continue
        q = min(flow[i, j], flow[u, v])
        # Correct endpoint is (u,j); the manuscript mistakenly prints (v,j).
        c = _candidate(instance, flow,
                       [(i,j,-q), (u,v,-q), (i,v,q), (u,j,q)], "reroute")
        if c:
            out.append(c)
            if max_candidates is not None and len(out)>=max_candidates:
                break
    return out


def reverse_neighbors(instance, flow, deadline=None, max_candidates=None, rng=None):
    schedule = evaluate(instance, flow)
    out = []
    arcs=critical_arcs(instance,flow)
    if rng is not None:
        rng.shuffle(arcs)
    for i, j in arcs:
        if deadline is not None and perf_counter() >= deadline:
            break
        if i == 0 or j == instance.sink:
            continue
        q = int(flow[i, j])
        incoming = sorted(np.flatnonzero(flow[:, i]),
                          key=lambda u: (schedule.starts[u]+instance.processing[u]
                                         +instance.travel[u][i], int(u)))
        outgoing = sorted(np.flatnonzero(flow[j]),
                          key=lambda v: (instance.travel[j][v], int(v)))
        # A priority prefix alone need not be inclusion-minimal. Prune redundant
        # members, preferring to retain earlier members of the priority ordering.
        def minimal_cover(nodes, amount):
            chosen, total = [], 0
            for node in nodes:
                chosen.append(node)
                total += amount(node)
                if total >= q:
                    break
            for node in list(reversed(chosen)):
                if total-amount(node) >= q:
                    chosen.remove(node)
                    total -= amount(node)
            return chosen
        incoming = minimal_cover(incoming,lambda u:int(flow[u,i]))
        outgoing = minimal_cover(outgoing,lambda v:int(flow[j,v]))
        changes = [(i, j, -q), (j, i, q)]
        remain = q
        for u in incoming:
            amount = min(remain, int(flow[u, i]))
            changes.extend([(int(u),i,-amount), (int(u),j,amount)])
            remain -= amount
            if remain == 0:
                break
        remain = q
        for v in outgoing:
            amount = min(remain, int(flow[j, v]))
            changes.extend([(j,int(v),-amount), (i,int(v),amount)])
            remain -= amount
            if remain == 0:
                break
        candidate = _candidate(instance, flow, changes, "reverse")
        if candidate:
            out.append(candidate)
            if max_candidates is not None and len(out)>=max_candidates:
                break
    return out


def tabu_search(instance, iterations=5000, seed=0, beta=0.5,
                tiny_min_candidates=True, initial=None, restart_best=300,
                restart_new=600, time_limit=None):
    """Table 4 candidate budgets; clamp for tiny n is an explicit deviation.

    Added-arc tabu, FIFO entries, best-so-far aspiration, and deterministic
    topological ties are fully specified here; draft omits some of these choices.
    This reference implementation prioritizes auditability over scalability.
    """
    started = perf_counter()
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit < 0):
        raise ValueError("Time limit must be finite and nonnegative")
    deadline = started+time_limit if time_limit is not None else None
    if not isinstance(iterations, int) or iterations < 0 or not 0 <= beta <= 1:
        raise ValueError("Invalid search configuration")
    rng = random.Random(seed)
    current = greedy(instance, seed) if initial is None else initial.copy()
    current_obj = evaluate(instance, current).makespan
    best, best_obj = current.copy(), current_obj
    initial_obj = current_obj
    divisors = (5,15,15)
    budgets = [instance.n//d for d in divisors]
    if tiny_min_candidates:
        budgets = [max(1,b) for b in budgets]
    capacities = [instance.n//8, instance.n//15]
    if tiny_min_candidates:
        capacities = [max(1,c) for c in capacities]
    tabu_added, tabu_cuts = [], []
    last_improve = 0
    history = [{"iteration":0, "current":current_obj, "best":best_obj,
                "move":"initial", "seconds":perf_counter()-started}]
    generated = {"reroute":0,"reverse":0,"redistribute":0}
    for it in range(1, iterations+1):
        if deadline is not None and perf_counter() >= deadline:
            break
        # Reconstructed trigger: iterations since improving the global best.
        stagnant = it-last_improve
        if restart_new and stagnant % restart_new == 0:
            current = greedy(instance, seed+it, randomize=True)
            current_obj = evaluate(instance,current).makespan
            tabu_added.clear(); tabu_cuts.clear()
        elif restart_best and stagnant % restart_best == 0:
            current, current_obj = best.copy(), best_obj
            tabu_added.clear(); tabu_cuts.clear()
        forbidden = set(a for entry in tabu_added for a in entry)
        candidates = []
        for neighbors,budget in [(reroute_neighbors,budgets[0]),
                                 (reverse_neighbors,budgets[1])]:
            if budget:
                # In timed experiments, sample without enumerating/evaluating
                # the entire valid neighborhood first. Random arc order and
                # stopping at K valid moves avoids a full neighborhood scan.
                pool = neighbors(instance,current,deadline=deadline,
                                 max_candidates=budget if deadline is not None else None,
                                 rng=rng if deadline is not None else None)
                rng.shuffle(pool)
                for cand in pool[:budget]:
                    generated[cand.kind] += 1
                    if not forbidden.intersection(cand.added) or cand.objective < best_obj-1e-9:
                        candidates.append(cand)
        if budgets[2]:
            cuts = list(range(1,instance.size))
            rng.shuffle(cuts)
            for cut in cuts[:budgets[2]]:
                if deadline is not None and perf_counter() >= deadline:
                    break
                new, result = redistribute_cut(instance,current,cut)
                generated["redistribute"] += 1
                if not np.array_equal(new,current):
                    cand = Candidate(new,result["objective"],"redistribute",[],[],cut)
                    if cut not in tabu_cuts or cand.objective < best_obj-1e-9:
                        candidates.append(cand)
        chosen = min(candidates,key=lambda c:c.objective) if candidates else None
        if chosen and chosen.objective >= current_obj-1e-9 and chosen.kind == "redistribute":
            alternatives = [c for c in candidates if c.kind != "redistribute"]
            if alternatives and rng.random() >= beta:
                chosen = min(alternatives,key=lambda c:c.objective)
        move = "no_admissible_move"
        if chosen:
            current,current_obj,move = chosen.flow,chosen.objective,chosen.kind
            if chosen.kind == "redistribute":
                tabu_cuts.append(chosen.cut)
                tabu_cuts = tabu_cuts[-capacities[1]:] if capacities[1] else []
            else:
                tabu_added.append(chosen.removed)
                tabu_added = tabu_added[-capacities[0]:] if capacities[0] else []
        if current_obj < best_obj-1e-9:
            best,best_obj,last_improve = current.copy(),current_obj,it
        history.append({"iteration":it,"current":current_obj,"best":best_obj,"move":move,
                        "seconds":perf_counter()-started})
    return {"status":"feasible", "algorithm":"TSR","objective":best_obj,
            "initial_objective":initial_obj,"flow":best,"history":history,
            "seed":seed,"beta":beta,"budgets":budgets,
            "tiny_min_candidates":tiny_min_candidates,"generated":generated,
            "seconds":perf_counter()-started,"iterations_completed":len(history)-1,
            "time_limit":time_limit,
            "stop_reason":"time_limit" if deadline is not None and perf_counter() >= deadline
                          else "iteration_limit"}
