"""PDP-CR memetic flow-based genetic algorithm. Reference OPT is never used by search."""
from dataclasses import dataclass
import random
from time import perf_counter
import numpy as np
from .core import checked_flow, evaluate, greedy
from .redistribute import redistribute_cut
from .search import reroute_neighbors, reverse_neighbors
from .flow_operators import min_cost_transport
from .flow_evolution import optimized_cycle_crossover, order_changing_mutation

@dataclass
class Individual:
    flow: np.ndarray
    objective: float
    order: tuple
    starts: np.ndarray
    tails: np.ndarray
    signature: bytes



def individual(instance,flow):
    flow=checked_flow(instance,flow)
    s=evaluate(instance,flow)
    return Individual(flow,s.makespan,tuple(s.order),s.starts,s.tails,flow.tobytes())



def random_cut_mutation(instance, parent, rng, beta=0.05, cut=None, deadline=None):
    """Random-cost feasible transport, bounded by (1+beta)*parent makespan."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('Mutation beta must be finite and nonnegative')
    if deadline is not None and perf_counter() >= deadline:
        return parent
    if cut is None:
        cut = rng.randrange(1, instance.size)
    if not 1 <= cut < instance.size:
        raise ValueError('Invalid cut')
    left, right = parent.order[:cut], parent.order[cut:]
    ix = np.ix_(left, right)
    old = parent.flow[ix]
    delta = np.array([[parent.starts[i]+instance.processing[i]+instance.travel[i][j]
                       +parent.tails[j] for j in right] for i in left])
    threshold = (1+beta)*parent.objective
    costs = np.array([[rng.randrange(1_000_001) for _ in right] for _ in left])
    crossing = min_cost_transport(old.sum(1), old.sum(0), delta <= threshold+1e-9,
                                  costs, deadline)
    if crossing is None:
        return parent
    flow = parent.flow.copy()
    flow[ix] = crossing
    child = individual(instance, flow)
    if child.objective > threshold+1e-7:
        raise AssertionError('Random cut mutation exceeded degradation limit')
    return child



def reroute_mutation(instance, parent, rng, max_attempts=64):
    """Random 2x2 flow exchange, NOT TSR's critical-arc neighborhood.

    New arcs must follow the parent's topological order. Delta is sampled
    uniformly from the positive feasible integers; u_ij = min(r_i, r_j).
    Return the unchanged parent if no valid pair is sampled in max_attempts.
    """
    arcs = [(int(i), int(j)) for i, j in zip(*np.nonzero(parent.flow))]
    position = {node: k for k, node in enumerate(parent.order)}
    if len(arcs) < 2:
        return parent
    for _ in range(max_attempts):
        (i, j), (u, v) = rng.sample(arcs, 2)
        if i == u or j == v or position[i] >= position[v] or position[u] >= position[j]:
            continue
        bound = min(
            int(parent.flow[i, j]), int(parent.flow[u, v]),
            min(instance.requirements[i], instance.requirements[v]) - int(parent.flow[i, v]),
            min(instance.requirements[u], instance.requirements[j]) - int(parent.flow[u, j]),
        )
        if bound <= 0:
            continue
        delta = rng.randint(1, bound)
        flow = parent.flow.copy()
        flow[i, j] -= delta
        flow[u, v] -= delta
        flow[i, v] += delta
        flow[u, j] += delta
        return individual(instance, flow)
    return parent



def mutate(instance, parent, rng, beta=0.05, strategy='mixed_order', deadline=None):
    """GA mutation: order changing (55%) or random feasible flow perturbations."""
    if strategy == 'mixed_order' and rng.random() < 0.55:
        return order_changing_mutation(instance, parent, rng, individual, deadline)
    if strategy in {'mixed', 'mixed_order'} and rng.random() < 0.5:
        return random_cut_mutation(instance, parent, rng, beta=beta, deadline=deadline), 'redistribute'
    return reroute_mutation(instance, parent, rng), 'reroute'



def local_descent(instance, parent, rng, max_steps=3, max_candidates=8, deadline=None):
    """Strict descent using the three existing feasible TSR neighborhoods.

    Each round samples up to K valid reroute moves, K valid reverse moves and
    K cuts, then accepts the best strict improvement found. This is bounded
    local search, not a full neighborhood-optimality claim. No tabu or restart.
    The deadline is the GA's shared wall-clock deadline, not an extra budget.
    """
    if (not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 0
            or not isinstance(max_candidates, int) or isinstance(max_candidates, bool)
            or max_candidates < 1):
        raise ValueError("Local-search steps must be nonnegative and candidates positive integers")
    started = perf_counter()
    current = parent
    stats = {"steps": 0, "reroute": 0, "reverse": 0, "redistribute": 0}

    def expired():
        return deadline is not None and perf_counter() >= deadline

    for _ in range(max_steps):
        if expired():
            break
        best_objective, best_flow = current.objective, None
        for neighbors, kind in ((reroute_neighbors, "reroute"), (reverse_neighbors, "reverse")):
            if expired():
                break
            pool = neighbors(instance, current.flow, deadline=deadline,
                             max_candidates=max_candidates, rng=rng)
            stats[kind] += len(pool)
            for candidate in pool:
                if candidate.objective < best_objective - 1e-9:
                    best_objective, best_flow = candidate.objective, candidate.flow
        cuts = rng.sample(range(1, instance.size), min(max_candidates, instance.size-1))
        for cut in cuts:
            if expired():
                break
            flow, result = redistribute_cut(instance, current.flow, cut)
            stats["redistribute"] += 1
            if result["objective"] < best_objective - 1e-9:
                best_objective, best_flow = result["objective"], flow
        if best_flow is None:
            break
        previous = current.objective
        current = individual(instance, best_flow)
        if current.objective >= previous - 1e-9:
            raise AssertionError("Local descent must strictly improve its input")
        stats["steps"] += 1
    stats.update(seconds=perf_counter()-started,
                 gain=parent.objective-current.objective)
    return current, stats



def select_survivors(candidates, size, duplicate_limit=1):
    """Best P parents+offspring, with an explicitly specified duplicate rule.

    Try the requested per-genotype limit first. If too few distinct candidates
    exist, raise that limit minimally to fill P. Never add random immigrants or
    introduce new candidates; the best candidate always survives.
    Returns (survivors, effective_limit).
    """
    if size < 1 or duplicate_limit < 1 or len(candidates) < size:
        raise ValueError("Insufficient candidates or invalid survivor settings")
    ordered = sorted(candidates, key=lambda x: x.objective)
    limit = duplicate_limit
    while True:
        selected, counts = [], {}
        for candidate in ordered:
            count = counts.get(candidate.signature, 0)
            if count < limit:
                selected.append(candidate)
                counts[candidate.signature] = count + 1
                if len(selected) == size:
                    return selected, limit
        limit += 1



def run_ga(instance, seed=0, population_size=16, generations=10**9,
           crossover_rate=0.9, mutation_rate=0.4, tournament_size=3,
           duplicate_limit=1, time_limit=None, mutation_beta=0.05,
           local_search_steps=2, local_search_candidates=8,
           cycle_trials=4, stagnation_generations=3):
    """Run the GA with flow-DAG crossover, mutation, adaptive diversity and local search.

    The time limit is per independent island. Initialization counts against it.
    Each island has its OWN full population, not a fraction of 16.
    """
    started = perf_counter()
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in
               (population_size, generations, tournament_size, duplicate_limit,
                local_search_steps, local_search_candidates, cycle_trials, stagnation_generations)):
        raise ValueError('GA integer settings must be integers')
    if (population_size < 2 or generations < 0 or tournament_size < 1 or duplicate_limit < 1
            or local_search_steps < 1 or local_search_candidates < 1
            or cycle_trials < 1 or stagnation_generations < 1):
        raise ValueError('Invalid GA settings')
    if not (0 <= crossover_rate <= 1 and 0 <= mutation_rate <= 1):
        raise ValueError('Invalid crossover/mutation probabilities')
    if mutation_beta < 0 or not np.isfinite(mutation_beta):
        raise ValueError('Invalid mutation beta')
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('Time limit must be positive and finite')
    deadline = None if time_limit is None else started + time_limit
    rng = random.Random(seed)
    best = individual(instance, greedy(instance))
    history = [{"seconds": perf_counter() - started, "generation": 0,
                "offspring": 0, "best": best.objective}]
    statistics = {"crossovers": 0, "redistribute": 0, "reroute": 0,
                  "mutations": 0, "clones_of_backbone": 0,
                  "initial_unique": 1, "max_duplicate_limit_used": duplicate_limit,
                  "local_search_calls": 0, "local_search_improved": 0,
                  "local_search_steps": 0, "local_search_gain": 0.0,
                  "local_search_seconds": 0.0, "local_search_reroute": 0,
                  "local_search_reverse": 0, "local_search_redistribute": 0}
    statistics.update(crossover_seconds=0.0, mutation_seconds=0.0,
                      parent_pairs=0, distinct_parent_pairs=0, novel_crossover_children=0,
                      swapped_parent_pairs=0, mutation_unchanged=0, mutation_improved=0,
                      mutation_equal=0, mutation_worsened=0,
                      offspring_novel_before_intensify=0, offspring_novel_after_intensify=0,
                      intensification_calls=0, intensification_improved=0,
                      intensification_solves=0, intensification_steps=0,
                      intensification_seconds=0.0, intensification_gain=0.0,
                      adaptive_triggers=0, immigrants=0, order_attempts=0,
                      order_changed=0)
    population_history = []
    diagnostic_rng = random.Random(seed ^ 0xBA51)

    def log_population(population, generation):
        # At most 32 sampled unordered pairs; a separate RNG cannot steer search.
        pairs = [(i, j) for i in range(len(population)) for j in range(i)]
        sample = diagnostic_rng.sample(pairs, min(32, len(pairs)))
        distance = sum(float(np.abs(population[i].flow-population[j].flow).sum())/2
                       for i, j in sample)/len(sample) if sample else 0.0
        unique = len({x.signature for x in population})
        population_history.append(dict(generation=generation, seconds=perf_counter()-started,
                                       unique=unique, duplicate_fraction=1-unique/len(population),
                                       mean_pairwise_flow_distance=distance, distance_pairs=len(sample),
                                       best=min(x.objective for x in population)))

    def expired():
        return deadline is not None and perf_counter() >= deadline

    def update(candidate, generation, offspring):
        nonlocal best
        if candidate.objective < best.objective - 1e-9:
            best = candidate
            history.append({"seconds": perf_counter() - started,
                            "generation": generation, "offspring": offspring,
                            "best": best.objective})

    # Proposal line 3: constructive heuristics plus flow-preserving perturbations.
    # Bounded attempts avoid hanging when fewer than P distinct DAGs exist.
    initial = [best]
    signatures = {best.signature}
    for attempt in range(8 * population_size):
        if expired() or len(initial) >= population_size:
            break
        flow = greedy(instance, rng.getrandbits(32), randomize=True)
        candidate = individual(instance, flow)
        if attempt % 2:
            # Baseline initialization retained from original GA implementation.
            candidate, _ = mutate(instance, candidate, rng, beta=mutation_beta,
                                  strategy='mixed', deadline=deadline)
        update(candidate, 0, 0)
        if candidate.signature not in signatures:
            initial.append(candidate)
            signatures.add(candidate.signature)
    statistics["initial_unique"] = len(signatures)
    # Fill only with existing valid candidates; no order decoder/immigrant.
    population = [initial[k % len(initial)] for k in range(population_size)]
    population, effective_limit = select_survivors(population, population_size, duplicate_limit)
    statistics["max_duplicate_limit_used"] = effective_limit
    statistics["initial_best"] = best.objective
    statistics['initialization_seconds'] = perf_counter()-started
    log_population(population, 0)
    offspring_total = completed = partial = 0
    generations_stagnant = 0
    effective_mutation = mutation_rate
    for generation in range(1, generations + 1):
        if expired():
            break
        offspring = []
        old_best = best.objective
        while len(offspring) < population_size and not expired():
            # Both parents come from the unchanged current generation.
            def tournament():
                return min(rng.sample(population, min(tournament_size, len(population))),
                           key=lambda x: x.objective)
            p, d = tournament(), tournament()
            statistics['parent_pairs'] += 1
            statistics['distinct_parent_pairs'] += int(p.signature != d.signature)
            parent_signatures = {p.signature, d.signature}
            if rng.random() < crossover_rate:
                crossing_started = perf_counter()
                first, second, cycle_stats = optimized_cycle_crossover(
                    instance, p, d, rng, individual, trials=cycle_trials, deadline=deadline)
                for key, value in cycle_stats.items():
                    statistics['cycle_'+key] = statistics.get('cycle_'+key, 0)+value
                statistics['crossover_seconds'] += perf_counter()-crossing_started
                children = (first, second)
                statistics["crossovers"] += 2
                statistics["clones_of_backbone"] += int(first.signature == p.signature)
                statistics["clones_of_backbone"] += int(second.signature == d.signature)
                statistics['novel_crossover_children'] += sum(c.signature not in parent_signatures for c in children)
                statistics['swapped_parent_pairs'] += int(p.signature != d.signature
                    and first.signature == d.signature and second.signature == p.signature)
            else:
                children = (p, d)
            for child in children:
                if len(offspring) == population_size:
                    break
                # Finish the first child even if its crossover crossed the deadline.
                if offspring and expired():
                    break
                if rng.random() < effective_mutation:
                    before, mutation_started = child, perf_counter()
                    child, kind = mutate(instance, child, rng, beta=mutation_beta,
                                         strategy='mixed_order', deadline=deadline)
                    statistics['mutation_seconds'] += perf_counter()-mutation_started
                    statistics["mutations"] += 1
                    statistics[kind] = statistics.get(kind, 0)+1
                    statistics['order_attempts'] += int(kind in {'reverse', 'free_2x2', 'order_unchanged','order_timeout'})
                    statistics['order_changed'] += int(kind in {'reverse','free_2x2'})
                    statistics['mutation_unchanged'] += int(child.signature == before.signature)
                    outcome = ('improved' if child.objective < before.objective-1e-9 else
                               'worsened' if child.objective > before.objective+1e-9 else 'equal')
                    statistics['mutation_'+outcome] += 1
                statistics['offspring_novel_before_intensify'] += int(child.signature not in parent_signatures)
                if not expired():
                    child, local = local_descent(instance, child, rng,
                                                  max_steps=local_search_steps,
                                                  max_candidates=local_search_candidates,
                                                  deadline=deadline)
                    statistics["local_search_calls"] += 1
                    statistics["local_search_improved"] += int(local["steps"] > 0)
                    for key in ("steps", "gain", "seconds", "reroute", "reverse", "redistribute"):
                        statistics[f"local_search_{key}"] += local[key]
                statistics['offspring_novel_after_intensify'] += int(child.signature not in parent_signatures)
                offspring.append(child)
                offspring_total += 1
                update(child, generation, offspring_total)
        if offspring:
            population, effective_limit = select_survivors(
                population + offspring, population_size, duplicate_limit)
            statistics["max_duplicate_limit_used"] = max(
                statistics["max_duplicate_limit_used"], effective_limit)
            log_population(population, generation)
            if best.objective < old_best - 1e-9:
                generations_stagnant = 0
            else:
                generations_stagnant += 1
            diversity = len({x.signature for x in population}) / len(population)
            if diversity < 0.55 or generations_stagnant >= stagnation_generations:
                effective_mutation = min(0.95, max(mutation_rate*2, 0.7))
                statistics['adaptive_triggers'] += 1
            else:
                effective_mutation = mutation_rate
            # Controlled immigrant rescue: only replaces a bad candidate,
            # never uses any reference optimum or CPLEX result.
            if generations_stagnant >= 2*stagnation_generations and not expired():
                immigrant = individual(instance,greedy(instance,rng.getrandbits(32),randomize=True))
                immigrant, _ = order_changing_mutation(instance,immigrant,rng,individual,deadline)
                population, effective_limit = select_survivors(
                    population[:-1] + [immigrant], population_size, duplicate_limit)
                statistics['immigrants'] += 1
                generations_stagnant = 0
        if len(offspring) == population_size:
            completed += 1
        else:
            partial = len(offspring)
            break
    elapsed = perf_counter() - started
    return {"status": "feasible", "algorithm": "GA", "objective": best.objective,
            "proposal_version": 2, "variant": "GA_memetic",
            "flow": best.flow.copy(), "history": history, "seconds": elapsed,
            "population_history": population_history,
            "generations_completed": completed, "partial_generation_offspring": partial,
            "offspring_completed": offspring_total, "statistics": statistics,
            "population_objectives": [candidate.objective for candidate in population],
            "config": {"seed": seed, "population_size": population_size,
                       "generations_limit": generations, "crossover_rate": crossover_rate,
                       "mutation_rate": mutation_rate, "tournament_size": tournament_size,
                       "duplicate_limit": duplicate_limit, "time_limit": time_limit,
                       "local_search_steps": local_search_steps,
                       "local_search_candidates": local_search_candidates,
                       "crossover_method": "cycle_best", "cycle_trials": cycle_trials,
                       "mutation_beta": mutation_beta, "mutation_strategy": "mixed_order",
                       "adaptive_diversity": True,
                       "stagnation_generations": stagnation_generations,
                       "intensification_probability": 0.0},
            "stop_reason": "time_limit" if expired() else "generation_limit"}

