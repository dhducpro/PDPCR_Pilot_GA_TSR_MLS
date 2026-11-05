"""Multi-start bounded descent baseline for PDP-CR. No GA population/crossover.

Each restart takes a fresh randomized greedy flow-DAG and applies exactly the
same local_descent implementation/budgets as the GA. This isolates GA benefits.
"""
from __future__ import annotations
import random
from time import perf_counter
from .core import greedy
from .genetic import individual, local_descent


def multi_start_descent(instance, seed=0, time_limit=30.0,
                        local_search_steps=2, local_search_candidates=8):
    if time_limit <= 0:
        raise ValueError('time_limit must be positive')
    start = perf_counter()
    deadline = start + time_limit
    rng = random.Random(seed)
    # Exactly the same deterministic common starting solution as GA.
    best = individual(instance, greedy(instance, 0))
    initial_obj = best.objective
    history = [{'seconds': perf_counter()-start, 'best': best.objective,
                'restart': 0}]
    restarts, improvements, local_calls = 0, 0, 0
    while perf_counter() < deadline:
        candidate = individual(instance, greedy(instance, rng.getrandbits(32), randomize=True))
        restarts += 1
        if perf_counter() < deadline:
            candidate, _ = local_descent(instance, candidate, rng,
                                         max_steps=local_search_steps,
                                         max_candidates=local_search_candidates,
                                         deadline=deadline)
            local_calls += 1
        if candidate.objective < best.objective - 1e-9:
            best = candidate
            improvements += 1
            history.append({'seconds':perf_counter()-start,'best':best.objective,
                            'restart':restarts})
    return {'algorithm':'MLS','objective':best.objective,'flow':best.flow,
            'initial_objective':initial_obj,'seconds':perf_counter()-start,
            'history':history,'iterations_completed':restarts,
            'local_search_calls':local_calls,'improvements':improvements,
            'stop_reason':'time_limit'}
