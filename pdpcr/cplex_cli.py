"""Time-limited PDP-CR MILP through the installed CPLEX executable.

No Python CPLEX package is needed. Flow conservation, arc activation, synchronized
start times, and independent order constraints describe the same flow-DAG model
as core.evaluate(), including zero-duration requests. A common greedy MIP start
is used; no GA/TSR/MLS results enter the solve.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math
import re
import subprocess
import textwrap
import xml.etree.ElementTree as ET

import numpy as np

from .core import decompose_routes, evaluate, greedy, replay_routes

DEFAULT_CPLEX_EXE = Path(
    r'C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cplex\bin\x64_win64\cplex.exe')
MODEL_VERSION = 'flow_dag_order_v1'


def instance_fingerprint(instance):
    raw = json.dumps(instance.__dict__, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf8')).hexdigest()


def write_model(instance, folder):
    """Write LP + a complete feasible greedy MIP start, without running a solver."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    flow = greedy(instance)
    schedule = evaluate(instance, flow)
    upper = schedule.makespan
    arcs = [(i, j) for i in range(instance.sink)
            for j in range(1, instance.size) if i != j]
    outgoing = [[] for _ in range(instance.size)]
    incoming = [[] for _ in range(instance.size)]
    for i, j in arcs:
        outgoing[i].append(f'x_{i}_{j}')
        incoming[j].append(f'x_{i}_{j}')

    def expression(terms):
        return ' '.join(f'{"+" if c >= 0 else "-"} {abs(c):.17g} {v}'
                        for c, v in terms if c) or '0 s_0'

    with (folder / 'model.lp').open('w', encoding='ascii') as f:
        def line(value):
            f.write(textwrap.fill(value, width=180, subsequent_indent='  ',
                                 break_long_words=False, break_on_hyphens=False) + '\n')
        line('Minimize')
        line(f' obj: s_{instance.sink}')
        line('Subject To')
        for i in range(instance.sink):
            line(f' out_{i}: ' + expression((1, v) for v in outgoing[i])
                 + f' = {instance.requirements[i]}')
        for j in range(1, instance.size):
            line(f' in_{j}: ' + expression((1, v) for v in incoming[j])
                 + f' = {instance.requirements[j]}')
        for i, j in arcs:
            x, y = f'x_{i}_{j}', f'y_{i}_{j}'
            capacity = min(instance.requirements[i], instance.requirements[j])
            delay = instance.processing[i] + instance.travel[i][j]
            big_m = upper + delay
            line(f' use_lo_{i}_{j}: {x} - {y} >= 0')
            line(f' use_hi_{i}_{j}: {x} - {capacity} {y} <= 0')
            line(f' time_{i}_{j}: ' + expression([
                (1, f's_{j}'), (-1, f's_{i}'), (-big_m, y)]) + f' >= {-upper:.17g}')
            # Time precedence alone cannot exclude a cycle of zero-cost arcs.
            line(f' order_{i}_{j}: q_{j} - q_{i} - {instance.size} {y}'
                 + f' >= {1-instance.size}')
        work = sum(r*c for r, c in zip(instance.requirements, instance.processing))
        line(' work: ' + expression([(instance.robots, f's_{instance.sink}')]
             + [(-instance.travel[i][j], f'x_{i}_{j}') for i, j in arcs]) + f' >= {work:.17g}')
        line('Bounds')
        for i in range(instance.size):
            line(f' 0 <= s_{i} <= {0 if i == 0 else upper:.17g}')
            line(f' 0 <= q_{i} <= {0 if i == 0 else instance.size-1}')
        for i, j in arcs:
            line(f' 0 <= x_{i}_{j} <= {min(instance.requirements[i], instance.requirements[j])}')
        line('Binary')
        for i, j in arcs:
            line(f' y_{i}_{j}')
        line('General')
        for i, j in arcs:
            line(f' x_{i}_{j}')
        line('End')

    root = ET.Element('CPLEXSolution', version='1.2')
    ET.SubElement(root, 'header', problemName='model.lp', solutionName='common_greedy',
                  MIPStartEffortLevel='1')
    variables = ET.SubElement(root, 'variables')
    def variable(name, value):
        ET.SubElement(variables, 'variable', name=name, value=format(float(value), '.17g'))
    for i, j in arcs:
        variable(f'x_{i}_{j}', flow[i, j])
        variable(f'y_{i}_{j}', int(flow[i, j] > 0))
    order = {node: k for k, node in enumerate(schedule.order)}
    for i in range(instance.size):
        variable(f's_{i}', schedule.starts[i])
        variable(f'q_{i}', order[i])
    ET.ElementTree(root).write(folder / 'start.mst', encoding='utf-8', xml_declaration=True)
    return {'greedy_upper_bound': upper, 'arc_count': len(arcs),
            'variable_count': 2*len(arcs)+2*instance.size}


def cplex_commands(seconds, seed, threads=1):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('CPLEX time limit must be finite and positive')
    if not isinstance(seed, int) or not 0 <= seed <= 2100000000:
        raise ValueError('Invalid CPLEX random seed')
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('CPLEX threads must be a positive integer')
    return ['read model.lp', 'set clocktype 2', f'set timelimit {seconds:.17g}',
            f'set threads {threads}', f'set randomseed {seed}',
            'set mip tolerances mipgap 0', 'set mip tolerances absmipgap 0',
            'set mip tolerances integrality 1e-9',
            'set simplex tolerances feasibility 1e-9', 'read start.mst',
            'optimize', 'display solution bestbound', 'display solution quality',
            'write solution.sol', 'quit']


NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def read_result(instance, folder, log):
    """Only CPLEX's exported incumbent is called a CPLEX solution."""
    folder = Path(folder)
    matches = re.findall(r'Best\s+(?:MIP\s+)?bound\s*(?:=|:)\s*(' + NUMBER + ')', log, re.I)
    lower_bound = float(matches[-1]) if matches else None
    timing = re.findall(r'Solution time\s*=\s*(' + NUMBER + r')\s*sec', log, re.I)
    result = {'status': 'no_incumbent', 'solver_status': None, 'message': '',
              'objective': None, 'solver_objective': None, 'lower_bound': lower_bound,
              'lower_bound_source': 'CPLEX display solution bestbound' if matches else None,
              'mip_gap': None, 'valid': False, 'flow': None, 'routes': None,
              'solver_wall_seconds': float(timing[-1]) if timing else None}
    status_lines = re.findall(r'^\s*MIP\s*-\s*(.+)$', log, re.M)
    if status_lines:
        result['message'] = status_lines[-1]
    solution = folder / 'solution.sol'
    if not solution.exists():
        if not re.search(r'no (?:integer )?solution|infeasible|time limit', log, re.I):
            raise RuntimeError(f'CPLEX did not export a solution; inspect {folder / "cplex.log"}')
        return result
    root = ET.parse(solution).getroot()
    header = root.find('header')
    if header is None or 'objectiveValue' not in header.attrib:
        raise ValueError('CPLEX solution has no incumbent objective')
    status = int(header.attrib['solutionStatusValue'])
    # 101 optimal, 102 optimal within tolerance; others require primal feasibility.
    if header.get('primalFeasible') != '1' and status not in (101, 102):
        raise ValueError('CPLEX exported a solution without primal feasibility')
    value = float(header.attrib['objectiveValue'])
    if not math.isfinite(value):
        raise ValueError('Nonfinite CPLEX objective')
    values = {}
    for variable in root.findall('./variables/variable'):
        name = variable.attrib['name']
        number = float(variable.attrib['value'])
        if name in values or not math.isfinite(number):
            raise ValueError('Duplicate or nonfinite CPLEX variable')
        values[name] = number
    flow = np.zeros((instance.size, instance.size), dtype=int)
    # SOL may omit zero-valued variables. Their value is zero, never a warm-start fallback.
    for i in range(instance.sink):
        for j in range(1, instance.size):
            if i == j:
                continue
            raw = values.get(f'x_{i}_{j}', 0.0)
            if abs(raw-round(raw)) > 1e-6:
                raise ValueError('Noninteger CPLEX robot flow')
            flow[i, j] = round(raw)
    schedule = evaluate(instance, flow)
    routes = decompose_routes(instance, flow)
    replay = replay_routes(instance, routes, schedule.starts)
    tolerance = 1e-6 * max(1, abs(value))
    if abs(replay['actual_return_max']-schedule.makespan) > tolerance or schedule.makespan > value+tolerance:
        raise ValueError('CPLEX objective disagrees with physical replay')
    # Independently left-shifting an incumbent schedule can improve its reported objective.
    actual = schedule.makespan
    if status == 101:
        if abs(actual-value) > tolerance:
            raise ValueError('Optimal CPLEX objective disagrees with replay')
        lower_bound = value
        result['lower_bound_source'] = 'CPLEX status 101 with zero gap tolerances'
    if lower_bound is not None and lower_bound > actual+tolerance:
        raise ValueError('CPLEX lower bound exceeds the replayed incumbent')
    result.update(status='optimal' if status == 101 else 'feasible', solver_status=status,
                  message=header.get('solutionStatusString', ''), objective=actual,
                  solver_objective=value, lower_bound=lower_bound,
                  mip_gap=max(0.0, (actual-lower_bound)/(1e-10+abs(actual))) if lower_bound is not None else None,
                  valid=True, flow=flow.tolist(), routes=routes,
                  starts=schedule.starts.tolist(), replay=replay,
                  mip_node_count=int(header.get('MIPNodes', '0')))
    return result


def solve_cplex(instance, folder, seconds, seed=0, threads=1, executable=DEFAULT_CPLEX_EXE):
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f'CPLEX executable not found: {executable}')
    commands = cplex_commands(float(seconds), seed, threads)
    started = perf_counter()
    folder = Path(folder).resolve()
    model = write_model(instance, folder)
    generation_seconds = perf_counter()-started
    (folder / 'solution.sol').unlink(missing_ok=True)
    (folder / 'commands.txt').write_text('\n'.join(commands)+'\n', encoding='ascii')
    cli_started = perf_counter()
    completed = subprocess.run([str(executable), '-c', *commands], cwd=folder,
                               capture_output=True, text=True, errors='replace')
    cli_seconds = perf_counter()-cli_started
    log = completed.stdout + completed.stderr
    (folder / 'cplex.log').write_text(log, encoding='utf8')
    errors = re.findall(r'CPLEX Error\s+(\d+)', log)
    if completed.returncode != 0 or any(code != '1217' for code in errors):
        raise RuntimeError(f'CPLEX failed; inspect {folder / "cplex.log"}\n{log[-1500:]}')
    result = read_result(instance, folder, log)
    result.update(instance=instance.name, n=instance.n, m=instance.robots, method='CPLEX',
                  seed=seed, threads=threads, time_limit_seconds=float(seconds),
                  clock_type='wall', warm_start='common deterministic greedy; no heuristic results',
                  cplex_executable=str(executable), artifacts_dir=str(folder),
                  model_version=MODEL_VERSION, instance_sha256=instance_fingerprint(instance),
                  model_generation_seconds=generation_seconds, cli_wall_seconds=cli_seconds,
                  end_to_end_seconds=perf_counter()-started, **model)
    return result
