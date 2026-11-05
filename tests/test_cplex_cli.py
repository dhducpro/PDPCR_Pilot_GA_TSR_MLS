"""MILP generation and CLI result handling; never run optimization in tests."""
from pathlib import Path
import contextlib
import io
import json
import os
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from pdpcr.core import Instance, decompose_routes, evaluate, greedy
from pdpcr.cplex_cli import cplex_commands, read_result, solve_cplex, write_model


def tiny_instance():
    return Instance('tiny_zero', 2, [2, 1, 1, 2], [0, 0, 1, 0],
                    [[0, 2, 3, 0], [0, 0, 1, 2], [0, 1, 0, 3], [0, 0, 0, 0]])


def fake_solution(folder, instance, status=107):
    root = ET.parse(folder / 'start.mst').getroot()
    header = root.find('header')
    upper = write_model(instance, folder)['greedy_upper_bound']
    header.set('solutionStatusValue', str(status))
    header.set('solutionStatusString', 'integer optimal solution' if status == 101 else 'time limit exceeded')
    header.set('primalFeasible', '1')
    header.set('objectiveValue', str(upper))
    ET.ElementTree(root).write(folder / 'solution.sol')
    return upper


class CplexCliTests(unittest.TestCase):
    def test_commands_use_wall_time_single_thread_and_seed(self):
        commands = cplex_commands(30, 4)
        self.assertIn('set clocktype 2', commands)
        self.assertIn('set timelimit 30', commands)
        self.assertIn('set threads 1', commands)
        self.assertIn('set randomseed 4', commands)
        self.assertLess(commands.index('read start.mst'), commands.index('optimize'))
        self.assertIn('display solution bestbound', commands)
        for seconds in [0, -1, float('inf')]:
            with self.assertRaises(ValueError):
                cplex_commands(seconds, 0)

    def test_greedy_start_satisfies_exported_constraints_including_order(self):
        instance = tiny_instance()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_model(instance, folder)
            root = ET.parse(folder / 'start.mst').getroot()
            values = {v.get('name'): float(v.get('value')) for v in root.findall('./variables/variable')}
            text = (folder / 'model.lp').read_text()
            constraints = text.split('Subject To\n')[1].split('Bounds\n')[0]
            expressions = re.split(r'\s+[a-z]+_?\w*:\s*', constraints)[1:]
            for expression in expressions:
                lhs, op, rhs = re.split(r'\s+(>=|<=|=)\s+', expression.strip())
                terms = re.findall(r'([+-]?)\s*(?:(\d+(?:\.\d*)?(?:e[+-]?\d+)?)\s+)?([a-z]_\d+(?:_\d+)?)', lhs)
                total = sum((-1 if sign == '-' else 1)*float(coef or 1)*values[name]
                            for sign, coef, name in terms)
                target = float(rhs)
                self.assertTrue({'=': abs(total-target) < 1e-7,
                                 '>=': total >= target-1e-7,
                                 '<=': total <= target+1e-7}[op], expression)
            self.assertGreater(len(expressions), 20)
            self.assertIn('order_1_2: q_2 - q_1 - 4 y_1_2 >= -3', text)
            self.assertIn('order_2_1: q_1 - q_2 - 4 y_2_1 >= -3', text)

    def test_time_limit_incumbent_gets_bound_and_gap_without_optimal_label(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            instance = tiny_instance()
            write_model(instance, folder)
            upper = fake_solution(folder, instance)
            result = read_result(instance, folder, 'Best bound = 1.25\nSolution time = 30.01 sec.')
            self.assertEqual(result['status'], 'feasible')
            self.assertEqual(result['solver_status'], 107)
            self.assertEqual(result['lower_bound'], 1.25)
            self.assertAlmostEqual(result['mip_gap'], (upper-1.25)/(1e-10+upper))
            self.assertEqual(result['solver_wall_seconds'], 30.01)
            self.assertTrue(result['replay']['valid'])

    def test_optimal_status_certifies_bound_but_102_does_not(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            instance = tiny_instance()
            write_model(instance, folder)
            for status in [101, 102]:
                fake_solution(folder, instance, status)
                result = read_result(instance, folder, '')
                self.assertEqual(result['status'], 'optimal' if status == 101 else 'feasible')
                self.assertEqual(result['mip_gap'], 0.0 if status == 101 else None)

    def test_no_incumbent_has_no_greedy_fallback_objective(self):
        with tempfile.TemporaryDirectory() as temp:
            result = read_result(tiny_instance(), temp,
                'MIP - Time limit exceeded, no integer solution.\nBest bound = 1.5')
            self.assertEqual(result['status'], 'no_incumbent')
            self.assertIsNone(result['objective'])
            self.assertIsNone(result['mip_gap'])
            self.assertEqual(result['lower_bound'], 1.5)

    def test_invalid_flow_and_invalid_bound_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            instance = tiny_instance()
            write_model(instance, folder)
            fake_solution(folder, instance)
            with self.assertRaisesRegex(ValueError, 'lower bound'):
                read_result(instance, folder, 'Best bound = 10000')
            root = ET.parse(folder / 'solution.sol').getroot()
            for v in root.findall('./variables/variable'):
                if v.get('name').startswith('x_'):
                    v.set('value', '0')
            ET.ElementTree(root).write(folder / 'solution.sol')
            with self.assertRaises(ValueError):
                read_result(instance, folder, '')

    def test_cli_launch_and_stale_solution_removal_without_optimization(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            executable = folder / 'cplex.exe'
            executable.touch()
            (folder / 'solution.sol').write_text('stale solution')
            instance = tiny_instance()
            def fake_run(args, **kwargs):
                self.assertFalse((folder / 'solution.sol').exists())
                fake_solution(folder, instance)
                self.assertNotIn('shell', kwargs)
                return SimpleNamespace(returncode=0, stdout='Best bound = 1.25', stderr='')
            with patch('pdpcr.cplex_cli.subprocess.run', side_effect=fake_run) as run:
                result = solve_cplex(instance, folder, 30, executable=executable)
            run.assert_called_once()
            self.assertEqual(result['time_limit_seconds'], 30)
            self.assertEqual(result['status'], 'feasible')
            self.assertEqual(result['threads'], 1)
            self.assertTrue((folder / 'commands.txt').is_file())

    def test_notebook_cell_compares_saved_ga_and_reuses_cplex_without_search(self):
        notebook = Path(__file__).resolve().parents[1] / 'run_pilot.ipynb'
        cell = ''.join(json.loads(notebook.read_text(encoding='utf8'))['cells'][-1]['source'])
        instance = tiny_instance()
        flow = greedy(instance)
        schedule = evaluate(instance, flow)
        routes = decompose_routes(instance, flow)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'instances').mkdir()
            (root / 'pdpcr').mkdir()
            (root / 'pdpcr/cplex_cli.py').write_text('mock runner identity')
            instance.save(root / 'instances/tiny_zero.json')
            output = root / 'results/mock'
            (output / 'best').mkdir(parents=True)
            (output / 'config.json').write_text(json.dumps({
                'seeds': list(range(5)), 'seconds_by_n': {'2': 7}}))
            (output / 'comparison.csv').write_text(
                f'instance,n,family,GA_best_of_5\ntiny_zero,2,TEST,{schedule.makespan}\n')
            (output / 'best/tiny_zero__GA.json').write_text(json.dumps({
                'flow': flow.tolist(), 'routes': routes}))
            def fake_solve(instance, folder, seconds, **kwargs):
                self.assertEqual(seconds, 7)
                self.assertEqual(kwargs['threads'], 1)
                Path(folder).mkdir(parents=True)
                return {'status': 'feasible', 'solver_status': 107,
                        'objective': schedule.makespan, 'flow': flow.tolist(), 'routes': routes,
                        'lower_bound': 1.0, 'mip_gap': 0.5,
                        'solver_wall_seconds': 7.01, 'end_to_end_seconds': 7.1}
            context = {'RESULT_NAME': 'mock', 'display': lambda value: None}
            try:
                os.chdir(root)
                with patch('pdpcr.cplex_cli.solve_cplex', side_effect=fake_solve) as solve:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(cell, 'notebook CPLEX cell', 'exec'), context)
                        exec(compile(cell, 'notebook CPLEX cell cached', 'exec'), context)
                    solve.assert_called_once()
                table = context['cplex_vs_ga']
                self.assertEqual(table.iloc[0]['GA_improvement_vs_CPLEX_pct'], 0)
                self.assertEqual(table.iloc[0]['time_limit_seconds'], 7)
                self.assertTrue((output / 'cplex_vs_ga/comparison_best_of_5_vs_cplex.csv').is_file())
                self.assertFalse((output / 'runs.csv').exists())
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
