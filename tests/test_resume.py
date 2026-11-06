"""Disconnect recovery tests; no optimization or worker processes are started."""
import contextlib
import ast
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pdpcr.core import Instance, decompose_routes, evaluate, greedy
import run_pilot as runner


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(runner, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.instance = Instance('tiny', 2, [2, 1, 1, 2], [0, 1, 1, 0],
                                 [[0, 1, 2, 0], [0, 0, 1, 1], [0, 1, 0, 2], [0, 0, 0, 0]])
        self.instance.save(self.root/'instance.json')
        self.selected = [{'name': 'tiny', 'n': 2, 'm': 2, 'family': 'TEST',
                          'instance_file': 'instance.json'}]
        (self.root/'manifest.json').write_text(json.dumps(self.selected))
        self.output = self.root/'results'
        (self.output/'seeds').mkdir(parents=True)
        (self.output/'best').mkdir()
        self.config = dict(sizes=[2], families=['TEST'], seeds=list(range(5)), methods=['GA'],
                           seconds_by_n={'2': 600.0}, workers=2, GA_population_per_process=16,
                           TSR_options={'beta': 0.5, 'tiny_min_candidates': False,
                                        'restart_best': 300, 'restart_new': 600},
                           MLS_options={'local_search_steps': 2, 'local_search_candidates': 8})

    def save_seed(self, seed=0, **extra):
        flow = greedy(self.instance)
        schedule = evaluate(self.instance, flow)
        record = dict(instance='tiny', n=2, m=2, method='GA', seed=seed,
                      objective=schedule.makespan, valid=True, time_limit_seconds=600.0,
                      solver_wall_seconds=600.0, solver_cpu_seconds=600.0,
                      time_limit_overrun_seconds=0, initial_objective=schedule.makespan,
                      flow=flow.tolist(), routes=decompose_routes(self.instance, flow),
                      history=[{'seconds': 0, 'best': schedule.makespan}], **extra)
        path = self.output/'seeds'/f'tiny__GA__seed{seed}.json'
        path.write_text(json.dumps(record))
        return path

    def prepare(self, resume=True, recover=True, config=None):
        return runner._prepare_run_config(self.output, self.selected,
                                          config or self.config, resume, recover)

    def test_missing_config_recovery_preserves_completed_seed(self):
        seed = self.save_seed()
        before = seed.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            restored = self.prepare()
        self.assertEqual(seed.read_bytes(), before)
        self.assertEqual(restored['recovery']['completed_seed_count'], 1)
        self.assertIn('GA_population_per_process', restored['recovery']['unverified_legacy_settings'])
        self.assertEqual(json.loads((self.output/'config.json').read_text()), restored)
        self.assertEqual(json.loads((self.output/'config.backup.json').read_text()), restored)

    def test_backup_restores_missing_or_truncated_main_config(self):
        self.save_seed()
        (self.output/'config.backup.json').write_text(json.dumps(self.config))
        (self.output/'config.json').write_text('{broken')
        restored = self.prepare(recover=False)
        self.assertEqual(restored, self.config)
        self.assertEqual((self.output/'config.json.unreadable').read_text(), '{broken')

    def test_recovery_rejects_different_budget_without_touching_seed(self):
        seed = self.save_seed()
        before = seed.read_bytes()
        config = dict(self.config, seconds_by_n={'2': 1800.0})
        with self.assertRaisesRegex(ValueError, 'different time limit'):
            self.prepare(config=config)
        self.assertEqual(seed.read_bytes(), before)
        self.assertFalse((self.output/'config.json').exists())

    def test_new_seed_config_rejects_different_population(self):
        self.save_seed(run_config=dict(self.config, GA_population_per_process=32))
        with self.assertRaisesRegex(ValueError, 'different experimental settings'):
            self.prepare()

    def test_known_config_mismatch_is_not_overridden_by_recovery(self):
        (self.output/'config.json').write_text(json.dumps(dict(self.config, workers=5)))
        with self.assertRaisesRegex(ValueError, 'workers: saved=5, requested=2'):
            self.prepare()

    def colab_restore(self):
        notebook = Path(__file__).resolve().parents[1]/'run_pilot_colab.ipynb'
        source = ''.join(json.loads(notebook.read_text(encoding='utf8'))['cells'][2]['source'])
        nodes = ast.parse(source).body
        first = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'OUTPUT' for t in node.targets))
        last = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and node.value.args
                    and isinstance(node.value.args[0], ast.Constant)
                    and node.value.args[0].value == 'Cloud project:')
        context = dict(ROOT=self.root, RESULT_NAME='', json=json,
                       SIZES=[50, 100, 200], FAMILIES=['LC1'], SEEDS=list(range(5)),
                       METHODS=('GA', 'TSR', 'MLS'), SECONDS_BY_N={50: 1800},
                       WORKERS=5, GA_POPULATION=32)
        block = ast.Module(body=nodes[first:last], type_ignores=[])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(block, 'Colab saved settings', 'exec'), context)
        return context

    def test_colab_restores_settings_instead_of_notebook_defaults(self):
        (self.output/'config.json').write_text(json.dumps(self.config))
        restored = self.colab_restore()
        self.assertEqual(restored['SIZES'], [2])
        self.assertEqual(restored['FAMILIES'], ['TEST'])
        self.assertEqual(restored['METHODS'], ('GA',))
        self.assertEqual(restored['WORKERS'], 2)
        self.assertEqual(restored['GA_POPULATION'], 16)
        self.assertEqual(restored['SECONDS_BY_N'], {2: 600.0})

    def test_colab_uses_backup_when_main_config_is_broken(self):
        (self.output/'config.json').write_text('{broken')
        (self.output/'config.backup.json').write_text(json.dumps(self.config))
        self.assertEqual(self.colab_restore()['SECONDS_BY_N'], {2: 600.0})

    def test_colab_new_folder_keeps_requested_defaults(self):
        restored = self.colab_restore()
        self.assertEqual(restored['WORKERS'], 5)
        self.assertEqual(restored['GA_POPULATION'], 32)

    def test_colab_restored_settings_resume_without_restarting_saved_seeds(self):
        (self.output/'config.json').write_text(json.dumps(self.config))
        paths = [self.save_seed(seed) for seed in range(5)]
        before = [p.read_bytes() for p in paths]
        restored = self.colab_restore()
        with patch.object(runner, 'ProcessPoolExecutor', side_effect=AssertionError('Unexpected search')):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = runner.run_pilot(sizes=restored['SIZES'], families=restored['FAMILIES'],
                    seeds=restored['SEEDS'], methods=restored['METHODS'],
                    seconds_by_n=restored['SECONDS_BY_N'], workers=restored['WORKERS'],
                    population=restored['GA_POPULATION'], manifest=self.root/'manifest.json',
                    output_dir=self.output, resume=True, recover_missing_config=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual([p.read_bytes() for p in paths], before)

    def test_resume_false_and_non_recovery_preserve_existing_runs(self):
        seed = self.save_seed()
        before = seed.read_bytes()
        with self.assertRaisesRegex(ValueError, 'resume=True'):
            self.prepare(resume=False)
        with self.assertRaisesRegex(ValueError, 'recover_missing_config=True'):
            self.prepare(recover=False)
        self.assertEqual(seed.read_bytes(), before)

    def test_complete_experiment_reuses_all_five_seeds_without_workers(self):
        paths = [self.save_seed(seed) for seed in range(5)]
        before = [path.read_bytes() for path in paths]
        with patch.object(runner, 'ProcessPoolExecutor', side_effect=AssertionError('Unexpected search')):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = runner.run_pilot(sizes=[2], families=['TEST'], methods=['GA'],
                    seeds=list(range(5)), seconds_by_n={2: 600.0}, workers=2, population=16,
                    manifest=self.root/'manifest.json', output_dir=self.output,
                    resume=True, recover_missing_config=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual([path.read_bytes() for path in paths], before)
        self.assertTrue((self.output/'comparison.csv').exists())


if __name__ == '__main__':
    unittest.main()
