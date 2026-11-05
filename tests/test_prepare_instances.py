"""Regression checks for nominal Li-Lim sizes and zero-distance requests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_instances as prep
from pdpcr.core import Instance, decompose_routes, evaluate, greedy, replay_routes
from pdpcr.li_lim import convert_li_lim


class PrepareInstancesTests(unittest.TestCase):
    def test_lc101_selects_exactly_50_complete_pairs(self):
        source = prep.DATA / 'lc101.txt'
        parent, metadata = convert_li_lim(source)
        self.assertEqual(parent.n, 53)
        with tempfile.TemporaryDirectory(dir=prep.ROOT) as folder:
            output = Path(folder)
            with patch.object(prep, 'OUT', output):
                records = prep.prepare([50], ['LC1'], download=False)
            record = records[0]
            instance = Instance.load(prep.ROOT / record['instance_file'])
            self.assertEqual(instance.n, 50)
            self.assertEqual(record['source_task_count'], 53)
            self.assertIn('first50', instance.name)
            self.assertEqual(record['selected_pairs'], [
                [t['pickup_id'], t['delivery_id']] for t in metadata['tasks'][:50]])
            self.assertEqual(instance.requirements[1:-1], parent.requirements[1:51])
            indices = [0] + list(range(1, 51)) + [parent.sink]
            self.assertEqual(instance.processing, [parent.processing[i] for i in indices])
            self.assertEqual(instance.travel, [
                [parent.travel[i][j] for j in indices] for i in indices])
            self.assertEqual(json.loads((output / 'manifest.json').read_text()), records)
            # Independently replay the selected requests, including zero durations.
            flow = greedy(instance)
            schedule = evaluate(instance, flow)
            replay = replay_routes(instance, decompose_routes(instance, flow), schedule.starts)
            self.assertAlmostEqual(replay['actual_return_max'], schedule.makespan)

    def test_exact_source_keeps_all_tasks_and_name(self):
        source = prep.DATA / 'LR2_2_6.pdp'
        parent, _ = convert_li_lim(source, expected_tasks=100)
        with tempfile.TemporaryDirectory(dir=prep.ROOT) as folder:
            with patch.object(prep, 'OUT', Path(folder)), patch.object(
                    prep, 'get_source', return_value=(source, 'local_cache')):
                record = prep.prepare([100], ['LR2'], download=False)[0]
            self.assertEqual(record['name'], parent.name)
            self.assertEqual(record['source_task_count'], 100)
            self.assertEqual(record['selection'], 'all complete requests')

    def test_insufficient_source_fails_without_manifest(self):
        source = prep.DATA / 'lc101.txt'
        with tempfile.TemporaryDirectory(dir=prep.ROOT) as folder:
            output = Path(folder)
            with patch.object(prep, 'OUT', output), patch.object(
                    prep, 'get_source', return_value=(source, 'local_cache')):
                with self.assertRaisesRegex(ValueError, 'source contains only 53'):
                    prep.prepare([100], ['LC1'], download=False)
            self.assertEqual(list(output.iterdir()), [])

    def test_strict_converter_still_rejects_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'Expected 50 complete tasks, found 53'):
            convert_li_lim(prep.DATA / 'lc101.txt', expected_tasks=50)

    def test_zero_processing_does_not_allow_cyclic_flows(self):
        instance = Instance('zero', 5, [5, 2, 2, 5], [0.0]*4, [[0.0]*4 for _ in range(4)])
        flow = np.zeros((4, 4), dtype=int)
        flow[0, 3], flow[1, 2], flow[2, 1] = 5, 2, 2
        with self.assertRaisesRegex(ValueError, 'Cyclic flow graph'):
            evaluate(instance, flow)

    def test_invalid_processing_still_rejected(self):
        for duration in [-1.0, float('nan'), float('inf')]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                Instance('invalid', 5, [5, 2, 5], [0.0, duration, 0.0], [[0.0]*3 for _ in range(3)])


if __name__ == '__main__':
    unittest.main()
