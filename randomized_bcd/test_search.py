import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import search
from bcd_sft.search import training_steps


class RandomizedTest(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).parent/'config.json').read_text())

    def run_trial(self, config, method, flat=False):
        calls = []
        def runner(params, path):
            calls.append(dict(params))
            loss = 1.0 if flat else ((params['lr_exponent']-15)**2
                + (10*(params['warmup_ratio']-0.1))**2
                + (100*(params['adam_beta2']-0.95))**2
                + (params['global_batch']/256-1)**2)
            path.write_text(json.dumps({'global_step':training_steps(config,params)[0],
                                        'eval_loss':loss}))
        answer = search.search(config,method,42,runner)
        output = search.REPO/config['output_dir']/method/'seed-42'
        events = json.loads((output/'coordinate_history.json').read_text())
        before = len(calls)
        self.assertEqual(search.search(config,method,42,runner),answer)
        self.assertEqual(len(calls),before)
        self.assertEqual(json.loads((output/'coordinate_history.json').read_text()),events)
        self.assertLessEqual(len(calls),config['iterations'])
        self.assertEqual(len(calls),len({json.dumps(p,sort_keys=True) for p in calls}))
        self.assertEqual(calls[0],config['defaults'])
        return answer, events, output

    def test_optimization_and_replay(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(search,'REPO',Path(folder)):
            for method in ['rp_bcd','r_cd']:
                config = self.config()
                answer, events, output = self.run_trial(config,method)
                self.assertEqual(answer['best_loss'],0)
                self.assertEqual(answer['stop_reason'],'coordinate_stationary')
                if method == 'rp_bcd':
                    for permutation in json.loads((output/'permutations.json').read_text()):
                        self.assertEqual(sorted(permutation),sorted(config['order']))
                else:
                    self.assertTrue(any(e['new_trials']==0 for e in events))

    def test_budget_cut_during_grid(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(search,'REPO',Path(folder)):
            for method in ['rp_bcd','r_cd']:
                config = {**self.config(),'iterations':2}
                answer, events, _ = self.run_trial(config,method)
                self.assertEqual(answer['completed'],2)
                self.assertEqual(answer['stop_reason'],'budget_exhausted')
                self.assertFalse(events[-1]['grid_complete'])
                self.assertFalse(events[-1]['accepted'])
                self.assertEqual(answer['accepted_params'],config['defaults'])

    def test_flat_loss_requires_all_coordinates_checked(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(search,'REPO',Path(folder)):
            for method in ['rp_bcd','r_cd']:
                config = self.config()
                answer, events, _ = self.run_trial(config,method,flat=True)
                self.assertEqual(answer['completed'],13)
                self.assertEqual({e['coordinate'] for e in events},set(config['order']))
                self.assertEqual(answer['stop_reason'],'coordinate_stationary')


if __name__ == '__main__':
    unittest.main()
