import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import search as bo
from bcd_sft import search as bcd
from utils.data_manifest import verify_data


class FixedDataTest(unittest.TestCase):
    def test_token_mask_contract_rejects_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = [{'input_ids':[1,2,3], 'assistant_masks':[0,1,1]}]
            trainer = SimpleNamespace(train_dataset=dataset, eval_dataset=dataset,
                                      args=SimpleNamespace(world_size=1,dataloader_drop_last=False))
            args = SimpleNamespace(max_steps=1,batch_size=1,grad_accum=1,eval_batch_size=8,seed=42,
                                   max_seq_length=1024,data_manifest=folder+'/contract.json',save_dir=folder)
            verify_data(trainer,args)
            verify_data(trainer,args)
            dataset[0]['assistant_masks'] = [0,0,1]
            with self.assertRaises(ValueError):
                verify_data(trainer,args)

    def test_steps_and_evaluation_batch(self):
        config = json.loads((Path(__file__).parent / 'config.json').read_text())
        for batch, steps, evaluation in [(32,1000,500),(64,500,250),(128,250,125),(256,125,63)]:
            params = {**config['defaults'], 'global_batch':batch}
            args = bcd.command(config, params, Path('/tmp/run'))
            self.assertEqual(args[args.index('--max-steps')+1], str(steps))
            self.assertEqual(args[args.index('--eval-steps')+1], str(evaluation))
            self.assertEqual(args[args.index('--eval-batch-size')+1], '8')
            self.assertEqual(steps * batch, config['train_examples'])

    def test_fine_grid_gp_suggestion(self):
        config = json.loads((Path(__file__).parent / 'config.json').read_text())
        self.assertEqual(len(bo.grid_values(config['grids']['lr_exponent'])),501)
        beta = bo.grid_values(config['grids']['adam_beta2'])
        self.assertAlmostEqual(min(beta),0.95)
        self.assertAlmostEqual(max(beta),0.999)
        bo.torch.set_num_threads(4)
        candidate = bo.suggest(config,[{'params':config['defaults'],'loss':0.916}])
        self.assertNotEqual(candidate, config['defaults'])
        self.assertIn(candidate['global_batch'],[32,64,128,256])
        self.assertTrue(13 <= candidate['lr_exponent'] <= 18)

    def test_bcd_budget_cap_and_batch_sweep(self):
        config = json.loads((Path(__file__).parents[1] / 'bcd_sft/config.json').read_text())
        config['iterations'] = 5
        config['order'] = ['global_batch','lr_exponent','warmup_ratio','adam_beta2']
        with tempfile.TemporaryDirectory() as folder, patch.object(bcd,'REPO',Path(folder)):
            calls = []
            def runner(params, path):
                calls.append(dict(params))
                steps, _ = bcd.training_steps(config,params)
                path.write_text(json.dumps({'global_step':steps,
                    'eval_loss':1 + 32/params['global_batch']}))
            answer = bcd.search(config,runner)
            self.assertEqual(len(calls),5)
            self.assertEqual(answer['completed'],5)
            self.assertEqual({p['global_batch'] for p in calls},{32,64,128,256})


if __name__ == '__main__':
    unittest.main()
