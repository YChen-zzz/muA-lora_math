import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import search


class BOTest(unittest.TestCase):
    def test_real_gp_budget_and_resume(self):
        config = json.loads((Path(__file__).parent / "config.json").read_text())
        config["iterations"] = 36
        config.pop("train_examples", None)
        config.pop("input_scales", None)
        config["grids"] = {"lr_exponent": [18,17,16,15,14,13],
                           "warmup_ratio": [0.01,0.05,0.1], "adam_beta2": [0.95,0.99,0.999]}
        calls = []
        search.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, patch.object(search, "REPO", Path(folder)):
            def runner(params, metrics_path):
                calls.append(dict(params))
                loss = ((params["lr_exponent"] - 15) ** 2
                        + 100 * (params["warmup_ratio"] - 0.1) ** 2
                        + 1000 * (params["adam_beta2"] - 0.95) ** 2)
                metrics_path.write_text(json.dumps({"global_step": 1000, "eval_loss": loss}))

            answer = search.search(config, runner)
            self.assertEqual(len(calls), 36)
            self.assertEqual(len({json.dumps(p, sort_keys=True) for p in calls}), 36)
            self.assertEqual(calls[0], config["defaults"])
            self.assertTrue(all(p["global_batch"] == 32 for p in calls))
            self.assertEqual(answer["best_loss"], 0.0)
            snapshot = Path(folder) / config["output_dir"] / "config.json"
            snapshot.write_text(json.dumps({**config, "iterations": 48}))
            self.assertEqual(search.search(config, runner), answer)
            self.assertEqual(len(calls), 36)


if __name__ == "__main__":
    unittest.main()
