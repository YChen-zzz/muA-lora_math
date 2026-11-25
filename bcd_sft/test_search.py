import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import search


class BcdTest(unittest.TestCase):
    def test_coordinate_search_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(search, "REPO", Path(directory)):
            config = {
                "output_dir": "results", "model_path": "/tmp/model", "nproc": 4,
                "rank": 0, "per_device_batch": 8, "train_set_size": 1,
                "test_set_size": 1, "max_steps": 2, "eval_steps": 1,
                "metric": "eval_loss", "max_rounds": 3, "min_improvement": 0.001,
                "defaults": {"lr_exponent": 15, "global_batch": 32, "warmup_ratio": 0.01,
                             "adam_beta2": 0.999},
                "grids": {"lr_exponent": [15, 16],
                          "warmup_ratio": [0.01, 0.05, 0.1], "adam_beta2": [0.95, 0.99, 0.999]},
                "order": ["lr_exponent", "warmup_ratio", "adam_beta2"],
            }
            calls = []

            def runner(params, metrics_path):
                calls.append(dict(params))
                loss = ((params["lr_exponent"] - 16) ** 2
                        + ((params["warmup_ratio"] - 0.1) * 10) ** 2
                        + ((params["adam_beta2"] - 0.95) * 100) ** 2)
                args = search.command(config, params, metrics_path.parent)
                self.assertEqual(params["global_batch"], 32)
                self.assertEqual(args[args.index("--grad-accum") + 1], "1")
                self.assertEqual(float(args[args.index("--adam-beta2") + 1]), params["adam_beta2"])
                metrics_path.write_text(json.dumps({"global_step": 2, "eval_loss": loss}))

            answer = search.search(config, runner)
            self.assertEqual(answer["best_params"],
                             {"lr_exponent": 16, "global_batch": 32, "warmup_ratio": 0.1,
                              "adam_beta2": 0.95})
            count = len(calls)
            self.assertEqual(search.search(config, runner)["best_params"], answer["best_params"])
            self.assertEqual(len(calls), count)


if __name__ == "__main__":
    unittest.main()
