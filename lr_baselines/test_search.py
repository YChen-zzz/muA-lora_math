import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import search


class BaselinesTest(unittest.TestCase):
    def test_budget_parameters_and_resume(self):
        config = json.loads((Path(__file__).parent / "config.json").read_text())
        with tempfile.TemporaryDirectory() as folder, patch.object(search, "REPO", Path(folder)):
            for method in config["methods"]:
                calls = []

                def runner(params, metrics_path):
                    calls.append(params)
                    x = -params["lr_exponent"]
                    self.assertTrue(-18 <= x <= -13)
                    self.assertEqual(params["global_batch"], 32)
                    self.assertEqual(params["warmup_ratio"], 0.05)
                    self.assertEqual(params["adam_beta2"], 0.999)
                    metrics_path.write_text(json.dumps({"global_step": 1000,
                                                       "eval_loss": (x + 15.3) ** 2 + 1}))

                answer = search.run(config, method, runner)
                self.assertEqual(len(calls), 6)
                self.assertEqual(len({p["lr_exponent"] for p in calls}), 6)
                self.assertEqual(search.run(config, method, runner), answer)
                self.assertEqual(len(calls), 6)
                if method == "bo":
                    self.assertEqual(calls[0]["lr_exponent"], 16)
                if method == "parabolic":
                    self.assertAlmostEqual(answer["best_loss"], 1.0)
                if method == "dichotomy":
                    self.assertEqual([p["lr_exponent"] for p in calls[:2]], [18, 13])


if __name__ == "__main__":
    unittest.main()
