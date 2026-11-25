"""Six-evaluation LR baselines; x = log2(LR), in [-18, -13]."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bcd_sft.search import REPO, result, write_json


class BudgetReached(Exception):
    pass


def golden(f, a, b):
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = f(c), f(d)
    while True:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = f(d)


def parabolic(f, a, b):
    points = {a: f(a), (a + b) / 2: f((a + b) / 2), b: f(b)}
    while True:
        # Fit the three nearest observations to the best observed point.
        best = min(points, key=points.get)
        u, v, w = sorted(sorted(points, key=lambda x: abs(x - best))[:3])
        fu, fv, fw = points[u], points[v], points[w]
        slope1, slope2 = (fv - fu) / (v - u), (fw - fv) / (w - v)
        curvature = (slope2 - slope1) / (w - u)
        x = (u + v - slope1 / curvature) / 2 if curvature > 0 else math.nan
        if not math.isfinite(x) or not a < x < b or any(abs(x - p) < 1e-5 for p in points):
            ordered = sorted(points)
            left, right = max(zip(ordered, ordered[1:]), key=lambda pair: pair[1] - pair[0])
            x = (left + right) / 2
        x = round(x, 6)
        points[x] = f(x)


def dichotomy(f, a, b):
    f(a)
    f(b)
    while True:
        middle = (a + b) / 2
        delta = min(0.05, (b - a) / 8)
        left, right = middle - delta, middle + delta
        if f(left) < f(right):
            b = right
        else:
            a = left


def bayes(f, a, b, config):
    import torch
    from bo_sft.search import suggest

    torch.set_num_threads(4)
    # A fine 0.001 grid in log2 space, not the six integer LR grid points.
    bo_config = {**config, "grids": {"lr_exponent":
                 [round(-b + i * 0.001, 6) for i in range(round((b - a) / 0.001) + 1)]}}
    observations = []
    while True:
        params = suggest(bo_config, observations)
        loss = f(-params["lr_exponent"])
        observations.append({"params": params, "loss": loss})


def run(config, method, runner=None):
    output = REPO / config["output_dir"] / method
    if not output.resolve().is_relative_to((REPO / "checkpoints/lr-baselines").resolve()):
        raise ValueError("Use an output_dir under checkpoints/lr-baselines/")
    snapshot = output / "config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        raise ValueError("Use a new output_dir when changing the config")
    write_json(snapshot, config)
    history, observed = [], {}

    def evaluate(x):
        x = round(x, 6)
        if x in observed:
            return observed[x]
        if len(history) == config["budget"]:
            raise BudgetReached
        params = {**config["defaults"], "lr_exponent": -x}
        loss = result(config, params, output, history, runner)
        observed[x] = loss
        best = min(history, key=lambda row: row["loss"])
        report = {"method": method, "completed": len(history), "budget": config["budget"],
                  "metric": config["metric"], "best_loss": best["loss"],
                  "best_params": best["params"], "best_lr": 2 ** -best["params"]["lr_exponent"],
                  "best_run_dir": best["run_dir"]}
        write_json(output / "best.json", report)
        print(json.dumps(report), flush=True)
        return loss

    a, b = config["bounds"]
    try:
        if 'initial_lr_exponent' in config:
            evaluate(-config['initial_lr_exponent'])
        if method == "bo":
            bayes(evaluate, a, b, config)
        else:
            {"golden": golden, "parabolic": parabolic, "dichotomy": dichotomy}[method](evaluate, a, b)
    except BudgetReached:
        pass
    return json.loads((output / "best.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("lr_baselines/config.json"))
    options = parser.parse_args()
    config = json.loads(options.config.read_text())
    reports = [run(config, method) for method in config["methods"]]
    write_json(REPO / config["output_dir"] / "summary.json", reports)
