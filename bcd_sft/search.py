"""Coordinate-wise LR, warmup and Adam beta2 search with fixed batch size."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_name(params):
    warmup = str(params["warmup_ratio"]).replace(".", "p")
    beta2 = str(params["adam_beta2"]).replace(".", "p")
    name = f"lr-{params['lr_exponent']}_batch-{params['global_batch']}_warmup-{warmup}_beta2-{beta2}"
    if 'adam_beta1' in params:
        name += '_beta1-' + str(params['adam_beta1']).replace('.', 'p')
    if 'final_lr_ratio' in params:
        name += '_final-' + str(params['final_lr_ratio']).replace('.', 'p')
    return name


def training_steps(config, params):
    if "train_examples" not in config:
        return config["max_steps"], config["eval_steps"]
    examples, batch = config["train_examples"], params["global_batch"]
    if examples % batch or 32 * config["train_set_size"] != examples:
        raise ValueError("Fixed-data training requires an exact dataset and whole optimizer steps")
    steps = (examples // batch) * config.get("epochs", 1)
    return steps, steps


def command(config, params, run_dir):
    world_size = config["nproc"]
    per_device = config["per_device_batch"]
    global_batch = params["global_batch"]
    if global_batch % (world_size * per_device):
        raise ValueError("global_batch must be divisible by nproc * per_device_batch")
    accumulation = global_batch // (world_size * per_device)
    steps, eval_steps = training_steps(config, params)
    args = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={world_size}", "run-llama.py",
        "--model-path", config["model_path"],
        "--rank", str(config["rank"]),
        "--batch-size", str(per_device),
        "--eval-batch-size", str(per_device),
        "--grad-accum", str(accumulation),
        "--train-set-size", str(config["train_set_size"]),
        "--test-set-size", str(config["test_set_size"]),
        "--max-steps", str(steps),
        "--epochs", str(config.get("epochs", 1)),
        "--eval-steps", str(eval_steps),
        "--learning-rate-exponent", str(params["lr_exponent"]),
        "--warmup-ratio", str(params["warmup_ratio"]),
        "--adam-beta2", str(params["adam_beta2"]),
        "--output-dir", str(run_dir),
    ]
    if 'adam_beta1' in params:
        args += ["--adam-beta1", str(params['adam_beta1'])]
    if 'final_lr_ratio' in params:
        args += ["--min-lr-rate", str(params['final_lr_ratio'])]
    if config.get("dataset_dir"):
        args += ["--dataset-dir", config["dataset_dir"], "--task", "metamathqa"]
    if config.get("max_seq_length"):
        args += ["--max-seq-length", str(config["max_seq_length"])]
    if config.get("save_steps"):
        args += ["--save-steps", str(config["save_steps"])]
    if not config.get("save_model", False):
        args.append("--no-save")
    if config["rank"]:
        args += ["--init-method", config["init_method"]]
    if config.get("data_manifest"):
        args += ["--data-manifest", str(REPO / config["data_manifest"])]
    return args


def result(config, params, output_dir, history, runner=None):
    run_dir = output_dir / "runs" / run_name(params)
    metrics_path = run_dir / "final_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print(f"reuse {run_dir}", flush=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        args = command(config, params, run_dir)
        write_json(run_dir / "params.json", params)
        print("run", " ".join(args), flush=True)
        if runner is None:
            with (run_dir / "train.log").open("w") as log:
                subprocess.run(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        else:
            runner(params, metrics_path)
        metrics = json.loads(metrics_path.read_text())
    expected_steps, _ = training_steps(config, params)
    if expected_steps > 0 and metrics["global_step"] != expected_steps:
        raise ValueError(f"Unexpected step count in {metrics_path}")
    loss = metrics[config["metric"]]
    if not math.isfinite(loss):
        raise ValueError(f"Non-finite {config['metric']} in {metrics_path}")
    history.append({"params": dict(params), "loss": loss, "run_dir": str(run_dir)})
    write_json(output_dir / "history.json", history)
    return loss


def search(config, runner=None):
    output_dir = REPO / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)
    params = dict(config["defaults"])
    history = []
    observed = {}
    def evaluate(candidate):
        key = run_name(candidate)
        if key not in observed:
            observed[key] = result(config, candidate, output_dir, history, runner)
        return observed[key]

    current_loss = evaluate(params)
    for round_number in range(1, config["max_rounds"] + 1):
        improved = False
        for name in config["order"]:
            best_value, best_loss = params[name], current_loss
            for value in config["grids"][name]:
                candidate = {**params, name: value}
                if run_name(candidate) not in observed and len(observed) >= config.get("iterations", math.inf):
                    break
                loss = evaluate(candidate)
                if loss < best_loss:
                    best_value, best_loss = value, loss
            if current_loss - best_loss > config["min_improvement"]:
                params[name], current_loss = best_value, best_loss
                improved = True
            print(f"round {round_number} {name}: {params[name]} loss={current_loss:.6f}", flush=True)
        write_json(output_dir / f"round_{round_number}.json",
                   {"params": params, "loss": current_loss, "improved": improved})
        if not improved or len(observed) >= config.get("iterations", math.inf):
            break
    best = min(history, key=lambda row: row["loss"])
    answer = {"best_params": params, "best_loss": current_loss,
              "best_observed_params": best["params"], "best_observed_loss": best["loss"],
              "completed": len(observed), "budget": config.get("iterations"),
              "metric": config["metric"], "rounds": round_number}
    write_json(output_dir / "best.json", answer)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    config = json.loads(options.config.read_text())
    if options.dry_run:
        print(" ".join(command(config, config["defaults"],
                               REPO / config["output_dir"] / "runs" / run_name(config["defaults"]))))
    else:
        print(search(config))


if __name__ == "__main__":
    main()
