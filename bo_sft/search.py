"""在离散超参数网格上逐次运行 SFT，并用 GP-UCB 选择下一组参数。"""

import argparse
import itertools
import math
import json
import sys
from pathlib import Path

import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 复用 BCD 实验的训练命令、指标读取和 JSON 写入；这里只负责 BO 选点。
from bcd_sft.search import REPO, result, write_json


def grid_values(spec):
    """把配置中的列表或范围说明展开为实际候选值。"""
    if isinstance(spec, list):
        return spec
    start, stop = spec["start"], spec["stop"]
    count = spec.get("count") or round((stop - start) / spec["step"]) + 1
    step = (stop - start) / (count - 1)
    values = [start + i * step for i in range(count)]
    # beta2 在 log10(1-beta2) 空间等距取点，再还原成训练使用的 beta2。
    return [round(1 - 10 ** v, 10) if spec.get("scale") == "log10_one_minus"
            else round(v, 10) for v in values]


def suggest(config, history):
    """根据已完成的训练记录，从未试过的组合中选出下一组参数。"""
    # 展开配置中各维的候选网格；只给候选打分，不在这里训练全部组合。
    grids = {name: grid_values(spec) for name, spec in config["grids"].items()}
    names = list(grids)
    candidates = [dict(zip(names, values)) for values in
                  itertools.product(*(grids[name] for name in names))]
    candidates = [{**config["defaults"], **point} for point in candidates]
    remaining = [p for p in candidates if p not in [h["params"] for h in history]]
    if not remaining:
        raise ValueError("Search grid exhausted")
    if not history:
        # 第一轮没有观测值可拟合 GP，直接使用配置中的默认起点。
        return dict(config["defaults"])

    def encode(points):
        """把候选参数转成 GP 输入：先按维度变换，再缩放到 [0, 1]。"""
        def transform(name, value):
            scale = config.get("input_scales", {}).get(name)
            if scale == "log10_one_minus":
                return math.log10(1 - value)
            if scale == "log2":
                return math.log2(value)
            return value
        x = torch.tensor([[transform(n, p[n]) for n in names] for p in points], dtype=torch.double)
        low = torch.tensor([min(transform(n,v) for v in grids[n]) for n in names], dtype=torch.double)
        high = torch.tensor([max(transform(n,v) for v in grids[n]) for n in names], dtype=torch.double)
        return (x - low) / (high - low).clamp_min(1e-12)

    torch.manual_seed(config["bo_seed"] + len(history))
    x = encode([h["params"] for h in history])
    # BoTorch 默认选分数最高的点，所以用负 loss 表示目标：loss 越低，分数越高。
    y = -torch.tensor([[h["loss"]] for h in history], dtype=torch.double)
    # 用已有的“参数 -> loss”观测拟合高斯过程，估计未试组合的表现及不确定性。
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=len(names)))
    model = SingleTaskGP(x, y, covar_module=kernel,
                         outcome_transform=Standardize(m=1, min_stdv=1e-8))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    # UCB 同时考虑预测分数和不确定性；beta 控制探索的权重。
    acquisition = UpperConfidenceBound(model, beta=config["ucb_beta"])
    with torch.no_grad():
        # 候选很多，分块算分数以限制一次送入模型的张量大小。
        scores = torch.cat([acquisition(encode(part).unsqueeze(1))
                            for part in [remaining[i:i+2048] for i in range(0,len(remaining),2048)]])
    return remaining[scores.argmax().item()]


def search(config, runner=None):
    """反复执行“选点 -> 训练 -> 记录”，直到达到 iterations 次评估。"""
    output = REPO / config["output_dir"]
    # 限制输出位置，并防止同一目录混入不同实验配置；只允许调整总次数。
    if not output.resolve().is_relative_to((REPO / "checkpoints/bo-sft").resolve()):
        raise ValueError("BO output_dir must be under checkpoints/bo-sft/")
    snapshot = output / "config.json"
    previous = json.loads(snapshot.read_text()) if snapshot.exists() else config
    if {**previous, "iterations": config["iterations"]} != config:
        raise ValueError("Use a new output_dir when changing the experiment config")
    write_json(snapshot, config)
    history_path = output / "history.json"
    # 重启时从磁盘恢复已经完成的训练，不重新计算这些点。
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    while len(history) < config["iterations"]:
        pending = output / "next_params.json"
        # 训练前先保存待运行参数；中断后会优先处理同一个点。
        params = json.loads(pending.read_text()) if pending.exists() else suggest(config, history)
        write_json(pending, params)
        if params not in [h["params"] for h in history]:
            # result() 启动训练、读取最终指标，并把记录写入 history.json。
            result(config, params, output, history, runner)
        # 只有训练结果已记录后才清除待运行标记。
        pending.unlink()
        report(config, output, history)
    return report(config, output, history)


def report(config, output, history):
    """从已完成的训练中找最低 loss，写出当前最佳结果。"""
    best = min(history, key=lambda row: row["loss"])
    answer = {"completed": len(history), "budget": config["iterations"],
              "metric": config["metric"], "best_params": best["params"],
              "best_loss": best["loss"], "best_run_dir": best["run_dir"]}
    write_json(output / "best.json", answer)
    print(json.dumps(answer), flush=True)
    return answer


if __name__ == "__main__":
    # 从仓库根目录运行时，默认读取 bo_sft/config.json。
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("bo_sft/config.json"))
    options = parser.parse_args()
    torch.set_num_threads(4)
    search(json.loads(options.config.read_text()))
