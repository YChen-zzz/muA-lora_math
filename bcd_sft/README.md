# BCD search for Llama SFT

## 当前实验：2026-11-23，固定32000条训练样本

p10任务：`pt-tdkoa721`，4张H100。真实小规模batch32/64训练已确认train/eval token及mask摘要一致。

运行 `bash bcd_sft/run.sh`。默认配置已更新：LR → warmup → beta2 → global batch，最多2轮、32次不同组合的训练，接受阈值 **1e-5**。一整轮没有接受更新仍会提前结束；`best_observed_loss`另外报告所有观测中的最低loss。

LR保留六个整数指数点，warmup保留0.01/0.05/0.1，beta2保留0.95/0.99/0.999；新增batch32/64/128/256，对应1000/500/250/125 optimizer steps。每次使用同一份32000条训练数据，完整训练一次；验证集固定32000条，每卡eval batch固定8。中途评估在半程附近（batch256为step63），末次评估对应相同训练样本量。

新输出：`checkpoints/bcd-sft/fixed32000-batch32-256-threshold1e-5-20261123/`。不复用旧实验，不保存权重。相同训练预算下，BCD采用原粗grid，BO采用更细候选，因此比较的是完整HPO策略，差异不只来自surrogate。

验证：`python bcd_sft/test_search.py`、`python bo_sft/test_fixed_data.py`。

每次训练前对实际预处理后的训练/验证tokens和loss mask做SHA256核对；两种方法共享`checkpoints/data-contracts/fixed32000-20261123.json`，每次运行目录保存data_manifest.json。若数据、mask、seed、截断或eval batch不同则报错；完整数据只训练一次。

## 以下为旧固定batch32实验说明（旧数据目录fft-1000）

This directory adapts the coordinate search in [BCD-nanogpt](https://github.com/YChen-zzz/BCD-nanogpt/blob/main/bcd_search.py) to this repository's `run-llama.py`. It searches **peak learning rate**, **warmup ratio**, and **Adam beta2**, one at a time, with a fixed global batch size of 32. Each candidate runs the same number of optimizer steps and is scored by the final `eval_loss` in `final_metrics.json`.

## Search rule

1. Start from `defaults` in `config.json` and evaluate that configuration.
2. In `order`, sweep one parameter's full `grids` list while holding the others fixed.
3. Adopt the grid minimum only when it improves `eval_loss` by more than `min_improvement`.
4. Repeat until an entire round makes no accepted change or `max_rounds` is reached.
5. Reuse a completed `final_metrics.json` for any repeated parameter combination. Re-running the controller resumes from those files.

The current config allows at most **2 rounds**. Counting grid entries gives 2 × (6+3+3) = 24 nominal evaluations, but cached combinations and early stopping reduce actual trainings; the initial default is evaluated separately. BO uses a separate fixed 24-trial budget, so actual training counts need not match.

This is **grid-based block coordinate descent**, not binary search. The port runs on one 4-GPU node and does not include the source project's NPU or multi-node coordination code.

## Parameter mapping

| BCD name | `run-llama.py` argument | Meaning |
| --- | --- | --- |
| `lr_exponent` | `--learning-rate-exponent` | Peak LR is `2 ** (-lr_exponent)` |
| `global_batch` (fixed) | `--grad-accum` | `global_batch / (nproc * per_device_batch)` |
| `warmup_ratio` | `--warmup-ratio` | Fraction of optimizer steps used for linear warmup |
| `adam_beta2` | `--adam-beta2` | Adam second-moment decay |

Default grids: LR exponents 18/17/16/15/14/13, warmup ratio 0.01/0.05/0.1, Adam beta2 0.95/0.99/0.999. Initial warmup is 0.05 and beta2 is 0.999 (the training script's original default). Search order follows that list. Global batch stays at 32 in every trial; it is present in `defaults` but absent from `grids` and `order`. Run directory names include beta2 so different beta2 values have separate results.

The example has `nproc=4`, `per_device_batch=8`, and gradient accumulation 1, giving global batch 32. Each candidate uses `--max-steps 1000`. `train_set_size` selects `32 * train_set_size` examples; it is not a step count. Keep it large enough to supply the chosen fixed steps, or the Trainer will repeat examples. The 1-GPU smoke config uses fixed global batch 8 for its two-step wiring check.

The default config tunes full finetuning (`rank=0`). To tune LoRA, set `rank` to a positive integer and `init_method` to `initB_constant1`. Use a separate `output_dir` for each rank and training budget. The model, dataset split, seed, evaluation set, and loss mask come from `run-llama.py` and remain the same across candidates.

## Run

From `/data/250010186/SFT_code/muA-lora`, on a 4-GPU SCO worker:

```bash
bash bcd_sft/run.sh
```

To submit that script from the login server to the p10 workspace:

```bash
~/.sco/bin/sco acp jobs create \
  --workspace-name=p10-intelligent-adaptation-and-optimization-for-domestic-ai \
  --aec2-name=share-cluster --job-name=mua-bcd-sft \
  --container-image-url=registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/yupengchen-torch270-v1.7.4_lm-eval_plus:torch270data-20251220034311 \
  --training-framework=pytorch --worker-spec=n6ls.iu.i40.4 --worker-nodes=1 \
  --priority=NORMAL --quota-type=reserved --wait \
  --command='cd /data/250010186/SFT_code/muA-lora && bash bcd_sft/run.sh' \
  --storage-mount=01995892-d478-76d8-aec7-13fd8284477e:/data
```

This submits one 4-GPU job; its grid points run sequentially inside that job. Edit `config.json` before submission. The example is a **1000-step search**, not the 320k-example paper sweep.

To inspect the initial command without training:

```bash
python bcd_sft/search.py --config bcd_sft/config.json --dry-run
```

Results go to `output_dir` in the config: `runs/<parameters>/final_metrics.json`, `train.log`, `params.json`, plus `history.json`, `round_N.json`, and `best.json`. `--no-save` avoids model checkpoints while `run-llama.py` still writes `trainer_state.json` and `final_metrics.json`. The controller stops on a failed run so an incomplete result cannot be selected as best.

To change the search budget, edit `max_steps`, `train_set_size`, and `output_dir` together. Start a new output directory when changing the model, data, step budget, rank, or metric; existing run files are reusable only under the same protocol.

## Verification

```bash
python bcd_sft/test_search.py
bash -n bcd_sft/run.sh
python bcd_sft/search.py --config bcd_sft/config.json --dry-run
```

`test_search.py` checks that coordinate updates find a known synthetic optimum and that a resumed run reuses completed results. For a quick end-to-end 1-GPU smoke run (two optimizer steps), use `bash bcd_sft/run.sh bcd_sft/smoke-1gpu.json`; a completed run produced `global_step=2`, `eval_loss=1.467998`, and `best.json` under `checkpoints/bcd-sft/smoke-1gpu`. The 4-GPU equivalent is `smoke.json`. Smoke losses are only a wiring check, not comparable to the full search.
