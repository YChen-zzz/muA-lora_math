# LR search baselines: six trainings each

## Run

```bash
bash /data/250010186/SFT_code/muA-lora/lr_baselines/run.sh
```

One shell runs golden section, safeguarded parabolic interpolation, dichotomous minimization, and 1D Bayesian optimization sequentially. Each gets **six different LR evaluations**, including initialization: at most 24 trainings in total. All are full finetuning (rank 0), 1000 optimizer steps, global batch 32 (4 GPUs × per-device 8 × accumulation 1), warmup 0.05, Adam beta2 0.999. Each evaluation starts from the base model and scores final `eval_loss`; eval runs every 500 steps and at the end. Model/data/seed/scheduler are inherited from `run-llama.py`. Data selection is 128k training examples and 32k validation examples.

Dependencies are installed from Tsinghua in `run.sh`, using existing PyTorch and adding missing BoTorch dependencies. The shell also installs TRL 0.19.1 as in the existing training launchers.

## Algorithms

Search coordinate is `x=log2(LR)` in [-18,-13], with fractional exponents allowed. The training entry receives `--learning-rate-exponent=-x`; the sign is reversed because it computes `2**(-exponent)`.

| Method | Initialization | Subsequent selection |
| --- | --- | --- |
| golden | x=-16.090170 and -14.909830 | Compare interior points, retain the better side and reuse one observation; one new training per reduction |
| parabolic | x=-18,-15.5,-13 | Fit a quadratic through the three observations nearest the current best and evaluate its minimum |
| dichotomy | x=-18,-13 | Evaluate midpoint ±0.05, retain the better half; two more rounds fit the remaining four evaluations |
| bo | x=-16 only | Fit Matérn 2.5 GP, maximize UCB(beta=2) of negative eval loss over unevaluated points; five adaptive trials |

Golden section does not spend evaluations on endpoints: its interval reduction needs two interior values, and the six-evaluation budget is small. It assumes a single valley; boundary optima and noisy losses can be challenging.

Parabolic interpolation falls back to the midpoint of the largest unsampled gap if curvature is nonpositive, the vertex is outside bounds, or it nearly repeats an observation. This is **safeguarded parabolic interpolation**, with midpoint fallback, not Brent's algorithm. New coordinates are rounded to six decimals.

Dichotomy is a function-minimization interval method, not root bisection. Endpoint values count toward the budget and final best, but do not determine the side to retain. For an equal interior loss it retains the right side. The ±0.05 separation is in log2 space; it can be sensitive to training noise.

BO reuses `bo_sft/search.py`'s GP/UCB selector with only the LR dimension. Its candidate grid is log2-spaced at resolution 0.001 (5001 points), so it can evaluate fractional exponents, not just the six integer grid-search LRs. CPU double precision is used for fitting. All methods see only their own observations.

## Outputs

Independent root: `checkpoints/lr-baselines/fft-b32-1000-sixtrials/`. Each method has its own subdirectory; no training results are shared between methods or with BCD/the separate 36-trial BO experiment.

- `<method>/best.json`: current best final eval loss, LR, full parameters and training directory; updated after each completed evaluation.
- `<method>/history.json`: ordered six evaluation records.
- `<method>/runs/<parameters>/`: `train.log`, `final_metrics.json`, `trainer_state.json`, and `params.json`.
- `summary.json`: all four reports after the shell finishes.

`--no-save` is used: model weights are not checkpointed. Interrupted runs can be resumed with the same shell. Algorithms replay their own completed evaluations from disk and only train missing candidates. Use a new output directory if changing the protocol/configuration.

```bash
cat checkpoints/lr-baselines/fft-b32-1000-sixtrials/summary.json
python lr_baselines/test_search.py
```

The test uses a synthetic quadratic objective and the real GP selector; it checks all four six-evaluation budgets, fixed training parameters, bounds, distinct candidates, and reuse on resume. It does not establish performance on actual SFT. Compare against LR grid search only when model, data split, training steps, batch, warmup, beta2 and other fixed parameters match.

The fractional-LR training path was also verified with `bash lr_baselines/run.sh lr_baselines/smoke.json`: two real 1-GPU, 2-step golden-section trials completed and produced metrics plus summary.json under a separate smoke directory. This is a wiring check only.

Production submission: p10 job `pt-uz9f7r36`, 4 H100 GPUs on one node, launching `bash lr_baselines/run.sh` with the default config (four methods, six trials each).
