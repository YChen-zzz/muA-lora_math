# SFT Bayesian optimization

## 当前实验：2026-11-23，固定训练数据量

p10任务：`pt-ybx23omu`，4张H100。真实小规模batch32/64训练已确认train/eval token及mask摘要一致。

运行 `bash bo_sft/run.sh`，默认配置已更新为四维搜索，总预算 **32次真实训练**（包含默认起点）。

| 参数 | 当前候选 |
| --- | --- |
| LR | 2^-18～2^-13，log2指数间隔0.01，共501点 |
| warmup | 0.01～0.10，间隔0.01，共10点 |
| beta2 | 对log10(1-beta2)等距取6点，范围0.95～0.999 |
| global batch | 32、64、128、256 |

每次使用同一份32000条训练样本，完整训练一次：batch32/64/128/256对应1000/500/250/125 optimizer steps。验证集固定32000条，每卡eval batch固定8；最终评估均在训练32000条样本后执行。中途评估在半程附近，batch256向上取整为step63（16128条）。固定数据量不保证loss相同。

默认起点仍为LR=2^-16、warmup=0.05、beta2=0.999、batch32。GP输入的beta2用log10(1-beta2)，batch用log2；所有维度再归一化。对120240个候选分块打分，不会逐个训练。

新输出：`checkpoints/bo-sft/fixed32000-batch32-256-finegrid-20261123/`。不复用旧实验。训练安装命令仍在run.sh内，使用清华源；不保存模型权重，保留log、final_metrics.json、trainer_state.json。验证：`python bo_sft/test_fixed_data.py`。

两种搜索器共享`checkpoints/data-contracts/fixed32000-20261123.json`：对Trainer实际预处理后的train/eval input_ids、attention mask及assistant/loss mask计算SHA256。每次训练前核对token总量、顺序、mask、seed、截断长度和eval batch；不一致会报错。每次运行目录另存data_manifest.json。`max_steps × global_batch`必须等于完整训练集大小，确保每个样本只使用一次。

## 以下为旧24次实验说明（旧数据目录fft-1000）

## 启动

在 4 卡训练容器内运行：

```bash
bash /data/250010186/SFT_code/muA-lora/bo_sft/run.sh
```

`run.sh` 从清华源安装 BoTorch 0.14.0 及缺失依赖，并安装训练需要的 TRL 0.19.1。安装不带 `--upgrade`；已检查当前 PyTorch 2.7 满足要求。

## 搜索方法

参考 [LoRA_HPO 的 BotorchOptimizer](https://github.com/kaist-ami/LoRA_HPO/blob/main/gollum/src/gollum/bo/optimizer.py)：每次拟合 Matérn 2.5 GP，以 UCB（beta=2）评估所有尚未尝试的离散候选，选择一个候选训练。使用 CPU double 拟合，输入各维缩放到 [0,1]，输出标准化，目标为最大化负最终 eval loss。这里直接使用 BoTorch，复用 `bcd_sft/search.py` 的训练调用和指标读取，未引入 gollum 的模型和配置框架。

初始只有 **1 个默认配置**；后续 **23 次由 BO 选择**，总计 **24 次不同组合的训练**，包括初始点。没有额外随机初始点。一点起步时 GP 的信息有限，早期选择主要依靠先验与不确定性；这不保证比其他初始化更好。

| 参数 | 初始值 | 候选 |
| --- | --- | --- |
| LR | 2^-16 | 2^-18、2^-17、2^-16、2^-15、2^-14、2^-13 |
| warmup ratio | 0.05 | 0.01、0.05、0.1 |
| Adam beta2 | 0.999 | 0.95、0.99、0.999 |
| 全局 batch | 32 | 固定：4 卡 × 每卡 8 × 累积 1 |

配置在 `bo_sft/config.json`。每次都是全参数训练（rank=0），从基础模型重新训练 1000 optimizer steps；每 500 steps eval，最后一步 eval，以最终 `eval_loss` 选优。训练数据选取 32×4000 条，验证数据选取 32×1000 条，数据处理沿用当前训练入口。这里只搜索上述 54 个离散组合中的 24 个。

## 查看结果与续跑

BO 所有输出独立写入 `checkpoints/bo-sft/fft-1000/`，不读取或复用 BCD 实验结果。共享训练函数只在传入的 BO 输出目录里查找结果。

- `best.json`：每次完成后更新，报告完成次数、总预算、当前最佳参数、最终 eval loss 与对应运行目录。
- `history.json`：按顺序记录所有完成的参数和最终 eval loss。
- `runs/<参数>/train.log`：完整训练输出。
- `runs/<参数>/final_metrics.json`：最终 train loss、末次 logged train loss、eval loss。
- `runs/<参数>/trainer_state.json`：训练日志状态。模型权重使用 `--no-save`，不保存 checkpoint。
- `next_params.json`：当前待完成候选；中断后重跑同一个 shell 会继续。

查看报告：

```bash
cat checkpoints/bo-sft/fft-1000/best.json
```

修改训练协议或搜索空间时使用新的 `output_dir`；只调整 `iterations` 总预算可在原目录续跑，已有完成次数计入新预算。当前仅实现 BO 结果报告；与 BCD 的比较后续再做。

## 验证

```bash
python bo_sft/test_search.py
bash -n bo_sft/run.sh
```

测试运行真实 GP/UCB 的完整 36 次循环，训练目标用已知最优值的函数代替，检查默认起点、候选无重复、固定 batch、找到最优值与续跑不重复调用训练。真实 SFT 的训练调用复用已验证的 BCD 入口；这个测试不代表完成了 36 次真实训练。

另外已完成 1 卡真实训练 smoke：默认点和 GP 推荐点各训练 2 steps，成功写出两条 history 与 best.json，最佳 eval loss 为 1.467149。这仅验证流程，不代表正式 1000-step 搜索结果。复跑命令为 `bash bo_sft/run.sh bo_sft/smoke.json`，输出独立保存在 `checkpoints/bo-sft/smoke-1gpu/`。
