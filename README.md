# MetaMathQA 上的学习率与多参数搜索

本目录从 [muA-LoRA](https://github.com/NanChanNN/muA-lora) 的代码复制而来，用于比较数学 SFT 中不同超参数搜索方法的效率。当前实验对 **Llama 3.2 1B 做全量微调**（`rank=0`），目标是在相同训练数据和验证集上，以尽量少的完整训练找到较低的最终验证损失。这里没有运行 LoRA，也没有运行 GSM8K 或 `lm-eval`。

原项目的介绍保存在 [`README_upstream.md`](README_upstream.md)。本实验的配置以 `scripts/math_*.json` 和本文件为准。

## 项目来源与改动

本项目基于 [NanChanNN/muA-lora](https://github.com/NanChanNN/muA-lora) 开发。原项目作者为 Nan Chen、Soledad Villar 和 Soufiane Hayou，代码按 [MIT License](LICENSE) 授权。本仓库保留并修改了原项目的训练入口与部分搜索代码，增加了 MetaMathQA 的固定训练/验证划分、以最终 `eval_loss` 为目标的数学任务配置，以及 BCD、BO、随机坐标方法和一维 LR 基线的运行脚本。原项目 README 另存为 [`README_upstream.md`](README_upstream.md)，便于区分原项目说明与本实验设置。

## 数据与评判指标

- 来源：[meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)，本地原始文件位于 `/data/250010186/hf/hub/datasets--meta-math--MetaMathQA/`。官方仓库只提供 train split。
- 固定划分：`data/metamathqa_38912_train_512_val_seed42/`，包含 **38,912 条训练样本、512 条验证样本**；随机种子为 42。`scripts/split_metamathqa.py` 生成划分，`manifest.json` 记录源文件及划分文件的 SHA256。
- 按 `original_question` 隔离：验证集取 512 道不同原题，每题选一条增强样本；同一道原题的其他增强样本全部排除在训练集之外。验证集是我们从 train 中建立的内部 holdout，不是官方测试集。
- `query` 和 `response` 分别转成 user、assistant 消息，使用 `chat_templates/llama_template.jinja`。`assistant_only_loss=True`；序列上限为 **1024 tokens**，不进行额外长度过滤。
- 所有候选共享同一划分、token IDs 和 loss mask。`utils/data_manifest.py` 对预处理后的训练/验证数据做 SHA256 校验，并要求每轮完整消费训练集。**搜索目标是最后一次 `eval_loss`，越低越好**；它不是数学题的生成准确率。

## 模型与单次训练

| 项目 | 设置 |
| --- | --- |
| 基座模型 | `/data/250010186/hf/Llama-3.2-1B` |
| 训练方式 | 全量微调，`rank=0` |
| 资源 | 单节点 4 张 H100；每卡 micro batch 8 |
| 训练长度 | **2 epoch**，共 77,824 次样本访问 |
| 精度 | BF16，允许 TF32；启用 gradient checkpointing |
| 优化器 | AdamW；weight decay 0.01；最大梯度范数 1.0 |
| 学习率日程 | 5% linear warmup，随后 cosine decay 到 `peak LR × final LR ratio` |
| 验证 | 每个候选在最后一步评估同一份 512 条验证集 |
| 随机种子 | 训练及划分均为 42；随机坐标顺序另用 search seed 42 |
| 存储 | **`--no-save`**：保留指标、日志、搜索历史和数据摘要，不保存模型或优化器 checkpoint |

Global batch 由 `4 × 每卡 8 × 梯度累积步数` 得到。改变 batch 时保持训练样本和实际 token 不变，调整优化器 step，而不是固定 step 数：

| Global batch | 梯度累积 | 每 epoch step | 2 epoch step |
| ---: | ---: | ---: | ---: |
| 32 | 1 | 1,216 | 2,432 |
| 64 | 2 | 608 | 1,216 |
| **128（初始值）** | **4** | **304** | **608** |
| 256 | 8 | 152 | 304 |

## 五维超参数空间

定义 `x = -log2(peak LR)`，因此 `peak LR = 2^-x`。所有方法从同一个完整配置开始：`x=11, batch=128, β1=0.9, β2=0.999, final LR ratio=0.1`。Warmup ratio 固定为 0.05。

| 参数 | BCD / 随机 BCD 候选 | BO 候选 |
| --- | --- | --- |
| Peak LR 指数 `x` | 整数 **11, 12, …, 20** | **11, 11.25, …, 20** |
| Global batch | 32, 64, 128, 256 | 相同 |
| Adam `β1` | 0.8, 0.9, 0.95 | 相同 |
| Adam `β2` | 0.95, 0.99, 0.999 | 相同 |
| Final LR ratio | 0, 0.01, 0.1, 0.3 | 相同 |

Final LR ratio 是 cosine 结束时学习率与 peak LR 的比值；例如初始配置的 peak LR 是 `2^-11`，结束时为 `0.1 × 2^-11`。

## 搜索方法与预算

| 方法 | 配置 | 规则与上限 |
| --- | --- | --- |
| BCD | `scripts/math_bcd.json` | 按 LR → batch → β1 → β2 → final LR ratio 扫完整坐标 grid，最多 2 轮、39 次不同配置；改善阈值 `1e-5` |
| RP-BCD | `scripts/math_randomized_bcd.json` | 每轮随机重排五个坐标，再逐坐标扫 grid；最多 2 轮、39 次不同配置 |
| 随机坐标 BCD（代码名 `r_cd`） | 同上 | 每次均匀随机选择一个坐标并扫 grid；最多 39 次不同配置，可因坐标停滞提前结束 |
| BO | `scripts/math_bo.json` | 五维 Matérn GP + UCB（`beta=2`），从未试候选中逐次选点；固定 39 次不同配置 |

另有三个 **仅搜索 LR** 的对照方法，固定其他四维为初始值，分别给 **10 次训练**预算，并把 `x=11` 的共同起点计入预算：黄金分割、带保护的抛物线插值、二分区间最小化。配置为 `scripts/math_lr_only.json`。这三个是一维对照；和五维搜索比较时需同时注明搜索空间及训练次数。

BCD 系列会复用已经评估过的配置，且可能提前停下。报告结果时应同时给出 **best eval loss 对实际完成训练次数的曲线**，而不能把“两轮”当成固定 39 次训练。

## 耗时与产物

截至 2026-11-25，23 次已完成候选训练的 `train_runtime` 中位数约 **496 秒（8.3 分钟）**，不含 GPU 排队以及每次启动和数据准备的全部开销。按每个五维方法最多 39 次训练估计，单个方法大约需要 **6–8 小时 GPU 节点运行时间**；提前停止会更短。实际总墙钟时间取决于各任务的排队和并行情况。

搜索结果写在 `checkpoints/` 下的 `lr-baselines/`、`bcd-sft/`、`bo-sft/` 与 `randomized-bcd/`。每次训练保留 `final_metrics.json`、`trainer_state.json`、`train.log`、`params.json` 和 `data_manifest.json`；方法目录保存 `history.json` 与 `best.json` 等搜索记录。**目录名虽为 checkpoints，里面没有模型 checkpoint**。

由于用户选择不保存模型权重，这批搜索结果可用于比较验证损失，但无法直接拿候选模型运行 `lm-eval`。如需对最佳配置做 GSM8K 或其他生成评测，须按该配置重新训练并另行保存模型。

## 建议的最终 GSM8K 评测

搜索全部结束后，先按同一验证集上的最终 `eval_loss` 选出配置，并记录方法、参数和实际完成的搜索预算。然后用固定的 38,912 条训练数据、seed 42、2 epoch **重训一次该配置**，只在重训结束时保存模型权重和 tokenizer，不保存优化器状态。当前搜索任务的 `--no-save` 不会产生可供推理的模型。

建议用 [EleutherAI lm-evaluation-harness 的 `gsm8k` 任务](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml) 对 GSM8K **test split** 做最终外部评测；该任务的标准设置为 5-shot、生成式回答、`exact_match`。固定任务名、few-shot 数、chat template、生成参数和 harness 版本，所有最终模型使用同一协议。下面的命令是**重训并保存模型之后**的评测模板，`FINAL_MODEL_DIR` 需替换为实际模型目录：

```bash
cd /data/250010186/SFT_code/muA-lora_math
export HF_HOME=/data/250010186/hf
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
python -m lm_eval \
  --model hf \
  --model_args pretrained="$FINAL_MODEL_DIR",dtype=bfloat16 \
  --tasks gsm8k \
  --num_fewshot 5 \
  --apply_chat_template \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/gsm8k_final
```

`lm-eval` 会在评测时加载所需的 GSM8K 数据；目前没有单独下载它。报告中分别列出内部 MetaMathQA 的 `eval_loss` 和 GSM8K 的 `exact_match`，注明采用的任务与 few-shot 协议。GSM8K 不参与这轮超参数搜索，以免把最终测试集用于选参。

## 运行环境

- 登录服务器：`ssh -p 10186 root@118.145.32.133`；代码及数据在共享的 `/data/250010186/` 下。
- 训练平台：SenseCore SCO，AEC2 `share-cluster`。五维任务位于 `p10-intelligent-adaptation-and-optimization-for-domestic-ai`；LR 单变量任务位于 `share-space`。资源规格 `n6ls.iu.i40.4`，每个任务单节点 **4 张 H100 80GB、88 vCPU、960 GiB 内存**；`/data` 通过 AFS 挂载。
- 容器镜像：`registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/yupengchen-torch270-v1.7.4_lm-eval_plus:torch270data-20251220034311`。
- 该镜像的环境探测记录显示：Python **3.10.12**、PyTorch **2.7.0+cu128**、Transformers **4.52.4**、Datasets **3.6.0**、Accelerate **1.8.1**、`lm_eval` **0.4.9.1**。启动脚本另安装 `trl==0.19.1`，BO 任务安装 `botorch==0.14.0`。
- 搜索启动脚本设置 `HF_HOME=/data/250010186/hf`、`HF_ENDPOINT=https://hf-mirror.com`，并开启 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 以读取已有模型和划分。最终 GSM8K 评测需关闭这两个离线开关，让 harness 加载测试数据。

## 运行与任务

训练启动脚本和数据划分脚本集中在 `scripts/`。在已有依赖和 4 张 GPU 的环境中，从仓库根目录运行：

```bash
bash scripts/run_math_search.sh lr_only
bash scripts/run_math_search.sh bcd
bash scripts/run_math_search.sh bo
bash scripts/run_math_search.sh rp_bcd
bash scripts/run_math_search.sh r_cd
```

`scripts/run_math_search.sh` 设置 `HF_HOME=/data/250010186/hf`、`HF_ENDPOINT=https://hf-mirror.com`，并从本地模型和划分加载数据。当前 LR 单变量任务在 share-space，四个五维任务已提交到 p10；任务 ID 与失败后续跑的记录见 [`MATH_SETUP.md`](MATH_SETUP.md)。运行控制器会复用已有的 `final_metrics.json`，无需重跑已完成候选。
