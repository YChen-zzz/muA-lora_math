#!/usr/bin/env bash
set -euo pipefail

cd /data/250010186/SFT_code/muA-lora_math
export HF_HOME=/data/250010186/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 WANDB_MODE=disabled MUA_SFT_DISABLE_VLLM=1

method="${1:?use lr_only, bcd, bo, rp_bcd, or r_cd}"
if [[ "$method" == bo ]]; then
  python -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple botorch==0.14.0
fi
python -m pip install -q --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple trl==0.19.1

case "$method" in
  lr_only) python -u lr_baselines/search.py --config scripts/math_lr_only.json ;;
  bcd) python -u bcd_sft/search.py --config scripts/math_bcd.json ;;
  bo) python -u bo_sft/search.py --config scripts/math_bo.json ;;
  rp_bcd) python -u randomized_bcd/search.py --config scripts/math_randomized_bcd.json --method rp_bcd --seed 42 ;;
  r_cd) python -u randomized_bcd/search.py --config scripts/math_randomized_bcd.json --method r_cd --seed 42 ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac
