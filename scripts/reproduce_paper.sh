#!/usr/bin/env bash
# Reproduce the paper's noise sweep for both architectures.
#
#   bash scripts/reproduce_paper.sh [DATA_ROOT] [OUT_ROOT]
#
# Each run takes a few hours on one A100. The pupil and phase pools are built
# once on the first run and reused by every run after it.
set -euo pipefail

DATA_ROOT="${1:-./data}"
OUT_ROOT="${2:-./runs}"

for model in unet mlp; do
  for sigma in 0 0.001 0.01 0.03; do
    echo "=== ${model}, sigma=${sigma} ==="
    pupil-design-train \
      --model "${model}" \
      --noise-sigma "${sigma}" \
      --epochs 4 \
      --data-root "${DATA_ROOT}" \
      --out "${OUT_ROOT}/${model}_sigma_${sigma}"

    pupil-design-eval "${OUT_ROOT}/${model}_sigma_${sigma}" \
      --data-root "${DATA_ROOT}" --split test --bins 30
  done
done
echo "All runs complete. Results under ${OUT_ROOT}."
