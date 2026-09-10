#!/bin/bash
# Slurm array job: one task per noise level.
#   sbatch --array=0-3 scripts/slurm_train.sh
#SBATCH --job-name=pupil-design
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
mkdir -p logs

SIGMAS=(0 0.001 0.01 0.03)
SIGMA="${SIGMAS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODEL:-unet}"

pupil-design-train --model "$MODEL" --noise-sigma "$SIGMA" --epochs 4 \
  --data-root "${DATA_ROOT:-./data}" --out "runs/${MODEL}_sigma_${SIGMA}"

pupil-design-eval "runs/${MODEL}_sigma_${SIGMA}" \
  --data-root "${DATA_ROOT:-./data}" --split test --bins 30
