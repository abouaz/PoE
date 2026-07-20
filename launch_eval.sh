#!/bin/bash
# Usage: sbatch launch_eval.sh 4
#SBATCH --job-name=eval_182m
#SBATCH --partition=agent-xlong
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=$HOME/sd-moe/logs/eval_%j.out
#SBATCH --error=$HOME/sd-moe/logs/eval_%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT

set -e

TOPK=${1:-8}
HERE="$HOME/sd-moe"
CKPT_DIR="$HERE/logs/182m_moe_n16_k${TOPK}_all_ckpts"

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate remoe

cd "$HERE"

python eval_checkpoints.py \
    --ckpt-dir "$CKPT_DIR" \
    --topk "$TOPK" \
    --eval-batch-size 32 \
    --wandb-project "PoE" \
    --wandb-entity "YOUR_WANDB_ENTITY" \
    --wandb-run "182m_moe_n16_k${TOPK}_eval"
