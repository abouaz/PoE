#!/bin/bash
# =============================================================================
# Portfolio of Experts (PoE_4) · 182M backbone · 1 node × 4 GPU (H100)
#
# Architecture: 12L × 768H, head_dim=64 (12 heads, 4 KV heads),
#               17 experts (1 shared + 16 routed), plain ReLU routing, FFN=3072.
#               N=16 via --num-routed-experts override; base config has N=8.
#
# MATCHES BASELINE EXACTLY (TopK k=8 / k=4 runs):
#   Same arch, data, batch, LR, steps, GPUs. Only change: --mode poe.
#
# PoE adds the portfolio divergence loss L_div (Eq. 16); the forward pass is
# plain ReLU routing identical to ReMoE.
#
# Usage:
#   sbatch launch_poe_182m.sh                # default: coactivation + ledoit_wolf
#   COV_ESTIMATOR=batchmean sbatch launch_poe_182m.sh   # ablation
#
# Ablation knobs (Section 3) — set as env vars before sbatch:
#   COV_ESTIMATOR  = coactivation | batchmean | routing_weighted |
#                    set_overlap | output_similarity      (default coactivation)
#   SHRINKAGE_MODE = ledoit_wolf | schedule | constant | none  (default ledoit_wolf)
#   B_TARGET       = risk aversion b (default 0.01)
# =============================================================================

#SBATCH --job-name=182m_poe
#SBATCH --partition=agent-xlong
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --output=$HOME/sd-moe/logs/182m_poe_%j.out
#SBATCH --error=$HOME/sd-moe/logs/182m_poe_%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT

set -e
set -o pipefail

# ===== Cluster config (single node, same as baseline) =====
GPUS_PER_NODE=4
# Auto-select free MASTER_PORT (avoids silent hangs from port conflicts)
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# ===== Training hparams (IDENTICAL to TopK k=8 baseline) =====
SEQ_LEN=1024
MICRO_BATCH=32
GLOBAL_BATCH=1536                         # 1536 × 1024 = 1.57M tokens/step
TRAIN_ITERS=63578                         # 63578 × 1.57M ≈ 100B tokens
WARMUP_ITERS=500
LR=3e-4
MIN_LR=3e-5
NUM_WORKERS=4
SAVE_INTERVAL=2000
LOG_INTERVAL=10

# ===== Expert / routing config (matches baseline) =====
NUM_ROUTED=16                             # overrides config default of 8
TOPK=8                                    # k=8 active experts per token

# ===== PoE hyperparameters =====
COV_ESTIMATOR=${COV_ESTIMATOR:-coactivation}
SHRINKAGE_MODE=${SHRINKAGE_MODE:-ledoit_wolf}
SHRINKAGE_GAMMA=${SHRINKAGE_GAMMA:-0.5}        # for SHRINKAGE_MODE=constant
SHRINKAGE_WARMUP=${SHRINKAGE_WARMUP:-2000}     # for SHRINKAGE_MODE=schedule
B_TARGET=${B_TARGET:-0.01}
B_WARMUP=${B_WARMUP:-500}                       # steps held at b=0
B_RAMP=${B_RAMP:-2000}                          # linear ramp length
BETA_COV=${BETA_COV:-0.05}
MU_EMA=${MU_EMA:-0.01}
RETURN_SCALE=${RETURN_SCALE:-rms}
POE_LAMBDA_INIT=${POE_LAMBDA_INIT:-1e-3}
POE_LAMBDA_GAMMA=${POE_LAMBDA_GAMMA:-0.01}      # η = 1 + this (Eq. 19 controller)
POE_TARGET_K=${POE_TARGET_K:-8.0}               # must match TOPK
PENALTY_SIGN=${PENALTY_SIGN:-redundancy_only}   # redundancy_only | signed

# ===== Paths (Huawei LRC Alpha cluster) =====
HERE="$HOME/sd-moe"
DATA_ROOT="$HOME/dolma_100b_gpt2"
RUN_TAG="182m_poe_n${NUM_ROUTED}_k${TOPK}_${COV_ESTIMATOR}_${SHRINKAGE_MODE}"
SAVE_DIR="${HERE}/logs/${RUN_TAG}"
TOKENIZER="gpt2"

# ===== WandB =====
WANDB_PROJECT="PoE"
WANDB_RUN="${RUN_TAG}_$(date +%m%d_%H%M)"
WANDB_FLAG=""
if [ -n "${NO_WANDB}" ]; then WANDB_FLAG="--no-wandb"; fi

# ===== Conda environment =====
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate remoe

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"

mkdir -p "${SAVE_DIR}"
mkdir -p "${HERE}/logs"

# ===== Verify batch divisibility =====
TOTAL_MICRO=$((MICRO_BATCH * GPUS_PER_NODE))
if [ $((GLOBAL_BATCH % TOTAL_MICRO)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH not divisible by MICRO_BATCH*GPUS=$TOTAL_MICRO"
    exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / TOTAL_MICRO))

echo "============================================================"
echo "  ARCH            : custom_182m  (182M backbone, ~1.5B total)"
echo "  MODE            : poe  (PoE_4, loss-only formulation)"
echo "  N (routed)      : $NUM_ROUTED  (+ 1 shared)"
echo "  k (active)      : $TOPK"
echo "  cov_estimator   : $COV_ESTIMATOR"
echo "  shrinkage_mode  : $SHRINKAGE_MODE"
echo "  b_target        : $B_TARGET  (warmup=$B_WARMUP ramp=$B_RAMP)"
echo "  penalty_sign    : $PENALTY_SIGN"
echo "  GPUs            : $GPUS_PER_NODE"
echo "  global_batch    : $GLOBAL_BATCH  (= ${GRAD_ACCUM} grad_accum × $TOTAL_MICRO micro)"
echo "  train_iters     : $TRAIN_ITERS  (~100B tokens)"
echo "  data            : $DATA_ROOT  (mmap format)"
echo "  save_dir        : $SAVE_DIR"
echo "============================================================"

cd "$HERE"
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    poe_runs/train.py \
    --arch custom_182m \
    --mode poe \
    --num-routed-experts $NUM_ROUTED \
    --moe-router-topk $TOPK \
    --data-root "$DATA_ROOT" \
    --data-format mmap \
    --tokenizer "$TOKENIZER" \
    --seq-len $SEQ_LEN \
    --micro-batch $MICRO_BATCH \
    --global-batch $GLOBAL_BATCH \
    --train-iters $TRAIN_ITERS \
    --warmup-iters $WARMUP_ITERS \
    --lr $LR \
    --min-lr $MIN_LR \
    --save-interval $SAVE_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --num-workers $NUM_WORKERS \
    --save-dir "$SAVE_DIR" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run "$WANDB_RUN" \
    --poe-cov-estimator "$COV_ESTIMATOR" \
    --poe-shrinkage-mode "$SHRINKAGE_MODE" \
    --poe-shrinkage-gamma "$SHRINKAGE_GAMMA" \
    --poe-shrinkage-warmup "$SHRINKAGE_WARMUP" \
    --poe-b-target "$B_TARGET" \
    --poe-b-warmup "$B_WARMUP" \
    --poe-b-ramp "$B_RAMP" \
    --poe-beta-cov "$BETA_COV" \
    --poe-mu-ema "$MU_EMA" \
    --poe-return-scale "$RETURN_SCALE" \
    --poe-lambda-init "$POE_LAMBDA_INIT" \
    --poe-lambda-gamma "$POE_LAMBDA_GAMMA" \
    --poe-target-k "$POE_TARGET_K" \
    --poe-penalty-sign "$PENALTY_SIGN" \
    $WANDB_FLAG \
    2>&1 | tee -a "$SAVE_DIR/train.log"
