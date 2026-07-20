#!/bin/bash
# =============================================================================
# 182M MoE TopK · 1 node × 5 GPU (507-15) · dolma_100b_gpt2 · 100B tokens
#
# Architecture: 12L × 768H, head_dim=64 (12 heads, 4 KV heads),
#               9 experts (1 shared + 8 routed), per-expert FFN=3072
#
# Usage:
#   # TopK k=1 (default):
#   sbatch launch_182m.sh moe 1
#
#   # TopK k=2:
#   sbatch launch_182m.sh moe 2
#
#   # TopK k=4:
#   sbatch launch_182m.sh moe 4
#
#   # TopK k=8 (all experts):
#   sbatch launch_182m.sh moe 8
#
#   # ReMoE:
#   sbatch launch_182m.sh remoe
# =============================================================================
#SBATCH --job-name=182m_moe
#SBATCH --partition=agent-xlong
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --output=$HOME/sd-moe/logs/182m_%j.out
#SBATCH --error=$HOME/sd-moe/logs/182m_%j.err
#SBATCH --account=YOUR_SLURM_ACCOUNT

set -e
set -o pipefail

# ===== Arguments =====
MODE=${1:-"moe"}
TOPK=${2:-"1"}

if [ "$MODE" != "moe" ] && [ "$MODE" != "remoe" ]; then
    echo "Usage: $0 <moe|remoe> [topk]"; exit 1
fi

# ===== Cluster config =====
GPUS_PER_NODE=4

# Auto-select free MASTER_PORT (avoids silent hangs from port conflicts)
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# ===== Training hyperparams (100B tokens) =====
SEQ_LEN=1024
MICRO_BATCH=32
GLOBAL_BATCH=1536                         # 1536 × 1024 = 1.57M tokens/step
TRAIN_ITERS=63578                         # 63578 × 1.57M ≈ 100B tokens
WARMUP_ITERS=500
LR=3e-4
MIN_LR=3e-5
NUM_WORKERS=4
SAVE_INTERVAL=5000
LOG_INTERVAL=10
AUX_LOSS_COEFF=1e-2
Z_LOSS_COEFF=1e-3

# ===== ReMoE =====
REMOE_LAMBDA_INIT=1e-3
REMOE_LAMBDA_GAMMA=0.1
REMOE_TARGET_K=${TOPK}                    # Match target_k to topk for fair comparison

# ===== Paths =====
HERE="$HOME/sd-moe"
DATA_ROOT="$HOME/dolma_100b_gpt2"

if [ "$MODE" = "remoe" ]; then
    RUN_NAME="182m_remoe"
else
    RUN_NAME="182m_moe_n16_k${TOPK}"
fi

SAVE_DIR="$HOME/sd-moe/logs/${RUN_NAME}"
TOKENIZER="gpt2"

# ===== WandB =====
WANDB_PROJECT="PoE"
WANDB_RUN="${RUN_NAME}_$(date +%m%d_%H%M)"
WANDB_FLAG=""
if [ -n "${NO_WANDB}" ]; then WANDB_FLAG="--no-wandb"; fi

# ===== Mode-specific args =====
REMOE_ARGS=""
TOPK_ARGS=""
if [ "$MODE" = "remoe" ]; then
    REMOE_ARGS="--remoe-lambda-init $REMOE_LAMBDA_INIT --remoe-lambda-gamma $REMOE_LAMBDA_GAMMA --remoe-target-k $REMOE_TARGET_K"
else
    TOPK_ARGS="--moe-router-topk $TOPK"
fi

# ===== Environment =====
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate remoe

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

mkdir -p "$SAVE_DIR"
mkdir -p "$HERE/logs"

# ===== Verify batch divisibility =====
TOTAL_MICRO=$((MICRO_BATCH * GPUS_PER_NODE))
if [ $((GLOBAL_BATCH % TOTAL_MICRO)) -ne 0 ]; then
    echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH not divisible by MICRO_BATCH*GPUS=$TOTAL_MICRO"
    exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / TOTAL_MICRO))

echo "============================================================"
echo "  ARCH            : custom_182m"
echo "  MODE            : $MODE  (topk=$TOPK)"
echo "  GPUs            : $GPUS_PER_NODE  (node=507-15)"
echo "  micro_batch     : $MICRO_BATCH"
echo "  global_batch    : $GLOBAL_BATCH  (= ${GRAD_ACCUM} grad_accum × $TOTAL_MICRO micro)"
echo "  seq_len         : $SEQ_LEN"
echo "  tokens/step     : $((GLOBAL_BATCH * SEQ_LEN)) (~$((GLOBAL_BATCH * SEQ_LEN / 1000000))M)"
echo "  train_iters     : $TRAIN_ITERS  (~$((TRAIN_ITERS * GLOBAL_BATCH * SEQ_LEN / 1000000000))B tokens)"
echo "  warmup          : $WARMUP_ITERS"
echo "  lr              : $LR  (min=$MIN_LR)"
echo "  data_format     : mmap"
echo "  data_root       : $DATA_ROOT"
echo "  save_dir        : $SAVE_DIR"
echo "  master_port     : $MASTER_PORT"
echo "============================================================"

cd "$HERE"de
torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    train.py \
    --arch custom_182m \
    --mode $MODE \
    --sharding-strategy full \
    --data-root "$DATA_ROOT" \
    --data-format mmap \
    --tokenizer "$TOKENIZER" \
    --micro-batch $MICRO_BATCH \
    --global-batch $GLOBAL_BATCH \
    --seq-len $SEQ_LEN \
    --train-iters $TRAIN_ITERS \
    --warmup-iters $WARMUP_ITERS \
    --lr $LR \
    --min-lr $MIN_LR \
    --aux-loss-coeff $AUX_LOSS_COEFF \
    --z-loss-coeff $Z_LOSS_COEFF \
    --save-interval $SAVE_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --num-workers $NUM_WORKERS \
    --save-dir "$SAVE_DIR" \
    --wandb-entity "YOUR_WANDB_ENTITY" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run "$WANDB_RUN" \
    --num-routed-experts 16 \
    $TOPK_ARGS \
    $REMOE_ARGS \
    $WANDB_FLAG \
    2>&1 | tee -a "$SAVE_DIR/train.log"