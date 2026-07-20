#!/bin/bash
SRC_DIR="$HOME/sd-moe/logs/182m_moe_n16_k4"
BAK_DIR="$HOME/sd-moe/logs/182m_moe_n16_k4_all_ckpts"
POLL_SECONDS=7200

mkdir -p "$BAK_DIR"
echo "[watch] source: $SRC_DIR"
echo "[watch] backup: $BAK_DIR"
echo "[watch] polling every 2 hours"
echo "[watch] started at $(date)"

while true; do
    for step_dir in "$SRC_DIR"/step_*; do
        [ -d "$step_dir" ] || continue
        step_name=$(basename "$step_dir")
        ckpt_file="$step_dir/checkpoint.pt"
        [ -f "$BAK_DIR/$step_name/checkpoint.pt" ] && continue
        [ ! -f "$ckpt_file" ] && continue
        [ -f "$ckpt_file.tmp" ] && continue
        echo "[watch] $(date +%H:%M:%S) backing up $step_name ..."
        mkdir -p "$BAK_DIR/$step_name"
        cp "$ckpt_file" "$BAK_DIR/$step_name/checkpoint.pt"
        echo "[watch] $(date +%H:%M:%S) done — $(du -sh "$BAK_DIR/$step_name/checkpoint.pt" | cut -f1)"
    done
    sleep "$POLL_SECONDS"
done
