#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-node FSDP training for MoE / ReMoE models on 2×8 GPU machines.

Key features:
  - HYBRID_SHARD: shard within each 8-GPU node, all-reduce across nodes
  - Activation checkpointing per DecoderLayer
  - bf16 mixed precision with fp32 gradient reduction
  - Cosine LR schedule with warmup
  - NaN detection and gradient skip
  - Sharded checkpoint save/load for fast multi-node I/O
  - WandB logging (optional)
"""
import argparse
import contextlib
import functools
import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from model import MoEConfig, MoEForCausalLM, DecoderLayer, count_active_params, count_params
from data import StreamingTokenDataset


# ============================== util ===================================== #

def log0(*a, **kw):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*a, **kw, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["moe", "remoe"], default="moe")
    p.add_argument("--arch", choices=["custom_182m", "custom_469m", "custom_978m",
                                      "custom_2b", "custom_7b"], default="custom_2b")

    # ReMoE
    p.add_argument("--remoe-lambda-init", type=float, default=1e-3)
    p.add_argument("--remoe-lambda-gamma", type=float, default=0.1)
    p.add_argument("--remoe-target-k", type=float, default=2.0)
    p.add_argument("--remoe-lambda-min", type=float, default=1e-6)
    p.add_argument("--remoe-lambda-max", type=float, default=1.0)

    # Data
    p.add_argument("--data-root",
                   default="/apdcephfs_hldy/share_304318596/nlperyin/dolma3_dolmino_mix-100B-1025")
    p.add_argument("--data-format", choices=["streaming", "mmap"], default="streaming",
                   help="streaming=.jsonl.zst via data.py, mmap=Megatron .bin/.idx via data_mmap.py")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subsets", nargs="+", default=None)

    # Model
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--moe-router-topk", type=int, default=None,
                   help="Override num_experts_per_tok from arch default (for k-sweep experiments)")
    p.add_argument("--num-routed-experts", type=int, default=None,
               help="Override num_routed_experts from arch default")
    p.add_argument("--aux-loss-coeff", type=float, default=1e-2)
    p.add_argument("--z-loss-coeff", type=float, default=1e-3)

    # Training
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--global-batch", type=int, default=4096)
    p.add_argument("--train-iters", type=int, default=12500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.95])

    # FSDP
    p.add_argument("--sharding-strategy", choices=["hybrid", "full"], default="hybrid",
                   help="hybrid=HYBRID_SHARD (intra-node shard), full=FULL_SHARD (global)")
    p.add_argument("--fsdp-wrap-mode", choices=["transformer", "root"], default="transformer",
                   help="transformer=auto-wrap DecoderLayer, root=only wrap top module")
    p.add_argument("--no-use-orig-params", action="store_true",
                   help="Set FSDP use_orig_params=False for compatibility")

    # Checkpoint / log
    p.add_argument("--save-dir",
                   default="/apdcephfs_hldy/share_304318596/nlperyin/remoe_logs/2b_moe_2node")
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--no-save", action="store_true",
                   help="Disable all checkpoint saving (including final step)")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--keep-last-n", type=int, default=3)

    # WandB
    p.add_argument("--wandb-entity", default="YOUR_WANDB_ENTITY")
    p.add_argument("--wandb-project", default="ReMoE-dolma3-scaling")
    p.add_argument("--wandb-run", default="2b_moe_2node")
    p.add_argument("--no-wandb", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-activation-ckpt", action="store_true")
    p.add_argument("--nan-skip-max", type=int, default=50,
                   help="Max consecutive NaN batches before aborting")
    return p.parse_args()


def build_model(args, vocab_size):
    """Build model from --arch flag. Returns (model, cfg, transformer_layer_cls)."""
    routing = "topk" if args.mode == "moe" else "relu"

    if args.arch == "custom_182m":
        cfg = MoEConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            head_dim=64,
            max_position_embeddings=args.seq_len,
            rope_theta=500000.0,
            num_routed_experts=8,
            num_shared_experts=1,
            num_experts_per_tok=1,
            moe_intermediate_size=3072,
            router_aux_loss_coef=args.aux_loss_coeff,
            router_z_loss_coef=args.z_loss_coeff,
            routing_mode=routing,
        )
    elif args.arch == "custom_469m":
        cfg = MoEConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=64,
            max_position_embeddings=args.seq_len,
            rope_theta=500000.0,
            num_routed_experts=8,
            num_shared_experts=1,
            num_experts_per_tok=1,
            moe_intermediate_size=4096,
            router_aux_loss_coef=args.aux_loss_coeff,
            router_z_loss_coef=args.z_loss_coeff,
            routing_mode=routing,
        )
    elif args.arch == "custom_978m":
        cfg = MoEConfig(
            vocab_size=vocab_size,
            hidden_size=1536,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=96,
            max_position_embeddings=args.seq_len,
            rope_theta=500000.0,
            num_routed_experts=8,
            num_shared_experts=1,
            num_experts_per_tok=1,
            moe_intermediate_size=6144,
            router_aux_loss_coef=args.aux_loss_coeff,
            router_z_loss_coef=args.z_loss_coeff,
            routing_mode=routing,
        )
    elif args.arch == "custom_2b":
        cfg = MoEConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=32,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=args.seq_len,
            rope_theta=500000.0,
            num_routed_experts=8,
            num_shared_experts=1,
            num_experts_per_tok=2,
            moe_intermediate_size=2048,
            router_aux_loss_coef=args.aux_loss_coeff,
            router_z_loss_coef=args.z_loss_coeff,
            routing_mode=routing,
        )
    elif args.arch == "custom_7b":
        cfg = MoEConfig(
            vocab_size=vocab_size,
            hidden_size=1536,
            num_hidden_layers=40,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=args.seq_len,
            rope_theta=500000.0,
            num_routed_experts=32,
            num_shared_experts=1,
            num_experts_per_tok=4,
            moe_intermediate_size=1024,
            router_aux_loss_coef=args.aux_loss_coeff,
            router_z_loss_coef=args.z_loss_coeff,
            routing_mode=routing,
        )
    else:
        raise ValueError(f"unknown arch: {args.arch}")
    
    if args.num_routed_experts is not None:
        cfg.num_routed_experts = args.num_routed_experts

    # Apply CLI overrides (e.g. --moe-router-topk for k-sweep)
    if args.moe_router_topk is not None:
        cfg.num_experts_per_tok = args.moe_router_topk

    model = MoEForCausalLM(cfg)
    return model, cfg, DecoderLayer


def lr_at(step, args):
    if step < args.warmup_iters:
        return args.lr * step / max(1, args.warmup_iters)
    progress = (step - args.warmup_iters) / max(1, args.train_iters - args.warmup_iters)
    progress = min(1.0, progress)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return args.min_lr + (args.lr - args.min_lr) * coeff


# ---- Process group helpers for HYBRID_SHARD ---- #

def create_hybrid_process_groups(rank, world_size, local_size):
    """Create (shard_group, replicate_group) for HYBRID_SHARD.

    shard_group: GPUs within the same node (intra-node).
    replicate_group: GPUs at the same local position across nodes (inter-node).
    """
    num_nodes = world_size // local_size
    shard_group = None
    replicate_group = None

    for node_i in range(num_nodes):
        ranks = list(range(node_i * local_size, (node_i + 1) * local_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            shard_group = g

    for local_i in range(local_size):
        ranks = list(range(local_i, world_size, local_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            replicate_group = g

    return shard_group, replicate_group


# ---- Checkpoint ---- #

def save_sharded_checkpoint(model, optimizer, step, save_dir, rank, world_size,
                            keep_last_n=3, extra=None):
    """Save per-rank sharded checkpoint for fast multi-node save."""
    step_dir = os.path.join(save_dir, f"step_{step:08d}")
    os.makedirs(step_dir, exist_ok=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        model_sd = model.state_dict()
        optim_sd = FSDP.optim_state_dict(model, optimizer)

    if rank == 0:
        payload = {"model": model_sd, "optimizer": optim_sd, "step": step}
        if extra:
            payload["extra"] = extra
        path = os.path.join(step_dir, "checkpoint.pt")
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        log0(f"[ckpt] saved -> {step_dir}")

        # Clean old checkpoints
        ckpt_dirs = sorted(
            d for d in os.listdir(save_dir)
            if d.startswith("step_") and os.path.isdir(os.path.join(save_dir, d))
        )
        for old in ckpt_dirs[:-keep_last_n]:
            old_path = os.path.join(save_dir, old)
            try:
                import shutil
                shutil.rmtree(old_path)
                log0(f"[ckpt] removed old {old}")
            except OSError:
                pass

    dist.barrier()


def maybe_resume(model, optimizer, save_dir, rank):
    """Resume from the latest checkpoint. Returns (step, extra_dict)."""
    if not os.path.isdir(save_dir):
        return 0, {}

    ckpt_dirs = sorted(
        d for d in os.listdir(save_dir)
        if d.startswith("step_") and os.path.isdir(os.path.join(save_dir, d))
    )
    if not ckpt_dirs:
        return 0, {}

    latest_dir = os.path.join(save_dir, ckpt_dirs[-1])
    ckpt_file = os.path.join(latest_dir, "checkpoint.pt")
    if not os.path.isfile(ckpt_file):
        return 0, {}

    log0(f"[resume] loading {ckpt_file}")
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
        model.load_state_dict(state["model"])

    try:
        osd = FSDP.optim_state_dict_to_load(state["optimizer"], model, optimizer)
        optimizer.load_state_dict(osd)
    except Exception as e:
        log0(f"[resume] WARNING: could not load optimizer state: {e}")
        log0(f"[resume] continuing with fresh optimizer (expect ~500 steps of warmup)")
    step = int(state.get("step", 0))
    extra = state.get("extra", {}) or {}
    log0(f"[resume] resumed at step={step}  extra_keys={list(extra.keys())}")
    return step, extra


# ============================== main ===================================== #

def main():
    args = parse_args()

    # ---- distributed init ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_master = rank == 0

    torch.manual_seed(args.seed + rank)

    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", min(8, world_size)))

    # ---- batch arithmetic ----
    assert args.global_batch % (args.micro_batch * world_size) == 0, \
        f"global_batch={args.global_batch} must divide by micro_batch*world_size={args.micro_batch*world_size}"
    grad_accum = args.global_batch // (args.micro_batch * world_size)

    # ---- tokenizer ----
    log0(f"[init] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Avoid noisy warnings for long raw documents in streaming tokenization.
    # We chunk to seq_len later, so this tokenizer-side max length is not a hard limit here.
    tokenizer.model_max_length = max(int(getattr(tokenizer, "model_max_length", 0) or 0), args.seq_len, 10**9)
    vocab_size = len(tokenizer)
    pad_to = 128
    padded_vocab = ((vocab_size + pad_to - 1) // pad_to) * pad_to
    log0(f"[init] tokenizer vocab={vocab_size} -> padded {padded_vocab}")

    # ---- model ----
    log0("[init] building model ...")
    model, cfg, transformer_layer_cls = build_model(args, padded_vocab)
    n_params = count_params(model)
    active_per_token = count_active_params(cfg)
    log0(f"[init] arch={args.arch}  total={n_params/1e9:.3f}B  "
         f"active/token={active_per_token/1e9:.3f}B (excl embed/lm_head)")
    log0(f"[init] config = layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
         f"head_dim={cfg.head_dim} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
         f"routed={cfg.num_routed_experts} shared={cfg.num_shared_experts} "
         f"topk={cfg.num_experts_per_tok} expert_ffn={cfg.moe_intermediate_size} "
         f"routing={cfg.routing_mode}")

    # ---- FSDP process groups for HYBRID_SHARD ----
    process_group = None
    if args.sharding_strategy == "hybrid" and world_size > local_size:
        log0(f"[init] creating HYBRID_SHARD groups: {world_size // local_size} nodes × {local_size} GPUs")
        shard_group, replicate_group = create_hybrid_process_groups(rank, world_size, local_size)
        process_group = (shard_group, replicate_group)
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD

    # ---- FSDP wrap ----
    auto_wrap = None
    if args.fsdp_wrap_mode == "transformer":
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

    log0(f"[init] wrapping FSDP (strategy={sharding_strategy.name}) ...")
    fsdp_kwargs = dict(
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=(not args.no_use_orig_params),
        limit_all_gathers=True,
    )
    if auto_wrap is not None:
        fsdp_kwargs["auto_wrap_policy"] = auto_wrap
    if process_group is not None:
        fsdp_kwargs["process_group"] = process_group
    model = FSDP(model, **fsdp_kwargs)

    # Activation checkpointing: apply after FSDP wrapping so auto-wrap can
    # still target raw DecoderLayer classes in transformer mode.
    if not args.no_activation_ckpt:
        log0("[init] applying activation checkpointing ...")
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: isinstance(m, transformer_layer_cls),
        )

    # ---- optimizer ----
    log0("[init] building optimizer ...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # ---- resume ----
    start_step, extra = maybe_resume(model, optimizer, args.save_dir, rank)
    resumed_lambda = extra.get("remoe_lambda") if args.mode == "remoe" else None

    # ---- data ----
    log0("[init] building dataset ...")
    if args.data_format == "mmap":
        from data_mmap import MMapTokenDataset
        dataset = MMapTokenDataset(
            data_root=args.data_root,
            seq_len=args.seq_len,
            rank=rank,
            world_size=world_size,
            infinite=True,
            seed=args.seed,
        )
    else:
        dataset = StreamingTokenDataset(
            root=args.data_root,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            rank=rank,
            world_size=world_size,
            subsets=args.subsets,
            infinite=True,
            shuffle_subsets_seed=args.seed,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    data_iter = iter(loader)

    # ---- wandb ----
    use_wandb = (not args.no_wandb) and is_master
    if use_wandb:
        import wandb
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run,
            config={**vars(args),
                    "world_size": world_size,
                    "local_size": local_size,
                    "grad_accum": grad_accum,
                    "tokens_per_step": args.global_batch * args.seq_len,
                    "total_params": n_params,
                    "active_params": active_per_token,
                    "padded_vocab": padded_vocab},
            resume="allow",
        )

    log0("=" * 70)
    log0(f"world_size      : {world_size}  ({world_size // local_size} nodes × {local_size} GPUs)")
    log0(f"sharding        : {sharding_strategy.name}")
    log0(f"micro_batch     : {args.micro_batch}")
    log0(f"global_batch    : {args.global_batch}")
    log0(f"grad_accum      : {grad_accum}")
    log0(f"seq_len         : {args.seq_len}")
    log0(f"tokens / step   : {args.global_batch * args.seq_len:,}")
    log0(f"train_iters     : {args.train_iters}")
    log0(f"total tokens    : {args.train_iters * args.global_batch * args.seq_len / 1e9:.2f} B")
    log0(f"start_step      : {start_step}")
    log0(f"mode            : {args.mode}")
    log0(f"save checkpoints: {not args.no_save}")
    log0(f"fsdp_wrap_mode  : {args.fsdp_wrap_mode}")
    log0(f"use_orig_params : {not args.no_use_orig_params}")
    log0("=" * 70)

    # ---- train loop ----
    model.train()
    t0 = time.time()
    log_t0 = time.time()
    log_tokens = 0
    nan_count = 0

    is_remoe = args.mode == "remoe"
    remoe_lambda = resumed_lambda if (is_remoe and resumed_lambda is not None) else args.remoe_lambda_init
    remoe_target_k = args.remoe_target_k
    remoe_gamma = args.remoe_lambda_gamma
    if is_remoe:
        log0(f"[init] ReMoE λ_init={remoe_lambda:.2e}  target_k={remoe_target_k}")

    for step in range(start_step, args.train_iters):
        lr = lr_at(step + 1, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        loss_local = 0.0
        ce_local = 0.0
        aux_local = 0.0
        active_local = 0.0
        l1_raw_local = 0.0

        for accum_i in range(grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if is_remoe:
                unwrapped = model.module if hasattr(model, "module") else model
                unwrapped.reset_remoe_stats()

            sync_ctx = model.no_sync() if accum_i < grad_accum - 1 else contextlib.nullcontext()
            with sync_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
                ce_loss = outputs.loss

                if is_remoe:
                    unwrapped = model.module if hasattr(model, "module") else model
                    l1_list, active_list = unwrapped.collect_remoe_stats()
                    l1_loss = torch.stack(l1_list).mean() if l1_list else torch.tensor(0., device=device)
                    active_mean = torch.stack(active_list).mean().detach() if active_list else torch.tensor(0., device=device)

                    total_loss = ce_loss + remoe_lambda * l1_loss
                    aux_local += (remoe_lambda * l1_loss).detach().float().item() / grad_accum
                    active_local += active_mean.float().item() / grad_accum
                    l1_raw_local += l1_loss.detach().float().item() / grad_accum
                    ce_local += ce_loss.detach().float().item() / grad_accum
                else:
                    total_loss = ce_loss
                    hf_aux = outputs.aux_loss.detach() if outputs.aux_loss is not None else torch.tensor(0., device=device)
                    aux_local += hf_aux.float().item() / grad_accum
                    ce_local += (ce_loss.detach() - args.aux_loss_coeff * hf_aux).float().item() / grad_accum

                finite_t = torch.isfinite(total_loss.detach()).to(torch.int32)
                dist.all_reduce(finite_t, op=dist.ReduceOp.MIN)
                if finite_t.item() == 0:
                    loss_local = float("nan")
                    break

                (total_loss / grad_accum).backward()
            loss_local += total_loss.detach().float().item() / grad_accum

        # NaN detection: skip this step if loss is NaN
        if not math.isfinite(loss_local):
            nan_count += 1
            log0(f"[WARN] NaN/Inf loss at step {step+1} (consecutive: {nan_count})")
            if nan_count > args.nan_skip_max:
                log0("[ABORT] Too many consecutive NaN steps, stopping training")
                break
            optimizer.zero_grad(set_to_none=True)
            continue
        nan_count = 0

        gnorm = model.clip_grad_norm_(args.clip_grad).item()
        optimizer.step()

        # Adaptive λ update (ReMoE)
        if is_remoe:
            active_t = torch.tensor([active_local], device=device)
            dist.all_reduce(active_t, op=dist.ReduceOp.AVG)
            global_active = active_t.item()
            if global_active > 0:
                ratio = global_active / max(remoe_target_k, 1e-6)
                remoe_lambda = remoe_lambda * (ratio ** remoe_gamma)
                remoe_lambda = max(args.remoe_lambda_min, min(args.remoe_lambda_max, remoe_lambda))

        log_tokens += args.global_batch * args.seq_len

        # ---- log ----
        if (step + 1) % args.log_interval == 0:
            stat_t = torch.tensor([loss_local, ce_local, aux_local, active_local, l1_raw_local], device=device)
            dist.all_reduce(stat_t, op=dist.ReduceOp.AVG)
            loss_avg, ce_avg, aux_avg, active_avg, l1_raw_avg = stat_t.tolist()

            now = time.time()
            window = now - log_t0
            tps = log_tokens / max(window, 1e-6)
            log_t0 = now
            log_tokens = 0

            elapsed = now - t0
            steps_done = step + 1 - start_step
            eta_h = (args.train_iters - step - 1) * (elapsed / max(1, steps_done)) / 3600
            tokens_seen_b = (step + 1) * args.global_batch * args.seq_len / 1e9

            if is_remoe:
                log0(f"[step {step+1:>6d}/{args.train_iters}] "
                     f"loss={loss_avg:.4f}  ce={ce_avg:.4f}  λ*L1={aux_avg:.4f}  "
                     f"L1={l1_raw_avg:.4f}  act/tok={active_avg:.2f}  λ={remoe_lambda:.2e}  "
                     f"lr={lr:.2e}  gnorm={gnorm:.2f}  "
                     f"tok/s={tps/1e3:.1f}k  seen={tokens_seen_b:.1f}B  eta={eta_h:.1f}h")
            else:
                log0(f"[step {step+1:>6d}/{args.train_iters}] "
                     f"loss={loss_avg:.4f}  ce={ce_avg:.4f}  aux={aux_avg:.4f}  "
                     f"lr={lr:.2e}  gnorm={gnorm:.2f}  "
                     f"tok/s={tps/1e3:.1f}k  seen={tokens_seen_b:.1f}B  eta={eta_h:.1f}h")

            if use_wandb:
                import wandb
                log_data = {
                    "train/loss": loss_avg,
                    "train/ce_loss": ce_avg,
                    "train/lr": lr,
                    "train/grad_norm": gnorm,
                    "train/tokens_per_sec": tps,
                    "train/tokens_seen_b": tokens_seen_b,
                    "train/eta_hours": eta_h,
                    "system/gpu_mem_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "system/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                }
                if is_remoe:
                    log_data.update({
                        "remoe/l1_raw": l1_raw_avg,
                        "remoe/lambda_times_l1": aux_avg,
                        "remoe/lambda": remoe_lambda,
                        "remoe/avg_active_per_token": active_avg,
                    })
                else:
                    log_data["train/aux_loss"] = aux_avg

                # Per-expert stats (both modes)
                unwrapped_log = model.module if hasattr(model, "module") else model
                avg_freq, avg_entropy = unwrapped_log.collect_expert_stats()
                if avg_freq is not None:
                    for ei in range(avg_freq.shape[0]):
                        log_data[f"experts/expert_{ei}_freq"] = avg_freq[ei].item()
                    log_data["experts/freq_std"] = avg_freq.std().item()
                    log_data["experts/max_min_ratio"] = (avg_freq.max() / avg_freq.min().clamp(min=1e-10)).item()
                if avg_entropy is not None:
                    log_data["experts/router_entropy"] = avg_entropy

                # z-loss (topk mode only)
                if not is_remoe:
                    z_terms = [l.moe.last_z_loss for l in unwrapped_log.layers
                               if hasattr(l.moe, 'last_z_loss') and l.moe.last_z_loss is not None]
                    if z_terms:
                        log_data["train/z_loss"] = torch.stack(z_terms).mean().item()

                wandb.log(log_data, step=step + 1)

        # ---- save ----
        if (not args.no_save) and ((step + 1) % args.save_interval == 0 or (step + 1) == args.train_iters):
            log0(f"[ckpt] saving @ step {step+1} ...")
            extra = {"mode": args.mode}
            if is_remoe:
                extra["remoe_lambda"] = remoe_lambda
            save_sharded_checkpoint(
                model, optimizer, step + 1, args.save_dir, rank, world_size,
                args.keep_last_n, extra=extra,
            )

    log0("[done] training finished")
    if use_wandb:
        import wandb
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()