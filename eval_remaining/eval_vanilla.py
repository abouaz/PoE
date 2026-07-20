#!/usr/bin/env python
"""
eval_checkpoints.py — evaluate all backed-up checkpoints on 7 benchmarks.
Logs results to W&B so you can see metric evolution over training.

Requirements (install once):
    pip install lm-eval>=0.4.0 --break-system-packages

Usage:
    python eval_checkpoints.py \
        --ckpt-dir .../182m_moe_n16_k8_all_ckpts \
        --wandb-project PoE \
        --wandb-run 182m_moe_n16_k8_eval

Or via SLURM:
    sbatch launch_eval.sh
"""

import argparse
import os
import re
import json
import glob
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# model.py must be in the same directory as this script
from model import MoEConfig, MoEForCausalLM

import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


# ======================= config =========================================== #

TASKS = [
    "arc_challenge",   # ARC-c   → acc_norm
    "arc_easy",        # ARC-e   → acc_norm
    "boolq",           # BoolQ   → acc
    "hellaswag",       # HellaSwag → acc_norm
    "lambada_openai",  # LAMBADA → acc, ppl
    "piqa",            # PIQA    → acc_norm
    "race",            # RACE    → acc
]

# Matches the 182M N=16 k=8 training config
MODEL_CONFIG = dict(
    vocab_size=50304,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    head_dim=64,
    max_position_embeddings=1024,
    rope_theta=500000.0,
    num_routed_experts=16,
    num_shared_experts=1,
    num_experts_per_tok=8,
    moe_intermediate_size=3072,
    routing_mode="topk",
)

# Which metric to extract per task for the summary table
PRIMARY_METRIC = {
    "arc_challenge":  "acc_norm,none",
    "arc_easy":       "acc_norm,none",
    "boolq":          "acc,none",
    "hellaswag":      "acc_norm,none",
    "lambada_openai": "acc,none",
    "piqa":           "acc_norm,none",
    "race":           "acc,none",
}


# ======================= lm-eval wrapper ================================== #

@register_model("custom_moe")
class CustomMoELM(LM):
    """Thin wrapper so lm-eval-harness can call our MoEForCausalLM."""

    def __init__(self, model, tokenizer, batch_size=32, device="cuda"):
        super().__init__()
        self._model = model.to(device).eval()
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = torch.device(device)

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, **kwargs):
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens, **kwargs):
        return self._tokenizer.decode(tokens)

    def _encode_pair(self, context, continuation):
        ctx_ids = self.tok_encode(context)
        cont_ids = self.tok_encode(continuation)
        return ctx_ids, cont_ids

    def loglikelihood(self, requests):
        """Compute log-prob of continuation given context."""
        results = []
        reqs = [req.args for req in requests]

        for i in range(0, len(reqs), self._batch_size):
            batch = reqs[i : i + self._batch_size]
            max_len = self.max_length

            all_logprobs = []
            all_greedy = []

            for context, continuation in batch:
                ctx_ids, cont_ids = self._encode_pair(context, continuation)

                # Truncate from the left if too long
                full_ids = ctx_ids + cont_ids
                if len(full_ids) > max_len:
                    full_ids = full_ids[-max_len:]
                    cont_len = len(cont_ids)
                else:
                    cont_len = len(cont_ids)

                input_ids = torch.tensor([full_ids], device=self._device)

                with torch.no_grad():
                    output = self._model(input_ids)
                    logits = output.logits[0]  # [T, V]

                # Log-probs of continuation tokens
                # logits[t] predicts token[t+1], so for continuation starting
                # at position (len - cont_len), we need logits from
                # (len - cont_len - 1) to (len - 2)
                shift_logits = logits[-(cont_len + 1):-1]  # [cont_len, V]
                shift_labels = torch.tensor(
                    full_ids[-cont_len:], device=self._device
                )

                log_probs = F.log_softmax(shift_logits.float(), dim=-1)
                token_log_probs = log_probs[
                    torch.arange(cont_len), shift_labels
                ]
                total_log_prob = token_log_probs.sum().item()

                # Check if greedy decoding matches
                greedy_ids = shift_logits.argmax(dim=-1)
                is_greedy = (greedy_ids == shift_labels).all().item()

                all_logprobs.append(total_log_prob)
                all_greedy.append(is_greedy)

            results.extend(zip(all_logprobs, all_greedy))

        return results

    def loglikelihood_rolling(self, requests):
        """Compute total log-prob of a string (no context split)."""
        results = []
        for req in requests:
            (string,) = req.args
            token_ids = self.tok_encode(string)

            max_len = self.max_length
            total_log_prob = 0.0

            # Process in chunks if longer than max_length
            for start in range(0, len(token_ids), max_len):
                chunk = token_ids[start : start + max_len]
                input_ids = torch.tensor([chunk], device=self._device)

                with torch.no_grad():
                    output = self._model(input_ids)
                    logits = output.logits[0]  # [T, V]

                shift_logits = logits[:-1].float()
                shift_labels = torch.tensor(chunk[1:], device=self._device)

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs[
                    torch.arange(len(shift_labels)), shift_labels
                ]
                total_log_prob += token_log_probs.sum().item()

            results.append((total_log_prob,))

        return results

    def generate_until(self, requests):
        """Greedy generation until stop string or max tokens."""
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            stop = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_ids = self.tok_encode(context)
            if len(ctx_ids) > self.max_length:
                ctx_ids = ctx_ids[-self.max_length:]

            generated = []
            input_ids = torch.tensor([ctx_ids], device=self._device)

            with torch.no_grad():
                for _ in range(max_gen):
                    if input_ids.shape[1] > self.max_length:
                        input_ids = input_ids[:, -self.max_length:]
                    output = self._model(input_ids)
                    next_id = output.logits[0, -1].argmax().item()
                    generated.append(next_id)

                    if next_id == self.eot_token_id:
                        break

                    gen_text = self.tok_decode(generated)
                    if any(s in gen_text for s in stop):
                        break

                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[next_id]], device=self._device)
                    ], dim=1)

            results.append(self.tok_decode(generated))

        return results


# ======================= helpers ========================================== #

def discover_checkpoints(ckpt_dir):
    """Find all step_NNNNNNNN.pt files, return sorted list of (step, path)."""
    entries = []
    for name in os.listdir(ckpt_dir):
        m = re.match(r"step_(\d+)\.pt$", name)
        if not m:
            continue
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            entries.append((int(m.group(1)), path))
    entries.sort()
    return entries


def load_model(ckpt_path, device="cuda"):
    """Build model from config and load fp16 weights."""
    cfg = MoEConfig(**MODEL_CONFIG)
    model = MoEForCausalLM(cfg)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)

    # Extract step number from filename (step_00005000.pt -> 5000)
    basename = os.path.basename(ckpt_path)
    m = re.match(r"step_(\d+)\.pt$", basename)
    step = int(m.group(1)) if m else 0

    model = model.to(device=device, dtype=torch.float16).eval()
    return model, cfg, step


def extract_metrics(results):
    """Pull primary metric per task from lm-eval results dict."""
    metrics = {}
    for task in TASKS:
        if task not in results["results"]:
            continue
        task_results = results["results"][task]
        key = PRIMARY_METRIC.get(task)
        if key and key in task_results:
            metrics[task] = task_results[key]
        # Also grab perplexity for LAMBADA if available
        if task == "lambada_openai" and "perplexity,none" in task_results:
            metrics["lambada_ppl"] = task_results["perplexity,none"]
    return metrics


# ======================= plotting ========================================= #

TASK_SHORT = {
    "arc_challenge":  "ARC-c",
    "arc_easy":       "ARC-e",
    "boolq":          "BoolQ",
    "hellaswag":      "HellaSwag",
    "lambada_openai": "LAMBADA",
    "piqa":           "PIQA",
    "race":           "RACE",
}

# Distinct colours per benchmark
TASK_COLORS = {
    "arc_challenge":  "#e6194b",
    "arc_easy":       "#3cb44b",
    "boolq":          "#4363d8",
    "hellaswag":      "#f58231",
    "lambada_openai": "#911eb4",
    "piqa":           "#42d4f4",
    "race":           "#f032e6",
}


def plot_progress(results_dir, plot_path):
    """Read all result JSONs, plot benchmark curves, save PNG."""
    json_files = sorted(glob.glob(os.path.join(results_dir, "step_*.json")))
    if not json_files:
        return

    # Load all results
    all_data = []
    for f in json_files:
        with open(f) as fp:
            all_data.append(json.load(fp))
    all_data.sort(key=lambda d: d.get("step", 0))

    tokens_b = [d["tokens_B"] for d in all_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each benchmark
    for task in TASKS:
        vals = [d.get(task) for d in all_data]
        if any(v is not None for v in vals):
            ax.plot(tokens_b, vals,
                    marker="o", markersize=4, linewidth=1.2,
                    color=TASK_COLORS.get(task, "gray"),
                    label=TASK_SHORT.get(task, task), alpha=0.7)

    # Plot average (bold)
    avgs = [d.get("avg") for d in all_data]
    if any(v is not None for v in avgs):
        ax.plot(tokens_b, avgs,
                marker="s", markersize=6, linewidth=2.5,
                color="black", label="Average", zorder=10)

    ax.set_xlabel("Tokens seen (B)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("182M MoE (N=16, k=8) — Eval over training", fontsize=13)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {plot_path}")


# ======================= main ============================================= #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                        help="Directory with step_*/checkpoint.pt backups")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="PoE")
    parser.add_argument("--wandb-entity", default="YOUR_WANDB_ENTITY")
    parser.add_argument("--wandb-run", default="182m_moe_n16_k8_eval")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--results-dir", default=None,
                        help="Save per-checkpoint JSON results here (default: ckpt-dir/eval_results)")
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(args.ckpt_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)

    # ---- discover checkpoints ----
    checkpoints = discover_checkpoints(args.ckpt_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for step, path in checkpoints:
        print(f"  step {step:>8d}  {path}")
    print()

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- wandb ----
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config={
                "eval_tasks": TASKS,
                "model_config": MODEL_CONFIG,
                "num_checkpoints": len(checkpoints),
            },
        )

    # ---- eval loop ----
    tokens_per_step = 1536 * 1024  # GLOBAL_BATCH * SEQ_LEN
    plot_path = os.path.join(results_dir, "eval_progress.png")

    for step, ckpt_path in checkpoints:
        result_file = os.path.join(results_dir, f"step_{step:08d}.json")

        # Skip if already evaluated
        if os.path.isfile(result_file):
            print(f"[step {step}] already evaluated, skipping")
            plot_progress(results_dir, plot_path)
            continue

        print(f"[step {step}] loading checkpoint ...")
        model, cfg, ckpt_step = load_model(ckpt_path, device=args.device)

        lm = CustomMoELM(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            device=args.device,
        )

        print(f"[step {step}] running {len(TASKS)} benchmarks ...")
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=TASKS,
            batch_size=args.eval_batch_size,
        )

        metrics = extract_metrics(results)
        avg = sum(v for k, v in metrics.items() if k != "lambada_ppl") / max(
            1, len([k for k in metrics if k != "lambada_ppl"])
        )
        metrics["avg"] = avg

        tokens_seen = step * tokens_per_step
        metrics["step"] = step
        metrics["tokens_B"] = tokens_seen / 1e9

        # Print
        print(f"[step {step}] tokens={tokens_seen/1e9:.1f}B")
        for task, val in sorted(metrics.items()):
            if task in ("step", "tokens_B"):
                continue
            print(f"  {task:20s}: {val:.4f}")
        print()

        # Save JSON
        with open(result_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Update plot with all results so far
        plot_progress(results_dir, plot_path)

        # Log to W&B
        if use_wandb:
            log_dict = {}
            for k, v in metrics.items():
                if k in ("step", "tokens_B"):
                    continue
                log_dict[f"eval/{k}"] = v
            log_dict["eval/avg"] = avg
            wandb.log(log_dict, step=step)

        # Free GPU memory before next checkpoint
        del model, lm
        torch.cuda.empty_cache()

    print("All checkpoints evaluated.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
