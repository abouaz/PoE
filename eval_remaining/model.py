#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoE Language Model with shared experts, GQA, SwiGLU, RoPE, SDPA.

Supports routing_mode="topk" (standard MoE) and "relu" (ReMoE).
Designed for PyTorch FSDP multi-node training.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEModelOutput:
    """Recognized as a dataclass by torch FSDP's _apply_to_tensors,
    so pre-backward hooks are registered on the contained tensors.
    Returning SimpleNamespace breaks FSDP (state stays IDLE -> backward asserts).
    """
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None


@dataclass
class MoEConfig:
    vocab_size: int = 50304
    hidden_size: int = 1024
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 2048
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # MoE
    num_routed_experts: int = 8
    num_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 2048
    router_aux_loss_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3
    routing_mode: str = "topk"  # "topk" or "relu"

    tie_word_embeddings: bool = False

    def __post_init__(self):
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.routing_mode in ("topk", "relu")


# ============================ Building blocks ============================= #

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(var + self.eps)
        return x.type_as(self.weight) * self.weight


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_pos=8192, base=500000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_pos = max_pos
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos = None
        self._sin = None
        self._cache_dev = None
        self._cache_dtype = None
        self._cache_len = 0

    def _build_cache(self, device, dtype, seq_len):
        need = (
            self._cos is None
            or self._cache_dev != device
            or self._cache_dtype != dtype
            or self._cache_len < seq_len
        )
        if not need:
            return
        n = max(seq_len, self.max_pos)
        t = torch.arange(n, device=device).float()
        freqs = t[:, None] * self.inv_freq.to(device)[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._cache_dev = device
        self._cache_dtype = dtype
        self._cache_len = n

    def forward(self, q, k, position_ids):
        seq_len = int(position_ids.max().item()) + 1
        self._build_cache(q.device, q.dtype, seq_len)
        cos = self._cos[position_ids].unsqueeze(1)
        sin = self._sin[position_ids].unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class Attention(nn.Module):
    """GQA + SDPA + independent head_dim."""
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.h = cfg.hidden_size
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.d = cfg.head_dim
        self.q_proj = nn.Linear(self.h, self.nh * self.d, bias=False)
        self.k_proj = nn.Linear(self.h, self.nkv * self.d, bias=False)
        self.v_proj = nn.Linear(self.h, self.nkv * self.d, bias=False)
        self.o_proj = nn.Linear(self.nh * self.d, self.h, bias=False)
        self.rope = RotaryEmbedding(self.d, cfg.max_position_embeddings, cfg.rope_theta)

    def forward(self, x, position_ids):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.d).transpose(1, 2)
        q, k = self.rope(q, k, position_ids)
        if self.nkv != self.nh:
            n_rep = self.nh // self.nkv
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.nh * self.d)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """Fused gate+up projection for reduced kernel launch overhead."""
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ============================ MoE block =================================== #

class MoEBlock(nn.Module):
    """Shared expert(s) + routed experts with TopK or ReLU routing."""
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.hidden_size
        self.num_routed = cfg.num_routed_experts
        self.num_shared = cfg.num_shared_experts
        self.top_k = cfg.num_experts_per_tok
        self.inter = cfg.moe_intermediate_size
        self.routing_mode = cfg.routing_mode

        self.gate = nn.Linear(self.h, self.num_routed, bias=False)
        self.routed_experts = nn.ModuleList([
            SwiGLU(self.h, self.inter) for _ in range(self.num_routed)
        ])
        self.shared_experts = nn.ModuleList([
            SwiGLU(self.h, self.inter) for _ in range(self.num_shared)
        ])

        self.last_aux_loss = None
        self.last_z_loss = None
        self.last_l1_loss = None
        self.last_active_count = None
        self.last_num_tokens = 0
        self.last_expert_freq = None     # per-expert token fraction [N]
        self.last_router_entropy = None  # mean entropy of routing distribution

    def reset_stats(self):
        self.last_aux_loss = None
        self.last_z_loss = None
        self.last_l1_loss = None
        self.last_active_count = None
        self.last_num_tokens = 0
        self.last_expert_freq = None
        self.last_router_entropy = None

    def forward(self, x):
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        N = flat.shape[0]

        router_logits = self.gate(flat)

        if self.routing_mode == "topk":
            probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
            top_w, top_i = torch.topk(probs, self.top_k, dim=-1)
            top_w = top_w / top_w.sum(dim=-1, keepdim=True)
            top_w = top_w.to(flat.dtype)

            out_routed = torch.zeros_like(flat)
            for e in range(self.num_routed):
                pos_mask = (top_i == e)
                if not pos_mask.any():
                    continue
                token_mask = pos_mask.any(dim=-1)
                idx = token_mask.nonzero(as_tuple=True)[0]
                w = (top_w[idx] * pos_mask[idx]).sum(dim=-1, keepdim=True)
                out_e = self.routed_experts[e](flat[idx]) * w
                out_routed.index_add_(0, idx, out_e.to(flat.dtype))

            # Switch Transformer load-balancing loss
            mean_probs = probs.mean(dim=0)
            one_hot = F.one_hot(top_i, num_classes=self.num_routed).float()
            freq = one_hot.sum(dim=(0, 1)) / (N * self.top_k)
            self.last_aux_loss = (self.num_routed * (mean_probs * freq).sum()).to(flat.dtype)

            # Per-expert stats for logging
            with torch.no_grad():
                self.last_expert_freq = freq.detach()
                log_probs = torch.log(probs + 1e-10)
                self.last_router_entropy = -(probs * log_probs).sum(dim=-1).mean().detach()

            # z-loss for router stability
            if self.cfg.router_z_loss_coef > 0 and self.training:
                z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()
                self.last_z_loss = z_loss.to(flat.dtype)
            else:
                self.last_z_loss = None

        else:
            # ReMoE (ReLU routing)
            scores = F.relu(router_logits)
            self.last_l1_loss = scores.sum() / max(1, N)
            with torch.no_grad():
                self.last_active_count = (scores > 0).float().sum() / max(1, N)
                self.last_num_tokens = N
                self.last_expert_freq = (scores > 0).float().mean(dim=0)
                # Entropy of normalized scores (for tokens with any active expert)
                score_sum = scores.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                norm_scores = scores / score_sum
                log_ns = torch.log(norm_scores + 1e-10)
                self.last_router_entropy = -(norm_scores * log_ns).sum(dim=-1).mean()

            out_routed = torch.zeros_like(flat)
            for e in range(self.num_routed):
                active = scores[:, e] > 0
                if not active.any():
                    continue
                idx = active.nonzero(as_tuple=True)[0]
                out_e = self.routed_experts[e](flat[idx]) * scores[idx, e:e + 1]
                out_routed.index_add_(0, idx, out_e.to(flat.dtype))

        out_shared = sum(expert(flat) for expert in self.shared_experts)
        return (out_routed + out_shared).view(B, T, H)


# ============================ Decoder layer & model ======================= #

class DecoderLayer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.moe = MoEBlock(cfg)

    def forward(self, x, position_ids):
        x = x + self.self_attn(self.input_layernorm(x), position_ids)
        x = x + self.moe(self.post_attention_layernorm(x))
        return x


class MoEForCausalLM(nn.Module):
    """HF-compatible CausalLM interface for FSDP training."""
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, m):
        std = self.config.initializer_range
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def reset_remoe_stats(self):
        for layer in self.layers:
            layer.moe.reset_stats()

    def collect_remoe_stats(self):
        l1, active = [], []
        for layer in self.layers:
            if layer.moe.last_l1_loss is not None:
                l1.append(layer.moe.last_l1_loss)
                active.append(layer.moe.last_active_count)
        return l1, active

    def collect_expert_stats(self):
        """Collect per-expert frequency and router entropy across all layers."""
        freqs, entropies = [], []
        for layer in self.layers:
            if layer.moe.last_expert_freq is not None:
                freqs.append(layer.moe.last_expert_freq)
            if layer.moe.last_router_entropy is not None:
                entropies.append(layer.moe.last_router_entropy)
        avg_freq = torch.stack(freqs).mean(dim=0) if freqs else None
        avg_entropy = torch.stack(entropies).mean().item() if entropies else None
        return avg_freq, avg_entropy

    def forward(self, input_ids, labels=None, **kwargs):
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        for layer in self.layers:
            x = layer(x, position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        aux_total = None
        z_loss_total = None
        if self.config.routing_mode == "topk":
            aux_terms = [l.moe.last_aux_loss for l in self.layers if l.moe.last_aux_loss is not None]
            z_terms = [l.moe.last_z_loss for l in self.layers
                       if hasattr(l.moe, 'last_z_loss') and l.moe.last_z_loss is not None]
            if aux_terms:
                aux_total = torch.stack(aux_terms).mean()
                if loss is not None:
                    loss = loss + self.config.router_aux_loss_coef * aux_total
            if z_terms:
                z_loss_total = torch.stack(z_terms).mean()
                if loss is not None:
                    loss = loss + self.config.router_z_loss_coef * z_loss_total

        return MoEModelOutput(loss=loss, logits=logits, aux_loss=aux_total, z_loss=z_loss_total)


# ============================ Util ======================================== #

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_active_params(cfg: MoEConfig):
    """Per-token activated params (excluding embed/lm_head)."""
    h = cfg.hidden_size
    d = cfg.head_dim
    inter = cfg.moe_intermediate_size

    attn = (
        h * cfg.num_attention_heads * d
        + h * cfg.num_key_value_heads * d * 2
        + cfg.num_attention_heads * d * h
    )
    # SwiGLU: gate_up (h→2*inter) + down (inter→h) = 3 * h * inter
    expert_params = 3 * h * inter
    moe_active = (cfg.num_experts_per_tok + cfg.num_shared_experts) * expert_params
    router = h * cfg.num_routed_experts
    norm = 2 * h
    per_layer_active = attn + moe_active + router + norm
    return per_layer_active * cfg.num_hidden_layers