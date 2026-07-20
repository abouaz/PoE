#!/usr/bin/env python
"""CPU smoke test for the PoE_4 implementation."""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model_poe4 import PoEConfig, PoEForCausalLM, poe_lambda_step, count_active_params_poe

torch.manual_seed(0)

def tiny_cfg(**kw):
    base = dict(
        vocab_size=256, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, num_routed_experts=6, num_shared_experts=1,
        num_experts_per_tok=2, moe_intermediate_size=128,
    )
    base.update(kw)
    return PoEConfig(**base)


def run_one(cov_estimator, shrinkage_mode, b=0.05, steps=6):
    cfg = tiny_cfg(cov_estimator=cov_estimator, shrinkage_mode=shrinkage_mode,
                   b_target=b, return_scale="rms")
    model = PoEForCausalLM(cfg)
    model.train()
    model.set_b(b)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B, T = 3, 32
    lam = 1e-3
    for step in range(steps):
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        labels = input_ids.clone()

        model.reset_poe_stats()
        out = model(input_ids=input_ids, labels=labels)
        l1, div, active = model.collect_poe_stats()
        assert len(l1) == cfg.num_hidden_layers, (len(l1), cfg.num_hidden_layers)
        assert len(div) == cfg.num_hidden_layers
        l1_mean = torch.stack(l1).mean()
        div_mean = torch.stack(div).mean()
        active_mean = torch.stack(active).mean()

        total = out.loss + lam * l1_mean + div_mean
        assert torch.isfinite(total), f"non-finite total at step {step}"
        # div must carry gradient into the gate from step 2 onward (Σ̂ != 0)
        opt.zero_grad(set_to_none=True)
        total.backward()

        # gate gradient should be present
        g = model.layers[0].moe.gate.weight.grad
        assert g is not None and torch.isfinite(g).all()

        model.update_portfolio_stats()
        opt.step()

        # lambda controller
        lam = poe_lambda_step(lam, active_mean.item(), target_k=cfg.num_experts_per_tok,
                              eta=1.01)

    # After several steps Σ̂ should be populated (unless full shrink / b ramp)
    Sig = model.layers[0].moe.Sigma_hat
    Chat = model.layers[0].moe.C_hat
    gamma = model.avg_shrinkage_gamma()
    diag_ok = torch.allclose(Sig.diag(), torch.zeros(cfg.num_routed_experts))
    sym = "n/a"
    if cov_estimator in ("coactivation", "routing_weighted", "batchmean",
                         "set_overlap", "output_similarity"):
        sym = float((Chat - Chat.t()).abs().max())
    print(f"  est={cov_estimator:18s} shrink={shrinkage_mode:11s} "
          f"loss={out.loss.item():.3f} div={div_mean.item():+.3e} "
          f"act/tok={active_mean.item():.2f} gamma={gamma:.3f} "
          f"|Sig|max={float(Sig.abs().max()):.3e} diag0={diag_ok} sym={sym}")
    assert diag_ok, "Sigma_hat diagonal must be zero"
    return True


def test_b_zero_recovers_relu():
    """b=0 -> L_div == 0 and no gradient contribution (Proposition 3)."""
    cfg = tiny_cfg(b_target=0.0)
    model = PoEForCausalLM(cfg)
    model.train()
    model.set_b(0.0)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    model.reset_poe_stats()
    out = model(input_ids=input_ids, labels=input_ids.clone())
    _, div, _ = model.collect_poe_stats()
    dsum = torch.stack(div).sum()
    assert float(dsum) == 0.0, f"L_div should be exactly 0 at b=0, got {float(dsum)}"
    print(f"  b=0: L_div={float(dsum)} (exact zero) OK")


def test_grad_only_into_gate_from_div():
    """L_div alone should produce gradient only on gate weights, not experts.

    Uses a single layer so there is no cross-layer residual path: with one
    layer, L_div depends on scores=ReLU(gate(input)) but not on that layer's
    expert outputs (y is not in L_div's path and Σ̂ is detached), so the layer's
    own experts must receive zero gradient from L_div.
    """
    cfg = tiny_cfg(b_target=0.1, shrinkage_mode="constant", shrinkage_gamma=0.0,
                   num_hidden_layers=1)
    model = PoEForCausalLM(cfg)
    model.train()
    model.set_b(0.1)
    # Prime Σ̂ with one full step.
    for _ in range(2):
        ids = torch.randint(0, cfg.vocab_size, (3, 32))
        model.reset_poe_stats()
        out = model(input_ids=ids, labels=ids.clone())
        l1, div, _ = model.collect_poe_stats()
        (out.loss + torch.stack(div).mean()).backward()
        model.update_portfolio_stats()
        model.zero_grad(set_to_none=True)

    # Now isolate L_div gradient.
    ids = torch.randint(0, cfg.vocab_size, (3, 32))
    model.reset_poe_stats()
    out = model(input_ids=ids, labels=ids.clone())
    _, div, _ = model.collect_poe_stats()
    div_mean = torch.stack(div).mean()
    model.zero_grad(set_to_none=True)
    div_mean.backward()
    moe0 = model.layers[0].moe
    gate_grad = moe0.gate.weight.grad
    expert_grad = moe0.routed_experts[0].down_proj.weight.grad
    has_gate = gate_grad is not None and float(gate_grad.abs().sum()) > 0
    expert_zero = expert_grad is None or float(expert_grad.abs().sum()) == 0.0
    print(f"  L_div grad -> gate present={has_gate}  experts_zero={expert_zero}")
    assert has_gate, "L_div must produce gate gradient"
    assert expert_zero, "L_div must NOT flow into expert weights (Σ̂ is constant)"


if __name__ == "__main__":
    print("active params (tiny):", count_active_params_poe(tiny_cfg()))
    print("\n[1] b=0 recovers ReLU:")
    test_b_zero_recovers_relu()
    print("\n[2] L_div gradient routing:")
    test_grad_only_into_gate_from_div()
    print("\n[3] all estimators x representative shrinkage modes:")
    for est in ["coactivation", "batchmean", "routing_weighted",
                "set_overlap", "output_similarity"]:
        run_one(est, "ledoit_wolf")
    print("\n[4] all shrinkage modes (coactivation):")
    for sk in ["ledoit_wolf", "schedule", "constant", "none"]:
        run_one("coactivation", sk)
    print("\nALL TESTS PASSED")
