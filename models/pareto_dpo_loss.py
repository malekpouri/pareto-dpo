#!/usr/bin/env python3
"""
Pareto-DPO loss — scalarization-free multi-objective preference alignment (RFC-001 §2.3).

The "Pareto" (multi-objective) content lives entirely in the DATA: preference pairs are
constructed from the 2-D Pareto-dominance partial order (scripts/01_build_preference_
pairs.py), so a "winner" Pareto-dominates its "loser" and no scalar objective weights
`w_k` are ever formed. Given such dominance pairs, the alignment objective is the DPO
loss over log-probability ratios to a frozen reference:

    h(y) = beta * ( log pi_theta(y) - log pi_ref(y) )          # implicit reward (no reward net)
    L    = - E_{(y_w, y_l) ~ D_dominance} [ log sigmoid( h(y_w) - h(y_l) ) ]

There is NO reward network in the graph -> nothing for the policy to saturate (RQ1).
This module is objective-agnostic: it consumes only log-probs + the dominance labelling
already imposed on the batch.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DPOStats:
    loss: torch.Tensor
    reward_w: torch.Tensor        # mean implicit reward of winners
    reward_l: torch.Tensor        # mean implicit reward of losers
    reward_margin: torch.Tensor   # mean (r_w - r_l)
    reward_accuracy: torch.Tensor # fraction of pairs with r_w > r_l


def pareto_dpo_loss(policy_logp_w: torch.Tensor, policy_logp_l: torch.Tensor,
                    ref_logp_w: torch.Tensor, ref_logp_l: torch.Tensor,
                    beta: float = 0.1, reduction: str = "mean"):
    """Scalarization-free DPO loss over dominance pairs.

    Args (all shape (B,)):
        policy_logp_w/l : log pi_theta(y_w / y_l)
        ref_logp_w/l    : log pi_ref(y_w / y_l)   (precomputed offline, RFC §4.2)
        beta            : implicit-KL temperature
    Returns: (loss_tensor, DPOStats)
    """
    pi_ratio_w = policy_logp_w - ref_logp_w        # log[pi_theta/pi_ref] for winners
    pi_ratio_l = policy_logp_l - ref_logp_l
    logits = beta * (pi_ratio_w - pi_ratio_l)      # = h(y_w) - h(y_l)
    per_pair = -F.logsigmoid(logits)               # numerically stable

    loss = per_pair.mean() if reduction == "mean" else \
        (per_pair.sum() if reduction == "sum" else per_pair)

    with torch.no_grad():
        r_w = beta * pi_ratio_w
        r_l = beta * pi_ratio_l
        stats = DPOStats(loss=loss.detach(),
                         reward_w=r_w.mean(), reward_l=r_l.mean(),
                         reward_margin=(r_w - r_l).mean(),
                         reward_accuracy=(r_w > r_l).float().mean())
    return loss, stats


class ParetoDPOLoss(torch.nn.Module):
    """Module wrapper. Optionally reports a per-objective DIAGNOSTIC decomposition
    (RFC-001 §2.3): for pairs whose dominance is driven by efficacy-only vs
    suppression-only vs both, the loss is reported separately. This is *diagnostic
    gradient attribution only* — the objectives are never fused into a scalar reward
    that the policy optimizes; the single training loss is the dominance DPO loss above.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l,
                dominance_kind: torch.Tensor | None = None):
        loss, stats = pareto_dpo_loss(policy_logp_w, policy_logp_l,
                                      ref_logp_w, ref_logp_l, self.beta)
        diag = {}
        if dominance_kind is not None:      # 0=both, 1=efficacy-only, 2=suppression-only
            with torch.no_grad():
                logits = self.beta * ((policy_logp_w - ref_logp_w) -
                                      (policy_logp_l - ref_logp_l))
                per = -F.logsigmoid(logits)
                for k, name in {0: "both", 1: "eff_only", 2: "supp_only"}.items():
                    m = dominance_kind.eq(k)
                    diag[name] = per[m].mean().item() if m.any() else float("nan")
        return loss, stats, diag


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests  (run:  python models/pareto_dpo_loss.py)
# ─────────────────────────────────────────────────────────────────────────────
def _tests():
    import math
    torch.manual_seed(0)
    B = 256
    LOG2 = math.log(2.0)

    # (1) policy == reference  ->  logits 0  ->  loss = -log sigmoid(0) = log 2
    z = torch.randn(B)
    loss, st = pareto_dpo_loss(z, z.clone(), z.clone(), z.clone(), beta=0.1)
    assert abs(loss.item() - LOG2) < 1e-5, loss.item()
    assert abs(st.reward_margin.item()) < 1e-6
    assert abs(st.reward_accuracy.item() - 0.0) < 1e-6 or st.reward_accuracy.item() == 0.0

    # (2) closed-form value for a fixed margin
    beta = 0.2
    plw = torch.tensor([1.0]); pll = torch.tensor([0.0])
    rlw = torch.tensor([0.0]); rll = torch.tensor([0.0])
    delta = beta * ((1.0 - 0.0) - (0.0 - 0.0))          # = 0.2
    expect = -math.log(1.0 / (1.0 + math.exp(-delta)))
    loss2, _ = pareto_dpo_loss(plw, pll, rlw, rll, beta=beta)
    assert abs(loss2.item() - expect) < 1e-6, (loss2.item(), expect)

    # (3) gradient direction: raising winner log-prob must LOWER the loss; raising
    #     loser log-prob must RAISE it.
    plw = torch.randn(B, requires_grad=True)
    pll = torch.randn(B, requires_grad=True)
    rlw, rll = torch.randn(B), torch.randn(B)
    loss3, _ = pareto_dpo_loss(plw, pll, rlw, rll, beta=0.1)
    loss3.backward()
    assert (plw.grad <= 1e-8).all(), "dL/dlogp_w should be <= 0"
    assert (pll.grad >= -1e-8).all(), "dL/dlogp_l should be >= 0"

    # (4) reward accuracy: winners strictly better -> accuracy 1.0
    plw = torch.full((B,), 2.0); pll = torch.full((B,), -2.0)
    rlw = torch.zeros(B); rll = torch.zeros(B)
    _, st4 = pareto_dpo_loss(plw, pll, rlw, rll, beta=0.5)
    assert st4.reward_accuracy.item() == 1.0
    assert st4.reward_margin.item() > 0

    # (5) module + per-objective diagnostic decomposition
    mod = ParetoDPOLoss(beta=0.1)
    kind = torch.randint(0, 3, (B,))
    l5, s5, diag = mod(plw, pll, rlw, rll, dominance_kind=kind)
    assert set(diag) == {"both", "eff_only", "supp_only"}

    print("all Pareto-DPO loss unit tests passed:")
    print(f"  (1) policy==ref loss = {loss.item():.6f}  (expected {LOG2:.6f})")
    print(f"  (2) closed-form loss = {loss2.item():.6f}  (expected {expect:.6f})")
    print(f"  (3) grad signs correct (dL/dw<=0, dL/dl>=0)")
    print(f"  (4) separable pairs -> reward_accuracy = {st4.reward_accuracy.item():.3f}")
    print(f"  (5) diagnostic decomposition keys = {list(diag)}")


if __name__ == "__main__":
    _tests()
