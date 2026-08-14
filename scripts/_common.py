#!/usr/bin/env python3
"""Shared Phase-4 utilities: DPO training, evaluation, sampling-based diversity.

Imported by 05_run_beta_sweep.py and 06_run_baseline_ladder.py so the training/eval
logic is defined once (matches scripts/04_train_pareto_dpo.py exactly).
"""
from __future__ import annotations
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from models.policy_decoder import ARDecoder, DecoderConfig, NucleotideTokenizer  # noqa: E402
from models.pareto_dpo_loss import pareto_dpo_loss                               # noqa: E402

PAIRS = PROJECT / "data" / "preference_pairs"
SFT = PROJECT / "models" / "checkpoints" / "sft_ref_base.pt"
TOK = NucleotideTokenizer()


def load_pref(name: str):
    """Load a *_ref.csv split -> (ids_w, ids_l, ref_w, ref_l)."""
    df = pd.read_csv(PAIRS / f"{name}_ref.csv")
    return (TOK.encode(df["y_w"].tolist()), TOK.encode(df["y_l"].tolist()),
            torch.tensor(df["ref_logp_w"].to_numpy(), dtype=torch.float32),
            torch.tensor(df["ref_logp_l"].to_numpy(), dtype=torch.float32))


def new_policy_from_sft(device) -> ARDecoder:
    st = torch.load(SFT, map_location=device, weights_only=False)
    m = ARDecoder(DecoderConfig(**st["config"])).to(device)
    m.load_state_dict(st["model"])
    return m


@torch.no_grad()
def dpo_eval(model, split, device, beta, amp, mb=256) -> dict:
    ids_w, ids_l, rw_all, rl_all = split
    model.eval()
    n = len(ids_w)
    agg = {k: 0.0 for k in ["loss", "reward_acc", "reward_margin",
                            "kl_to_ref", "implicit_reward_var", "disp_chosen"]}
    for i in range(0, n, mb):
        s = slice(i, i + mb)
        iw, il = ids_w[s].to(device), ids_l[s].to(device)
        rw, rl = rw_all[s].to(device), rl_all[s].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pw = model.sequence_logprob(iw).float()
            pl = model.sequence_logprob(il).float()
        loss, st = pareto_dpo_loss(pw, pl, rw, rl, beta=beta)
        bs = len(iw)
        agg["loss"] += loss.item() * bs
        agg["reward_acc"] += st.reward_accuracy.item() * bs
        agg["reward_margin"] += st.reward_margin.item() * bs
        agg["kl_to_ref"] += (pw - rw).mean().item() * bs
        agg["implicit_reward_var"] += torch.cat([beta * (pw - rw), beta * (pl - rl)]).var().item() * bs
        agg["disp_chosen"] += (pw - rw).mean().item() * bs        # chosen-logprob displacement
    return {k: v / n for k, v in agg.items()}


def dpo_train(model, train, val, device, *, beta, epochs, micro_batch=16,
              accum=4, lr=1e-4, seed=42, amp=True, verbose=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n = len(train[0])
    spe = int(np.ceil(n / micro_batch / accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * spe, eta_min=lr / 20)
    best_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + ep))
        opt.zero_grad(); micro = 0
        for i in range(0, n, micro_batch):
            idx = perm[i:i + micro_batch]
            iw, il = train[0][idx].to(device), train[1][idx].to(device)
            rw, rl = train[2][idx].to(device), train[3][idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pw = model.sequence_logprob(iw).float()
                pl = model.sequence_logprob(il).float()
            loss, _ = pareto_dpo_loss(pw, pl, rw, rl, beta=beta)
            (loss / accum).backward()
            micro += 1
            if micro % accum == 0 or i + micro_batch >= n:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); sched.step()
        ev = dpo_eval(model, val, device, beta, amp)
        if ev["reward_acc"] > best_acc:
            best_acc = ev["reward_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    ep{ep:02d} val_acc={ev['reward_acc']:.3f} KL={ev['kl_to_ref']:.1f} "
                  f"margin={ev['reward_margin']:.2f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


# ─── diversity of a generated pool ───────────────────────────────────────────
def diversity_metrics(seqs: list[str], k: int = 3, n_pairs: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    arr = np.array([[c for c in s] for s in seqs])
    N, L = arr.shape
    # positional Shannon entropy (bits), averaged over positions; max 2.0
    pos_ent = []
    for j in range(L):
        _, cnt = np.unique(arr[:, j], return_counts=True)
        p = cnt / cnt.sum()
        pos_ent.append(-(p * np.log2(p)).sum())
    # k-mer (3-mer) Shannon entropy over the pool
    kmers = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            kmers[s[i:i + k]] += 1
    tot = sum(kmers.values())
    p = np.array(list(kmers.values())) / tot
    kmer_ent = float(-(p * np.log2(p)).sum())
    # mean pairwise Hamming (sampled) / L  == normalized k-mer distance proxy
    idx = rng.integers(0, N, size=(min(n_pairs, N * (N - 1) // 2), 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    ham = (arr[idx[:, 0]] != arr[idx[:, 1]]).sum(1).mean() / L
    return {"positional_entropy_bits": float(np.mean(pos_ent)),
            "kmer3_entropy_bits": kmer_ent,
            "kmer3_entropy_pct_max": float(kmer_ent / math.log2(4 ** k) * 100),
            "mean_pairwise_hamming_frac": float(ham),
            "unique_frac": len(set(seqs)) / N, "n": N}
