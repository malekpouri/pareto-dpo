#!/usr/bin/env python3
"""
Phase 2.2 — SFT the autoregressive reference decoder on the CRISPGen pool (RFC-001
Decision 3). This realises "initialise from the CRISPGen SFT base" *in distribution*:
the AR decoder is trained by next-token maximum likelihood on CRISPGen-generated 20-nt
guides, yielding a normalized generative reference pi_ref with an exact, cheap
log pi_ref(y) for the offline DPO precompute (RFC §4.2).

Output: models/checkpoints/sft_ref_base.pt  = {"model": state_dict, "config": {...}}
        results/json/sft_metrics.json
Memory: tiny model (~1.8M params) over 20-nt sequences -> fits the 8 GB budget trivially
        (BF16 autocast on CUDA). This is base-model pretraining, NOT alignment.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from models.policy_decoder import ARDecoder, DecoderConfig, NucleotideTokenizer, PAD  # noqa: E402

CKPT = PROJECT / "models" / "checkpoints" / "sft_ref_base.pt"
METRICS = PROJECT / "results" / "json" / "sft_metrics.json"
VALID = set("ACGT")


def load_sequences(n: int, seed: int) -> list[str]:
    df = pd.read_csv(NOTEBOOK / "report" / "CRISPGen_Mass_Pool.csv", usecols=["seq"])
    s = df["seq"].astype(str).str.strip().str.upper()
    s = s[s.str.len().eq(20) & s.apply(lambda x: set(x) <= VALID)]
    if n and n < len(s):
        s = s.sample(n=n, random_state=seed)
    return s.tolist()


@torch.no_grad()
def evaluate(model, loader, device, amp) -> float:
    model.eval()
    tot, ntok = 0.0, 0
    for (ids,) in loader:
        ids = ids.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               ids[:, 1:].reshape(-1), ignore_index=PAD, reduction="sum")
        n = ids[:, 1:].ne(PAD).sum().item()
        tot += loss.item(); ntok += n
    return tot / max(ntok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000, help="sequences sampled from the 3M pool")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.cuda.manual_seed_all(args.seed); torch.cuda.reset_peak_memory_stats()

    seqs = load_sequences(args.n, args.seed)
    tok = NucleotideTokenizer()
    ids = tok.encode(seqs)                                  # (N, 22)
    n_val = int(round(args.val_frac * len(ids)))
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed))
    val_ids, tr_ids = ids[perm[:n_val]], ids[perm[n_val:]]
    tr_loader = DataLoader(TensorDataset(tr_ids), batch_size=args.batch, shuffle=True,
                           generator=torch.Generator().manual_seed(args.seed), drop_last=True)
    va_loader = DataLoader(TensorDataset(val_ids), batch_size=args.batch, shuffle=False)
    print(f"[data] {len(tr_ids):,} train / {len(val_ids):,} val 20-nt guides "
          f"(sampled from CRISPGen 3M pool)")

    cfg = DecoderConfig()
    model = ARDecoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 50)
    uniform_ce = float(np.log(4))                          # random 20-mer baseline (nats/token)
    print(f"[model] ARDecoder params={model.n_params:,} | device={device} | "
          f"uniform-4 baseline CE={uniform_ce:.4f} (ppl {np.exp(uniform_ce):.2f})")

    hist, best = [], float("inf")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); run, nb = 0.0, 0
        for (b,) in tr_loader:
            b = b.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(b[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       b[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item(); nb += 1
        sched.step()
        tr_ce = run / nb
        va_ce = evaluate(model, va_loader, device, amp)
        hist.append({"epoch": ep, "train_ce": round(tr_ce, 5), "val_ce": round(va_ce, 5),
                     "val_ppl": round(float(np.exp(va_ce)), 4)})
        flag = ""
        if va_ce < best:
            best = va_ce
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "val_ce": va_ce, "seed": args.seed}, CKPT)
            flag = "  <- best (saved)"
        print(f"  ep {ep:02d}/{args.epochs}  train_CE={tr_ce:.4f}  "
              f"val_CE={va_ce:.4f}  val_ppl={np.exp(va_ce):.3f}{flag}")

    meta = {"n_train": len(tr_ids), "n_val": len(val_ids), "epochs": args.epochs,
            "best_val_ce": round(best, 5), "best_val_ppl": round(float(np.exp(best)), 4),
            "uniform4_baseline_ce": round(uniform_ce, 5), "params": model.n_params,
            "wall_seconds": round(time.time() - t0, 1), "history": hist,
            "checkpoint": str(CKPT.relative_to(PROJECT))}
    if amp:
        meta["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(meta, indent=2))
    print(f"[done] best val CE {best:.4f} (ppl {np.exp(best):.3f}) | "
          f"peak VRAM {meta.get('peak_vram_mb','-')} MB | {meta['wall_seconds']}s -> {CKPT.name}")


if __name__ == "__main__":
    main()
