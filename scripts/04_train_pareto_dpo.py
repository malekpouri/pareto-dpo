#!/usr/bin/env python3
"""
Phase 3 — Multi-objective Pareto-DPO training loop  (RFC-001 §2.3, §4).

Aligns the policy toward the 2-D Pareto frontier using the scalarization-free DPO loss
(models/pareto_dpo_loss.py) over dominance pairs (scripts/01), with the reference
log-probs read from the precomputed CSVs (scripts/02) so the reference model is NEVER
resident in VRAM (RFC §4.2). Policy is initialised from the SFT reference (Decision 3),
so at step 0 policy == reference (margin 0, loss = log 2) — a clean DPO start.

Memory: BF16 autocast, micro-batch 16 + gradient accumulation, tiny policy (~1.8M).
Outputs: models/checkpoints/pareto_dpo_best.pt   (best by val reward-accuracy)
         results/json/training_metrics.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
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
BEST = PROJECT / "models" / "checkpoints" / "pareto_dpo_best.pt"
METRICS = PROJECT / "results" / "json" / "training_metrics.json"


def load_split(name, tok):
    df = pd.read_csv(PAIRS / f"{name}_ref.csv")
    ids_w = tok.encode(df["y_w"].tolist())
    ids_l = tok.encode(df["y_l"].tolist())
    ref_w = torch.tensor(df["ref_logp_w"].to_numpy(), dtype=torch.float32)
    ref_l = torch.tensor(df["ref_logp_l"].to_numpy(), dtype=torch.float32)
    return ids_w, ids_l, ref_w, ref_l


def init_policy(device):
    state = torch.load(SFT, map_location=device, weights_only=False)
    cfg = DecoderConfig(**state["config"])
    model = ARDecoder(cfg).to(device)
    model.load_state_dict(state["model"])                 # init policy = SFT reference
    return model


@torch.no_grad()
def evaluate(model, split, device, beta, amp, mb=256):
    ids_w, ids_l, ref_w, ref_l = split
    model.eval()
    losses, margins, accs, kls, rvar = [], [], [], [], []
    for i in range(0, len(ids_w), mb):
        sl = slice(i, i + mb)
        iw, il = ids_w[sl].to(device), ids_l[sl].to(device)
        rw, rl = ref_w[sl].to(device), ref_l[sl].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pw = model.sequence_logprob(iw).float()
            pl = model.sequence_logprob(il).float()
        loss, st = pareto_dpo_loss(pw, pl, rw, rl, beta=beta)
        losses.append(loss.item() * len(iw)); margins.append(st.reward_margin.item() * len(iw))
        accs.append(st.reward_accuracy.item() * len(iw))
        kls.append(((pw - rw).mean().item()) * len(iw))
        rvar.append(torch.cat([beta * (pw - rw), beta * (pl - rl)]).var().item() * len(iw))
    n = len(ids_w)
    return {"loss": sum(losses) / n, "reward_margin": sum(margins) / n,
            "reward_acc": sum(accs) / n, "kl_to_ref": sum(kls) / n,
            "implicit_reward_var": sum(rvar) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation (eff batch = mb*accum)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.cuda.manual_seed_all(args.seed); torch.cuda.reset_peak_memory_stats()
    tok = NucleotideTokenizer()

    train = load_split("train", tok)
    val = load_split("val", tok)
    ood = load_split("ood", tok)
    n_tr = len(train[0])
    print(f"[data] train {n_tr} | val {len(val[0])} | ood {len(ood[0])} dominance pairs "
          f"| eff batch = {args.micro_batch * args.accum}")

    model = init_policy(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = int(np.ceil(n_tr / args.micro_batch / args.accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch, eta_min=args.lr / 20)
    print(f"[model] policy={model.n_params:,} params (init=SFT ref) | beta={args.beta} | device={device}")

    e0 = evaluate(model, val, device, args.beta, amp)
    print(f"[init]  val loss={e0['loss']:.4f}  reward_acc={e0['reward_acc']:.3f}  "
          f"(policy==ref -> ~log2={np.log(2):.3f}, acc~0.5)")

    hist, best_acc = [], -1.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(args.seed + ep))
        run_loss = run_acc = 0.0; nb = 0
        opt.zero_grad()
        micro = 0
        for i in range(0, n_tr, args.micro_batch):
            idx = perm[i:i + args.micro_batch]
            iw, il = train[0][idx].to(device), train[1][idx].to(device)
            rw, rl = train[2][idx].to(device), train[3][idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pw = model.sequence_logprob(iw).float()
                pl = model.sequence_logprob(il).float()
            loss, st = pareto_dpo_loss(pw, pl, rw, rl, beta=args.beta)
            (loss / args.accum).backward()
            run_loss += loss.item(); run_acc += st.reward_accuracy.item(); nb += 1
            micro += 1
            if micro % args.accum == 0 or i + args.micro_batch >= n_tr:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); sched.step()
        tr = {"loss": run_loss / nb, "reward_acc": run_acc / nb}
        ev = evaluate(model, val, device, args.beta, amp)
        row = {"epoch": ep, "train_loss": round(tr["loss"], 5),
               "train_reward_acc": round(tr["reward_acc"], 4),
               "val_loss": round(ev["loss"], 5), "val_reward_acc": round(ev["reward_acc"], 4),
               "val_reward_margin": round(ev["reward_margin"], 4),
               "val_kl_to_ref": round(ev["kl_to_ref"], 4),
               "val_implicit_reward_var": round(ev["implicit_reward_var"], 6)}
        hist.append(row)
        flag = ""
        if ev["reward_acc"] > best_acc:
            best_acc = ev["reward_acc"]
            BEST.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "config": model.cfg.__dict__, "beta": args.beta,
                        "val_reward_acc": ev["reward_acc"], "seed": args.seed}, BEST)
            flag = "  <- best (saved)"
        print(f"  ep {ep:02d}/{args.epochs}  train_loss={tr['loss']:.4f} "
              f"acc={tr['reward_acc']:.3f} | val_loss={ev['loss']:.4f} "
              f"val_acc={ev['reward_acc']:.3f} margin={ev['reward_margin']:.3f} "
              f"KL={ev['kl_to_ref']:.2f}{flag}")

    ood_eval = evaluate(model, ood, device, args.beta, amp)
    meta = {"beta": args.beta, "micro_batch": args.micro_batch, "accum": args.accum,
            "effective_batch": args.micro_batch * args.accum, "epochs": args.epochs,
            "lr": args.lr, "seed": args.seed, "n_train_pairs": n_tr,
            "best_val_reward_acc": round(best_acc, 4),
            "init_val": {k: round(v, 5) for k, v in e0.items()},
            "final_ood": {k: round(v, 5) for k, v in ood_eval.items()},
            "wall_seconds": round(time.time() - t0, 1),
            "history": hist, "checkpoint": str(BEST.relative_to(PROJECT))}
    if amp:
        meta["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(meta, indent=2))
    print(f"[ood]   held-out chr22: loss={ood_eval['loss']:.4f} "
          f"reward_acc={ood_eval['reward_acc']:.3f} margin={ood_eval['reward_margin']:.3f}")
    print(f"[done] best val reward-acc {best_acc:.3f} | peak VRAM "
          f"{meta.get('peak_vram_mb','-')} MB | {meta['wall_seconds']}s -> {BEST.name}")


if __name__ == "__main__":
    main()
