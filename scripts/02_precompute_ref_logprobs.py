#!/usr/bin/env python3
"""
Phase 2 — offline reference log-prob precompute  (RFC-001 §4.2, the 8 GB memory trick).

DPO needs log pi_ref(y_w) and log pi_ref(y_l) for every preference pair. Because the
reference policy is FROZEN, we compute these ONCE, here, and store them. During DPO
training the reference model is then never loaded into VRAM — only these cached scalars
are read. This removes the classic DPO memory doubling and is exact (no approximation).

Inputs : data/preference_pairs/{train,val,ood}.csv  (from 01_build_preference_pairs.py)
Model  : a frozen ARDecoder reference checkpoint (--ckpt); with --self-test a fresh
         (untrained) decoder is used to validate the pipeline + report the memory
         footprint end-to-end.
Outputs: data/preference_pairs/{split}_ref.csv  (pairs + ref_logp_w, ref_logp_l)
         data/preference_pairs/ref_logprob_meta.json  (model, timing, peak memory)
"""
from __future__ import annotations
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from models.policy_decoder import ARDecoder, DecoderConfig, NucleotideTokenizer  # noqa: E402

PAIRS = PROJECT / "data" / "preference_pairs"


@torch.no_grad()
def logp_for_column(model, tok, seqs, batch, device, use_bf16) -> np.ndarray:
    out = np.empty(len(seqs), dtype=np.float64)
    model.eval()
    amp = (device.type == "cuda" and use_bf16)
    for i in range(0, len(seqs), batch):
        chunk = list(seqs[i:i + batch])
        ids = tok.encode(chunk, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lp = model.sequence_logprob(ids)
        out[i:i + len(chunk)] = lp.float().cpu().numpy()
    return out


def load_reference(ckpt: str | None, device) -> ARDecoder:
    if ckpt:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = DecoderConfig(**state["config"]) if isinstance(state, dict) and "config" in state \
            else DecoderConfig()
        model = ARDecoder(cfg).to(device)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(sd)
        print(f"[ref] loaded frozen reference from {ckpt}")
    else:
        torch.manual_seed(0)
        model = ARDecoder().to(device)
        print("[ref] --self-test: using a FRESH (untrained) decoder to validate mechanics")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="frozen reference decoder checkpoint")
    ap.add_argument("--self-test", action="store_true",
                    help="run with a fresh untrained decoder (validate pipeline + memory)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "ood"])
    args = ap.parse_args()

    if not args.ckpt and not args.self_test:
        ap.error("provide --ckpt <reference> or --self-test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok = NucleotideTokenizer()
    model = load_reference(args.ckpt, device)
    print(f"[ref] device={device} params={model.n_params:,} bf16={args.bf16 and device.type=='cuda'}")

    meta = {"model_params": model.n_params, "device": str(device),
            "checkpoint": args.ckpt or "SELF_TEST_UNTRAINED",
            "batch": args.batch, "bf16": bool(args.bf16 and device.type == "cuda"),
            "splits": {}}
    t0 = time.time()
    for split in args.splits:
        path = PAIRS / f"{split}.csv"
        if not path.exists():
            print(f"[skip] {split}: {path} not found")
            continue
        df = pd.read_csv(path)
        lp_w = logp_for_column(model, tok, df["y_w"].tolist(), args.batch, device, args.bf16)
        lp_l = logp_for_column(model, tok, df["y_l"].tolist(), args.batch, device, args.bf16)
        df["ref_logp_w"] = lp_w
        df["ref_logp_l"] = lp_l
        out = PAIRS / f"{split}_ref.csv"
        df.to_csv(out, index=False)
        meta["splits"][split] = {"n_pairs": int(len(df)),
                                 "ref_logp_w_mean": float(lp_w.mean()),
                                 "ref_logp_l_mean": float(lp_l.mean()),
                                 "out": str(out.relative_to(PROJECT))}
        print(f"[{split}] {len(df):>6} pairs | ref_logp_w mean {lp_w.mean():.3f} "
              f"| ref_logp_l mean {lp_l.mean():.3f} -> {out.name}")

    peak_cpu_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    meta["wall_seconds"] = round(time.time() - t0, 2)
    meta["peak_cpu_rss_mb"] = round(peak_cpu_mb, 1)
    if device.type == "cuda":
        meta["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        print(f"[mem] peak VRAM {meta['peak_vram_mb']} MB "
              f"(budget 8192 MB) | peak CPU RSS {meta['peak_cpu_rss_mb']} MB")
    else:
        print(f"[mem] CPU run | peak CPU RSS {meta['peak_cpu_rss_mb']} MB")
    (PAIRS / "ref_logprob_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {meta['wall_seconds']}s | wrote *_ref.csv + ref_logprob_meta.json")


if __name__ == "__main__":
    main()
