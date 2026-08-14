#!/usr/bin/env python3
"""
Phase-4 EXTERNAL evaluation oracles for scoring GENERATED sequence pools.

IMPORTANT (integrity): the real CRISPRon software is NOT installed in this environment
(only its input-prep script exists). We therefore score on-target efficacy with a
**CRISPRon SURROGATE** — a small CNN regressor trained on 300,000 real (sequence,
CRISPRon) labels — and report its held-out fidelity so its status as a proxy is
explicit. Off-target is scored with a REAL (if partial-genome) Cas-OFFinder-style
structural screen against chromosome 22 extracted from GRCh38.

Nothing here fabricates scores: efficacy = a validated learned oracle (fidelity
reported); off-target = real near-match counts against real genomic sites.
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT / "models" / "checkpoints"
CACHE = PROJECT / "data" / "raw"
NUC = {"A": 0, "C": 1, "G": 2, "T": 3}


def _onehot(seqs: list[str]) -> torch.Tensor:
    x = torch.zeros(len(seqs), 4, 20)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            x[i, NUC[c], j] = 1.0
    return x


# ─────────────────────────────────────────────────────────────────────────────
# CRISPRon SURROGATE  (efficacy oracle)
# ─────────────────────────────────────────────────────────────────────────────
class CRISPRonSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_crispron_surrogate(device, retrain=False, epochs=6, seed=42):
    """Return (predict_fn, fidelity_dict). Trains once on 300k CRISPRon labels and caches."""
    from scipy import stats
    import pandas as pd
    ckpt = CKPT_DIR / "crispron_surrogate.pt"
    model = CRISPRonSurrogate().to(device)
    if ckpt.exists() and not retrain:
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        fidelity = st["fidelity"]
    else:
        df = pd.read_csv(NOTEBOOK / "report" / "crispgen_mass_sample_300k_results.csv")
        df = df[df["Sequence"].str.len().eq(20)].reset_index(drop=True)
        y = df["CRISPRon_Score"].to_numpy(np.float32)
        torch.manual_seed(seed); np.random.seed(seed)
        perm = np.random.permutation(len(df)); n_val = 20000
        val_i, tr_i = perm[:n_val], perm[n_val:]
        Xtr = _onehot(df["Sequence"].iloc[tr_i].tolist())
        ytr = torch.tensor(y[tr_i])
        Xva = _onehot(df["Sequence"].iloc[val_i].tolist()).to(device)
        yva = y[val_i]
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        bs = 4096
        for ep in range(epochs):
            model.train(); order = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), bs):
                idx = order[i:i + bs]
                xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
                loss = F.mse_loss(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva).cpu().numpy()
        fidelity = {"heldout_pearson": float(stats.pearsonr(pred, yva)[0]),
                    "heldout_spearman": float(stats.spearmanr(pred, yva)[0]),
                    "heldout_rmse": float(np.sqrt(np.mean((pred - yva) ** 2))),
                    "n_train": len(tr_i), "n_val": n_val}
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "fidelity": fidelity}, ckpt)

    model.eval()

    @torch.no_grad()
    def predict(seqs: list[str]) -> np.ndarray:
        out = np.empty(len(seqs), np.float32)
        for i in range(0, len(seqs), 8192):
            xb = _onehot(seqs[i:i + 8192]).to(device)
            out[i:i + xb.shape[0]] = model(xb).cpu().numpy()
        return out

    return predict, fidelity


# ─────────────────────────────────────────────────────────────────────────────
# chr22 STRUCTURAL OFF-TARGET  (real, partial-genome Cas-OFFinder-style)
# ─────────────────────────────────────────────────────────────────────────────
def _read_chr22() -> str:
    fa = NOTEBOOK.parent / "hg38.fa.gz"
    seq, capture = [], False
    with gzip.open(fa, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if capture:
                    break
                capture = (line[1:].split()[0] == "chr22")
            elif capture:
                seq.append(line.strip().upper())
    return "".join(seq)


def get_chr22_sites(n_sample: int = 500_000, seed: int = 0) -> np.ndarray:
    """All forward-strand NGG-adjacent 20-mers on chr22 (sampled). Cached as uint8 (M,20)."""
    cache = CACHE / f"chr22_ngg_sites_{n_sample}.npy"
    if cache.exists():
        return np.load(cache)
    s = _read_chr22()
    arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    code = np.full(256, 255, np.uint8)
    for c, v in NUC.items():
        code[ord(c)] = v
    coded = code[arr]
    G = NUC["G"]
    # protospacer at i..i+19, PAM NGG at i+20..i+22 -> positions i+21,i+22 == G
    n = len(coded)
    starts = np.arange(0, n - 22)
    is_ngg = (coded[starts + 21] == G) & (coded[starts + 22] == G)
    valid_pam = starts[is_ngg]
    # keep only sites whose 20-mer has no ambiguous base (code<4)
    idx = valid_pam[:, None] + np.arange(20)[None, :]
    sites = coded[idx]
    good = (sites < 4).all(axis=1)
    sites = sites[good].astype(np.uint8)
    if n_sample and n_sample < len(sites):
        rng = np.random.default_rng(seed)
        sites = sites[rng.choice(len(sites), n_sample, replace=False)]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, sites)
    return sites


@torch.no_grad()
def offtarget_counts(seqs: list[str], sites: np.ndarray, device, max_mm: int = 3,
                     site_block: int = 20_000, guide_chunk: int = 128) -> np.ndarray:
    """Near-match count (Hamming <= max_mm) of each guide against chr22 sites.
    Memory-safe: chunks BOTH guides and sites so no intermediate exceeds ~50 MB."""
    g_all = torch.tensor([[NUC[c] for c in s] for s in seqs], dtype=torch.uint8)  # (G,20)
    S = torch.from_numpy(sites)
    counts = np.zeros(len(seqs), dtype=np.int64)
    for gi in range(0, len(seqs), guide_chunk):
        g = g_all[gi:gi + guide_chunk].to(device)            # (Gc,20)
        acc = torch.zeros(g.shape[0], dtype=torch.int64, device=device)
        for si in range(0, len(sites), site_block):
            sb = S[si:si + site_block].to(device)            # (Sb,20)
            hd = (g[:, None, :] != sb[None, :, :]).sum(dim=2)  # (Gc,Sb) small
            acc += (hd <= max_mm).sum(dim=1)
            del sb, hd
        counts[gi:gi + g.shape[0]] = acc.cpu().numpy()
        del g, acc
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return counts
