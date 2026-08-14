#!/usr/bin/env python3
"""
Small autoregressive nucleotide decoder — the likelihood-based policy / reference for
Pareto-DPO (RFC-001 §4.1).

Why autoregressive: DPO needs a tractable sequence likelihood log pi(y) = sum_t
log pi(y_t | y_<t). CRISPGen's diffusion base does not expose this cheaply, so the
policy is a compact AR decoder SFT-pretrained on the CRISPGen pool (RFC Decision 3).

Design: GPT-style causal Transformer over a 7-token nucleotide vocabulary, sized for
the 8 GB budget (default ~4-8 M params; configurable up to ~30 M). Deliberately tiny —
sequences are only 20 nt, so capacity, not memory, is the ceiling.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed vocabulary (do not reorder — checkpoints depend on it)
PAD, BOS, EOS = 0, 1, 2
NUC2ID = {"A": 3, "C": 4, "G": 5, "T": 6}
ID2NUC = {v: k for k, v in NUC2ID.items()}
VOCAB = 7


class NucleotideTokenizer:
    """Maps 20-nt strings <-> padded id tensors with BOS/EOS."""

    def encode(self, seqs: list[str], device=None) -> torch.Tensor:
        rows = [[BOS] + [NUC2ID[c] for c in s.strip().upper()] + [EOS] for s in seqs]
        L = max(len(r) for r in rows)
        out = torch.full((len(rows), L), PAD, dtype=torch.long)
        for i, r in enumerate(rows):
            out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return out.to(device) if device is not None else out


@dataclass
class DecoderConfig:
    n_layer: int = 4
    d_model: int = 192
    n_head: int = 6
    d_ff: int = 768
    max_len: int = 24          # BOS + 20 + EOS + slack
    dropout: float = 0.1


class ARDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig = DecoderConfig()):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(VOCAB, cfg.d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_head, cfg.d_ff, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layer)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB, bias=False)
        self.head.weight = self.tok_emb.weight            # weight tying

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids (B,T) -> logits (B,T,VOCAB) with causal masking."""
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.tok_emb(ids) + self.pos_emb(pos)[None]
        causal = torch.triu(torch.ones(T, T, device=ids.device, dtype=torch.bool), 1)
        pad_mask = ids.eq(PAD)
        h = self.blocks(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.head(self.ln_f(h))

    def sequence_logprob(self, ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced log pi(y) = sum_t log p(y_t | y_<t), summed over non-PAD targets.
        Returns (B,) log-probabilities. EOS is scored; BOS is only a conditioning input."""
        logits = self.forward(ids[:, :-1])                # predict positions 1..T-1
        targets = ids[:, 1:]
        logp = F.log_softmax(logits, dim=-1)
        tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (B,T-1)
        mask = targets.ne(PAD).float()
        return (tok_lp * mask).sum(dim=1)

    @torch.no_grad()
    def sample(self, n: int, device, temperature: float = 1.0,
               seed: int | None = None) -> list[str]:
        """Autoregressively sample `n` 20-nt sequences (nucleotides only)."""
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        ids = torch.full((n, 1), BOS, device=device, dtype=torch.long)
        nuc_ids = torch.tensor([NUC2ID[c] for c in "ACGT"], device=device)  # [3,4,5,6]
        for _ in range(20):
            logits = self.forward(ids)[:, -1, :][:, nuc_ids] / max(temperature, 1e-6)
            nxt = nuc_ids[torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)]
            ids = torch.cat([ids, nxt[:, None]], dim=1)
        return ["".join(ID2NUC[i] for i in row[1:].tolist()) for row in ids]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":  # smoke test
    tok = NucleotideTokenizer()
    m = ARDecoder()
    ids = tok.encode(["GTAGACGAGTAGGGTAAGAG", "AAGGGGGGTGAAAGTTTGCA"])
    lp = m.sequence_logprob(ids)
    print(f"params={m.n_params:,} | ids{tuple(ids.shape)} | seq_logp={lp.tolist()}")
    assert lp.shape == (2,) and torch.isfinite(lp).all()
    print("policy_decoder OK")
