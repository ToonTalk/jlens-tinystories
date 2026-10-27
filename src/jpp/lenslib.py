"""Readout plumbing shared by the J++ fitting/eval scripts.

A lens here is ``{layer: J_l}`` (``None`` = the logit lens). Readout follows
jlens.apply exactly: residual at the layer OUTPUT -> J_l @ h -> final RMSNorm
-> LM head. Readout Filtering (J++) is an exclusion mask applied to ranks.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from jpp_common import (
    APP_BANDS,
    ARTIFACTS,
    JPP_OUT,
    load_existing_J,
)


@torch.no_grad()
def record(lens_model, prompt: str, layers, max_len: int = 512):
    """One forward pass; (input_ids [S], {layer: [S, d] fp32 on device})."""
    from jlens.hooks import ActivationRecorder

    ids = lens_model.encode(prompt, max_length=max_len)
    with ActivationRecorder(lens_model.layers, at=list(layers)) as rec:
        lens_model.forward(ids)
        acts = {l: rec.activations[l][0].detach().float() for l in layers}
    return ids[0], acts


@torch.no_grad()
def lens_logits(lens_model, J: torch.Tensor | None, h: torch.Tensor) -> torch.Tensor:
    """[P, d] residuals -> [P, V] fp32 lens logits (J None: logit lens)."""
    x = h if J is None else h @ J.to(h.device).T
    return lens_model.unembed(x).float()


def ranks_of(logits_PV: torch.Tensor, cand: list[int], excluded: torch.Tensor | None) -> torch.Tensor:
    """0-based rank of the best-scoring candidate per row (paper rule:
    #(logits > best candidate)); excluded ids are pushed to -inf first."""
    if excluded is not None:
        logits_PV = logits_PV.masked_fill(excluded[None, :], float("-inf"))
    best = logits_PV[:, cand].amax(dim=1, keepdim=True)
    return (logits_PV > best).sum(dim=1)


def exclusion_mask(non_semantic: set[int], spare: set[int], vocab: int, device) -> torch.Tensor:
    m = torch.zeros(vocab, dtype=torch.bool)
    ids = sorted(non_semantic - spare)
    m[ids] = True
    return m.to(device)


def load_dev(model_key: str, split: str | None = None) -> list[dict]:
    """Dev items the model gets right (verify_devset.py), joined with their
    prompts; ``split`` filters to "fit" or "held-out"."""
    items = {it["name"]: it for it in json.loads((JPP_OUT / "devset" / "items.json").read_text(encoding="utf-8"))}
    rows = json.loads((JPP_OUT / "devset" / f"correct_{model_key}.json").read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if not r["correct"] or (split and r["split"] != split):
            continue
        out.append({**items[r["name"]], **r})
    return out


def _load_pt(path: Path) -> dict[int, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {int(l): J.float() for l, J in state["J"].items()}


def lens_variants(model_key: str) -> dict[str, dict[int, torch.Tensor] | None]:
    """Every lens variant whose files exist, by report name."""
    d = JPP_OUT / model_key
    v: dict[str, dict[int, torch.Tensor] | None] = {"logit": None}
    v["J"] = load_existing_J(model_key, 1000)
    v["J n100"] = load_existing_J(model_key, 100)
    files = {
        "LRP-only": "pooled_rlens_skip16_n1000.pt",
        "LRP-only n100": "pooled_rlens_skip16_n100.pt",
        "pooled-noLRP": "pooled_none_skip16_n1000.pt",
        "J++": "jpp_rlens_skip16_n1000.pt",
        "J++ n100": "jpp_rlens_skip16_n100.pt",
        "J++ mixed": "jpp_rlens_skip16_n1000_mixed.pt",
        "J++ mixed n100": "jpp_rlens_skip16_n100_mixed.pt",
        "experts-noLRP": "jpp_none_skip16_n1000.pt",
        "experts-noLRP n100": "jpp_none_skip16_n100.pt",  # 110M ablation runs at n=100 only
        "J skip1": "jskip1_n1000.pt",
        "J skip1 n100": "jskip1_n100.pt",  # 110M: compare with "J n100"
    }
    for name, fn in files.items():
        if (d / fn).exists():
            v[name] = _load_pt(d / fn)
    return v


def band(model_key: str) -> list[int]:
    return APP_BANDS[model_key]


def experts_path(model_key: str, run: str, n: int) -> Path:
    return ARTIFACTS / model_key / f"experts_{run}_n{n}.pt"
