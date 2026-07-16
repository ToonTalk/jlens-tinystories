"""Step 3: run the 12-prompt eval set for each fitted lens plus the logit-lens
baseline. Teacher-forced, all fitted layers, all positions.

Per fit id (A, B, C) writes out/eval/<id>/readouts.json and slice-page HTMLs;
also writes out/eval/divergence.json aggregated across prompts.

Usage: python src/evaluate.py [A B C]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from common import (
    EVAL_PROMPTS,
    FITS,
    OUT,
    load_model,
    piece_str,
    set_seeds,
    word_start_alpha_ids,
    write_json,
)

TOP_RAW = 20  # raw top-k kept for the appendix
TOP_FILTERED = 8  # filtered top-k in the main tables
OVERLAP_K = 10  # k for the top-k-overlap divergence metric

# Tokens whose full-vocab rank is tracked at every (position, layer), both
# lenses: the Lily check (" L", "ily") plus Max analogues.
TRACK_PIECES = ["▁L", "ily", "▁Lily", "▁M", "ax", "▁Max"]


def best_lens_file(fit_id: str) -> Path:
    d = OUT / "lenses" / fit_id
    for n in (1000, 100):
        p = d / f"lens_{n}.pt"
        if p.exists():
            return p
    raise FileNotFoundError(f"no lens for fit {fit_id} in {d}")


def full_ranks(logits: torch.Tensor) -> torch.Tensor:
    """[P, V] logits -> [P, V] float ranks (0 = top)."""
    order = logits.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    arange = torch.arange(logits.shape[-1]).expand_as(order)
    ranks.scatter_(1, order, arange)
    return ranks.float()


def spearman(ranks_a: torch.Tensor, ranks_b: torch.Tensor) -> torch.Tensor:
    """Spearman rho per row from two [P, V] rank tensors."""
    v = ranks_a.shape[-1]
    d2 = ((ranks_a - ranks_b) ** 2).sum(dim=-1)
    return 1.0 - 6.0 * d2 / (v * (v * v - 1.0))


def topk_lists(logits_1d, tokenizer, filter_ids, k_raw, k_filt):
    raw_idx = logits_1d.topk(k_raw).indices.tolist()
    raw = [piece_str(tokenizer, t) for t in raw_idx]
    order = logits_1d.argsort(descending=True).tolist()
    filt = []
    for t in order:
        if t in filter_ids:
            filt.append(piece_str(tokenizer, t))
            if len(filt) >= k_filt:
                break
    return raw, filt


def evaluate_fit(fit_id: str, model_cache: dict) -> dict:
    from jlens import JacobianLens
    from jlens.vis import build_page, compute_slice

    model_key, corpus_key = FITS[fit_id]
    if model_key not in model_cache:
        model_cache[model_key] = load_model(model_key)
    lens_model, hf, tok, info = model_cache[model_key]

    lens_file = best_lens_file(fit_id)
    lens = JacobianLens.load(str(lens_file))
    vocab_size = hf.config.vocab_size
    filter_ids = word_start_alpha_ids(tok, vocab_size)

    tracked = {}
    for piece in TRACK_PIECES:
        tid = tok.convert_tokens_to_ids(piece)
        if tid is not None and tid != tok.unk_token_id:
            tracked[piece.replace("▁", " ")] = tid

    eval_dir = OUT / "eval" / fit_id
    slice_dir = eval_dir / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "fit_id": fit_id,
        "lens_file": lens_file.name,
        "model_key": model_key,
        "fit_corpus": corpus_key,
        "n_layers": lens_model.n_layers,
        "source_layers": lens.source_layers,
        "tracked_token_ids": tracked,
        "prompts": [],
    }
    # divergence accumulators: layer -> list over (prompt, position)
    overlap_acc = {l: [] for l in lens.source_layers}
    rho_acc = {l: [] for l in lens.source_layers}

    for pi, prompt in enumerate(EVAL_PROMPTS):
        t0 = time.time()
        jl, model_logits, input_ids = lens.apply(lens_model, prompt)
        ll, _, _ = lens.apply(lens_model, prompt, use_jacobian=False)
        ids = input_ids[0].tolist()
        n_pos = len(ids)
        ctx = [piece_str(tok, t) for t in ids]

        pdata = {
            "prompt": prompt,
            "n_tokens": n_pos,
            "context_tokens": ctx,
            "model_top5_last": [
                piece_str(tok, t)
                for t in model_logits[-1].topk(5).indices.tolist()
            ],
            "layers": {},
            "tracked_ranks": {},  # name -> {"jlens"/"logit": [n_pos][n_layers]}
        }

        model_ranks = None  # per-layer loop computes lens ranks; model not needed
        for layer in lens.source_layers:
            j_logits, l_logits = jl[layer], ll[layer]
            j_raw, j_filt = topk_lists(
                j_logits[-1], tok, filter_ids, TOP_RAW, TOP_FILTERED
            )
            l_raw, l_filt = topk_lists(
                l_logits[-1], tok, filter_ids, TOP_RAW, TOP_FILTERED
            )
            pdata["layers"][str(layer)] = {
                "jlens_raw_top20_last": j_raw,
                "logit_raw_top20_last": l_raw,
                "jlens_filtered_top8_last": j_filt,
                "logit_filtered_top8_last": l_filt,
            }

            # divergence over positions >= 1 (skip BOS)
            jr = full_ranks(j_logits[1:])
            lr = full_ranks(l_logits[1:])
            rho_acc[layer].extend(spearman(jr, lr).tolist())
            j_top = j_logits[1:].topk(OVERLAP_K).indices
            l_top = l_logits[1:].topk(OVERLAP_K).indices
            for p in range(j_top.shape[0]):
                inter = len(set(j_top[p].tolist()) & set(l_top[p].tolist()))
                overlap_acc[layer].append(inter / OVERLAP_K)

        # tracked-token ranks at every (position, layer), both lenses
        for name, tid in tracked.items():
            entry = {"jlens": [], "logit": []}
            for kind, source in (("jlens", jl), ("logit", ll)):
                per_layer = []
                for layer in lens.source_layers:
                    lg = source[layer]  # [n_pos, V]
                    ranks = (lg > lg[:, tid : tid + 1]).sum(dim=-1)
                    per_layer.append(ranks.tolist())
                # transpose to [n_pos][n_layers]
                entry[kind] = [list(col) for col in zip(*per_layer)]
            pdata["tracked_ranks"][name] = entry

        result["prompts"].append(pdata)

        # slice page (repo visualisation, mask_display like the paper pages)
        pinned = (
            {tracked[" L"], tracked["ily"]}
            if pi in (0, 7) and " L" in tracked and "ily" in tracked
            else set(tracked.values())
        )
        slice_data = compute_slice(
            lens_model, lens, prompt, mask_display=True, pinned_token_ids=pinned
        )
        page, _, _ = build_page(
            slice_data,
            prompt,
            title=f"fit {fit_id} ({model_key}, {corpus_key}) - prompt {pi + 1}",
            description=f"J-lens slice; lens file {lens_file.name}",
            mode="embed",
        )
        (slice_dir / f"prompt{pi + 1:02d}.html").write_text(page, encoding="utf-8")
        print(f"[eval {fit_id}] prompt {pi + 1}/12 done in {time.time() - t0:.1f}s")

    result["divergence"] = {
        str(l): {
            "top10_overlap_mean": sum(overlap_acc[l]) / len(overlap_acc[l]),
            "spearman_mean": sum(rho_acc[l]) / len(rho_acc[l]),
            "n_samples": len(overlap_acc[l]),
        }
        for l in lens.source_layers
    }
    write_json(eval_dir / "readouts.json", result)
    return {fit_id: result["divergence"]}


def main() -> None:
    set_seeds()
    fit_ids = sys.argv[1:] or ["A", "B", "C"]
    model_cache: dict = {}
    divergence = {}
    for fit_id in fit_ids:
        divergence.update(evaluate_fit(fit_id, model_cache))
    # merge with any existing divergence file (partial runs)
    div_path = OUT / "eval" / "divergence.json"
    if div_path.exists():
        import json

        old = json.loads(div_path.read_text(encoding="utf-8"))
        old.update(divergence)
        divergence = old
    write_json(div_path, divergence)
    print(f"Wrote {div_path}")


if __name__ == "__main__":
    main()
