"""J++ step 2e: learn the expert weights on the TinyStories dev set's fit split.

Usage: python src/jpp/fit_weights.py <stories110M|stories15M> <run> <n>
  run = rlens_skip16 (J++) or none_skip16 (ablation: expert weighting
        without LRP); n = 1000 or 100 (the n=100 stability fit).

The J++ recipe's stage 4 (scripts/jpp_cli.py fit-weights) on our items: per
layer, cache every expert's logits at each fit item's readout residual, then
Adam (lr 0.05, 200 steps, from 3 starts: uniform / best single expert / top-3
experts) on the smooth surrogate of that layer's macro recall@10 over the dev
families, keeping the iterate with the best exact macro. Readout Filtering
(non-semantic ids out of the ranking) is on, as in the recipe. The combined
lens is J_l = sum_e w_e J_{l,e}; weights are unit-L2 and may be negative.

Writes out/jpp/<model>/jpp_<run>_n<n>.pt (fp32, all source layers) and
..._weights.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import JPP_OUT, load_model, non_semantic_ids, save_jlens_pt  # noqa: E402
from lenslib import exclusion_mask, experts_path, load_dev, record  # noqa: E402


def readouts(items: list[dict], positions: str) -> list[dict]:
    """One readout per item at the recipe position (the final period), or with
    positions="mixed" also at the sentence's last word where valid
    (add_word_positions.py), so the weights see non-period residuals too."""
    out = [{**it, "pos": it["readout_pos"]} for it in items]
    if positions == "mixed":
        out += [{**it, "pos": it["word_pos"]} for it in items if it.get("word_ok")]
    return out


def main() -> None:
    model_key, run, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    positions = sys.argv[4] if len(sys.argv) > 4 else "period"
    assert positions in ("period", "mixed")
    from workspace_lens.fitting.condense_experts import CondenseConfig, ExpertJacobians
    from workspace_lens.fitting.condense_experts.fitter import (
        build_search_starting_points,
        fit_expert_weights,
    )
    from workspace_lens.fitting.condense_experts.rank_objective import (
        build_pair_table,
        expert_logits,
        expert_pair_pass_scores,
    )

    t0 = time.time()
    torch.manual_seed(0)
    lens_model, hf, tok = load_model(model_key)
    experts = ExpertJacobians.load(str(experts_path(model_key, run, n)))
    layers = experts.layers
    items = readouts(load_dev(model_key, split="fit"), positions)
    assert items, "no fit items"
    device = lens_model.input_device
    cands = [it["candidate_ids"] for it in items]
    spare = {c for cs in cands for c in cs}
    mask = exclusion_mask(non_semantic_ids(tok, hf.config.vocab_size), spare, hf.config.vocab_size, device)

    resid = {l: [] for l in layers}
    for it in items:
        _, acts = record(lens_model, it["prompt"], layers)
        for l in layers:
            resid[l].append(acts[l][it["pos"]])
    resid = {l: torch.stack(v) for l, v in resid.items()}

    pairs = build_pair_table(
        [{"answer": c} for c in cands], [it["family"] for it in items], list(range(len(items))), device=device
    )
    cfg = CondenseConfig()
    combined, report = {}, {"config": cfg.__dict__, "positions": positions, "n_fit_readouts": len(items),
                            "n_fit_items": len({it["name"] for it in items}),
                            "fit_families": sorted({it["family"] for it in items}), "by_layer": {}}
    for l in layers:
        E = experts.experts_L_dict_EFN[l].to(device=device, dtype=torch.float32)
        transports = torch.einsum("pj,eij->pei", resid[l], E)
        logits = expert_logits(lens_model.unembed, transports)
        scores = expert_pair_pass_scores(logits, pairs, excluded_mask_Bool_V=mask, k=cfg.pass_k)
        learned = fit_expert_weights(
            logits, pairs, excluded_mask_Bool_V=mask,
            initial_weightings=build_search_starting_points(scores, top_n=cfg.top_n),
            condense_config=cfg,
        )
        combined[l] = torch.einsum("e,eij->ij", learned.weights_E.to(E), E).cpu()
        report["by_layer"][str(l)] = {
            "weights_unit_l2": [round(x, 4) for x in learned.weights_E.tolist()],
            "signed_shares": [round(x, 4) for x in learned.signed_shares_E().tolist()],
            "single_expert_recall10": [round(x, 3) for x in scores.tolist()],
            "train_macro_recall10": round(learned.train_macro_pass, 3),
            "init": learned.init_name,
            "step": learned.step,
            "position_counts": experts.position_counts_L_dict_E[l].tolist(),
        }
        print(f"[{model_key} {run} n{n} {positions}] L{l}: single {report['by_layer'][str(l)]['single_expert_recall10']} "
              f"-> {learned.train_macro_pass:.3f} ({learned.init_name}@{learned.step}) "
              f"w={report['by_layer'][str(l)]['signed_shares']}")
        del logits, transports
    report["wall_s"] = round(time.time() - t0, 1)
    out = JPP_OUT / model_key / f"jpp_{run}_n{n}{'_mixed' if positions == 'mixed' else ''}.pt"
    save_jlens_pt(combined, out, n, {"lens": f"J++ ({run}) expert-weighted", "weights": str(out.with_name(out.stem + "_weights.json").name)})
    out.with_name(out.stem + "_weights.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"-> {out} ({report['wall_s']}s)")


if __name__ == "__main__":
    main()
