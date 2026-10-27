"""How far apart J++'s eight LRP experts are, and how the learned weights mix them.

Per layer: cosine of each expert Jacobian to the pooled one (= LRP-only) and
its relative distance ||E - P|| / ||P||; the learned weights' negative share
(signed shares, sum |w| = 1); and the cosine of the resulting J++ map to P.
The J++ map is sum_e w_e E_e, so its part along P is sum_e w_e <E_e, P> / |P|^2;
when the signed weights drive that near 0 or below, the shared part cancels
and only the differences between experts are left (cos(J++, P) <= ~0).

Usage: python src/jpp/expert_spread.py   (writes out/jpp/<model>/expert_spread.json)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import JPP_OUT  # noqa: E402
from lenslib import experts_path, lens_variants  # noqa: E402

RUN, N = "rlens_skip16", 1000


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def main() -> None:
    for model_key in ("stories15M", "stories110M"):
        state = torch.load(experts_path(model_key, RUN, N), map_location="cpu", weights_only=True)
        jpp = lens_variants(model_key)["J++"]
        weights = json.loads((JPP_OUT / model_key / f"jpp_{RUN}_n{N}_weights.json").read_text(encoding="utf-8"))
        rows = {}
        for layer in sorted(state["experts"]):
            E = state["experts"][layer].float()
            P = state["pooled"][layer].float()
            cosines = [cos(e, P) for e in E]
            dists = [float((e - P).norm() / P.norm()) for e in E]
            shares = weights["by_layer"][str(layer)]["signed_shares"]
            rows[layer] = {
                "expert_cos_min": min(cosines),
                "expert_cos_median": statistics.median(cosines),
                "expert_rel_dist_median": statistics.median(dists),
                "weight_negative_share": -sum(s for s in shares if s < 0),
                "jpp_cos_pooled": cos(jpp[layer], P),
            }
            print(model_key, f"L{layer}", {k: round(v, 2) for k, v in rows[layer].items()})
        out = JPP_OUT / model_key / "expert_spread.json"
        out.write_text(json.dumps({"run": RUN, "n": N, "by_layer": rows}, indent=1), encoding="utf-8")
        print("->", out)


if __name__ == "__main__":
    main()
