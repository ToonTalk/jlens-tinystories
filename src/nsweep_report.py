"""Step 0a report: filtered top-8 agreement at L4-L8 vs n=1000, per fit size
n in {1, 5, 10, 25, 100}, over the 12 eval prompts (last position), plus slice
pages for eval prompts 4 and 9 per n. Writes out/nsweep/report.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from common import EVAL_PROMPTS, OUT, load_model, word_start_alpha_ids

NS = [1, 5, 10, 25, 100]
BAND = [4, 5, 6, 7, 8]
SLICE_PROMPTS = [3, 8]  # eval prompts 4 and 9 (0-based)


def main() -> None:
    from jlens import JacobianLens
    from jlens.vis import build_page, compute_slice

    out_dir = OUT / "nsweep"
    out_dir.mkdir(exist_ok=True)
    model, hf, tok, info = load_model("stories110M")
    filter_ids = word_start_alpha_ids(tok, hf.config.vocab_size)
    ref = JacobianLens.load(str(OUT / "lenses" / "A" / "lens_1000.pt"))

    def top8(lens, prompt, layer):
        jl, _, _ = lens.apply(model, prompt, layers=[layer], positions=[-1])
        out = []
        for t in jl[layer][0].argsort(descending=True).tolist():
            if t in filter_ids:
                out.append(t)
                if len(out) == 8:
                    break
        return set(out)

    ref_tops = {
        (pi, l): top8(ref, EVAL_PROMPTS[pi], l)
        for pi in range(len(EVAL_PROMPTS))
        for l in BAND
    }

    rows = []
    wall = {}
    for n in NS:
        lens = JacobianLens.load(str(OUT / "lenses" / "A" / f"lens_{n}.pt"))
        cfg = json.loads(
            (OUT / "lenses" / "A" / f"config_{n}.json").read_text(encoding="utf-8")
        )
        wall[n] = cfg["wall_clock_s"]
        per_layer = {}
        for l in BAND:
            scores = []
            for pi in range(len(EVAL_PROMPTS)):
                s = top8(lens, EVAL_PROMPTS[pi], l)
                scores.append(len(s & ref_tops[(pi, l)]) / 8)
            per_layer[l] = sum(scores) / len(scores)
        # J-matrix cosine vs n=1000, mid band mean
        cos = sum(
            torch.nn.functional.cosine_similarity(
                lens.jacobians[l].flatten(), ref.jacobians[l].flatten(), dim=0
            ).item()
            for l in BAND
        ) / len(BAND)
        rows.append((n, per_layer, cos))
        print(f"n={n}: band agreement "
              + " ".join(f"L{l}={per_layer[l]:.2f}" for l in BAND)
              + f" cos={cos:.4f}")

        for pi in SLICE_PROMPTS:
            sd = compute_slice(model, lens, EVAL_PROMPTS[pi], mask_display=True)
            page, _, _ = build_page(
                sd, EVAL_PROMPTS[pi],
                title=f"n-sweep n={n} - eval prompt {pi + 1}",
                description="J-lens slice, stories110M/TinyStories fit",
                mode="embed",
            )
            (out_dir / f"n{n:04d}_prompt{pi + 1:02d}.html").write_text(
                page, encoding="utf-8"
            )

    md = ["# n-sweep: how few fitting prompts does the 110M lens need?\n",
          "Filtered top-8 agreement with the n=1000 lens at the last position "
          "of the 12 eval prompts, per mid-band layer; plus mean J cosine over "
          "L4-L8 and fit wall-clock.\n",
          "| n | " + " | ".join(f"L{l}" for l in BAND) + " | mean | J cos | fit wall-clock |",
          "|---|" + "---|" * (len(BAND) + 3)]
    for n, per_layer, cos in rows:
        mean = sum(per_layer.values()) / len(per_layer)
        md.append(
            f"| {n} | "
            + " | ".join(f"{per_layer[l]:.2f}" for l in BAND)
            + f" | **{mean:.2f}** | {cos:.4f} | {wall[n]:.0f} s |"
        )
    md.append(
        f"| 1000 (ref) | " + " | ".join("1.00" for _ in BAND)
        + " | **1.00** | 1.0000 | 6327 s |"
    )
    md.append("\nSlice pages per n for eval prompts 4 and 9: `n****_prompt04/09.html`.\n")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
