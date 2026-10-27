"""J++ step 2 side-question: does skipping the first 16 positions make sense here?

The paper (and jlens) drop positions 0-15 from every fit as attention sinks.
TinyStories fit prompts are 128 tokens, so that drops 15 of 126 usable
positions -- but kids' prompts are short and the app reads out at EVERY
position, including 1-15. Two checks:

  profile  per layer, the residual norm and the cosine to the mean residual
           direction at each position (mean over the first 200 fit prompts):
           a sink shows up as an outlier norm / direction.
  refit    a plain J with skip_first=1 (only BOS dropped), same estimator:
           stories15M on all 1000 prompts, stories110M on the first 100
           (compare with the shipped n=100 fit); evaluate_jpp.py then scores
           it as "J skip1", with the future-word metric split at position 16.

Usage: python src/jpp/skip_check.py <stories110M|stories15M> [profile|refit]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import ARTIFACTS, JPP_OUT, ROWS_PER_PASS, SEQ_LEN, fit_prompts, harden_atomic_writes, load_model, save_jlens_pt, setup_logging  # noqa: E402
from lenslib import record  # noqa: E402


def profile(model_key: str) -> None:
    lens_model, hf, tok = load_model(model_key)
    layers = list(range(lens_model.n_layers))
    norms = {l: torch.zeros(SEQ_LEN) for l in layers}
    sums = {l: torch.zeros(SEQ_LEN, lens_model.d_model) for l in layers}
    prompts = fit_prompts(200)
    for p in prompts:
        _, acts = record(lens_model, p, layers, max_len=SEQ_LEN)
        for l in layers:
            a = acts[l].cpu()
            norms[l][: a.shape[0]] += a.norm(dim=-1)
            sums[l][: a.shape[0]] += a / a.norm(dim=-1, keepdim=True)
    out = {}
    for l in layers:
        n = norms[l] / len(prompts)
        mean_dir = sums[l][16:127].sum(0)
        mean_dir = mean_dir / mean_dir.norm()
        unit = sums[l] / len(prompts)
        cos = (unit @ mean_dir) / unit.norm(dim=-1).clamp_min(1e-9)
        out[l] = {"norm": [round(float(x), 2) for x in n[:127]], "cos_to_mean_dir": [round(float(x), 3) for x in cos[:127]]}
    path = JPP_OUT / model_key / "position_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    for l in (0, len(layers) // 2, len(layers) - 2):
        n = out[l]["norm"]
        c = out[l]["cos_to_mean_dir"]
        print(f"L{l}: norm pos0={n[0]} pos1-3={n[1:4]} pos4-15 mean={sum(n[4:16]) / 12:.2f} pos16+ mean={sum(n[16:]) / len(n[16:]):.2f}"
              f" | cos pos0={c[0]} pos1-3={c[1:4]} pos4-15 mean={sum(c[4:16]) / 12:.3f} pos16+ mean={sum(c[16:]) / len(c[16:]):.3f}")


def refit(model_key: str) -> None:
    from workspace_lens.config import LensConfig
    from workspace_lens.fitting.jacobian_fitting import LensTrainer
    from workspace_lens.utils import hf_model_name_of

    n = 1000 if model_key == "stories15M" else 100
    setup_logging(JPP_OUT / model_key / f"fit_jskip1_n{n}.log")
    harden_atomic_writes()
    lens_model, hf, tok = load_model(model_key)
    config = LensConfig(
        hf_model_name=hf_model_name_of(lens_model),
        checkpoint_name=f"{model_key}/jskip1_n{n}",
        checkpoint_every_n_prompts=None,
        creation_time="2026-10-26_00-00-00",
        artifacts_base_dir=str(ARTIFACTS),
        source_layers=list(range(lens_model.n_layers - 1)),
        jacobian_rows_per_pass=ROWS_PER_PASS,
        max_seq_len=SEQ_LEN,
        skip_first_n_positions=1,
    )
    lens = LensTrainer(config, lens_model, fit_prompts(n)).fit()
    save_jlens_pt(lens.jacobians_L_dict_FN, JPP_OUT / model_key / f"jskip1_n{n}.pt", n,
                  {"lens": "plain J, skip_first=1"})


if __name__ == "__main__":
    model_key = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "profile"
    profile(model_key) if mode == "profile" else refit(model_key)
