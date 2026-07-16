"""Step 2: fit one lens (A, B, or C) and save lens.pt + config JSON.

Usage: python src/fit_lens.py <A|B|C> <n_prompts>

Deviations from repo defaults, logged here and in the config JSON:
  - dim_batch=32 (repo default 8). Pure performance knob: fewer, larger
    backward passes; the estimator is unchanged. These models are tiny so
    GPU memory is not a constraint.
  - checkpoint_every=25 (repo default 1) to avoid rewriting the multi-MB
    checkpoint after every prompt.
Everything else (skip_first=16, target_layer=final, source_layers=all below
target, max_seq_len=128) is the repo/paper default.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from common import FITS, OUT, SEED, SEQ_LEN, build_corpus, load_model, set_seeds

DIM_BATCH = 32
CHECKPOINT_EVERY = 25


def main() -> None:
    fit_id, n_prompts = sys.argv[1], int(sys.argv[2])
    model_key, corpus_key = FITS[fit_id]
    out_dir = OUT / "lenses" / fit_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / f"fit_{n_prompts}.log", encoding="utf-8"),
        ],
    )

    set_seeds()
    import jlens

    lens_model, hf, tok, info = load_model(model_key)
    prompts = build_corpus(corpus_key, n_prompts, tok)

    t0 = time.time()
    lens = jlens.fit(
        lens_model,
        prompts,
        dim_batch=DIM_BATCH,
        max_seq_len=SEQ_LEN,
        checkpoint_path=str(out_dir / f"ckpt_{n_prompts}.pt"),
        checkpoint_every=CHECKPOINT_EVERY,
    )
    wall = time.time() - t0

    # NaN check (acceptance criterion 1).
    for layer, J in lens.jacobians.items():
        assert torch.isfinite(J).all(), f"non-finite values in J_{layer}"

    lens_path = out_dir / f"lens_{n_prompts}.pt"
    lens.save(str(lens_path))
    config = {
        "fit_id": fit_id,
        "model": info,
        "corpus": corpus_key,
        "n_prompts": n_prompts,
        "n_prompts_used": lens.n_prompts,
        "seq_len": SEQ_LEN,
        "dim_batch": DIM_BATCH,
        "skip_first": 16,
        "target_layer": info["n_layers"] - 1,
        "source_layers": lens.source_layers,
        "seed": SEED,
        "wall_clock_s": round(wall, 1),
        "s_per_prompt": round(wall / max(lens.n_prompts, 1), 2),
        "hardware": info["gpu"] or "cpu",
        "deviations": {
            "dim_batch": f"{DIM_BATCH} (repo default 8; perf only)",
            "checkpoint_every": f"{CHECKPOINT_EVERY} (repo default 1; perf only)",
        },
    }
    (out_dir / f"config_{n_prompts}.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(
        f"[fit {fit_id}] n={lens.n_prompts} wall={wall:.0f}s "
        f"({wall / max(lens.n_prompts, 1):.1f}s/prompt) -> {lens_path}"
    )


if __name__ == "__main__":
    main()
