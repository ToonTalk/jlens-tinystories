"""Step 0 smoke test: walkthrough.ipynb flow (load -> fit -> apply -> slice)
on a tiny random HF Llama before touching the real targets."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import transformers

import jlens
from jlens.vis import build_page, compute_slice

REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = transformers.AutoTokenizer.from_pretrained(REPO)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=torch.float32
    ).to(device)
    model = jlens.from_hf(hf, tok)
    print(f"loaded {model!r} on {device}")

    prompts = [
        "One day a little cat walked to the park and saw a big red ball near the tree "
        "and wanted to play with it all day long.",
        "The sun was warm and the wind was soft and all the children ran outside to "
        "play games together in the green grass by the school.",
    ]
    lens = jlens.fit(model, prompts, dim_batch=8, max_seq_len=64)
    for layer, J in lens.jacobians.items():
        assert torch.isfinite(J).all(), f"NaN in J_{layer}"
    print(f"fitted {lens!r}")

    lens_logits, model_logits, _ = lens.apply(model, prompts[0], positions=[-1])
    assert all(torch.isfinite(v).all() for v in lens_logits.values())
    print(f"apply ok: {len(lens_logits)} layers, logits {model_logits.shape}")

    sd = compute_slice(model, lens, prompts[0], mask_display=False)
    page, _, _ = build_page(sd, prompts[0], title="smoke", description="smoke", mode="embed")
    assert "<html" in page.lower() or "<div" in page.lower()
    print(f"slice page ok ({len(page)} chars)")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
