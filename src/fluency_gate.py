"""Step 1 sanity gate: greedy completion of 'Once upon a time' on both models
must be fluent TinyStories prose. Writes out/fluency.json."""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import torch

from common import MODELS, OUT, load_model, set_seeds, write_json


def main() -> None:
    set_seeds()
    results = {}
    for key in MODELS:
        lens_model, hf, tok, info = load_model(key)
        ids = tok("Once upon a time", return_tensors="pt").input_ids.to(hf.device)
        with torch.no_grad():
            out = hf.generate(
                ids,
                max_new_tokens=120,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        results[key] = {"info": info, "greedy_completion": text}
        print(f"\n=== {key} ({info['repo']} @ {info['revision'][:8]}) ===")
        print(text.encode("ascii", "replace").decode())
        del hf, lens_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(OUT / "fluency.json", results)
    print(f"\nWrote {OUT / 'fluency.json'}")


if __name__ == "__main__":
    main()
