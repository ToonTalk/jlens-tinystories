"""J++ step 2b': a second readout position per dev item, off the period.

Every dev prompt ends in a sentence-final period, so every recipe readout is a
period residual -- and the router puts all of them in ONE cluster per layer.
Kids click any token. This adds the readout one token earlier (the sentence's
last word, e.g. "outside" in "...very cold outside."): when the model's own
greedy next token there IS that period, its continuation is the verified one
shifted by a token, so the item's label (answer said soon, not next) carries
over unchanged. Items failing that check get word_ok = false.

Adds word_pos / word_ok to out/jpp/devset/correct_<model>.json.
Usage: python src/jpp/add_word_positions.py stories110M stories15M
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import JPP_OUT, load_model  # noqa: E402


@torch.no_grad()
def main(model_key: str) -> None:
    lens_model, hf, tok = load_model(model_key)
    items = {it["name"]: it for it in json.loads((JPP_OUT / "devset" / "items.json").read_text(encoding="utf-8"))}
    path = JPP_OUT / "devset" / f"correct_{model_key}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    n_ok = 0
    for r in rows:
        ids = lens_model.encode(items[r["name"]]["prompt"], max_length=512)
        p = r["readout_pos"] - 1
        greedy = int(hf(input_ids=ids[:, : p + 1], use_cache=False).logits[0, -1].argmax())
        r["word_pos"] = p
        r["word_ok"] = bool(r["correct"] and greedy == int(ids[0, p + 1]) and int(ids[0, p]) not in r["candidate_ids"])
        n_ok += r["word_ok"]
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[{model_key}] word readouts valid for {n_ok}/{sum(r['correct'] for r in rows)} correct items")


if __name__ == "__main__":
    for key in sys.argv[1:]:
        main(key)
