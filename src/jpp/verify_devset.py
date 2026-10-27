"""J++ step 2b: which dev items does each model actually "hold in mind"?

The paper scores only items the model answers correctly. The TinyStories
analog: an item counts for a model when the model's own greedy continuation
says one of the item's answer words SOON but NOT NEXT -- the answer is not the
greedy next token, and it appears among generated tokens 2..48. That is the
J-lens's own promise ("words the model is disposed to say, now or a few
sentences later"), checked against the model's behavior, with the trivially
next-token cases removed so the items test latent content.

Items are also dropped if an answer word has no single-token surface, or (for
all families except "name") if an answer token already occurs in the prompt.

Writes out/jpp/devset/correct_<model>.json. Usage:
  python src/jpp/verify_devset.py stories110M stories15M
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import JPP_OUT, candidate_ids, load_model  # noqa: E402

MAX_NEW = 48


@torch.no_grad()
def verify(model_key: str) -> None:
    lens_model, hf, tok = load_model(model_key)
    items = json.loads((JPP_OUT / "devset" / "items.json").read_text(encoding="utf-8"))
    rows = []
    for it in items:
        cands = candidate_ids(tok, it["answers"])
        ids = lens_model.encode(it["prompt"], max_length=512)
        out = hf.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id
        )[0, ids.shape[1]:].tolist()
        prompt_ids = set(ids[0].tolist())
        hits = [i for i, t in enumerate(out) if t in cands]
        reason = None
        missing = [w for w in it["answers"] if not candidate_ids(tok, [w])]
        if missing:
            reason = f"multi-token answer {missing}"
        elif not it["copy_allowed"] and prompt_ids & set(cands):
            reason = "answer already in prompt"
        elif out and out[0] in cands:
            reason = "answer is the next token"
        elif not hits:
            reason = "answer not said within 48 tokens"
        rows.append(
            {
                "name": it["name"],
                "family": it["family"],
                "split": it["split"],
                "correct": reason is None,
                "reason": reason,
                "first_hit": (hits[0] + 1) if hits else None,
                "readout_pos": ids.shape[1] - 1,
                "candidate_ids": cands,
                "continuation": tok.decode(out),
            }
        )
    path = JPP_OUT / "devset" / f"correct_{model_key}.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter

    kept = Counter((r["family"], r["split"]) for r in rows if r["correct"])
    total = Counter((r["family"], r["split"]) for r in rows)
    print(f"[{model_key}] kept {sum(kept.values())}/{len(rows)}")
    for key in sorted(total):
        print(f"   {key[0]:8s} {key[1]:9s} {kept[key]:3d}/{total[key]}")
    print("   reasons:", Counter(r["reason"] for r in rows if r["reason"]))


if __name__ == "__main__":
    for key in sys.argv[1:] or ["stories110M", "stories15M"]:
        verify(key)
