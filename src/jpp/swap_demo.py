"""J++ step 3 (causal): the app's swap, reimplemented exactly, per lens.

Usage: python src/jpp/swap_demo.py <stories110M|stories15M>

App semantics (app/jlens-explorer.html makeEditHook): at each layer in the
swap range and each position >= the anchor (the last prompt token),
  beta = h . s_hat ;  h += strength * beta * (t_hat - s_hat)
where s_hat / t_hat are the unit readout directions (wcls[tok] * gamma) @ J_l
of the take-out / put-in words, applied to the layer OUTPUT. Greedy rewrite
(temperature 0) of up to 120 new tokens; the no-swap rewrite is the control.
Directions are scale- and sign-invariant, so J++'s unit-L2 expert weights do
not matter here.

The headline is the README demo ("...Max dug a", hole -> castle, L4-L8,
strength 0.4; the shipped J writes "moat"); a small battery of extra swaps
(take-out word = both models' greedy next word at the anchor) checks it is not
a one-off. Scored on the FIRST word after the anchor, which is what the demo is
about: "exact" = it is the put-in word, "switched" = it is no longer the
take-out word (the demo's "moat" is switched, not exact); the first words are
reported so a reader can judge whether a switch is on-concept.

Writes out/jpp/<model>/swaps.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import APP_BANDS, JPP_OUT, load_model  # noqa: E402
from lenslib import lens_variants  # noqa: E402

DEMO = ("Lily and Max went to the beach. Lily built a castle. Max dug a", "hole", "castle")
BATTERY = [  # every take-out word is BOTH models' greedy next word at the anchor
    DEMO,
    ("Ann played in the hot sun all day. She was very thirsty. She asked her mom for some", "water", "milk"),
    ("Kate found an old chest in the attic, but it was locked. She needed a", "key", "box"),
    ("Tom had a pet cat. Every morning, he gave milk to his", "cat", "dog"),
    ("The little bird flew high up into the", "sky", "tree"),
    ("Ben was hungry, so he ate a big red", "apple", "cookie"),
    ("The dog barked and wagged its", "tail", "head"),
]
STRENGTH = 0.4
MAX_NEW = 120


def directions(hf, J, tid_s, tid_t):
    g = hf.lm_head.weight * hf.model.norm.weight  # [V, d]: wcls[t] * gamma
    out = {}
    for l, Jl in J.items():
        Jl = Jl.to(g.device)
        s = g[tid_s] @ Jl
        t = g[tid_t] @ Jl
        out[l] = (s / s.norm(), t / t.norm())
    return out


@torch.no_grad()
def rewrite(hf, ids, edit=None):
    """Greedy, full recompute each step (positions are causal, so this equals
    the app's cached forward). edit = (dirs, from_pos, strength)."""
    handles = []
    if edit is not None:
        dirs, from_pos, strength = edit
        for l, (s, t) in dirs.items():
            def hook(module, inputs, output, s=s, t=t):
                h = output if torch.is_tensor(output) else output[0]
                x = h[:, from_pos:, :]
                beta = (x @ s)[..., None]
                h[:, from_pos:, :] = x + strength * beta * (t - s)
                return output
            handles.append(hf.model.layers[l].register_forward_hook(hook))
    try:
        seq = ids.clone()
        for _ in range(MAX_NEW):
            logits = hf(input_ids=seq, use_cache=False).logits[0, -1]
            nxt = int(logits.argmax())
            if nxt in (1, 2):  # BOS / EOS end the story, as in the app
                break
            seq = torch.cat([seq, torch.tensor([[nxt]], device=seq.device)], dim=1)
    finally:
        for h in handles:
            h.remove()
    return seq[0, ids.shape[1]:].tolist()


def main() -> None:
    model_key = sys.argv[1]
    lens_model, hf, tok = load_model(model_key)
    variants = {k: v for k, v in lens_variants(model_key).items()
                if k in ("J", "LRP-only", "J++", "J++ mixed", "experts-noLRP", "experts-noLRP n100")}
    bands = {"band": APP_BANDS[model_key]}
    if model_key == "stories15M":
        bands["L1-L4"] = [1, 2, 3, 4]
    else:
        bands["L2-L8"] = [2, 3, 4, 5, 6, 7, 8]
    results = []
    for prompt, src, tgt in BATTERY:
        ids = lens_model.encode(prompt, max_length=512)
        sid, tid = tok.convert_tokens_to_ids("▁" + src), tok.convert_tokens_to_ids("▁" + tgt)
        assert tok.unk_token_id not in (sid, tid), (src, tgt)  # app: single-token words only
        anchor = ids.shape[1] - 1
        base = rewrite(hf, ids)
        row = {"prompt": prompt, "src": src, "tgt": tgt, "no_swap": tok.decode(base[:60]),
               "control_first_is_src": bool(base) and base[0] == sid, "swaps": {}}
        for name, J in variants.items():
            for bname, layers in bands.items():
                dirs = directions(hf, {l: J[l] for l in layers}, sid, tid)
                out = rewrite(hf, ids, (dirs, anchor, STRENGTH))
                row["swaps"][f"{name} {bname}"] = {
                    "layers": layers,
                    "text": tok.decode(out[:60]),
                    "first_words": " ".join(tok.decode(out[:12]).split()[:4]),
                    "exact": bool(out) and out[0] == tid,
                    "switched": bool(out) and out[0] != sid,
                    "changed": out != base,
                }
        results.append(row)
        print(f"\n== {prompt!r} {src}->{tgt}\n   no swap: {row['no_swap'][:90]!r}")
        for k, v in row["swaps"].items():
            mark = "EXACT" if v["exact"] else "SWITCH" if v["switched"] else "--"
            print(f"   {k:24s} {mark:6s} {v['text'][:90]!r}")
    out = JPP_OUT / model_key / "swaps.json"
    out.write_text(json.dumps({"strength": STRENGTH, "max_new": MAX_NEW, "bands": bands, "results": results},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
