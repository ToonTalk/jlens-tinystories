"""Phase-1 app prep (brief step 0b-0d):

b. Export lens_1000.pt -> .jlens binaries:
     stories110M.jlens  (layers 4-8, d=768)
     stories15M.jlens   (layers 2-3, d=288)
   Format (little-endian): magic 'JLNS', uint32 version=1, uint32 d_model,
   uint32 n_layers, n_layers x uint32 layer index, then per layer J as fp32
   row-major [d_model, d_model] with rows = OUTPUT dims (transport = J @ h,
   matching torch's lens.jacobians[l] layout).

c. vocab sidecar JSON: pruned token ids (TinyStories train sample, count>=5),
   word-start display mask, multi-token-name alias table, theme vocabulary.

d. fixtures.json: eval prompts 1, 4, 9; all positions; layers 4-8; top-20
   ids + fp32 scores for J-lens AND logit lens, straight from the Python
   pipeline (jlens.apply semantics: residual -> J@h -> final norm -> unembed).

Everything goes to app/.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from common import EVAL_PROMPTS, OUT, ROOT, load_model, resolve_pins

APP = ROOT / "app"
FIXTURE_PROMPTS = [0, 3, 8]  # eval prompts 1, 4, 9 (0-based)
BANDS = {"stories110M": [4, 5, 6, 7, 8], "stories15M": [2, 3]}
VOCAB_SAMPLE_STORIES = 100_000  # deterministic prefix of TinyStories train
COUNT_THRESHOLD = 5
NAME_COUNT_MIN = 50
THEME_COUNT_MIN = 2000


def export_jlens(lens_path: Path, layers: list[int], out_path: Path) -> None:
    from jlens import JacobianLens

    lens = JacobianLens.load(str(lens_path))
    d = lens.d_model
    with open(out_path, "wb") as fh:
        fh.write(b"JLNS")
        fh.write(struct.pack("<III", 1, d, len(layers)))
        for l in layers:
            fh.write(struct.pack("<I", l))
        for l in layers:
            J = lens.jacobians[l].to(torch.float32).contiguous()
            assert J.shape == (d, d)
            assert torch.isfinite(J).all()
            fh.write(J.numpy().astype("<f4").tobytes())
    print(f"[export] {out_path.name}: d={d} layers={layers} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


def build_vocab_sidecar(tok) -> dict:
    from datasets import load_dataset

    pins = resolve_pins()
    d = pins["datasets"]["tinystories"]
    t0 = time.time()
    stream = load_dataset(
        d["repo"], d["config"], split=d["split"], revision=d["sha"], streaming=True
    )
    counts: Counter[int] = Counter()
    word_counts: Counter[str] = Counter()
    n_stories = 0
    batch: list[str] = []

    def flush(batch):
        for ids in tok(batch, add_special_tokens=False).input_ids:
            counts.update(ids)

    for row in stream:
        text = row["text"].strip()
        if not text:
            continue
        batch.append(text)
        for w in text.split():
            word_counts[w.strip('.,!?";:()').strip("'")] += 1
        if len(batch) >= 512:
            flush(batch)
            batch = []
        n_stories += 1
        if n_stories >= VOCAB_SAMPLE_STORIES:
            break
    if batch:
        flush(batch)
    print(f"[vocab] tokenized {n_stories} stories in {time.time() - t0:.0f}s; "
          f"{len(counts)} distinct ids")

    pruned = sorted(tid for tid, c in counts.items() if c >= COUNT_THRESHOLD)

    # word-start display mask (relative to pruned list)
    def piece(tid):
        return tok.convert_ids_to_tokens(tid)

    word_start = [
        tid for tid in pruned
        if (p := piece(tid)) and len(p) > 1 and p.startswith("▁") and p[1:].isalpha()
    ]

    # multi-token-name alias table: capitalized corpus words, count>=NAME_COUNT_MIN,
    # whose " Word" tokenization is multi-token -> first piece id aliases the name.
    alias: dict[int, dict] = {}
    names = [
        (w, c) for w, c in word_counts.most_common()
        if c >= NAME_COUNT_MIN and w[:1].isupper() and w.isalpha() and len(w) >= 3
    ]
    for w, c in names:
        ids = tok(" " + w, add_special_tokens=False).input_ids
        if len(ids) < 2:
            continue
        first = ids[0]
        entry = alias.setdefault(
            first, {"piece": piece(first).replace("▁", " "), "names": []}
        )
        if len(entry["names"]) < 6 and w not in [n["name"] for n in entry["names"]]:
            entry["names"].append({"name": w, "count": c})
    for first, entry in alias.items():
        top = entry["names"][0]["name"]
        frag = entry["piece"].strip()
        entry["display"] = f"{frag}… ({top}?)"

    # keep alias fragments (first pieces AND their continuation pieces for the
    # top names) in the pruned set even if below threshold
    pruned_set = set(pruned)
    for first, entry in alias.items():
        pruned_set.add(first)
        for n in entry["names"][:3]:
            for tid in tok(" " + n["name"], add_special_tokens=False).input_ids:
                pruned_set.add(tid)
    pruned = sorted(pruned_set)

    # theme vocabulary: frequent single-token lowercase story words (prompt picker)
    theme = []
    for w, c in word_counts.most_common():
        if c < THEME_COUNT_MIN or not w.isalpha() or not w.islower() or len(w) < 3:
            continue
        ids = tok(" " + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            theme.append(w)
    theme = theme[:300]

    sidecar = {
        "source": f"first {n_stories} stories of TinyStories train "
                  f"@ {d['sha'][:10]}, count >= {COUNT_THRESHOLD}",
        "prunedIds": pruned,
        "wordStartIds": word_start,
        "alias": {str(k): v for k, v in sorted(alias.items())},
        "themeWords": theme,
    }
    print(f"[vocab] pruned={len(pruned)} wordStart={len(word_start)} "
          f"aliases={len(alias)} theme={len(theme)}")
    return sidecar


def build_fixtures(model_key: str, lens_path: Path, layers: list[int]) -> dict:
    from jlens import JacobianLens

    lens_model, hf, tok, info = load_model(model_key)
    lens = JacobianLens.load(str(lens_path))
    out = {
        "model": info["repo"],
        "revision": info["revision"],
        "lens_file": lens_path.name,
        "layers": layers,
        "top_k": 20,
        "note": "scores are fp32 logits exactly as the Python pipeline computes "
                "them: J-lens = unembed(final_norm(J_l @ h)); logit lens = "
                "unembed(final_norm(h)); h = block-output residual.",
        "prompts": [],
    }
    for pi in FIXTURE_PROMPTS:
        prompt = EVAL_PROMPTS[pi]
        jl, model_logits, input_ids = lens.apply(lens_model, prompt, layers=layers)
        ll, _, _ = lens.apply(lens_model, prompt, layers=layers, use_jacobian=False)
        n_pos = input_ids.shape[1]
        entry = {
            "eval_index": pi + 1,
            "text": prompt,
            "token_ids": input_ids[0].tolist(),
            "jlens": [],   # [layer][pos] {ids, scores}
            "logit": [],
        }
        for kind, source in (("jlens", jl), ("logit", ll)):
            per_layer = []
            for l in layers:
                per_pos = []
                for p in range(n_pos):
                    top = source[l][p].topk(20)
                    per_pos.append({
                        "ids": top.indices.tolist(),
                        "scores": [round(s, 5) for s in top.values.tolist()],
                    })
                per_layer.append(per_pos)
            entry[kind] = per_layer
        out["prompts"].append(entry)
        print(f"[fixtures] prompt {pi + 1}: {n_pos} positions x {len(layers)} layers")
    return out


def main() -> None:
    APP.mkdir(exist_ok=True)

    export_jlens(OUT / "lenses" / "A" / "lens_1000.pt", BANDS["stories110M"],
                 APP / "stories110M.jlens")
    export_jlens(OUT / "lenses" / "C" / "lens_1000.pt", BANDS["stories15M"],
                 APP / "stories15M.jlens")

    _, _, tok, _ = load_model("stories110M")
    sidecar = build_vocab_sidecar(tok)
    (APP / "vocab.json").write_text(
        json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[vocab] wrote {APP / 'vocab.json'}")

    fixtures = build_fixtures(
        "stories110M", OUT / "lenses" / "A" / "lens_1000.pt", BANDS["stories110M"]
    )
    (APP / "fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")
    print(f"[fixtures] wrote {APP / 'fixtures.json'} "
          f"({(APP / 'fixtures.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
