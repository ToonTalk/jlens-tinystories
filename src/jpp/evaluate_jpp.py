"""J++ step 3: compare J, J+filter, LRP-only and J++ (and ablations) on one model.

Usage: python src/jpp/evaluate_jpp.py <stories110M|stories15M>

Metrics (each per layer; "band" = the app's band, 110M L4-L8, 15M L2-L3):
  overlap   top-10 overlap with the logit lens over every position >= 1 of the
            12 eval prompts (the out/report.md divergence metric), unfiltered
            and with Readout Filtering applied to BOTH lenses.
  recall    dev-set recall@10 (min rank over a layer set <= 10, the paper's
            rule) on the held-out items the model gets right; per family and
            macro over families. Unfiltered and filtered.
  pins      the 12 eval prompts' last-position readouts as the app displays
            them (word-start story vocab, app/vocab.json wordStartIds), the
            headline "pin" words from out/report.md, and two legibility
            proxies: share of display slots held by prompt-independent tokens
            (in the top-8 of >= 6 of the 12 prompts) and share of the RAW top-10
            that are displayable story words.
  lily      full-vocab rank of " L" (and " Max") at every position of eval
            prompts 1 and 8, min over the band, both filter settings.
  future    label-free, open-ended: on 60 held-out TinyStories VALIDATION
            stories, at every position, the share of the app-display top-10
            that the story actually says in the next 2..21 tokens and has not
            said yet (novel upcoming content words; frequent function words
            excluded). Split at position 16 for the skip-first question.
  stability n=100 vs n=1000 fit: app-display top-8 agreement at the band
            layers (12 eval prompts, last position) and J cosine.

Writes out/jpp/<model>/eval.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import (  # noqa: E402
    APP_BANDS,
    EVAL_PROMPTS,
    JPP_OUT,
    ROOT,
    fit_prompts,
    load_model,
    non_semantic_ids,
    piece_str,
    resolve_pins,
)
from lenslib import exclusion_mask, lens_logits, lens_variants, load_dev, ranks_of, record  # noqa: E402

PIN_WORDS = {  # out/report.md headline readouts (eval prompt index -> words)
    3: ["hid", "hide", "hiding"],
    8: ["shadow", "investigate", "secret", "noise", "sound"],
    7: ["dig", "buried", "pit", "mole", "deeper"],
    4: ["chose", "choose", "Which", "what"],
    5: ["Finally", "finally", "persistent", "success", "mistakes"],
}
N_VAL = 60
FUTURE = (2, 22)  # tokens p+2 .. p+21
N_STOP = 100


def validation_stories(tok) -> list[str]:
    path = JPP_OUT / "corpora" / f"validation_{N_VAL}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from datasets import load_dataset

    d = resolve_pins()["datasets"]["tinystories"]
    stream = load_dataset(d["repo"], split="validation", revision=d["sha"], streaming=True)
    out = []
    for row in stream:
        text = row["text"].strip()
        if text and len(tok(text).input_ids) >= 128:
            out.append(text)
        if len(out) >= N_VAL:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def layer_sets(model_key: str, layers: list[int]) -> dict[str, list[int]]:
    b = APP_BANDS[model_key]
    sets = {"band": b, "all": layers}
    sets["early"] = [l for l in layers if 1 <= l < b[0]] or [layers[0]]
    if model_key == "stories110M":
        sets["L1-L8"] = [l for l in layers if 1 <= l <= 8]
    else:
        sets["L1-L3"] = [l for l in layers if 1 <= l <= 3]
    return sets


@torch.no_grad()
def main() -> None:
    model_key = sys.argv[1]
    t0 = time.time()
    lens_model, hf, tok = load_model(model_key)
    V = hf.config.vocab_size
    dev = lens_model.input_device
    variants = lens_variants(model_key)
    layers = sorted(variants["J"])
    sets = layer_sets(model_key, layers)
    print(f"[{model_key}] variants: {list(variants)}")

    ns = non_semantic_ids(tok, V)
    filt = exclusion_mask(ns, set(), V, dev)
    vocab = json.loads((ROOT / "app" / "vocab.json").read_text(encoding="utf-8"))
    display = torch.zeros(V, dtype=torch.bool)
    display[vocab["wordStartIds"]] = True
    display = display.to(dev)
    not_display = ~display

    def logits_for(name, l, h):
        J = variants[name]
        return lens_logits(lens_model, None if J is None else J[l], h)

    res: dict = {"model": model_key, "variants": list(variants), "layers": layers, "layer_sets": sets}

    # ---------- eval prompts: overlap, pins, lily, stability ----------
    rec = [record(lens_model, p, layers) for p in EVAL_PROMPTS]
    overlap = {f: {n: {} for n in variants if n != "logit"} for f in ("raw", "filtered")}
    pins = {n: {} for n in variants}
    lily = {n: {} for n in variants}
    L_id, Max_id = tok.convert_tokens_to_ids("▁L"), tok.convert_tokens_to_ids("▁Max")
    for l in layers:
        base = {}
        for pi, (ids, acts) in enumerate(rec):
            lg = logits_for("logit", l, acts[l][1:])
            base[pi] = {"raw": lg.topk(10).indices, "filtered": lg.masked_fill(filt, -1e30).topk(10).indices}
        for n in variants:
            disp_lists, raw10_word = [], []
            for pi, (ids, acts) in enumerate(rec):
                lg = logits_for(n, l, acts[l])
                if n != "logit":
                    for f in ("raw", "filtered"):
                        top = (lg[1:] if f == "raw" else lg[1:].masked_fill(filt, -1e30)).topk(10).indices
                        ov = (top[:, :, None] == base[pi][f][:, None, :]).any(-1).float().mean(-1)
                        overlap[f][n].setdefault(l, []).extend(ov.tolist())
                last = lg[-1]
                disp = last.masked_fill(not_display, -1e30).topk(8).indices.tolist()
                disp_lists.append(disp)
                raw10 = last.topk(10).indices
                raw10_word.append(display[raw10].float().mean().item())
                if pi in (0, 7):
                    for tid, tname in ((L_id, " L"), (Max_id, " Max")):
                        for f in ("raw", "filtered"):
                            r = ranks_of(lg, [tid], filt if f == "filtered" else None).tolist()
                            lily[n].setdefault(f"{pi}|{tname}|{f}", {})[l] = r
            counts = Counter(t for d in disp_lists for t in set(d))
            recurring = {t for t, c in counts.items() if c >= 6}
            pins[n][l] = {
                "display_top8": [[piece_str(tok, t) for t in d] for d in disp_lists],
                "recurring_share": sum(t in recurring for d in disp_lists for t in d) / (8 * len(disp_lists)),
                "raw_top10_wordshare": sum(raw10_word) / len(raw10_word),
            }
    res["overlap"] = {f: {n: {l: sum(v) / len(v) for l, v in d.items()} for n, d in g.items()} for f, g in overlap.items()}
    # pin hits: headline words present in the band's display top-8 (union over band layers)
    hits = {}
    for n in variants:
        h = {}
        for pi, words in PIN_WORDS.items():
            shown = {w.strip() for l in sets["band"] for w in pins[n][l]["display_top8"][pi]}
            h[pi] = sorted(set(words) & shown)
        hits[n] = h
    res["pins"] = pins
    res["pin_hits"] = hits
    res["lily"] = {n: {k: {str(l): r for l, r in d.items()} for k, d in v.items()} for n, v in lily.items()}
    res["lily_tokens"] = {pi: [piece_str(tok, t) for t in rec[pi][0].tolist()] for pi in (0, 7)}

    # stability
    stab = {}
    for a, b in (("J n100", "J"), ("LRP-only n100", "LRP-only"), ("J++ n100", "J++")):
        if a in variants and b in variants:
            agree = {}
            for l in sets["band"]:
                da, db = pins[a][l]["display_top8"], pins[b][l]["display_top8"]
                agree[l] = sum(len(set(x) & set(y)) / 8 for x, y in zip(da, db)) / len(da)
            cos = {
                l: float(torch.nn.functional.cosine_similarity(variants[a][l].flatten(), variants[b][l].flatten(), dim=0))
                for l in layers
            }
            stab[b] = {"top8_agreement": agree, "J_cosine": cos}
    res["stability"] = stab

    # ---------- dev recall ----------
    recall = {}
    for split, kind in (("held-out", "period"), ("held-out", "word"), ("fit", "period"), ("fit", "word")):
        items = load_dev(model_key, split=split)
        if kind == "word":
            items = [{**it, "readout_pos": it["word_pos"]} for it in items if it.get("word_ok")]
        if not items:
            continue
        spare = {c for it in items for c in it["candidate_ids"]}
        excl = exclusion_mask(ns, spare, V, dev)
        ranks = {n: {f: [] for f in ("raw", "filtered")} for n in variants}
        for it in items:
            _, acts = record(lens_model, it["prompt"], layers)
            for n in variants:
                rows = {"raw": {}, "filtered": {}}
                for l in layers:
                    lg = logits_for(n, l, acts[l][it["readout_pos"] : it["readout_pos"] + 1])
                    rows["raw"][l] = int(ranks_of(lg, it["candidate_ids"], None)[0])
                    rows["filtered"][l] = int(ranks_of(lg, it["candidate_ids"], excl)[0])
                for f in ("raw", "filtered"):
                    ranks[n][f].append(rows[f])
        fams = sorted({it["family"] for it in items})
        summary = {}
        for n in variants:
            summary[n] = {}
            for f in ("raw", "filtered"):
                s = {}
                for sname, ls in {**sets, **{f"L{l}": [l] for l in layers}}.items():
                    per_fam = {}
                    for fam in fams:
                        hits_ = [min(r[l] for l in ls) < 10 for it, r in zip(items, ranks[n][f]) if it["family"] == fam]
                        per_fam[fam] = sum(hits_) / len(hits_)
                    s[sname] = {"macro": sum(per_fam.values()) / len(per_fam), "by_family": per_fam}
                summary[n][f] = s
        recall[f"{split}|{kind}"] = {
            "n_items": len(items), "families": {fam: sum(it["family"] == fam for it in items) for fam in fams},
            "summary": summary,
            "ranks": {n: {f: v for f, v in d.items()} for n, d in ranks.items()},
            "items": [it["name"] for it in items]}
        print(f"[{model_key}] recall {split} at {kind}: {len(items)} items")
    res["recall"] = recall

    # ---------- future-word precision on validation stories ----------
    ws = set(vocab["wordStartIds"])
    freq = Counter()
    for p in fit_prompts(1000):
        freq.update(t for t in tok(p, truncation=True, max_length=128).input_ids if t in ws)
    stop = {t for t, _ in freq.most_common(N_STOP)}
    stories = validation_stories(tok)
    fut = {n: {l: {"early": [0.0, 0, 0], "late": [0.0, 0, 0]} for l in layers} for n in variants}
    for text in stories:
        ids_t, acts = record(lens_model, text, layers, max_len=128)
        ids = ids_t.tolist()
        S = len(ids)
        targets = []
        for p in range(1, S - FUTURE[1]):
            prefix = set(ids[: p + 2])  # includes the next token
            F = {t for t in ids[p + FUTURE[0] : p + FUTURE[1]] if t in ws and t not in stop and t not in prefix}
            if F:
                targets.append((p, F))
        if not targets:
            continue
        pos = torch.tensor([p for p, _ in targets], device=dev)
        for n in variants:
            for l in layers:
                top = logits_for(n, l, acts[l][pos]).masked_fill(not_display, -1e30).topk(10).indices.tolist()
                for (p, F), tl in zip(targets, top):
                    k = sum(t in F for t in tl)
                    bucket = fut[n][l]["early" if p < 16 else "late"]
                    bucket[0] += k / 10
                    bucket[1] += int(k > 0)
                    bucket[2] += 1
    res["future"] = {
        n: {l: {b: {"precision10": v[0] / max(v[2], 1), "hit_rate": v[1] / max(v[2], 1), "n": v[2]} for b, v in d.items()}
            for l, d in g.items()}
        for n, g in fut.items()
    }
    res["wall_s"] = round(time.time() - t0, 1)
    out = JPP_OUT / model_key / "eval.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out} ({res['wall_s']}s)")


if __name__ == "__main__":
    main()
