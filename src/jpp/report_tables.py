"""Markdown tables for out/jpp/report.md, built from out/jpp/<model>/*.json.

Each lens CONFIGURATION is (variant, readout filter):
  J            the shipped J-lens (out/lenses/<fit>/lens_1000.pt = app/*.jlens)
  J+filter     the same J, J++ Readout Filtering on
  LRP-only     the pooled LRP-Jacobian (R-Lens of the same 1000 prompts) + filter
  J++          LRP experts (K=8) x weights learned on period readouts + filter (the recipe)
  J++ mixed    same experts, weights learned on period AND last-word readouts + filter
  no-LRP exp.  ablation: plain-gradient experts x learned weights + filter
Display-based metrics (app view: word-start story vocab) are untouched by the
filter, which only removes tokens the display already hides.

Usage: python src/jpp/report_tables.py <model>   (prints the tables)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jpp_common import APP_BANDS, JPP_OUT  # noqa: E402

CONFIGS = [  # (label, variant, filter)
    ("logit", "logit", "raw"),
    ("J", "J", "raw"),
    ("J+filter", "J", "filtered"),
    ("LRP-only", "LRP-only", "filtered"),
    ("J++", "J++", "filtered"),
    ("J++ mixed", "J++ mixed", "filtered"),
    ("no-LRP exp.", "experts-noLRP", "filtered"),
]
N100 = {"J": "J n100", "LRP-only": "LRP-only n100", "J++": "J++ n100", "J++ mixed": "J++ mixed n100",
        "experts-noLRP": "experts-noLRP n100"}  # by variant


def load(model_key: str) -> dict:
    return json.loads((JPP_OUT / model_key / "eval.json").read_text(encoding="utf-8"))


USE_N100 = False  # --n100: read the n=100 fits (early look / stability)


def configs(ev: dict, model_key: str, n100: bool | None = None) -> list[tuple[str, str, str]]:
    n100 = USE_N100 if n100 is None else n100
    out = []
    for label, var, f in CONFIGS:
        if model_key == "stories110M" and var == "experts-noLRP" and not n100:
            var = "experts-noLRP n100"  # 110M ablation exists at n=100 only
            label = "no-LRP exp. (n100)"
        if n100 and var in N100:
            var = N100[var]
        if var in ev["variants"]:
            out.append((label, var, f))
    return out


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def overlap_table(ev: dict, model_key: str) -> str:
    layers = ev["layers"]
    rows = []
    for label, var, f in configs(ev, model_key):
        if var == "logit":
            continue
        o = ev["overlap"][f][var]
        rows.append([label] + [f"{o[str(l)]:.2f}" for l in layers])
    return table(["lens"] + [f"L{l}" for l in layers], rows)


def recall_cell(ev, key, var, f, lset):
    r = ev["recall"].get(key)
    if r is None:
        return "-"
    return pct(r["summary"][var][f][lset]["macro"])


def recall_table(ev: dict, model_key: str) -> str:
    sets = list(ev["layer_sets"])
    cols, rows = [], []
    for kind in ("period", "word"):
        for s in sets:
            cols.append(f"{kind} {s}")
    for label, var, f in configs(ev, model_key):
        rows.append([label] + [recall_cell(ev, f"held-out|{kind}", var, f, s) for kind in ("period", "word") for s in sets])
    n_p = ev["recall"]["held-out|period"]["n_items"]
    n_w = ev["recall"].get("held-out|word", {}).get("n_items", 0)
    names = {s: f"{s} (L{ev['layer_sets'][s][0]}-L{ev['layer_sets'][s][-1]})" for s in sets}
    hdr = ["lens"] + [f"{c.split()[0]} {names[c.split()[1]]}" for c in cols]
    return table(hdr, rows) + f"\n\nHeld-out items: {n_p} period readouts, {n_w} last-word readouts."


def recall_layer_table(ev: dict, model_key: str, kind: str = "period") -> str:
    layers = ev["layers"]
    rows = []
    for label, var, f in configs(ev, model_key):
        rows.append([label] + [recall_cell(ev, f"held-out|{kind}", var, f, f"L{l}") for l in layers])
    return table(["lens"] + [f"L{l}" for l in layers], rows)


def recall_family_table(ev: dict, model_key: str, kind: str = "period", lset: str = "band") -> str:
    key = f"held-out|{kind}"
    fams = list(ev["recall"][key]["families"])
    rows = []
    for label, var, f in configs(ev, model_key):
        bf = ev["recall"][key]["summary"][var][f][lset]["by_family"]
        rows.append([label] + [pct(bf[fam]) for fam in fams])
    counts = ev["recall"][key]["families"]
    return table(["lens"] + [f"{fam} (n={counts[fam]})" for fam in fams], rows)


def fit_recall_table(ev: dict, model_key: str) -> str:
    rows = []
    for label, var, f in configs(ev, model_key):
        rows.append([label] + [recall_cell(ev, f"fit|{k}", var, f, "band") for k in ("period", "word")]
                    + [recall_cell(ev, f"held-out|{k}", var, f, "band") for k in ("period", "word")])
    return table(["lens", "fit split, period", "fit split, word", "held-out, period", "held-out, word"], rows)


def pin_table(ev: dict, model_key: str) -> str:
    band = ev["layer_sets"]["band"]
    from evaluate_jpp import PIN_WORDS

    rows = []
    for label, var, f in configs(ev, model_key):
        if f == "filtered" and var == "J":
            continue  # display identical to J
        hits = ev["pin_hits"][var]
        n_hit = sum(len(v) for v in hits.values())
        n_pins = sum(1 for v in hits.values() if v)
        rec = statistics.mean(ev["pins"][var][str(l)]["recurring_share"] for l in band)
        ws = statistics.mean(ev["pins"][var][str(l)]["raw_top10_wordshare"] for l in band)
        rows.append([label, f"{n_pins}/5", str(n_hit), pct(rec), pct(ws)])
    return table(["lens (app display)", "pins showing a headline word", "headline words shown",
                  "display slots held by prompt-independent tokens", "raw top-10 that are displayable words"], rows)


def readout_rows(ev: dict, model_key: str, prompt_idx: int, layer: int) -> str:
    rows = []
    for label, var, f in configs(ev, model_key):
        if f == "filtered" and var == "J":
            continue
        rows.append([label, ", ".join(f"`{w}`" for w in ev["pins"][var][str(layer)]["display_top8"][prompt_idx])])
    return table(["lens", f"L{layer} display top-8"], rows)


def lily_table(ev: dict, model_key: str, prompt_idx: int, token: str = " L") -> str:
    band = [str(l) for l in ev["layer_sets"]["band"]]
    toks = ev["lily_tokens"][str(prompt_idx)]
    cfgs = [c for c in configs(ev, model_key) if c[0] in ("logit", "J", "J+filter", "LRP-only", "J++", "J++ mixed")]
    rows = []
    for p in range(1, len(toks)):
        row = [str(p), f"`{toks[p]}`"]
        for label, var, f in cfgs:
            r = ev["lily"][var][f"{prompt_idx}|{token}|{f}"]
            row.append(str(min(r[l][p] for l in band)))
        rows.append(row)
    return table(["pos", "token"] + [c[0] for c in cfgs], rows)


def lily_summary(ev: dict, model_key: str) -> str:
    """Median band rank of ' L' at the positions where Lily is the active character
    (out/report.md's '<-' rows), per lens."""
    band = [str(l) for l in ev["layer_sets"]["band"]]
    pos = {"0": [9, 11, 12, 13, 14], "7": [1, 2, 3, 9, 10, 15]}
    rows = []
    for label, var, f in configs(ev, model_key):
        if var not in ev["lily"]:
            continue
        ranks = []
        for pi, ps in pos.items():
            r = ev["lily"][var][f"{pi}|{' L'}|{f}"]
            ranks += [min(r[l][p] for l in band) for p in ps]
        top10 = sum(x < 10 for x in ranks)
        rows.append([label, str(int(statistics.median(ranks))), f"{top10}/{len(ranks)}"])
    return table(["lens", "median ' L' rank (band, 11 Lily positions)", "in top-10"], rows)


def future_table(ev: dict, model_key: str, bucket: str = "late") -> str:
    layers = ev["layers"]
    rows = []
    seen = set()
    for label, var, f in configs(ev, model_key) + [("J skip1", "J skip1", "raw"), ("J skip1 n100", "J skip1 n100", "raw"), ("J n100", "J n100", "raw")]:
        if var in seen or var not in ev["future"]:
            continue
        seen.add(var)
        if label == "J+filter":
            continue
        fu = ev["future"][var]
        rows.append([label] + [f"{100 * fu[str(l)][bucket]['precision10']:.1f}" for l in layers])
    n = ev["future"]["J"][str(layers[0])][bucket]["n"]
    return table(["lens"] + [f"L{l}" for l in layers], rows) + f"\n\n({n} story positions.)"


def stability_table(ev: dict, model_key: str) -> str:
    band = [str(l) for l in ev["layer_sets"]["band"]]
    rows = []
    for name, s in ev["stability"].items():
        agree = statistics.mean(s["top8_agreement"][l] for l in band)
        cos = statistics.mean(s["J_cosine"][l] for l in band)
        rows.append([name, pct(agree), f"{cos:.3f}"] + [pct(s["top8_agreement"][l]) for l in band])
    return table(["lens", "top-8 agreement n100 vs n1000 (band mean)", "J cosine (band mean)"] + [f"L{l}" for l in band], rows)


def swap_table(model_key: str, band_only: bool = False) -> str:
    sw = json.loads((JPP_OUT / model_key / "swaps.json").read_text(encoding="utf-8"))
    keys = [k for k in sw["results"][0]["swaps"] if not band_only or k.endswith(" band")]
    rows = []
    for r in sw["results"]:
        row = [f"{r['src']}→{r['tgt']}", f"`{' '.join(r['no_swap'].split()[:4])}`"]
        for k in keys:
            v = r["swaps"][k]
            mark = "**=**" if v["exact"] else "≠" if v["switched"] else "·"
            row.append(f"{mark} `{v['first_words']}`")
        rows.append(row)
    tally = ["**switched / exact**", ""] + [
        f"**{sum(r['swaps'][k]['switched'] for r in sw['results'])} / {sum(r['swaps'][k]['exact'] for r in sw['results'])}**"
        for k in keys]
    return table(["swap", "no swap"] + keys, rows + [tally])


def cosine_table(model_key: str) -> str:
    """Cosine (flattened) between each lens's J_l and the shipped J_l."""
    import torch

    from lenslib import lens_variants

    v = lens_variants(model_key)
    layers = sorted(v["J"])
    rows = []
    for name in ("pooled-noLRP", "LRP-only", "J++", "J++ mixed", "experts-noLRP", "experts-noLRP n100", "J skip1", "J skip1 n100"):
        if name not in v:
            continue
        ref = v["J n100"] if name.endswith("n100") else v["J"]
        rows.append([name] + [f"{float(torch.nn.functional.cosine_similarity(v[name][l].flatten(), ref[l].flatten(), dim=0)):.2f}" for l in layers])
    return table(["lens vs shipped J"] + [f"L{l}" for l in layers], rows)


def spread_table(model_key: str) -> str:
    """Expert spread and weight cancellation per layer (out/jpp/<model>/expert_spread.json)."""
    s = json.loads((JPP_OUT / model_key / "expert_spread.json").read_text(encoding="utf-8"))["by_layer"]
    band = APP_BANDS[model_key]
    rows = [[f"**L{l}**" if int(l) in band else f"L{l}",
             f"{v['expert_cos_min']:.2f} / {v['expert_cos_median']:.2f}",
             f"{v['expert_rel_dist_median']:.2f}",
             pct(v["weight_negative_share"]),
             f"{v['jpp_cos_pooled']:+.2f}"] for l, v in s.items()]
    return table(["layer", "expert vs average: cosine, min / median", "expert vs average: distance, median",
                  "weight that is negative", "J++ map vs average: cosine"], rows)


def weights_table(model_key: str, stem: str) -> str:
    w = json.loads((JPP_OUT / model_key / f"{stem}_weights.json").read_text(encoding="utf-8"))
    rows = []
    for l, v in w["by_layer"].items():
        rows.append([f"L{l}", " ".join(f"{x:+.2f}" for x in v["signed_shares"]),
                     " ".join(f"{x:.2f}" for x in v["single_expert_recall10"]),
                     f"{v['train_macro_recall10']:.2f}", f"{v['init']}@{v['step']}"])
    return table(["layer", "expert weights (signed, sum |w| = 1)", "each expert alone: fit recall@10",
                  "learned: fit recall@10", "winning start@step"], rows)


def main() -> None:
    global USE_N100
    model_key = sys.argv[1]
    USE_N100 = "--n100" in sys.argv
    ev = load(model_key)
    print("## overlap\n" + overlap_table(ev, model_key))
    print("\n## recall\n" + recall_table(ev, model_key))
    print("\n## recall by layer (period)\n" + recall_layer_table(ev, model_key))
    print("\n## recall by layer (word)\n" + recall_layer_table(ev, model_key, "word"))
    print("\n## recall by family (band, period)\n" + recall_family_table(ev, model_key))
    print("\n## fit vs held-out\n" + fit_recall_table(ev, model_key))
    print("\n## pins\n" + pin_table(ev, model_key))
    mid = 6 if model_key == "stories110M" else 3
    early = 2 if model_key == "stories110M" else 1
    for pi in (3, 7, 8):
        print(f"\n## readouts prompt {pi + 1}\n" + readout_rows(ev, model_key, pi, mid))
        print(readout_rows(ev, model_key, pi, early))
    print("\n## lily\n" + lily_summary(ev, model_key))
    print("\n## future late\n" + future_table(ev, model_key))
    print("\n## future early\n" + future_table(ev, model_key, "early"))
    print("\n## stability\n" + stability_table(ev, model_key))
    if (JPP_OUT / model_key / "swaps.json").exists():
        print("\n## swaps\n" + swap_table(model_key))
    stem = "jpp_rlens_skip16_n100" if USE_N100 else "jpp_rlens_skip16_n1000"
    print("\n## weights J++\n" + weights_table(model_key, stem))


if __name__ == "__main__":
    main()
