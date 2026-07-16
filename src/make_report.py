"""Step 4: assemble out/report.md + out/appendix.md from the eval artifacts.

All tables are computed from out/eval/*/readouts.json, out/eval/divergence.json,
out/lenses/*/config_*.json, out/fluency.json. Analysis sentences are factual
summaries of the numbers; the human evaluates legibility.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import EVAL_PROMPTS, FITS, OUT

# Layer bands (fitted source layers; final layer excluded — it IS the output).
BANDS = {
    "stories110M": {"early": [1, 2, 3], "mid": [4, 5, 6, 7, 8], "late": [9, 10]},
    "stories15M": {"early": [1], "mid": [2, 3], "late": [4]},
}
# Representative layer per band for the top-8 tables.
BAND_REP = {
    "stories110M": {"early": 2, "mid": 6, "late": 9},
    "stories15M": {"early": 1, "mid": 3, "late": 4},
}


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_toks(toks):
    return ", ".join(f"`{t}`" for t in toks) if toks else "(none)"


def diff_col(j_filt, l_filt, l_raw):
    out = []
    for t in j_filt:
        if t not in l_filt:
            out.append(t + ("*" if t not in l_raw else ""))
    return out


def divergence_png(divergence, readouts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for fit_id, div in sorted(divergence.items()):
        model_key = FITS[fit_id][0]
        layers = sorted(int(l) for l in div)
        frac = [l / (readouts[fit_id]["n_layers"] - 1) for l in layers]
        ov = [div[str(l)]["top10_overlap_mean"] for l in layers]
        rho = [div[str(l)]["spearman_mean"] for l in layers]
        label = f"{fit_id}: {model_key}/{FITS[fit_id][1]}"
        axes[0].plot(frac, ov, marker="o", label=label)
        axes[1].plot(frac, rho, marker="o", label=label)
    axes[0].set_ylabel("mean top-10 overlap (J-lens vs logit lens)")
    axes[1].set_ylabel("mean Spearman rho (full vocab)")
    for ax in axes:
        ax.set_xlabel("layer / (n_layers - 1)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("J-lens vs logit-lens divergence by layer (12 prompts, all positions)")
    fig.tight_layout()
    path = OUT / "eval" / "divergence.png"
    fig.savefig(path, dpi=140)
    return path


def lily_section(readouts, fit_id):
    """Rank trajectories of ' L' and 'ily' on prompts 1 and 8 (index 0, 7)."""
    r = readouts[fit_id]
    model_key = r["model_key"]
    mid = BANDS[model_key]["mid"]
    src = r["source_layers"]
    mid_idx = [src.index(l) for l in mid if l in src]
    lines = []
    for pi in (0, 7):
        p = r["prompts"][pi]
        lines.append(f"\n**Prompt {pi + 1}:** \"{p['prompt']}\"\n")
        lines.append("| pos | token | ` L` J rank | ` L` logit rank | `ily` J rank | `ily` logit rank |")
        lines.append("|---|---|---|---|---|---|")
        tl = p["tracked_ranks"][" L"]
        ti = p["tracked_ranks"]["ily"]
        for pos in range(1, p["n_tokens"]):
            jl_l = min(tl["jlens"][pos][i] for i in mid_idx)
            lg_l = min(tl["logit"][pos][i] for i in mid_idx)
            jl_i = min(ti["jlens"][pos][i] for i in mid_idx)
            lg_i = min(ti["logit"][pos][i] for i in mid_idx)
            tok = p["context_tokens"][pos].replace("|", "\\|")
            hot = " <-" if (jl_l <= 10 or jl_i <= 10) else ""
            lines.append(
                f"| {pos} | `{tok}` | {jl_l} | {lg_l} | {jl_i} | {lg_i} |{hot}"
            )
        lines.append("")
        lines.append(
            "(Rank = full-vocab rank, 0 = top; min over mid layers "
            f"L{mid[0]}-L{mid[-1]}; `<-` marks positions where the J-lens puts "
            "a Lily fragment in its top-10.)"
        )
    return "\n".join(lines)


def main() -> None:
    fluency = load_json(OUT / "fluency.json")
    divergence = load_json(OUT / "eval" / "divergence.json")
    readouts = {
        fid: load_json(OUT / "eval" / fid / "readouts.json")
        for fid in FITS
        if (OUT / "eval" / fid / "readouts.json").exists()
    }
    configs = {}
    for fid in FITS:
        for n in (100, 1000):
            p = OUT / "lenses" / fid / f"config_{n}.json"
            if p.exists():
                configs[(fid, n)] = load_json(p)

    png = divergence_png(divergence, readouts)

    md = []
    md.append("# J-lens on TinyStories models: feasibility report\n")
    md.append(
        "Question: does a 12-layer whole-word-register TinyStories model "
        "(stories110M) produce mid-layer J-lens readouts that are legible and "
        "not reproducible by a plain logit lens? Secondary: stories15M gap, "
        "and TinyStories-fit vs WikiText-fit.\n"
    )

    # --- setup + fluency ---
    md.append("## Setup and fluency gate\n")
    for key, r in fluency.items():
        info = r["info"]
        md.append(
            f"- **{key}** = `{info['repo']}` @ `{info['revision'][:10]}`, "
            f"{info['n_layers']} layers, d_model {info['d_model']}, fp32 on "
            f"{info['gpu'] or 'CPU'}; torch {info['torch']}, "
            f"transformers {info['transformers']}."
        )
    md.append("\nGreedy completions of \"Once upon a time\" (gate: fluent TinyStories prose):\n")
    for key, r in fluency.items():
        md.append(f"**{key}:**\n\n> {r['greedy_completion']}\n")

    # --- wall-clock ---
    md.append("## Fitting wall-clock\n")
    md.append("| fit | model | corpus | n_prompts | wall-clock | s/prompt | hardware |")
    md.append("|---|---|---|---|---|---|---|")
    for (fid, n), c in sorted(configs.items()):
        md.append(
            f"| {fid} | {c['model']['model_key']} | {c['corpus']} | "
            f"{c['n_prompts_used']} | {c['wall_clock_s']:.0f} s | "
            f"{c['s_per_prompt']} | {c['hardware']} |"
        )
    md.append("")

    # --- divergence ---
    md.append("## Where the J-lens diverges from the logit lens\n")
    md.append(f"![divergence]({png.relative_to(OUT).as_posix()})\n")
    for fid, r in readouts.items():
        div = divergence[fid]
        layers = sorted(int(l) for l in div)
        md.append(
            f"**Fit {fid}** ({r['model_key']}, {r['fit_corpus']}): "
            + "; ".join(
                f"L{l}: overlap {div[str(l)]['top10_overlap_mean']:.2f} / "
                f"rho {div[str(l)]['spearman_mean']:.2f}"
                for l in layers
            )
        )
        md.append("")

    # --- per prompt x band tables ---
    for fid, r in readouts.items():
        model_key = r["model_key"]
        md.append(
            f"\n## Fit {fid} ({model_key}, fit on {r['fit_corpus']}, "
            f"{r['lens_file']}): top-8 readouts at the last position\n"
        )
        for pi, p in enumerate(r["prompts"]):
            md.append(f"\n### Prompt {pi + 1}: \"{p['prompt']}\"\n")
            md.append(
                f"Model's actual next-token top-5: {fmt_toks(p['model_top5_last'])}\n"
            )
            md.append("| band (layer) | J-lens top-8 (filtered) | logit-lens top-8 (filtered) | DIFF (J-only) |")
            md.append("|---|---|---|---|")
            for band, rep in BAND_REP[model_key].items():
                lay = p["layers"][str(rep)]
                j_f = lay["jlens_filtered_top8_last"]
                l_f = lay["logit_filtered_top8_last"]
                d = diff_col(j_f, l_f, lay["logit_raw_top20_last"])
                md.append(
                    f"| {band} (L{rep}) | {fmt_toks(j_f)} | {fmt_toks(l_f)} | {fmt_toks(d)} |"
                )
            md.append("")
    md.append(
        "\nDIFF = tokens in the J-lens filtered top-8 that are absent from the "
        "logit-lens filtered top-8; `*` = also absent from the logit lens's raw "
        "top-20.\n"
    )

    # --- A vs B ---
    if "A" in readouts and "B" in readouts:
        md.append("\n## A vs B: TinyStories fit vs WikiText fit (same model)\n")
        mid_rep = BAND_REP["stories110M"]["mid"]
        scored = []
        for pi in range(len(EVAL_PROMPTS)):
            a = readouts["A"]["prompts"][pi]["layers"][str(mid_rep)]["jlens_filtered_top8_last"]
            b = readouts["B"]["prompts"][pi]["layers"][str(mid_rep)]["jlens_filtered_top8_last"]
            scored.append((len(set(a) & set(b)), pi))
        picks = [pi for _, pi in sorted(scored)[:3]]
        for pi in sorted(picks):
            pa = readouts["A"]["prompts"][pi]
            pb = readouts["B"]["prompts"][pi]
            md.append(f"\n**Prompt {pi + 1}:** \"{pa['prompt']}\" (mid band, L{mid_rep})\n")
            md.append("| lens | J-lens top-8 (filtered) |")
            md.append("|---|---|")
            md.append(f"| A (TinyStories fit) | {fmt_toks(pa['layers'][str(mid_rep)]['jlens_filtered_top8_last'])} |")
            md.append(f"| B (WikiText fit) | {fmt_toks(pb['layers'][str(mid_rep)]['jlens_filtered_top8_last'])} |")
            md.append("")

    # --- A vs C ---
    if "A" in readouts and "C" in readouts:
        md.append("\n## A vs C: 110M vs 15M (both TinyStories fit)\n")
        rep_a = BAND_REP["stories110M"]["mid"]
        rep_c = BAND_REP["stories15M"]["mid"]
        for pi in (0, 7, 10):
            pa = readouts["A"]["prompts"][pi]
            pc = readouts["C"]["prompts"][pi]
            md.append(f"\n**Prompt {pi + 1}:** \"{pa['prompt']}\"\n")
            md.append("| model | mid-band J-lens top-8 (filtered) |")
            md.append("|---|---|")
            md.append(f"| stories110M (L{rep_a}) | {fmt_toks(pa['layers'][str(rep_a)]['jlens_filtered_top8_last'])} |")
            md.append(f"| stories15M (L{rep_c}) | {fmt_toks(pc['layers'][str(rep_c)]['jlens_filtered_top8_last'])} |")
            md.append("")

    # --- Lily check ---
    if "A" in readouts:
        md.append("\n## The Lily check (fit A, mid layers)\n")
        md.append(lily_section(readouts, "A"))
        if "C" in readouts:
            md.append("\n### Same check on stories15M (fit C)\n")
            md.append(lily_section(readouts, "C"))

    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")

    # --- appendix: raw top-20, unfiltered ---
    ap = ["# Appendix: unfiltered raw top-20 readouts (last position)\n"]
    for fid, r in readouts.items():
        ap.append(f"\n## Fit {fid} ({r['model_key']}, {r['fit_corpus']})\n")
        for pi, p in enumerate(r["prompts"]):
            ap.append(f"\n### Prompt {pi + 1}: \"{p['prompt']}\"\n")
            for layer in r["source_layers"]:
                lay = p["layers"][str(layer)]
                ap.append(f"- **L{layer} J-lens:** {fmt_toks(lay['jlens_raw_top20_last'])}")
                ap.append(f"- **L{layer} logit:** {fmt_toks(lay['logit_raw_top20_last'])}")
            ap.append("")
    (OUT / "appendix.md").write_text("\n".join(ap), encoding="utf-8")
    print(f"Wrote {OUT / 'report.md'} and {OUT / 'appendix.md'}")


if __name__ == "__main__":
    main()
