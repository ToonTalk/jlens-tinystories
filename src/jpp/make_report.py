"""Assemble out/jpp/report.md from src/jpp/report_template.md.

The template holds the prose; every {{name:model[:arg]}} placeholder is
replaced by a table from report_tables.py built from the JSON results, so the
numbers in the tables always match the data on disk.

Usage: python src/jpp/make_report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_tables as rt  # noqa: E402
from jpp_common import JPP_OUT  # noqa: E402

HERE = Path(__file__).resolve().parent


def render(name: str, model: str, arg: str | None) -> str:
    ev = rt.load(model) if name not in ("swaps", "swaps_band", "weights", "cosine", "spread") else None
    if name == "cosine":
        return rt.cosine_table(model)
    if name == "spread":
        return rt.spread_table(model)
    if name == "overlap":
        return rt.overlap_table(ev, model)
    if name == "recall":
        return rt.recall_table(ev, model)
    if name == "recall_layer":
        return rt.recall_layer_table(ev, model, arg or "period")
    if name == "recall_family":
        return rt.recall_family_table(ev, model, arg or "period")
    if name == "fit_recall":
        return rt.fit_recall_table(ev, model)
    if name == "pins":
        return rt.pin_table(ev, model)
    if name == "readout":
        pi, layer = (int(x) for x in arg.split("@"))
        return rt.readout_rows(ev, model, pi - 1, layer)
    if name == "lily":
        return rt.lily_summary(ev, model)
    if name == "lily_positions":
        return rt.lily_table(ev, model, int(arg) - 1)
    if name == "future":
        return rt.future_table(ev, model, arg or "late")
    if name == "stability":
        return rt.stability_table(ev, model)
    if name == "swaps":
        return rt.swap_table(model)
    if name == "swaps_band":
        return rt.swap_table(model, band_only=True)
    if name == "weights":
        return rt.weights_table(model, arg)
    raise KeyError(name)


def main() -> None:
    text = (HERE / "report_template.md").read_text(encoding="utf-8")
    # __NAME__ markers pull in src/jpp/report_name.md (prose sections kept apart)
    text = re.sub(
        r"__([A-Z0-9]+)__",
        lambda m: (HERE / f"report_{m.group(1).lower()}.md").read_text(encoding="utf-8").strip(),
        text,
    )
    # a placeholder indented inside a list item gets every line of its table indented
    text = re.sub(
        r"(?m)^([ \t]+)\{\{(\w+):(stories\w+)(?::([^}]+))?\}\}",
        lambda m: "\n".join(m.group(1) + line for line in render(m.group(2), m.group(3), m.group(4)).split("\n")),
        text,
    )
    out = re.sub(
        r"\{\{(\w+):(stories\w+)(?::([^}]+))?\}\}",
        lambda m: render(m.group(1), m.group(2), m.group(3)),
        text,
    )
    (JPP_OUT / "report.md").write_text(out, encoding="utf-8")
    print(f"-> {JPP_OUT / 'report.md'} ({len(out)} chars)")


if __name__ == "__main__":
    main()
