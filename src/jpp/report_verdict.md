## Verdict

### stories110M: keep the J-lens. J++ is not worth it, not even as an option.

What J++ gets right, measured on held-out TinyStories items:
- **Recall goes up in the app band (L4–L8).** At the period: 64% vs J's 52% (56% with the filter alone). At the last word: 37% vs 31%. The answer also appears a layer earlier (L5: 15% vs 0%), the pattern the paper reports.
- **Names read better**: ` L` (Lily's first piece) is in the top 10 at 8 of 11 Lily positions, vs 5.
- Some L5–L8 bubbles are richer, e.g. the persistent bird: `effort, struggle, eventually, determined`.

Why it still doesn't earn a place:
1. **The bottom of the band breaks.** At L4, J++ shows the same off-register list on every prompt (`Nelson, Ny, Kent, Aless, Stein…`). Its overlap with the logit lens there is 0.01 and its label-free precision collapses (0.2 vs 0.9). A J++ app would have to drop L4, so the band *shrinks* rather than widens.
2. **Nothing gains below L4.** J, LRP-only and J++ are all at 0–1% recall at L1–L3, and the bubbles there are junk for J and J++ alike. The paper's early-layer gains need early layers that hold something.
3. **Less of what makes the bubble worth showing.** J++ shows 13 of the headline words (J: 22). It drops `shadow / investigate / secret` from the dark-night starter prompt, and prompt-independent tokens fill 12% of its band display (J: 0%).
4. **It is unstable.** Refitting on 100 instead of 1000 prompts changes 46% of the band display (J: 16%; 100% at L4). Which words a child sees would depend on fitting noise. That also rules out in-browser fitting.
5. **Swap is no better.** Every lens writes "Max dug a **moat**". Over 7 swaps J++ switches 4, J 5.
6. **Its gains are on its own exam.** The improvements are in the dev-set format the weights were trained on (sentence-final readouts with target words). The label-free check on real stories shows J++ level with or slightly ahead of J at L5–L9 (L6: 2.1% vs 1.7%; L8: 2.4% vs 2.5%), and far worse at L4. That is a modest edge, not one that outweighs points 1–4.

**As an option, no.** A J/J++ toggle would show children two different, partly contradictory bubbles, one of them less stable, which is exactly the unexplained part Ken's design rule rules out. If one lens has to go, it's J++.

### stories15M: not worth it, and it doesn't open the door for Tiny Mind's Microscope.

J++ lifts period recall from 5% to 14%, but the level stays tiny, and at the last word it barely beats the logit lens (23% vs 21%). Its bubbles fill with the same `the / was / sad / did` on every prompt. Label-free precision halves at L3–L4, and its display changes 73% between a 100- and a 1000-prompt fit. Swaps do nothing for any lens at the app band. **J++ makes 15M less useful, not more.** So the Microscope decision rests where `out/report.md` left it: if Tiny Mind gets a thought bubble on stories15M, it would be the plain J at L2–L3, with surface-level content.

### The pieces separately

- **Readout Filtering alone:** +4 points of period recall in full-vocab ranking. It is already covered, more strictly, by the app's word-only display, so there is nothing to ship.
- **LRP alone (the R-Lens):** the most stable lens (94% n=100/1000 agreement) and slightly better label-free precision. But it is closer to the logit lens (L6 overlap 0.43 vs 0.29), so the J-only "hidden thought" words thin out. Recall falls 52% → 34%, and headline words fall 22 → 10. It's not a better Thought Bubble.
- **Expert weighting** is where J++'s recall gain comes from, and it is also where the instability and the junk layers come from.

**No `.jlens` export**, since J++ did not win. `src/jpp/export_jlens.py` writes the app's exact format (and round-trips it) if Ken wants to try one. Nothing in `app/` was touched.

**If someone revisits this later**, the promising piece is J++'s weighting at L5–L8 only, re-fitted with a larger dev set whose readouts sit at varied tokens, plus non-negative or regularized weights. Non-negative weights would stop the cancellation behind the junk layers, but because the experts are near copies of their average ("Why", point 1), such a lens could not move far from LRP-only. "J++ mixed" (weights also fitted at last-word readouts) is a small step that way: steadier at the bottom of the band and better on names, but not enough to change this verdict.
