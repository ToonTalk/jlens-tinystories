# J++ vs the J-lens on the TinyStories models

**Question (Ken, via the Thought Bubble session):** can J++ (Ayonrinde & Lindsey, 2026, a "Jacobian filtering" upgrade of the J-lens) be an option in the Thought Bubble app, or replace the J-lens if it is better? And does it make stories15M good enough for a thought bubble in Tiny Mind's Microscope?

**Short answer: not worth it, for either model.** The shipped J-lens stays. J++ was fitted exactly as the paper describes, with TinyStories in place of WikiText and a hand-built TinyStories dev set for its weights. At 110M it does raise target-word recall in the app band (+8 to +12 points) and reads names better. But it breaks the bottom band layer (L4 shows the same off-register names on every prompt), gains nothing below it, shows fewer of the app's signature readouts, is no better at swaps, and changes half its display between two fits. At 15M it makes things worse, so it does not open the door to a Microscope thought bubble. Offering it as a toggle would add exactly the kind of unexplained part Ken's design rule rules out. No `.jlens` export is proposed.

__VERDICT__

## What was compared

| lens | what it is |
|---|---|
| **J** | the shipped J-lens: `out/lenses/{A,C}/lens_1000.pt` = `app/stories110M.jlens` / `app/stories15M.jlens` (1000 TinyStories prompts, plain gradients) |
| **J+filter** | the same J with J++'s *Readout Filtering*: tokens with no letter or digit are left out of the ranking (1,738 of the 32,000 Llama-2 ids) |
| **LRP-only** | one Jacobian per layer averaged over the same 1000 prompts, but through the LRP backward graph (J++'s second ingredient; this is the paper's "R-Lens"), + filter |
| **J++** | the full recipe: 8 expert Jacobians per layer (k-means clusters of the activations), through LRP, combined with 8 learned signed weights per layer, + filter |
| **J++ mixed** | *my variant, not the paper's*: the same experts, weights learned on readouts at the period AND at the sentence's last word (see "the dev set" for why) |
| no-LRP exp. | ablation: expert weighting with plain gradients (15M at n=1000; 110M at n=100 only, to save GPU time) |
| logit | the plain logit lens, as a floor |

**Recipe, as implemented** (the `jpp_lens` library itself, cloned to `C:\Users\toont\Documents\GitHub\jpp_lens` with its own venv: torch 2.11+cu128, transformers 5.14.1; the app's `.venv` is untouched):
- *Router*, per layer: unit-normalize, center, PCA to 64 dims, k-means with K = 8 (k-means++, seed 0) on the valid positions (positions 0–15 and the last are skipped) of the same 1000 TinyStories prompts fit A / fit C used.
- *Experts*: one Jacobian per cluster through J++'s LRP surgery: the LN rule on every residual RMSNorm, identity + half rules on the SwiGLU MLP, plain gradients through attention. Target = final block output; all source layers (110M L0–L10, 15M L0–L4). The surgery refuses unknown classes, so I registered `LlamaRMSNorm` (its forward is the library's Qwen2 form) and `LlamaMLP`, after asserting their source. The patched forward matches stock to 2e-6 (15M) and 1e-5 (110M). No expert fell under the 50-position fallback.
- *Weights*: J++'s own fitter (Adam, lr 0.05, 200 steps from 3 starts, smooth surrogate of each layer's macro recall@10, readout filter on), run on the dev set's fit split.
- *Parity check*: the J++ pipeline with plain gradients and the clusters pooled reproduces the shipped 15M J to a relative error of 5e-7 at every layer. The port is faithful, so the differences below come from the method, not from plumbing.

**Cost.** 15M: 450 s for the 1000-prompt LRP expert fit (the shipped J took 278 s), plus 35 s for the router and ~30 s for the weights. 110M: ~6.0 s per prompt for the LRP expert fit when the GPU is not shared (100-prompt shards took 597–629 s; the whole 1000-prompt run took 115 min of wall-clock alongside other jobs), vs 6.3 s per prompt for the shipped J. Add 85 s for the router and 1–2 min for the weights. Inference cost is unchanged, since J++ is still one 768×768 map per layer, the same `.jlens` shape.

stories110M:

{{cosine:stories110M}}

stories15M:

{{cosine:stories15M}}

*Cosine between each lens's map and the shipped J's, per layer (n=100 rows compare with the shipped n=100 fit).* LRP-only is a mild rotation of J. J++ is a different map wherever its weights had little to fit: anti-aligned with J at 110M L0–L2, orthogonal at L4 (see "Why"), and nearly orthogonal at most 15M layers.

## The TinyStories dev set (step 2)

J++ learns its weights from labeled "latent-variable" items. The paper's items (multi-hop facts, typos, translations, poetry) are beyond a TinyStories model, so I built story-register items with a known word the model is disposed to say **soon but not next**, the `hid / hide before mom arrives` pattern from `out/report.md`. 338 hand-written items from 54 templates × name fillers, in 5 families (`src/jpp/devset.py`, `out/jpp/devset/items.json`):

| family | example (readout at the final period) | answer words |
|---|---|---|
| need | "Tom wanted to play in the snow, but it was very cold outside." | coat, hat |
| feeling | "Kim lost her favorite teddy bear at the park. She looked everywhere, but she could not find it." | sad, cry, cried |
| name | "Long ago, a little girl called Jane lived in a small house near the woods. She liked to pick berries." | Jane |
| riddle | "Andy heard a tiny squeak and saw a little gray animal with a long tail run under the table." | mouse |
| hide | "Amy broke her dad's favorite cup. Then she heard her dad coming." | hide, hid, hiding |

- **Correctness filter**, per model (the paper scores only items the model gets right): an item counts if the model's own greedy continuation says an answer word among tokens 2–48 but **not** as the very next token, so the items test held-in-mind content rather than next-token guessing. Answers are single tokens (multi-token words like "umbrella" or "scared" were left out, as in the paper). 110M keeps 199 of 338 items; 15M keeps 161.
- **Split by template**, never by filler, so no sentence frame is shared between fitting and scoring: ~30% of the need/feeling/name/riddle templates fit the weights (110M: 52 items, 15M: 39). Everything else is held out (110M: 147, 15M: 122). **All of "hide" is held out**, because it is the concept of pinned eval prompt 4 ("Ben broke his mom's favorite vase…"). That keeps the pin honest and doubles as a leave-one-task-out test.
- **Disjointness**: no item shares a 6-word run with any of the 12 eval prompts (asserted in code; this caught and removed "Once upon a time, there was a little…"). The items are hand-written, so none overlaps the pinned TinyStories fit corpora (`out/pins.json`, `out/corpora/`).
- **Second readout position.** Every dev prompt ends in a period, so every readout the recipe sees is a period residual, and the router puts all of them in **one cluster per layer** (checked: all 39 fit readouts on 15M land in one cluster at each layer). Kids click any token. So I also read out one token earlier, at the sentence's last word, which is valid when the model's greedy next token there is that period (110M: 155 items, 15M: 141). That gives "word" recall below, and the "J++ mixed" weights.

## Results: stories110M

__R110__

## Results: stories15M

__R15__

## Does J++ make stories15M useful? (step 4)

__STEP4__

## Side question: skipping the first 16 positions

The paper (and jlens) drop positions 0–15 from every fit as attention sinks. The fit prompts are 128 tokens, so this drops 15 of 126 usable positions. The app, though, reads out at every position, and children's prompts are short. Residual norms by position, mean over 200 fit prompts (`out/jpp/*/position_profile.json`):

| model, layer | pos 0 (BOS) | pos 1–3 | pos 4–15 | pos ≥ 16 |
|---|---|---|---|---|
| 110M L2 | 120 | 51, 48, 85 | 19.5 | 12.0 |
| 110M L6 | 177 | 85, 82, 131 | 40.6 | 26.7 |
| 110M L10 | 150 | 81, 89, 121 | 66.8 | 58.5 |
| 15M L2 | 17.9 | 4.2, 3.2, 3.1 | 3.6 | 3.0 |
| 15M L4 | 21.5 | 7.4, 6.7, 7.2 | 8.1 | 7.2 |

So 110M does have sink-like early positions: positions 1–3 run 3–6× the norm of positions ≥16 through the middle layers, and positions 4–15 about 1.5×. Fit prompts nearly always open "Once upon a…", so part of this may be those tokens rather than their positions. 15M has only BOS. **The skip still doesn't matter.** Refitting the plain J with only BOS skipped:
- stories110M (100 prompts, vs the shipped n=100 fit): cosine 0.99–1.00 at every layer. Period recall in the band 51% vs 52%, last word 33% vs 39% (about 7 items), headline words 18 vs 18. Future-word precision at positions 1–15 rises by at most 0.2 points (L6–L7).
- stories15M (1000 prompts): cosine 0.987–0.996. Recall, overlap and readouts are unchanged within noise. At positions 1–15, future-word precision is not better (L3: 0.7 vs 0.8).

The fit is dominated by the ~110 later positions either way, and a J fitted on early positions does not read early positions noticeably better. Keep the default.

## Why J++ does not transfer to these models

__WHY__

## Caveats

- The dev set is small (39–52 fit items, 122–147 held-out) and hand-built. Differences of a few points in recall are within noise: one item is 0.7–0.8 points, and family macro-averaging magnifies small families. The verdict rests on effects that are large, or consistent across metrics.
- "Soon but not next" is a behavioral label: the model *said* the word within 48 tokens. That matches the J-lens's own promise, but it is not the paper's latent-variable ground truth.
- Legibility proxies (prompt-independent share, raw-top-10 word share, the headline pins) stand in for a human look. The readout tables are there for that look.
- The 110M no-LRP ablation ran at n=100 only.
- J++ was evaluated as published. With a larger, position-diverse dev set, or with non-negative weights, it might behave differently, though a non-negative mix cannot move far from LRP-only here ("Why", point 1). "J++ mixed" is one step in that direction, and it helped some metrics but did not change the verdict.

## Files

- Code: `src/jpp/` (`devset.py`, `verify_devset.py`, `add_word_positions.py`, `fit_experts.py`, `fit_weights.py`, `evaluate_jpp.py`, `swap_demo.py`, `skip_check.py`, `export_jlens.py`, `report_tables.py`, `make_report.py`; runners `run_fits.sh`, `finish.sh`). Run with `C:\Users\toont\Documents\GitHub\jpp_lens\.venv\Scripts\python.exe` and `PYTHONIOENCODING=utf-8`.
- Results: `out/jpp/devset/` (items, per-model correctness), `out/jpp/<model>/eval.json` (every metric, ranks and readouts), `swaps.json`, the `*_weights.json` files (learned expert weights), `router_*.json` (cluster sizes), `fit_*.log`, and the fitted lenses as fp32 `.pt` (`pooled_rlens_*` = LRP-only, `jpp_*` = J++).
- Size: `out/jpp/` is 247 MB, almost all of it the nine 25 MB fp32 110M lens files. They are regenerable from the scripts (about 2 GPU-hours), so Ken may prefer not to commit them. Everything else in `out/jpp/` and `src/jpp/` is small.
- Large intermediates (per-shard expert sums, expert Jacobians, 100–200 MB each, 3 GB in all) are in the jpp_lens clone's gitignored `artifacts/tinystories/`, not in this repo.
- Nothing in `app/` was changed. No `.jlens` was exported, because J++ did not win (`src/jpp/export_jlens.py` is ready if Ken wants one anyway).
