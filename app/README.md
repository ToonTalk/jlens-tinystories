# J-Lens Explorer — Phase 1 prototype (CPU, shipped lens)

A single-file, zero-dependency, in-browser instrument that lets a middle-schooler
watch what a TinyStories model is "holding in mind" — per token, per layer (L4–L8) —
while it reads or writes a story, and change those contents and see the story bend.
Fourth Tiny Mind app (working name; final name TBD by Ken).

## Running it

Serve this folder over HTTP and open `jlens-explorer.html`, e.g.:

```
python -m http.server 8749 --directory app
# then open http://localhost:8749/jlens-explorer.html and press "fetch files"
```

From `file://` the fetch button can't work — drag the four files onto the drop
zone instead: `stories110M.bin`, `tokenizer.bin`, `stories110M.jlens`, `vocab.json`.

`stories110M.bin` (~420 MB) is Karpathy's fp32 llama2.c checkpoint
([karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)); it is not
committed to the repo. `repro.sh` + `src/step0_app_prep.py` regenerate every other
file here from the fitted lenses.

## What's in the folder

| file | what |
|---|---|
| `jlens-explorer.html` | the app — one file, no build, no CDN; compute in a Web Worker |
| `stories110M.jlens` | J₄…J₈ for stories110M, fp32, 11.8 MB (fit A, 1000 TinyStories prompts) |
| `stories15M.jlens` | J₂…J₃ for stories15M, 0.7 MB (Phase 2; exported now while tooling was open) |
| `vocab.json` | pruned TinyStories vocab (7912 ids, count ≥ 5 in 100 K stories), word-start display mask (5226), name-alias table (182 first-piece entries), theme words (300) |
| `fixtures.json` | golden fixtures: eval prompts 1/4/9, all positions, L4–L8, top-20 ids+scores for BOTH lenses, straight from the Python pipeline |
| `../test/parity.mjs` | Node harness: fixture parity + timing (extracts the worker source from the HTML, so it tests the shipped code path) |
| `../out/nsweep/report.md` | step-0a n-sweep (also summarized below) |

## Acceptance criteria — measured results

**1. Fixture parity — PASS (perfect).** Tokenization matches the HF tokenizer
exactly on all three fixture prompts. Full-vocab top-10 set overlap vs the Python
pipeline = **1.0000 over all 550 (position, layer, lens) cells** (threshold ≥ 0.9;
worst cell 1.0). The pruned display vocab covers 100 % of fixture top-10 tokens.
Run: `node test/parity.mjs`.

**2. Analyze 200 tokens — PASS.** 207-token story, all captures + full readout
precompute for both lenses. Phase 1.5 added a K-worker compute pool
(K = min(6, hardwareConcurrency − 2); weights row-sharded so total memory stays
~1× the model) with layer-batched barriers and a positions×rows batched matmul
kernel — residuals verified **bit-identical** to the single-thread path
(`dev-compare` returns max diff 0):

| where | forward (capture) | readout precompute | analyze total |
|---|---|---|---|
| Chrome, Phase 1.5 pool (K=6) | 5.6–7.6 s | ~5 s | **10.6–12.5 s** |
| Chrome, Phase 1 single thread | 27.0 s | 23.9 s | 50.9 s |
| Node 24, single thread (parity path) | 24.1 s | 19.2 s | 43.2 s |

Dictionary build `D_l = (W_U[pruned]⊙γ)·J_l`: cached in IndexedDB after the
first build (keyed by model+lens+vocab content hashes); a cold build runs in
the pool during the first analyze, and revisits skip it entirely.
Hardware: Intel Xeon W-2223 @ 3.6 GHz, 4 cores / 8 threads. Phase-2 headroom: SIMD/WebGPU.

**3. Readout latency — PASS.** 28–40 ms per token selection (budget < 100 ms);
possible because analyze precomputes both lenses' score tensors over the pruned
vocab (~40 MB for 200 tokens), so selection is a sort, and the heat map a scan (~40 ms).

**4. Swap demo on eval prompt 8 — PASS.** Anchor = the final ` a` of
"…Max dug a", swap ` hole` → ` castle`, layers L4–L8, strength 0.4, greedy
rewrites (temperature 0, same seed both runs — the ONLY difference is the
residual edit). Phase 1.5 verified the rewrites are byte-identical across a
full page reload (deterministic swap, acceptance 1.5-1):

> **before:** Lily and Max went to the beach. Lily built a castle. Max dug a
> **hole**. They were happy. "Look at my castle!" Lily said. "It is big and
> pretty." "Look at my **hole**!" Max said. "It is deep and dark." They wanted to
> see what was in the hole. They went closer. They saw something shiny. […] They
> dug and dug. They found a metal box. It was old and rusty.

> **after:** Lily and Max went to the beach. Lily built a castle. Max dug a
> **moat**. They were having fun. Suddenly, Lily saw a big wave coming. She
> shouted, "Max, we have to go!" Max said, "Wait, I want to finish my **moat**
> first." […]

Note what happened: pushed from *hole* toward *castle* in the middle layers, the
model wrote **moat** — the castle-adjacent thing you dig. The edit lands on the
concept, not the word, and the story stays coherent. (Generation of ~140 tokens
took 40 s — full-vocab sampling per token; fine for watching it write.)

**5. n-sweep (step 0a)** — filtered top-8 agreement with the n=1000 lens at
L4–L8 (12 eval prompts, last position), mean J cosine, fit wall-clock (RTX A4000):

| n | L4 | L5 | L6 | L7 | L8 | mean | J cos | wall-clock |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.28 | 0.36 | 0.44 | 0.54 | 0.72 | **0.47** | 0.892 | 4 s |
| 5 | 0.49 | 0.55 | 0.62 | 0.71 | 0.76 | **0.63** | 0.959 | 16 s |
| 10 | 0.57 | 0.62 | 0.70 | 0.71 | 0.79 | **0.68** | 0.967 | 42 s |
| 25 | 0.58 | 0.65 | 0.67 | 0.72 | 0.80 | **0.68** | 0.971 | 131 s |
| 100 | 0.78 | 0.82 | 0.82 | 0.88 | 0.92 | **0.84** | 0.995 | 515 s |
| 1000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 1.000 | 6327 s |

Read: a 10–25-prompt fit already lands in the right neighborhood (J cosine ≈ 0.97)
but readout top-8s still churn ~30 %; n=100 is where readouts stabilize. For
Phase-2 in-browser fitting, n≈100 on the 15M model (couple of minutes even at
10× WebGPU slowdown) looks right; for 110M, ship the pre-fitted matrices.
Slice pages per n for eval prompts 4 and 9: `out/nsweep/n*_prompt*.html`.

## Implementation notes (for the next phase)

- **Reused from Tiny Mind** byte-for-byte where possible: bin-reader, the
  SentencePiece tokenizer port, the tap-hook Capture pattern. The forward pass
  drops the `Math.fround` C-parity discipline per the brief and uses the
  Trainer's register-blocked matmul (4 output rows, f64 accumulators) — parity
  vs the Python (HF fp32) pipeline is unaffected (see criterion 1).
- **Lens semantics** (from the jlens package source, and what the fixtures encode):
  `J-lens(h) = wcls · (γ ⊙ (J_l·h) / rms(J_l·h))` with `rms(u)=√(mean(u²)+1e-5)`,
  h = layer-OUTPUT residual; logit lens is the same with J = I. The 1/rms scalar
  is rank-neutral and applied only so displayed scores match Python's.
- **Readout dictionaries**: `D_l = (wcls[pruned] ⊙ γ) · J_l` built once per layer
  at load; a readout is then one `D_l·h` matvec. Analyze additionally precomputes
  all (position × layer) scores for both lenses, which is what makes token
  selection, DIFF and the heat strip real-time.
- **Swap** is a residual-edit hook at the layer output (same point the lens
  reads): `h += strength · (h·d̂_src) · (d̂_tgt − d̂_src)` with d = the layer's
  readout direction `(wcls[t]⊙γ)·J_l`, for positions ≥ anchor, layers in range.

## Swap semantics (Phase 1.5)

Three panes: **original** (the analyzed/generated story, verbatim),
**rewrite (no swap)**, and **rewrite (with swap)**. From the anchor token, the
model rewrites the story twice with the same seed and temperature; the ONLY
difference between the two rewrites is the residual edit. Rewrites default to
temperature 0, so running the same swap twice gives byte-identical panes
("use story heat" opts into the sampling temperature). The rewrites regenerate
from the **analyzed token ids** up to the anchor — not re-encoded text — so
token boundaries can't drift from what's on screen. If the swapped word is
already written before the anchor, it stays: the swap changes the model's
*state* from the anchor onward, not text it already wrote. Both rewrite panes
stream tokens live; the swap button is disabled with a spinner while running.

## Background tabs (Phase 1.5)

Compute is timer-free in the worker, so a plainly-hidden desktop tab runs full
speed. The residual risks are tab discard and OS/browser energy modes:

- Progress shows in the **tab title** ("⏳ reading 43% — J-Lens") during
  analyze / generate / dictionary builds.
- The app keeps a foreground tokens/s baseline; if throughput while hidden
  drops > 3× below it, a banner on return says the browser slowed the tab.
  (Test hook: `__forceThroughputBaseline(1e9)` then hide.)
- All four input files AND the readout dictionaries live in IndexedDB, keyed
  by content hashes (~600 MB of origin storage). A revisit or a discarded-tab
  recovery is **interactive in < 1 s** with zero clicks (measured 0.8 s); a
  cold cache defers the dictionary build to the first analyze (in the pool,
  a few seconds) rather than stalling the load path.

## Known issues

- The "raw" toggle is honest about filtering but still shows the *pruned* vocab
  (7912 ids), not all 32 000; the full-vocab path exists in the worker
  (`readoutFull`, used by the parity tests) and covered 100 % of fixture
  top-10s, so the practical difference on story text is nil.
- Swap source/target must be single-token words from the story vocabulary (the
  UI guards and suggests via the datalist; multi-token words like "treasure"
  are rejected rather than silently mis-tokenized).
- stories15M.jlens ships, but the app's file map is hardwired to the 110M
  filenames — a dropped 15M bundle is currently ignored (Phase 2).
- Generation runs ~3.5 tokens/s (full-vocab unembed per sampled token,
  sequential — the pool accelerates analyze and the post-generation readout
  precompute, not the sampling loop).
- The compute pool relies on nested Workers (Chrome/Firefox/Safari 15+ OK);
  where unavailable the app falls back to the single-threaded path.
- IndexedDB caching stores ~600 MB per origin (model + dictionaries); no
  eviction UI yet — clear site data to reclaim.
- No mobile layout, no voice guide (explicitly out of Phase-1 scope).
