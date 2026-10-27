The app band is L4–L8; all layers are shown so the "J++ helps early layers" claim can be checked.

### Step 1: readout filtering alone (no refit)

Comparing the J and J+filter rows below: the filter leaves the logit-lens overlap within ±0.01 and leaves last-word recall and the Lily ranks unchanged. It lifts period-readout recall in the band from 52% to 56%, mostly from "hide" (79% → 95%), where newline and quote tokens otherwise crowd the top of the ranking right after a period. **It changes nothing in the app's normal view**: the app already displays only word-start story tokens (`vocab.json` `wordStartIds`), a strict subset of what the filter keeps. The only view it would touch is the "raw" toggle, which exists to show everything, punctuation included. Adding the filter there would undo the toggle's purpose. Verdict on the filter alone: free, harmless, and already covered by the app's display.

### Held-out recall@10

{{recall:stories110M}}

{{recall_layer:stories110M:period}}

*Per-layer held-out recall at the period.* This is where J++ earns its keep. Over the L4–L8 band it reaches 64% at the period vs J's 52% (56% with the filter), and 37% vs 31% at the last word. The answer also surfaces a layer earlier: 15% at L5, where J has 0%. That matches the paper's headline pattern. **Below L4 nothing changes**: J, LRP-only and J++ are all at 0–1% at L1–L3 (the no-LRP ablation reaches 7%), so J++ does not widen the band downward. Most of the gain is one family (feeling: 38% → 78%), with smaller gains on names and riddles and a loss on "need":

{{recall_family:stories110M:period}}

{{fit_recall:stories110M}}

*Fit split vs held-out (band).* The held-out split scores higher than the fit split for every lens. Its templates are simply easier, so the split is not leaking.

### Overlap with the logit lens

{{overlap:stories110M}}

J++'s overlap jumps around from layer to layer: 0.00 at L0–L2, 0.34 at L3, **0.01 at L4**, then 0.13–0.38. The L4 value is the real problem, because L4 is the bottom of the app band:

### What a child sees (pins, legibility)

{{pins:stories110M}}

*"Headline" = the out/report.md examples: `hid/hide/hiding` (prompt 4), `shadow/investigate/secret/noise/sound` (9), `dig/buried/pit/mole/deeper` (8), `chose/choose/Which/what` (5), `Finally/persistent/success/mistakes` (6), counted over the band's top-8s.* J++ keeps all five pins, but with 13 headline words instead of 22, and 12% of its band display slots go to prompt-independent tokens (J: 0%). The worst of those are at L4, where J++ shows **the same list on every prompt**: `Nelson, Ny, Kent, Aless, Stein, â, Rena, Kin`. These are out-of-register name fragments. Three of the app's starter prompts, at the last position:

Prompt 4, "Ben broke his mom's favorite vase. He heard her coming.", at L4:

{{readout:stories110M:4@4}}

Prompt 9, "It was a dark night. Tim heard a strange noise in the garden.", at L6 and L8:

{{readout:stories110M:9@6}}

{{readout:stories110M:9@8}}

Prompt 6, "The little bird could not fly. Every day it tried and tried.", at L7:

{{readout:stories110M:6@7}}

At L5–L8, J++ is sometimes *nicer*: the persistent bird gets `effort, struggle, eventually, determined`, and the dark night gets `curious, brave, wondering, fear`. But it also loses the starter prompt's signature `shadow / investigate / secret` and puts `an, two, a, was, on, went` on "One day" (prompt 1, L6). Early layers stay unreadable for every lens (prompt 8, "…Lily built a castle. Max dug a", at L2):

{{readout:stories110M:8@2}}

### Lily first-piece check

{{lily:stories110M}}

J++ is better here: ` L` ranks in the top 10 at 8 of the 11 Lily positions (J: 5), with median rank 5 vs 10. Its "name" family recall is also higher (72% vs 62%).

### Swap demo and battery

{{swaps_band:stories110M}}

*First word after the anchor at the app band (L4–L8), strength 0.4, greedy. "=" means it is the put-in word, "≠" means it is no longer the take-out word, "·" means unchanged.* **Every lens reproduces the README demo**, "Max dug a **moat**". The no-swap and J texts match the README word for word, which validates this port of the app's swap. Over the battery, J and LRP-only switch 5 of 7 (3 exact); J++ switches 4 (2 exact; it loses tail → head), and J++ mixed 3. The full table with the wider L2–L8 range is in `out/jpp/stories110M/swaps.json`. It doesn't change the picture.

### Stability, n=100 vs n=1000

{{stability:stories110M}}

J's display is 84% the same whether fitted on 100 or 1000 prompts, LRP-only's 94%. **J++'s is 54%, and 0% at L4.** The 8 learned weights per layer change from fit to fit (cosine 0.66 between the two maps), so the words a child sees depend on fitting noise. For the planned in-browser fitting that matters a lot.

### Open-ended, label-free check: future-word precision

Share (%) of the app-display top-10 that the story actually says 2–21 tokens later, novel words only, on 60 held-out validation stories, positions ≥ 16:

{{future:stories110M:late}}

At positions 1–15:

{{future:stories110M:early}}

With no target word at all, the lenses are close at L5–L9. J++ is a little ahead at L6–L7 (2.1 vs 1.7, 2.3 vs 2.1) and level at L8. It is far behind at L4 (0.2 vs 0.9) and at L0–L2. LRP-only is the steadiest here, ahead of J at every layer up to L8.
