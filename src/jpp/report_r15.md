The app band for 15M is L2–L3 (`out/report.md`); all five fitted layers (L0–L4) are shown.

**Held-out recall@10** (the target word in the top 10 at any layer of the set; macro over families):

{{recall:stories15M}}

{{recall_layer:stories15M:period}}

*Per-layer held-out recall at the period.* J++ does lift period recall: 5% → 14% in the band and 5% → 18% over all layers, with a little at L0–L1, where J has nothing. But the absolute level stays low. At the last-word position over all layers, J++'s 23% is barely above the **logit lens's 21%** (J: 17%), so at 15M none of these lenses reads much that the logit lens doesn't. Almost all of J++'s period gain is one family (feeling: 23% → 50%); hide, name and riddle stay at 0% for every lens:

{{recall_family:stories15M:period}}

**Overlap with the logit lens** (top-10 sets over every position of the 12 eval prompts; filtered variants are compared with the filtered logit lens):

{{overlap:stories15M}}

J++ drops to 0.01–0.17. That is not "more hidden content": the readouts become a different and mostly prompt-independent list:

**Legibility (app display):**

{{pins:stories15M}}

*The headline words are the 110M report's, so this column means little at 15M. The prompt-independent share is the useful one.* What a child would see at L3 (the band's upper layer). Prompt 4, "Ben broke his mom's favorite vase. He heard her coming.":

{{readout:stories15M:4@3}}

Prompt 8, "…Lily built a castle. Max dug a":

{{readout:stories15M:8@3}}

Prompt 9, "It was a dark night. Tim heard a strange noise in the garden.":

{{readout:stories15M:9@3}}

Across all 12 prompts, J++ at L3 is led by `the / was / sad / did / food` regardless of the story. The shipped J at L2 still shows `Finally / until / eventually` for the persistent bird, `months / days / weeks` for the seed, and `hole / castle / fort` for the digging prompt.

**Lily first-piece check** (rank of ` L`, min over L2–L3, at the 11 positions where Lily is the active character):

{{lily:stories15M}}

**Stability, n=100 vs n=1000** (app-display top-8 at L2–L3, 12 prompts):

{{stability:stories15M}}

J++ is unstable: refitting from 100 instead of 1000 prompts changes three quarters of what a child sees (J cosine 0.59). Its eight signed weights per layer, learned from 39 items, swing a lot. J and LRP-only reproduce at 90–92%.

**Open-ended, label-free check:** future-word precision. On 60 held-out TinyStories *validation* stories, at every position: the share of the app-display top-10 that the story actually says in the next 2–21 tokens and has not said before (frequent function words excluded), in %, at positions ≥ 16:

{{future:stories15M:late}}

J++ is lower than J at L1–L4 (L3: 0.6 vs 1.2; L4: 0.3 vs 1.6). The LRP-only lens is slightly above J at L1–L3.

**Swap** (the app's edit, strength 0.4, greedy; first word after the anchor; "=" means it is the put-in word, "≠" means it is no longer the take-out word):

{{swaps:stories15M}}

At the app band (L2–L3), no lens moves 15M at all. Widening to L1–L4, J and LRP-only switch three of the seven: "hole" becomes "big hole … Max dug a big castle", water becomes juice ("…drink some milk first"), and apple becomes cookie (J) or cake (LRP-only). J++ switches only the first, and not toward the concept.
