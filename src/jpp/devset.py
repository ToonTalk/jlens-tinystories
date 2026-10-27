"""J++ step 2a: the TinyStories dev set of narrative-anticipation items.

J++ learns 8 expert weights per layer on labeled "latent-variable" items.
The paper's items (multihop facts, typos, translations) are beyond a
TinyStories model, so this builds the story-register analog: short story
openings after which the model is disposed to say a known word *soon* but
not *next* (the "hid / hide before mom arrives" pattern from out/report.md).

Five families ("evals" in the paper's sense):

  need     a situation that calls for an object      cold -> coat / hat
  feeling  an event that calls for an emotion        lost toy -> sad / cry
  name     the active character's name               ... -> Tom
  riddle   a cue that implies a thing, never named   "a soft meow" -> cat
  hide     a misdeed + someone coming -> hide / hid / hiding

Every answer word is a single token of the Llama-2 vocab (checked by
verify_devset.py; multi-token words like "umbrella" or "scared" are left out,
as in the paper). The readout position is the prompt's final token.

Split (by TEMPLATE, never by filler, so no sentence frame is shared between
the weight-fitting items and the scored items): ~30% of the templates of
need/feeling/name/riddle go to "fit"; the rest, plus ALL of "hide", are
"held-out". "hide" is held out entirely because it is the concept of pinned
eval prompt 4 ("Ben broke his mom's favorite vase. He heard her coming."):
keeping it out of the fit keeps that pin honest and doubles as a
leave-one-task-out test. No item repeats any of the 12 EVAL_PROMPTS (checked
below) and none comes from the TinyStories fit corpora (they are hand-written).

Writes out/jpp/devset/items.json. Usage: python src/jpp/devset.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import EVAL_PROMPTS, OUT  # noqa: E402

JPP_OUT = OUT / "jpp"
FIT_FRACTION = 0.3
SPLIT_SEED = 0

BOYS = ["Tom", "Jack", "Bob", "Joe", "Mark", "John", "Ted", "Billy", "Roy", "Leo", "Peter", "Paul", "Andy"]
GIRLS = ["Sue", "Lucy", "Kate", "Amy", "Emma", "Jane", "Jen", "Ann", "Kim", "Ruby", "Lisa", "Mary"]
PRONOUNS = {
    "boy": {"He": "He", "he": "he", "his": "his", "His": "His", "him": "him", "kind": "boy", "elder": "man"},
    "girl": {"He": "She", "he": "she", "his": "her", "His": "Her", "him": "her", "kind": "girl", "elder": "woman"},
}

# (template id, text, answer words). {N} is the main character; {N2} a second
# character (other gender); {He}/{he}/{his}/{His}/{him}/{kind} follow {N},
# {he2}/{his2} follow {N2}. An answer of "{N}" / "{N2}" means that name.
TEMPLATES: dict[str, list[tuple[str, str, list[str]]]] = {
    "need": [
        ("cold", "{N} wanted to play in the snow, but it was very cold outside.", ["coat", "hat"]),
        ("thirsty", "{N} played in the hot sun all day. {He} was very thirsty.", ["water", "milk"]),
        ("hungry", "{N} came home after a long day at school. {His} tummy was rumbling.", ["food", "eat", "cookie", "cookies", "bread", "soup", "apple"]),
        ("dark", "{N} walked into the dark room. {He} could not see anything at all.", ["light", "lamp"]),
        ("shelf", "The jar was on a very high shelf. {N} was too small to reach it.", ["chair", "box"]),
        ("locked", "{N} found an old chest in the attic, but it was locked.", ["key"]),
        ("island", "{N} wanted to visit the little island in the middle of the lake.", ["boat", "ship", "bridge"]),
        ("catch", "{N} saw a big fish in the pond and wanted to catch it.", ["net", "rod", "pole"]),
        ("tired", "It was very late at night, and {N} was very tired.", ["bed", "sleep", "nap"]),
        ("dirty", "{N} played in the mud all afternoon and came home very dirty.", ["bath", "soap", "clean", "water"]),
        ("broken", "{N}'s favorite doll broke into two pieces.", ["fix"]),
        ("garden", "The flowers in {N}'s garden were dry and drooping.", ["water", "rain"]),
        ("bowl", "{N}'s dog looked at his empty bowl and whined.", ["food", "eat", "water"]),
        ("read", "It was dark, and {N} wanted to read {his} new book in bed.", ["light", "lamp"]),
    ],
    "feeling": [
        ("lost", "{N} lost {his} favorite teddy bear at the park. {He} looked everywhere, but {he} could not find it.", ["sad", "cry", "cried"]),
        ("gift", "On {his} birthday, {N} opened a big box and found the puppy {he} had always wanted.", ["happy", "excited", "smile"]),
        ("thunder", "In the middle of the night, a loud boom of thunder shook {N}'s window.", ["afraid"]),
        ("grabbed", "{N2} grabbed {N}'s new kite and broke it on purpose.", ["angry", "mad", "sad"]),
        ("painting", "{N} worked hard all morning and finished a beautiful painting for {his} mom.", ["proud", "happy"]),
        ("pushed", "{N} pushed {his} little sister, and she fell down and cried.", ["sorry"]),
        ("moved", "{N}'s best friend had to move far away to a new town.", ["sad", "miss"]),
        ("splash", "The puppy jumped into the bath and splashed water all over {N}.", ["laugh", "laughed"]),
        ("coin", "{N} found a shiny gold coin under a rock in the garden.", ["surprised", "excited", "happy"]),
        ("brokecar", "{N} broke {his} friend's toy car. The friend started to cry.", ["sorry", "sad"]),
        ("forest", "{N} was all alone in the big dark forest, and a wolf howled.", ["afraid"]),
        ("won", "{N}'s team won the big race at school.", ["happy", "proud", "excited"]),
    ],
    "name": [
        ("named", "Long ago, a little {kind} called {N} lived in a small house near the woods. {He} liked to pick berries.", ["{N}"]),
        ("ball", "{N} had a big red ball. {He} loved to play with it every day in the yard.", ["{N}"]),
        ("store", "{N} and {his} mom went to the store to buy some apples. The store was big and full of people.", ["{N}"]),
        ("elder", "There was a kind old {elder} in the village. Everyone called {him} {N}. {He} had a big garden full of roses.", ["{N}"]),
        ("sick", "{N} was sad because {his} friend {N2} was sick and could not come to school.", ["{N2}"]),
        ("shy", "In a small town, there lived a {kind} named {N}. {He} was very shy and did not have many friends.", ["{N}"]),
        ("cake", "{N} and {N2} were baking a cake. {N2} cracked the eggs, and the bowl fell on the floor.", ["{N2}"]),
    ],
    "riddle": [
        ("meow", "{N} heard a soft meow coming from under the bed.", ["cat"]),
        ("woof", "{N} heard a loud woof from the yard next door.", ["dog"]),
        ("moo", "On the farm, {N} heard a big animal go moo.", ["cow"]),
        ("feathers", "{N} saw something with feathers singing on a branch.", ["bird"]),
        ("fins", "{N} looked into the pond. Something with shiny scales and fins was swimming there.", ["fish"]),
        ("neigh", "{N} gave some hay to the big animal that said neigh.", ["horse"]),
        ("wool", "{N} visited the farm. An animal on the hill had soft white wool and said baa.", ["sheep"]),
        ("fruit", "{N} picked a round red fruit from the tree and took a bite.", ["apple"]),
        ("drink", "The baby was hungry, so {N} poured the white drink from the cow into a cup.", ["milk"]),
        ("night", "{N} looked up at the night sky and saw a big round light shining.", ["moon"]),
        ("squeak", "{N} heard a tiny squeak and saw a little gray animal with a long tail run under the table.", ["mouse"]),
        ("morning", "In the morning, the big yellow ball rose in the sky and made everything warm.", ["sun"]),
        ("growl", "{N} saw a big brown animal in the woods. It stood on two legs and growled.", ["bear"]),
        ("nest", "On the farm, {N} found something white and round in the hen's nest.", ["egg", "eggs"]),
    ],
    "hide": [
        ("cup", "{N} broke {his} dad's favorite cup. Then {he} heard {his} dad coming.", ["hide", "hid", "hiding"]),
        ("cookies", "{N} ate all the cookies that mom had baked for the party. Then {he} heard mom open the door.", ["hide", "hid", "hiding"]),
        ("paint", "{N} spilled red paint all over the new rug. {He} heard footsteps coming down the hall.", ["hide", "hid", "hiding"]),
        ("crayon", "{N} drew on the kitchen wall with a crayon. Then {he} heard {his} mom calling.", ["hide", "hid", "hiding"]),
        ("lamp", "While playing ball inside, {N} knocked over the lamp and it broke. {He} heard the front door open.", ["hide", "hid", "hiding"]),
        ("seek", "{N} and {his} friends wanted to play a game. {N2} closed {his2} eyes and started to count.", ["hide", "hid", "hiding"]),
        ("jar", "{N} took a cookie from the jar without asking. Then {he} heard {his} sister coming into the kitchen.", ["hide", "hid", "hiding"]),
    ],
}
FILLERS_PER_TEMPLATE = {"need": 6, "feeling": 6, "name": 8, "riddle": 6, "hide": 6}
HELD_OUT_ONLY = {"hide"}


def fillers(family: str, template_index: int, count: int) -> list[tuple[str, str, str]]:
    """Deterministic (name, gender, second name) fillers, alternating genders and
    rotating through the name lists so each template gets different names."""
    out = []
    for i in range(count):
        gender = "boy" if (template_index + i) % 2 == 0 else "girl"
        own = BOYS if gender == "boy" else GIRLS
        other = GIRLS if gender == "boy" else BOYS
        k = template_index * count + i
        name = own[k % len(own)]
        second = other[(k * 7 + 3) % len(other)]
        out.append((name, gender, second))
    return out


def render(text: str, name: str, gender: str, second: str) -> str:
    p = PRONOUNS[gender]
    p2 = PRONOUNS["girl" if gender == "boy" else "boy"]
    return text.format(
        N=name, N2=second, he2=p2["he"], his2=p2["his"], **p
    )


def build_items() -> list[dict]:
    rng = random.Random(SPLIT_SEED)
    items = []
    for family, templates in TEMPLATES.items():
        ids = [tid for tid, _, _ in templates]
        if family in HELD_OUT_ONLY:
            fit_ids = set()
        else:
            n_fit = max(1, round(FIT_FRACTION * len(ids)))
            fit_ids = set(rng.sample(ids, n_fit))
        for ti, (tid, text, answers) in enumerate(templates):
            for name, gender, second in fillers(family, ti, FILLERS_PER_TEMPLATE[family]):
                prompt = render(text, name, gender, second)
                words = [
                    name if a == "{N}" else second if a == "{N2}" else a for a in answers
                ]
                items.append(
                    {
                        "family": family,
                        "template": f"{family}/{tid}",
                        "name": f"{family}/{tid}/{name}",
                        "prompt": prompt,
                        "answers": words,
                        "split": "fit" if tid in fit_ids else "held-out",
                        "copy_allowed": family == "name",
                    }
                )
    # Disjointness from the eval prompts (exact and near-exact: shared 6-grams).
    def grams(s: str) -> set[tuple[str, ...]]:
        w = s.lower().replace(".", " ").replace(",", " ").split()
        return {tuple(w[i : i + 6]) for i in range(len(w) - 5)}

    eval_grams = set().union(*(grams(p) for p in EVAL_PROMPTS))
    for it in items:
        assert it["prompt"] not in EVAL_PROMPTS
        shared = grams(it["prompt"]) & eval_grams
        assert not shared, (it["name"], shared)
    names = [it["name"] for it in items]
    assert len(set(names)) == len(names), "duplicate item names"
    return items


def main() -> None:
    items = build_items()
    out = JPP_OUT / "devset" / "items.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=1), encoding="utf-8")
    from collections import Counter

    c = Counter((it["family"], it["split"]) for it in items)
    for (fam, split), n in sorted(c.items()):
        print(f"{fam:8s} {split:9s} {n}")
    print(f"{len(items)} items -> {out}")


if __name__ == "__main__":
    main()
