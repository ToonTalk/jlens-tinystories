"""Shared helpers for the J-lens TinyStories feasibility experiment.

Everything deterministic: model revisions and dataset revisions are resolved
to commit SHAs once and cached in out/pins.json; corpora are materialized to
JSON files so every fit and eval reuses byte-identical prompts.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SEED = 0

MODELS = {
    "stories110M": "Xenova/llama2.c-stories110M",
    "stories15M": "Xenova/llama2.c-stories15M",
}

# Expected configs from the brief; loading asserts against these.
EXPECTED = {
    "stories110M": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12),
    "stories15M": dict(hidden_size=288, num_hidden_layers=6, num_attention_heads=6),
}

FITS = {
    # fit id -> (model key, corpus key)
    "A": ("stories110M", "tinystories"),
    "B": ("stories110M", "wikitext"),
    "C": ("stories15M", "tinystories"),
}

DATASETS = {
    "tinystories": ("roneneldan/TinyStories", None, "train"),
    "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train"),
}

SEQ_LEN = 128  # fit sequence length (tokens), per the brief

EVAL_PROMPTS = [
    "Once upon a time there was a little girl named Lily. She had a dog named Max. One day",
    "Tom saw a big dark cloud in the sky. He kept playing outside.",
    "Sara put her red ball in the box. Then she went to eat lunch. When she came back",
    "Ben broke his mom's favorite vase. He heard her coming.",
    "Anna was very hungry. She looked in the kitchen and saw an apple, a banana, and a cake.",
    "The little bird could not fly. Every day it tried and tried.",
    "First Anna put on her socks, then her shoes, then",
    "Lily and Max went to the beach. Lily built a castle. Max dug a",
    "It was a dark night. Tim heard a strange noise in the garden.",
    "Mia planted a tiny seed. She watered it every day. After many days",
    "The dragon was not mean. He was just lonely.",
    "Sam had three cookies. He gave one to his sister.",
]


def pins_path() -> Path:
    return OUT / "pins.json"


def resolve_pins() -> dict:
    """Resolve model + dataset revisions to commit SHAs once; cache to disk."""
    path = pins_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from huggingface_hub import HfApi

    api = HfApi()
    pins = {"models": {}, "datasets": {}}
    for key, repo in MODELS.items():
        pins["models"][key] = {"repo": repo, "sha": api.model_info(repo).sha}
    for key, (repo, config, split) in DATASETS.items():
        pins["datasets"][key] = {
            "repo": repo,
            "config": config,
            "split": split,
            "sha": api.dataset_info(repo).sha,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pins, indent=2), encoding="utf-8")
    return pins


def load_model(model_key: str, device: str | None = None):
    """Load model + tokenizer at the pinned revision, wrap for jlens.

    Returns (lens_model, hf_model, tokenizer, info_dict).
    """
    import jlens
    import transformers

    pins = resolve_pins()
    repo = pins["models"][model_key]["repo"]
    sha = pins["models"][model_key]["sha"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = transformers.AutoTokenizer.from_pretrained(repo, revision=sha)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        repo, revision=sha, torch_dtype=torch.float32
    ).to(device)

    cfg = hf.config
    for attr, want in EXPECTED[model_key].items():
        got = getattr(cfg, attr)
        assert got == want, f"{model_key}.{attr}: expected {want}, got {got}"

    lens_model = jlens.from_hf(hf, tok)
    info = {
        "model_key": model_key,
        "repo": repo,
        "revision": sha,
        "device": device,
        "dtype": "float32",
        "n_layers": lens_model.n_layers,
        "d_model": lens_model.d_model,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return lens_model, hf, tok, info


def corpus_path(corpus_key: str, n: int) -> Path:
    return OUT / "corpora" / f"{corpus_key}_{n}.json"


def build_corpus(corpus_key: str, n: int, tokenizer) -> list[str]:
    """Materialize n prompts of >= SEQ_LEN tokens; cached to JSON.

    tinystories: first n stories (in dataset order) whose tokenization is
        >= SEQ_LEN tokens. Each prompt is one story (fit truncates to 128).
    wikitext: article text concatenated in dataset order, tokenized
        incrementally and emitted as decoded SEQ_LEN-token chunks.
    """
    path = corpus_path(corpus_key, n)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    from datasets import load_dataset

    pins = resolve_pins()
    d = pins["datasets"][corpus_key]
    t0 = time.time()
    stream = load_dataset(
        d["repo"], d["config"], split=d["split"], revision=d["sha"], streaming=True
    )

    prompts: list[str] = []
    if corpus_key == "tinystories":
        for row in stream:
            text = row["text"].strip()
            if not text:
                continue
            n_tok = len(tokenizer(text, truncation=False).input_ids)
            if n_tok >= SEQ_LEN:
                prompts.append(text)
            if len(prompts) >= n:
                break
    elif corpus_key == "wikitext":
        buf_ids: list[int] = []
        for row in stream:
            text = row["text"]
            if not text.strip():
                continue
            buf_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
            while len(buf_ids) >= SEQ_LEN:
                chunk, buf_ids = buf_ids[:SEQ_LEN], buf_ids[SEQ_LEN:]
                prompts.append(tokenizer.decode(chunk))
            if len(prompts) >= n:
                break
        prompts = prompts[:n]
    else:
        raise ValueError(corpus_key)

    assert len(prompts) == n, f"only found {len(prompts)} prompts for {corpus_key}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[corpus] {corpus_key} n={n} built in {time.time() - t0:.0f}s -> {path}")
    return prompts


def word_start_alpha_ids(tokenizer, vocab_size: int) -> set[int]:
    """Display filter from the brief: leading-space alphabetic tokens only.

    For the Llama SentencePiece vocab this means pieces of the form
    '<U+2581><alpha>+' (word-starts). Drops punctuation, continuation
    fragments, digits, and special tokens.
    """
    keep = set()
    for tid in range(vocab_size):
        piece = tokenizer.convert_ids_to_tokens(tid)
        if (
            isinstance(piece, str)
            and len(piece) > 1
            and piece.startswith("▁")
            and piece[1:].isalpha()
        ):
            keep.add(tid)
    return keep


def piece_str(tokenizer, tid: int) -> str:
    """Human-readable single-token string, space shown as leading blank."""
    piece = tokenizer.convert_ids_to_tokens(int(tid))
    return piece.replace("▁", " ")


def set_seeds() -> None:
    import random

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8"
    )
