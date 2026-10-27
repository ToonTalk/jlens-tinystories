"""Shared helpers for the J++ evaluation (runs in the jpp_lens environment).

The J++ code lives in a separate clone, C:/Users/toont/Documents/GitHub/jpp_lens,
installed editable into its own venv (torch 2.11 + cu128, transformers 5.14.1)
so the app pipeline's .venv is untouched. Its ``jlens`` package is a copy of
anthropics/jacobian-lens's hooks + HF adapter, so model wrapping is identical.

Big intermediate files (router activations, per-shard expert sums, expert
Jacobians: ~100-200 MB each for 110M) go to the jpp_lens clone's gitignored
``artifacts/tinystories/``; only small outputs (dev set, routers, weights,
final lenses, evals, report) land in out/jpp/.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # src/, for common.py

import torch  # noqa: E402

from common import (  # noqa: E402,F401
    EVAL_PROMPTS,
    MODELS,
    OUT,
    ROOT,
    SEQ_LEN,
    piece_str,
    resolve_pins,
    word_start_alpha_ids,
    write_json,
)

JPP_OUT = OUT / "jpp"
ARTIFACTS = Path("C:/Users/toont/Documents/GitHub/jpp_lens/artifacts/tinystories")
FIT_CORPUS = OUT / "corpora" / "tinystories_1000.json"  # fit A / fit C prompts
EXISTING_FIT = {"stories110M": "A", "stories15M": "C"}  # the shipped J's fit ids
APP_BANDS = {"stories110M": [4, 5, 6, 7, 8], "stories15M": [2, 3]}
SKIP_FIRST = 16  # paper + jlens default
NUM_CLUSTERS = 8
PROJECTION_DIM = 64
ROWS_PER_PASS = 32  # = the original fits' dim_batch (perf only)
NUM_SHARDS = 10  # 10 x 100 prompts: shard 0 alone is the n=100 fit


def fit_prompts(n: int = 1000) -> list[str]:
    prompts = json.loads(FIT_CORPUS.read_text(encoding="utf-8"))
    assert len(prompts) >= n
    return prompts[:n]


def load_model(model_key: str, device: str = "cuda"):
    """HF model at the pinned revision in fp32 (as the shipped J was fitted),
    wrapped as a jlens LensModel. Returns (lens_model, hf_model, tokenizer)."""
    import jlens
    import transformers

    pins = resolve_pins()
    repo = pins["models"][model_key]["repo"]
    sha = pins["models"][model_key]["sha"]
    tok = transformers.AutoTokenizer.from_pretrained(repo, revision=sha)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        repo, revision=sha, dtype=torch.float32
    ).to(device)
    lens_model = jlens.from_hf(hf, tok)
    return lens_model, hf, tok


def register_llama_lrp() -> None:
    """Teach the J++ LRP surgery the Llama classes the TinyStories models use.

    The surgery refuses unknown RMSNorm / gated-MLP classes rather than guess.
    transformers' LlamaRMSNorm forward is the Qwen2 form (fp32 rsqrt, weight
    applied after the downcast, eps attr ``variance_epsilon``) and LlamaMLP is
    ``down(act(gate(x)) * up(x))`` with SiLU; both are asserted from source
    here, and :func:`check_lrp_value_exact` checks the patched forward."""
    from transformers.models.llama import modeling_llama as ml

    from workspace_lens.lrp import lrp as lrp_mod

    norm_src = inspect.getsource(ml.LlamaRMSNorm.forward)
    assert "variance_epsilon" in norm_src and "self.weight * hidden_states.to(input_dtype)" in norm_src, norm_src
    mlp_src = inspect.getsource(ml.LlamaMLP.forward)
    assert "self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))" in mlp_src, mlp_src
    lrp_mod._RMSNORM_LRP_FORWARDS["LlamaRMSNorm"] = lrp_mod._qwen2_rmsnorm_lrp_forward
    if "LlamaMLP" not in lrp_mod._KNOWN_GATED_MLPS:
        lrp_mod._KNOWN_GATED_MLPS = (*lrp_mod._KNOWN_GATED_MLPS, "LlamaMLP")


@torch.no_grad()
def check_lrp_value_exact(lens_model, prompt: str) -> float:
    """Max |stock - patched| over the final block output: the surgery must not
    change forward values (only gradients)."""
    from jlens.hooks import ActivationRecorder

    from workspace_lens.lrp import apply_lrp_rules_for_mode

    ids = lens_model.encode(prompt, max_length=SEQ_LEN)
    last = lens_model.n_layers - 1
    with ActivationRecorder(lens_model.layers, at=[last]) as rec:
        lens_model.forward(ids)
        stock = rec.activations[last].clone()
    with apply_lrp_rules_for_mode(lens_model, "rlens") as report:
        with ActivationRecorder(lens_model.layers, at=[last]) as rec:
            lens_model.forward(ids)
            patched = rec.activations[last].clone()
    assert len(report.ln_rule) >= 2 * lens_model.n_layers and len(report.mlp_rule) == lens_model.n_layers
    return float((stock - patched).abs().max())


def load_existing_J(model_key: str, n: int = 1000) -> dict[int, torch.Tensor]:
    """The shipped J-lens: out/lenses/<fit>/lens_<n>.pt (fp16 on disk, the same
    values app/*.jlens holds as fp32)."""
    fit = EXISTING_FIT[model_key]
    state = torch.load(OUT / "lenses" / fit / f"lens_{n}.pt", map_location="cpu", weights_only=True)
    return {int(l): J.float() for l, J in state["J"].items()}


def load_lens_jacobians(path: Path) -> dict[int, torch.Tensor]:
    """A lens .pt written by this pipeline or by workspace_lens BaseLens.save."""
    from workspace_lens.utils import load_lens_file

    if path.suffix == ".pt" and "jlens_format" in path.stem:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return {int(l): J.float() for l, J in state["J"].items()}
    lens = load_lens_file(str(path))
    return {int(l): J.float() for l, J in lens.jacobians_L_dict_FN.items()}


def save_jlens_pt(jacobians: dict[int, torch.Tensor], path: Path, n_prompts: int, meta: dict) -> None:
    """Save in the anthropics jlens JacobianLens.save layout (keys J, n_prompts,
    source_layers, d_model) but fp32, plus a meta dict."""
    layers = sorted(jacobians)
    d = jacobians[layers[0]].shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "J": {l: jacobians[l].float().contiguous() for l in layers},
            "n_prompts": n_prompts,
            "source_layers": layers,
            "d_model": d,
            "meta": meta,
        },
        path,
    )


def candidate_ids(tok, words: list[str]) -> list[int]:
    """Single-token surfaces of each word: '▁w', '▁W', 'w' (leading-space and
    bare forms, original and capitalized case), as the paper's surface_forms."""
    ids: set[int] = set()
    for w in words:
        for form in {w, w.lower(), w.capitalize()}:
            for piece in ("▁" + form, form):
                tid = tok.convert_tokens_to_ids(piece)
                if tid is not None and tid != tok.unk_token_id:
                    ids.add(int(tid))
    return sorted(ids)


def non_semantic_ids(tok, vocab_size: int) -> set[int]:
    """J++ Readout Filtering: ids whose decoded string has no letter or digit
    (the paper's 'no letter or digit' rule, via the J++ library function)."""
    from lens_evals.eval_utils import all_non_semantic_token_ids

    return set(all_non_semantic_token_ids(tok, vocab_size))


def harden_atomic_writes(tries: int = 20, wait_s: float = 3.0) -> None:
    """Windows: a scanner can briefly lock a just-written 200 MB checkpoint, and
    the library's os.replace then fails with WinError 5 (seen once on the 110M
    fit). Retry the rename instead of losing the run. Patches the trainer's
    _atomic_write at runtime; the jpp_lens clone itself is untouched."""
    import os
    import time

    from workspace_lens.fitting.jacobian_fitting import LensTrainer

    def _atomic_write(self, path, write):
        tmp = f"{path}.tmp.{os.getpid()}"
        write(tmp)
        for i in range(tries):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if i == tries - 1:
                    raise
                time.sleep(wait_s)

    LensTrainer._atomic_write = _atomic_write


def setup_logging(path: Path | None = None) -> None:
    import logging

    handlers = [logging.StreamHandler()]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers, force=True)
