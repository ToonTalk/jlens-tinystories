"""J++ steps 2c-2d: fit the per-layer router and the expert Jacobians.

Usage: python src/jpp/fit_experts.py <stories110M|stories15M> <rlens|none> [skip_first] [n_shards]
  n_shards defaults to 10 (1000 prompts), except the 110M plain-gradient
  ablation, which runs 1 shard (100 prompts; the paper fit its experts on 64).

Follows the J++ recipe (scripts/jpp_cli.py fit-router / fit-shard /
merge-experts) through the jpp_lens library calls, with TinyStories in place
of WikiText:

  router   unit-normalize -> center -> PCA-64 -> k-means (K=8, k-means++,
           seed 0) per layer, on the valid positions (skip the first 16 and
           the last) of the same 1000 TinyStories prompts fit A / fit C used.
  experts  one Jacobian per cluster, averaged over that cluster's positions,
           through the LRP backward graph (lrp_mode "rlens": LN-rule on the
           RMSNorms, identity + half rules on the SwiGLU MLP, plain gradients
           through attention) or plain gradients ("none", the ablation that
           also reproduces the shipped J as its pooled lens).
           Fitted as 10 contiguous 100-prompt shards (exact sums), so shard 0
           alone is the n=100 fit for the stability check.
  merge    experts with < 50 kept positions fall back to the pooled Jacobian
           (the recipe's min_kept_positions). The pooled lens of an "rlens" fit
           is the LRP-only lens (the R-Lens of the same data).

All source layers below the final block (110M: L0-L10, 15M: L0-L4); target =
final block output, as the shipped J. fp32 model; 32 rows per backward pass.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from jpp_common import (  # noqa: E402
    ARTIFACTS,
    JPP_OUT,
    NUM_CLUSTERS,
    NUM_SHARDS,
    PROJECTION_DIM,
    ROWS_PER_PASS,
    SEQ_LEN,
    check_lrp_value_exact,
    fit_prompts,
    harden_atomic_writes,
    load_model,
    register_llama_lrp,
    save_jlens_pt,
    setup_logging,
)

logger = logging.getLogger("fit_experts")
CREATION_TIME = "2026-10-26_00-00-00"  # fixed, so checkpoint paths are stable across resumes


def router_path(model_key: str, skip: int) -> Path:
    return JPP_OUT / model_key / f"router_K{NUM_CLUSTERS}_skip{skip}.pt"


def fit_or_load_router(lens_model, model_key, prompts, source_layers, skip):
    from workspace_lens.routing.cluster import collect_valid_position_activations
    from workspace_lens.routing.router import ActivationRouterCollection

    path = router_path(model_key, skip)
    if path.exists():
        return ActivationRouterCollection.load(str(path))
    t0 = time.time()
    acts = collect_valid_position_activations(
        lens_model, prompts, source_layers, skip_first_n_positions=skip, max_seq_len=SEQ_LEN
    )
    router = ActivationRouterCollection(NUM_CLUSTERS, PROJECTION_DIM, seed=0)
    router.fit(acts)
    # cluster sizes on the router's own data, for the report
    sizes = {
        l: torch.bincount(router.assign(acts[l], l), minlength=NUM_CLUSTERS).tolist()
        for l in source_layers
    }
    del acts
    path.parent.mkdir(parents=True, exist_ok=True)
    router.save(str(path))
    (path.with_suffix(".json")).write_text(
        json.dumps({"n_prompts": len(prompts), "skip_first": skip, "cluster_sizes": sizes,
                    "wall_s": round(time.time() - t0, 1)}, indent=1),
        encoding="utf-8",
    )
    logger.info("router fitted in %.0fs -> %s", time.time() - t0, path)
    return router


def main() -> None:
    model_key, lrp_mode = sys.argv[1], sys.argv[2]
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    n_shards = int(sys.argv[4]) if len(sys.argv) > 4 else (
        1 if (model_key, lrp_mode) == ("stories110M", "none") else NUM_SHARDS)
    run_name = f"{lrp_mode}_skip{skip}"
    setup_logging(JPP_OUT / model_key / f"fit_{run_name}.log")
    from workspace_lens.config import LensConfig
    from workspace_lens.fitting.condense_experts import ExpertJacobians
    from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer
    from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
    from workspace_lens.fitting.utils import expert_checkpoint_filename, merge_expert_checkpoints
    from workspace_lens.utils import hf_model_name_of

    register_llama_lrp()
    harden_atomic_writes()
    torch.manual_seed(0)
    lens_model, hf, tok = load_model(model_key)
    if lrp_mode != "none":
        diff = check_lrp_value_exact(lens_model, fit_prompts(1)[0])
        logger.info("LRP surgery forward max|diff| = %.3g (must be ~0)", diff)
        assert diff < 1e-4
    prompts = fit_prompts(1000)
    source_layers = list(range(lens_model.n_layers - 1))
    router = fit_or_load_router(lens_model, model_key, prompts, source_layers, skip)

    trainer_cls = ExpertJacobianRelPTrainer if lrp_mode != "none" else ExpertJacobianTrainer
    shard_dirs = []
    per = len(prompts) // NUM_SHARDS
    t_fit = 0.0
    for s in range(n_shards):
        shard_prompts = prompts[s * per : (s + 1) * per]
        config = LensConfig(
            hf_model_name=hf_model_name_of(lens_model),
            checkpoint_name=f"{model_key}/{run_name}/shard{s}of{NUM_SHARDS}",
            checkpoint_every_n_prompts=25,
            creation_time=CREATION_TIME,
            artifacts_base_dir=str(ARTIFACTS),
            source_layers=source_layers,
            jacobian_rows_per_pass=ROWS_PER_PASS,
            max_seq_len=SEQ_LEN,
            skip_first_n_positions=skip,
            lrp_mode=lrp_mode,
        )
        ckpt = Path(config.checkpoint_path) / expert_checkpoint_filename(NUM_CLUSTERS)
        shard_dirs.append(config.checkpoint_path)
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            if state["config"]["next_prompt_idx"] >= len(shard_prompts):
                logger.info("shard %d already complete", s)
                continue
            trainer = trainer_cls.from_checkpoint_dir(
                config.checkpoint_path, shard_prompts,
                router_collections_K_dict={NUM_CLUSTERS: router}, model=lens_model,
            )
        else:
            trainer = trainer_cls(
                config, lens_model, shard_prompts, router_collections_K_dict={NUM_CLUSTERS: router}
            )
        t0 = time.time()
        trainer.fit()
        t_fit += time.time() - t0
        logger.info("shard %d done in %.0fs", s, time.time() - t0)

    out_dir = JPP_OUT / model_key
    merges = [(per, shard_dirs[:1])] + ([(len(prompts), shard_dirs)] if n_shards == NUM_SHARDS else [])
    for n, dirs in merges:
        merged = merge_expert_checkpoints(
            [str(Path(d) / expert_checkpoint_filename(NUM_CLUSTERS)) for d in dirs], device="cpu"
        )
        experts = ExpertJacobians.from_sums(merged.fit_sums, merged.config, min_kept_positions=50)
        experts.save(str(ARTIFACTS / model_key / f"experts_{run_name}_n{n}.pt"))
        pooled = experts.pooled_lens(checkpoint_name=f"{model_key}/{run_name}/pooled_n{n}")
        save_jlens_pt(
            pooled.jacobians_L_dict_FN, out_dir / f"pooled_{run_name}_n{n}.pt", n,
            {"lens": f"pooled {lrp_mode} (LRP-only lens when rlens)", "skip_first": skip},
        )
        fallbacks = {l: experts.fallback_L_dict_Bool_E[l].tolist() for l in experts.layers}
        counts = {l: experts.position_counts_L_dict_E[l].tolist() for l in experts.layers}
        (out_dir / f"experts_{run_name}_n{n}.json").write_text(
            json.dumps({"n_prompts": n, "position_counts": counts, "fallback": fallbacks}, indent=1),
            encoding="utf-8",
        )
        logger.info("merged n=%d: fallbacks %s", n, {l: sum(f) for l, f in fallbacks.items()})
    stamp = out_dir / f"fit_{run_name}_stamp.json"
    prev = json.loads(stamp.read_text()) if stamp.exists() else {}
    stamp.write_text(json.dumps({
        "model": model_key, "lrp_mode": lrp_mode, "skip_first": skip, "n_prompts": per * n_shards,
        "num_clusters": NUM_CLUSTERS, "projection_dim": PROJECTION_DIM,
        "rows_per_pass": ROWS_PER_PASS, "fit_wall_s_this_run": round(t_fit, 1),
        "fit_wall_s_total": round(prev.get("fit_wall_s_total", 0) + t_fit, 1),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
