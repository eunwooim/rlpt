#!/usr/bin/env python
"""Launch one GRPO arm of the six-arm campaign on the frozen VisualPRM subset.

Reads grpo_arms/data/train_subset.jsonl (sha256-checked), materializes bounded
images + verl parquet under the run dir, composes the verl main_ppo command
(cloned from the proven src/train/rlvr_grpo.py skeleton incl. Edit 3), and
execs it. The reward is grpo_arms/arm_reward.py, configured via RLPT_* env set
by the sbatch wrapper (which also owns the scoring-server lifecycle).

The log opens with the arm config header: arm id, reward, segmentation, tau,
subset sha256, model path — so arms can be audited to differ only where intended.

Fixed split: the LAST 120 rows of the frozen file (after a seed-0 shuffle of
indices) are verl-val; the remaining 5,880 train. Identical for every arm.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

REPO = Path("/scratch/sghos104/rlpt")
sys.path.insert(0, str(REPO / "src/train"))
from train_common import (  # noqa: E402
    command_to_text,
    ensure_output_layout,
    seed_everything,
    write_bounded_image,
    write_json,
    write_text,
)

SUBSET = REPO / "grpo_arms/data/train_subset.jsonl"
SUBSET_SHA = REPO / "grpo_arms/data/train_subset.sha256"
VAL_N = 120
SPLIT_SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, help="1..6, log label only")
    p.add_argument("--base_model", required=True)
    p.add_argument("--reward", required=True, choices=("match", "match_v2", "vprm"))
    p.add_argument("--segmentation", required=True, choices=("chunker", "marker", "native", "vprm"))
    p.add_argument("--tau", type=float, default=None, help="match threshold (match arms)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--rollout_n", type=int, default=5)
    p.add_argument("--ppo_mini_batch_size", type=int, default=8)
    p.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_response_length", type=int, default=1024)
    p.add_argument("--max_steps", type=int, default=-1, help="-1 = one epoch over the train split")
    p.add_argument("--save_freq", type=int, default=20)
    p.add_argument("--test_freq", type=int, default=20)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    p.add_argument("--n_gpus_per_node", type=int, default=2)
    p.add_argument("--rollout_tp_size", type=int, default=1)
    p.add_argument("--max_train_samples", type=int, default=None, help="smoke: cap train rows")
    p.add_argument("--prepare_data_only", action="store_true")
    p.add_argument("--config_only", action="store_true")
    p.add_argument("--verl_extra_override", action="append", default=[])
    # v3 (2026-09-11): extra val parquet(s) logged under their own data_source; custom entrypoint (filter_groups port);
    # total_epochs > 1 lets the dataloader wrap when filter_groups consumes >1 gen batch per step
    p.add_argument("--extra_val_files", action="append", default=[])
    p.add_argument("--entrypoint", default="verl.trainer.main_ppo", help="python module run with -m")
    p.add_argument("--total_epochs", type=int, default=1)
    return p.parse_args()


def load_subset():
    sha = hashlib.sha256(SUBSET.read_bytes()).hexdigest()
    expected = SUBSET_SHA.read_text().strip()
    if sha != expected:
        raise RuntimeError(f"train_subset.jsonl sha256 {sha} != recorded {expected}")
    rows = [json.loads(l) for l in SUBSET.open()]
    return rows, sha


def build_rows(rows, args, output_dir: Path):
    """Materialize bounded images and produce verl parquet rows."""
    img_root = output_dir / "data" / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(rows):
        src = REPO / r["image"]
        tgt = img_root / r["image"].replace("data/visualprm_v11_raw/images_extracted/images/", "")
        if not tgt.exists():
            tgt.parent.mkdir(parents=True, exist_ok=True)
            write_bounded_image(src.read_bytes(), tgt)
        gt = {"answer": r["answer"], "gold_steps": r["gold_steps"],
              "question": r["question"], "image_path": str(tgt)}
        out.append({
            "data_source": r["source_file"],
            "prompt": [{"role": "user", "content": r["question"]}],
            "images": [{"image": str(tgt)}],
            "ability": "vl_reasoning",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
            "extra_info": {"index": i, "source_file": r["source_file"],
                           "line_index": r["line_index"], "n_gold_steps": len(r["gold_steps"])},
        })
        if (i + 1) % 1000 == 0:
            print(f"[data] materialized {i + 1}/{len(rows)}", flush=True)
    return out


def write_parquet(rows, path: Path):
    import datasets

    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(path))


def main():
    args = parse_args()
    if args.reward in ("match", "match_v2"):
        assert args.tau is not None, "--tau required for match arms"
        assert args.segmentation in ("chunker", "marker", "native")
    else:
        assert args.segmentation == "vprm"
    seed_everything(args.seed)

    rows, sha = load_subset()
    header = {
        "arm": args.arm, "reward": args.reward, "segmentation": args.segmentation,
        "tau": args.tau,
        "tau_note": ("gate-1 calibrated 2026-08-22 (see grpo_arms/TAU_DECISION.md): "
                     "0.45 keeps all merge/split perturbation kinds >=0.79 survival at "
                     "hn FPR 2.1%; the FPR<=1% alternative 0.56 was rejected because it "
                     "sits on the split-survival cliff and would penalize the marker "
                     "arms' over-segmentation as if it were reasoning error"
                     ) if args.reward in ("match", "match_v2") else None,
        "subset_sha256": sha, "n_subset_rows": len(rows),
        "base_model": args.base_model, "seed": args.seed,
        "train_batch_size": args.train_batch_size, "rollout_n": args.rollout_n,
        "learning_rate": args.learning_rate,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "reward_mode": os.environ.get("REWARD_MODE", "match"),
        "extra_val_files": args.extra_val_files, "entrypoint": args.entrypoint, "total_epochs": args.total_epochs,
        "image_pixel_cap": os.environ.get("RLVR_MAX_IMAGE_PIXELS", str(640 * 28 * 28)),
        "reward_weights": ({"answer": 5.0, "match": 2.0, "format": 1.0, "pun": -1.0,
                            "spec": "reward_redesign/reward_v2.py score_new FINAL 2026-09-11 (no tau gate)"}
                           if args.reward == "match_v2" else {"format": 1.0, "match_or_prm": 2.0, "answer": 5.0}),
        "env": {k: os.environ.get(k) for k in
                ("RLPT_ARM_REWARD", "RLPT_SEGMENTATION", "RLPT_MATCH_TAU", "RLPT_SCORER_SOCKET")},
    }
    print("[ARM CONFIG] " + json.dumps(header, indent=2), flush=True)

    output_dir = ensure_output_layout(args.output_dir, force=True)
    write_json(output_dir / "configs" / "arm_config.json", header)

    # env sanity: the reward module reads these at score time
    assert os.environ.get("RLPT_ARM_REWARD") == args.reward, "RLPT_ARM_REWARD mismatch"
    if args.reward in ("match", "match_v2"):
        assert os.environ.get("RLPT_SEGMENTATION") == args.segmentation
        assert abs(float(os.environ.get("RLPT_MATCH_TAU", "nan")) - args.tau) < 1e-9

    # fixed cross-arm split
    idx = list(range(len(rows)))
    random.Random(SPLIT_SEED).shuffle(idx)
    val_idx = set(idx[-VAL_N:])
    train_rows = [rows[i] for i in idx if i not in val_idx]
    val_rows = [rows[i] for i in sorted(val_idx)]
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
        val_rows = val_rows[: max(2, min(len(val_rows), 8))]

    max_steps = args.max_steps
    if max_steps <= 0:
        max_steps = len(train_rows) // args.train_batch_size
    header["max_steps"] = max_steps
    header["n_train"] = len(train_rows)
    header["n_val"] = len(val_rows)
    write_json(output_dir / "configs" / "arm_config.json", header)

    if args.config_only:
        print("[arm] config-only; stopping before data prep")
        return

    train_file = output_dir / "data" / "train.parquet"
    val_file = output_dir / "data" / "val.parquet"
    if train_file.exists() and val_file.exists():
        print("[arm] parquet already present (resume); skipping re-materialization", flush=True)
    else:
        write_parquet(build_rows(train_rows, args, output_dir), train_file)
        write_parquet(build_rows(val_rows, args, output_dir), val_file)

    reward_path = REPO / "grpo_arms/arm_reward.py"
    overrides = [
        "algorithm.adv_estimator=grpo",
        f"data.train_files=[{train_file}]",
        "data.val_files=[" + ",".join([str(val_file)] + list(args.extra_val_files)) + "]",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"actor_rollout_ref.model.path={args.base_model}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.strategy=fsdp",
        # FSDP CPU offload: the June-campaign recipe for the FSDP<->vLLM weight
        # handoff on A100-80GB, needed here because the scoring server holds
        # ~5 GB on GPU 1 (smoke 61997527 OOMed at vLLM wake_up without it)
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        "reward_model.enable=False",
        "reward_model.reward_manager=naive",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        "trainer.project_name=rlpt_six_arms",
        f"trainer.experiment_name=arm{args.arm}",
        f"trainer.default_local_dir={output_dir / 'checkpoints'}",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.total_training_steps={max_steps}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        "trainer.logger=[console]",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
        "trainer.max_actor_ckpt_to_keep=2",
        f"trainer.rollout_data_dir={output_dir / 'rollouts'}",
    ]
    overrides.extend(args.verl_extra_override)
    command = [sys.executable, "-m", args.entrypoint, *overrides]
    write_text(output_dir / "logs" / "verl_command.txt", command_to_text(command) + "\n")
    print(command_to_text(command), flush=True)

    if args.prepare_data_only:
        print(f"[arm] data + command prepared under {output_dir}")
        return

    with (output_dir / "logs" / "verl_stdout_stderr.log").open("a") as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
