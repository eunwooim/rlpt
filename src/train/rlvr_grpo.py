#!/usr/bin/env python
"""veRL GRPO entrypoint for RLPT RLVR baselines."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from train_common import (
    command_to_text,
    ensure_output_layout,
    import_or_raise,
    infer_max_steps,
    load_hf_dataset,
    materialize_visualprm_images,
    normalize_supervised_example,
    scaled_dataset,
    seed_everything,
    write_json,
    write_text,
)


SUPPORTED_PROCESS_REWARDS = {"rlvr", "none", "nli_heuristic", "ours", "sft"}
BASELINE_REWARD_COMPONENTS = {"accuracy", "format"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--data_ratio", type=float, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--token_budget", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config_only", action="store_true")
    parser.add_argument("--prepare_data_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--report_to", default=os.environ.get("REPORT_TO", "none"), choices=("none", "wandb"))
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "rlpt_baselines"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb_tags", default=os.environ.get("WANDB_TAGS", ""))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))

    parser.add_argument("--rl_algorithm", default="grpo", choices=("grpo",))
    parser.add_argument("--process_reward", default="rlvr", choices=sorted(SUPPORTED_PROCESS_REWARDS))
    parser.add_argument("--reward_components", default="accuracy,format")
    parser.add_argument("--reward_ablation", default="accuracy_format")

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=8)
    parser.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--rollout_n", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--total_epochs", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--test_freq", type=int, default=50)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--project_name", default="rlpt_baselines")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n_gpus_per_node", type=int, default=None)
    parser.add_argument("--rollout_tp_size", type=int, default=1)
    parser.add_argument("--reward_manager", default="naive")
    parser.add_argument("--logger", default="console")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_remove_padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verl_module", default="verl.trainer.main_ppo")
    parser.add_argument(
        "--verl_extra_override",
        action="append",
        default=[],
        help="Additional Hydra override passed to veRL. May be repeated.",
    )
    return parser.parse_args()


def validate_rewards(args: argparse.Namespace) -> list[str]:
    if args.process_reward == "ours":
        raise ValueError("process_reward=ours is intentionally not implemented for baseline testing.")
    if args.process_reward == "nli_heuristic":
        raise ValueError("process_reward=nli_heuristic is not part of the first baseline run.")
    if args.process_reward == "sft":
        raise ValueError("process_reward=sft is not a valid RLVR training reward.")
    if args.process_reward == "none":
        raise ValueError("process_reward=none cannot train RLVR without a reward function.")

    components = [part.strip() for part in args.reward_components.split(",") if part.strip()]
    unsupported = sorted(set(components) - BASELINE_REWARD_COMPONENTS)
    if unsupported:
        raise ValueError(f"Unsupported baseline reward components: {unsupported}")
    if not components:
        raise ValueError("At least one baseline reward component is required.")
    return components


def row_from_example(example: dict[str, Any], index: int, args: argparse.Namespace, components: list[str]) -> dict[str, Any]:
    normalized = normalize_supervised_example(example)
    images = normalized.get("images") or []
    row = {
        "data_source": args.dataset_name,
        "prompt": [{"role": "user", "content": normalized["prompt"]}],
        "ability": "vl_reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": normalized["response"],
        },
        "extra_info": {
            "split": args.dataset_split,
            "index": index,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "reward_ablation": args.reward_ablation,
            "reward_components": components,
            "num_images": len(images),
            "image_policy": "omitted_from_verl_parquet",
        },
    }
    if images:
        row["images"] = [{"image": image} for image in images]
        row["extra_info"]["image_policy"] = "local_paths_in_images_column"
    return row


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    datasets = import_or_raise(
        "datasets",
        "Activate the pinned RLPT environment before preparing veRL parquet files.",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(path))


def prepare_verl_data(args: argparse.Namespace, output_dir: Path, components: list[str]) -> tuple[Path, Path, int, int]:
    raw_dataset = load_hf_dataset(args)
    selected = scaled_dataset(raw_dataset, args.data_ratio, args.seed, args.max_train_samples)
    materialized_images = 0
    if isinstance(selected, list):
        materialized_images = materialize_visualprm_images(selected, args, output_dir)
    rows = [row_from_example(example, index, args, components) for index, example in enumerate(selected)]
    if not rows:
        raise ValueError("No rows available after applying data_ratio/max_train_samples.")

    if len(rows) == 1:
        train_rows = rows
        val_rows = rows
    else:
        val_count = max(1, int(len(rows) * args.eval_ratio))
        val_count = min(val_count, len(rows) - 1)
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]

    train_file = output_dir / "data" / "train.parquet"
    val_file = output_dir / "data" / "val.parquet"
    write_parquet(train_rows, train_file)
    write_parquet(val_rows, val_file)
    write_json(
        output_dir / "configs" / "data_manifest.json",
        {
            "train_file": train_file,
            "val_file": val_file,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "materialized_images": materialized_images,
            "data_ratio": args.data_ratio,
            "seed": args.seed,
        },
    )
    return train_file, val_file, len(train_rows), len(val_rows)


def gpu_count_from_env() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        return len([part for part in visible.split(",") if part.strip()])
    return 1


def build_verl_command(
    args: argparse.Namespace,
    output_dir: Path,
    train_file: Path,
    val_file: Path,
    max_steps: int,
) -> list[str]:
    reward_path = Path(__file__).resolve().with_name("baseline_rewards.py")
    experiment_name = args.experiment_name or output_dir.parent.name
    n_gpus = args.n_gpus_per_node or gpu_count_from_env()
    logger = [part.strip() for part in args.logger.split(",") if part.strip()]
    if args.report_to == "wandb" and "wandb" not in logger:
        logger.append("wandb")
    logger_override = "[" + ",".join(f"'{part}'" for part in logger) + "]"

    overrides = [
        "algorithm.adv_estimator=grpo",
        f"data.train_files=[{train_file}]",
        f"data.val_files=[{val_file}]",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"actor_rollout_ref.model.path={args.base_model}",
        f"actor_rollout_ref.model.trust_remote_code={str(args.trust_remote_code)}",
        f"actor_rollout_ref.model.use_remove_padding={str(args.use_remove_padding)}",
        f"actor_rollout_ref.actor.optim.lr={args.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(args.gradient_checkpointing)}",
        f"actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rollout_tp_size}",
        f"reward_model.enable=False",
        f"reward_model.reward_manager={args.reward_manager}",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={experiment_name}",
        f"trainer.default_local_dir={output_dir / 'checkpoints'}",
        f"trainer.total_epochs={args.total_epochs}",
        f"trainer.total_training_steps={max_steps}",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        f"trainer.logger={logger_override}",
        f"trainer.nnodes={args.nnodes}",
        f"trainer.n_gpus_per_node={n_gpus}",
    ]
    overrides.extend(args.verl_extra_override)
    return [sys.executable, "-m", args.verl_module, *overrides]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    components = validate_rewards(args)
    output_dir = ensure_output_layout(args.output_dir, force=args.force)
    write_json(output_dir / "configs" / "args.json", vars(args))

    max_steps = args.max_steps
    if max_steps <= 0:
        max_steps = infer_max_steps(
            token_budget=args.token_budget,
            max_seq_len=args.max_seq_len,
            per_device_train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=1,
            world_size=max(1, args.nnodes * (args.n_gpus_per_node or gpu_count_from_env())),
        )

    plan = {
        "mode": "rlvr_grpo",
        "base_model": args.base_model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "data_ratio": args.data_ratio,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "token_budget": args.token_budget,
        "max_steps": max_steps,
        "process_reward": args.process_reward,
        "reward_components": components,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "configs" / "planned_rlvr_grpo.json", plan)

    if args.config_only:
        print(f"Config-only RLVR plan written to {output_dir}")
        return

    train_file, val_file, train_rows, val_rows = prepare_verl_data(args, output_dir, components)
    command = build_verl_command(args, output_dir, train_file, val_file, max_steps)
    command_text = command_to_text(command)
    write_text(output_dir / "logs" / "verl_command.txt", command_text + "\n")
    write_json(
        output_dir / "configs" / "verl_launch.json",
        {
            "command": command,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "reward_source": str(Path(__file__).resolve().with_name("baseline_rewards.py")),
        },
    )
    print(command_text)

    if args.prepare_data_only:
        print(f"Prepared veRL data and command under {output_dir}")
        return

    env = os.environ.copy()
    env["RLPT_REWARD_COMPONENTS"] = ",".join(components)
    if args.report_to == "wandb":
        env.setdefault("WANDB_PROJECT", args.wandb_project)
        env.setdefault("WANDB_MODE", args.wandb_mode)
        env.setdefault("WANDB_DIR", str(output_dir / "logs" / "wandb"))
        env.setdefault("WANDB_NAME", args.wandb_run_name or output_dir.parent.name + "/" + output_dir.name)
        if args.wandb_entity:
            env.setdefault("WANDB_ENTITY", args.wandb_entity)
        if args.wandb_group:
            env.setdefault("WANDB_RUN_GROUP", args.wandb_group)
        if args.wandb_tags:
            env.setdefault("WANDB_TAGS", args.wandb_tags)
    with (output_dir / "logs" / "verl_stdout_stderr.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
