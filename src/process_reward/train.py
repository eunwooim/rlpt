#!/usr/bin/env python
"""Fine-tune the DeBERTaV3 LoRA+MLP process-reward classifier."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import os
import random

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import (
    append_log,
    create_stage_layout,
    default_output_dir,
    parse_csv,
    print_config,
    write_json,
)
from process_reward.dataset import TokenizedPairDataset, load_reward_examples
from process_reward.model import DEFAULT_BASE_MODEL, DEFAULT_LORA_TARGETS, build_pair_classifier, model_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_target_modules", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=(
            "Path to pairs.jsonl or its containing data directory. "
            "Existing pairs_train/val/test.jsonl files are reused."
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", choices=("none", "wandb"), default="none")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "rlpt_process_reward"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--config_only", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    temporary_path.replace(path)


def resolve_data_paths(
    data_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    data_path = Path(data_path).expanduser().resolve()

    if data_path.is_dir():
        data_directory = data_path
        pairs_path = data_directory / "pairs.jsonl"
    else:
        data_directory = data_path.parent
        pairs_path = data_path

    return (
        pairs_path,
        data_directory / "pairs_train.jsonl",
        data_directory / "pairs_val.jsonl",
        data_directory / "pairs_test.jsonl",
    )


def create_grouped_splits(
    pairs_path: Path,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            "Split ratios must sum to 1.0, but received "
            f"{train_ratio}, {validation_ratio}, and {test_ratio}."
        )

    if not pairs_path.is_file():
        raise FileNotFoundError(f"Missing unsplit dataset: {pairs_path}")

    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row_number, row in enumerate(iter_jsonl(pairs_path), start=1):
        source_sample_id = row.get("source_sample_id")
        if not isinstance(source_sample_id, str) or not source_sample_id:
            raise ValueError(
                f"{pairs_path}:{row_number} has no valid "
                "'source_sample_id'. Grouped splitting is required to "
                "prevent source leakage."
            )

        rows_by_source[source_sample_id].append(row)

    source_ids = sorted(rows_by_source)
    random.Random(seed).shuffle(source_ids)

    number_of_sources = len(source_ids)
    train_end = round(number_of_sources * train_ratio)
    validation_end = train_end + round(
        number_of_sources * validation_ratio
    )

    train_sources = set(source_ids[:train_end])
    validation_sources = set(source_ids[train_end:validation_end])
    test_sources = set(source_ids[validation_end:])

    if not train_sources or not validation_sources or not test_sources:
        raise ValueError(
            "The dataset is too small for non-empty train, validation, "
            "and test splits with the requested ratios."
        )

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for source_sample_id, source_rows in rows_by_source.items():
        if source_sample_id in train_sources:
            train_rows.extend(source_rows)
        elif source_sample_id in validation_sources:
            validation_rows.extend(source_rows)
        elif source_sample_id in test_sources:
            test_rows.extend(source_rows)
        else:
            raise AssertionError(
                f"Unassigned source sample: {source_sample_id}"
            )

    # Verify that no source appears in more than one split.
    assert train_sources.isdisjoint(validation_sources)
    assert train_sources.isdisjoint(test_sources)
    assert validation_sources.isdisjoint(test_sources)

    write_jsonl_atomic(train_path, train_rows)
    write_jsonl_atomic(validation_path, validation_rows)
    write_jsonl_atomic(test_path, test_rows)

    print(
        "Created dataset splits:\n"
        f"  train: {train_path} "
        f"({len(train_sources)} sources, {len(train_rows)} pairs)\n"
        f"  val:   {validation_path} "
        f"({len(validation_sources)} sources, "
        f"{len(validation_rows)} pairs)\n"
        f"  test:  {test_path} "
        f"({len(test_sources)} sources, {len(test_rows)} pairs)"
    )


def prepare_data_splits(
    data_path: str | Path,
    *,
    train_ratio: float = 0.90,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    (
        pairs_path,
        train_path,
        validation_path,
        test_path,
    ) = resolve_data_paths(data_path)

    split_paths = (train_path, validation_path, test_path)
    existing = [path.exists() for path in split_paths]

    if all(existing):
        print(
            "Using existing dataset splits:\n"
            f"  train: {train_path}\n"
            f"  val:   {validation_path}\n"
            f"  test:  {test_path}"
        )
        return train_path, validation_path, test_path

    if any(existing):
        missing = [
            str(path)
            for path, exists in zip(split_paths, existing)
            if not exists
        ]
        present = [
            str(path)
            for path, exists in zip(split_paths, existing)
            if exists
        ]
        raise RuntimeError(
            "Found a partial split set. Refusing to overwrite or mix "
            f"splits.\nPresent: {present}\nMissing: {missing}\n"
            "Delete all split files to regenerate them."
        )

    create_grouped_splits(
        pairs_path,
        train_path,
        validation_path,
        test_path,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    return train_path, validation_path, test_path


def compute_binary_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted = predictions.argmax(axis=-1)
    labels = eval_prediction.label_ids
    total = int(len(labels))
    correct = int((predicted == labels).sum()) if total else 0
    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "accuracy": correct / total if total else 0.0,
        "f1": (2 * true_positive / denominator) if denominator else 0.0,
    }


def filtered_logging_callback(transformers: Any, output_dir: Path, wandb_run: Any = None) -> Any:
    log_path = output_dir / "logs" / "training_dynamics.jsonl"

    class FilteredLoggingCallback(transformers.TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> None:
            if not state.is_world_process_zero or not logs:
                return
            filtered: dict[str, Any] = {}
            mapping = {
                "loss": "train/loss",
                "eval_loss": "validation/loss",
                "eval_accuracy": "validation/accuracy",
                "eval_f1": "validation/f1",
                "learning_rate": "train/learning_rate",
            }
            for source, destination in mapping.items():
                if source in logs:
                    filtered[destination] = logs[source]
            if not filtered:
                return
            row = {"step": int(state.global_step), **filtered}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if wandb_run is not None:
                wandb_run.log(filtered, step=int(state.global_step))

    return FilteredLoggingCallback()


def main() -> None:
    args = parse_args()
    targets = parse_csv(args.lora_target_modules)
    if not targets:
        raise ValueError("--lora_target_modules must contain at least one module")
    architecture = model_configuration(
        args.base_model,
        targets,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.mlp_hidden_dim,
        args.mlp_dropout,
    )
    if args.config_only:
        resolved_output = Path(args.output_dir) if args.output_dir else default_output_dir("training")
        config = {**vars(args), "lora_target_modules": targets, "output_dir": str(resolved_output)}
        print_config({"training": config, "model": architecture})
        return
    if args.bf16 and args.fp16:
        raise ValueError("Use at most one of --bf16 and --fp16")
    for name in (
        "max_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")

    import numpy as np
    import torch
    import transformers

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA for the NCCL backend")
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank() if distributed else 0
    is_main_process = rank == 0

    if args.output_dir:
        resolved_output = Path(args.output_dir)
    else:
        output_values: list[str | None] = [
            str(default_output_dir("training")) if is_main_process else None
        ]
        if distributed:
            torch.distributed.broadcast_object_list(output_values, src=0)
        resolved_output = Path(str(output_values[0]))
    config = {**vars(args), "lora_target_modules": targets, "output_dir": str(resolved_output)}

    if is_main_process:
        output_dir = create_stage_layout(resolved_output, include_checkpoints=True)
        write_json(output_dir / "configs" / "args.json", config)
        write_json(output_dir / "configs" / "model_config.json", architecture)
        train_path, validation_path, test_path = prepare_data_splits(
            args.data_path,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
    else:
        output_dir = resolved_output
        _, train_path, validation_path, test_path = resolve_data_paths(args.data_path)
    if distributed:
        torch.distributed.barrier()

    train_examples = load_reward_examples(train_path)
    validation_examples = load_reward_examples(validation_path)
    test_examples = load_reward_examples(test_path)

    if not train_examples:
        raise ValueError("No training examples remain after splitting")
    if not validation_examples:
        raise ValueError("No validation examples remain after loading the validation split")
    if not test_examples:
        raise ValueError("No test examples remain after loading the test split")
    manifest = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "train_source_samples": len({item.source_sample_id for item in train_examples}),
        "validation_source_samples": len({item.source_sample_id for item in validation_examples}),
        "test_source_samples": len({item.source_sample_id for item in test_examples}),
        "training_fields": ["anchor", "generated", "compatibility_label"],
        "nli_features_used_for_training": False,
    }
    if is_main_process:
        write_json(output_dir / "configs" / "data_manifest.json", manifest)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    train_dataset = TokenizedPairDataset(train_examples, tokenizer, args.max_length)
    validation_dataset = (
        TokenizedPairDataset(validation_examples, tokenizer, args.max_length) if validation_examples else None
    )
    test_dataset = TokenizedPairDataset(test_examples, tokenizer, args.max_length)
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    model = build_pair_classifier(
        architecture,
        torch_dtype=dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if is_main_process and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    wandb_run = None
    if is_main_process and args.report_to == "wandb":
        import wandb

        wandb_dir = output_dir / "logs" / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or output_dir.name,
            mode=args.wandb_mode,
            dir=str(wandb_dir),
            config={"training": config, "model": architecture, "data": manifest},
            reinit=True,
        )

    has_validation = validation_dataset is not None
    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=args.eval_steps if has_validation else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_binary_metrics if has_validation else None,
        callbacks=[filtered_logging_callback(transformers, output_dir, wandb_run)],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_metrics = dict(result.metrics)
    train_metrics["train_examples"] = len(train_examples)
    validation_metrics = trainer.evaluate() if has_validation else {}
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )

    final_dir = output_dir / "checkpoints" / "final"
    summary = {
        "output_dir": str(output_dir),
        "final_checkpoint": str(final_dir),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "nli_features_used_for_training": False,
    }
    if is_main_process:
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        if wandb_run is not None:
            wandb_run.finish()
        trainer.model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        write_json(final_dir / "model_config.json", architecture)
        trainer.save_state()
        write_json(output_dir / "metrics" / "train_metrics.json", train_metrics)
        write_json(output_dir / "metrics" / "validation_metrics.json", validation_metrics)
        write_json(output_dir / "metrics" / "test_metrics.json", test_metrics)
        write_json(output_dir / "metrics" / "summary.json", summary)
        append_log(output_dir, json.dumps(summary, sort_keys=True))
        print_config(summary)
    if distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=7,6,5,4 \
TOKENIZERS_PARALLELISM=false \
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  src/data/process_reward/train.py \
  --data_path src/outputs/process_reward_generation/v1_1/data \
  --train_ratio 0.90 \
  --validation_ratio 0.05 \
  --test_ratio 0.05 \
  --split_seed 42 \
  --output_dir src/outputs/process_reward_judge/v1_1 \
  --max_length 256 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --eval_steps 250 \
  --save_steps 250 \
  --bf16 \
  --no-gradient_checkpointing \
  --report_to none
"""