#!/usr/bin/env python
"""Supervised fine-tuning entrypoint for RLPT baselines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from train_common import (
    build_sft_messages,
    ensure_output_layout,
    get_tqdm,
    import_or_raise,
    infer_max_steps,
    load_hf_dataset,
    normalize_supervised_example,
    prepare_images_for_sft,
    scaled_dataset,
    seed_everything,
    setup_wandb,
    write_json,
)


class SupervisedDataset:
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return normalize_supervised_example(self.dataset[index])


# Qwen2.5-VL spends one token per 28x28 pixel block; an unbounded image can expand past
# max_seq_len and get truncated mid-image-block, which the processor rejects.
MAX_IMAGE_PIXELS = int(os.environ.get("SFT_MAX_IMAGE_PIXELS", str(1280 * 28 * 28)))


class VLMDataCollator:
    def __init__(self, processor: Any, max_seq_len: int) -> None:
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.image_load_failures = 0
        tokenizer = getattr(processor, "tokenizer", processor)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        # Vision placeholder tokens must not be prediction targets.
        unk = getattr(tokenizer, "unk_token_id", None)
        self.vision_token_ids = []
        for special in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"):
            tid = tokenizer.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid >= 0 and tid != unk:
                self.vision_token_ids.append(tid)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        all_images: list[Any] = []
        has_images = False

        for feature in features:
            loaded_images: list[Any] = []
            for image in feature.get("images") or []:
                try:
                    loaded_images.append(self._load_image(image))
                except Exception as exc:
                    self.image_load_failures += 1
                    if self.image_load_failures <= 5:
                        print(f"Warning: dropping unreadable image {image!r}: {exc}", flush=True)
                    elif self.image_load_failures == 6:
                        print("Warning: suppressing further unreadable image messages", flush=True)

            messages = build_sft_messages(feature["prompt"], feature["response"], loaded_images)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            if loaded_images:
                has_images = True
                all_images.append(loaded_images)  # uniform nesting; processor flattens

        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "truncation": True,
            "max_length": self.max_seq_len,
            "return_tensors": "pt",
        }
        if has_images:
            processor_kwargs["images"] = all_images
        batch = self.processor(**processor_kwargs)
        labels = batch["input_ids"].clone()
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        for tid in self.vision_token_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch

    @staticmethod
    def _bound_pixels(image: Any) -> Any:
        total = image.width * image.height
        if total <= MAX_IMAGE_PIXELS:
            return image
        scale = (MAX_IMAGE_PIXELS / total) ** 0.5
        return image.resize((max(28, int(image.width * scale)), max(28, int(image.height * scale))))

    @classmethod
    def _load_image(cls, image: Any) -> Any:
        if isinstance(image, (str, Path)) and Path(image).exists():
            from PIL import Image

            with Image.open(image) as pil_image:
                return cls._bound_pixels(pil_image.convert("RGB").copy())
        if isinstance(image, str) and image.startswith("zip://") and "::" in image:
            import zipfile

            from PIL import Image

            archive_name, member_name = image[len("zip://") :].split("::", 1)
            with zipfile.ZipFile(archive_name) as archive:
                with archive.open(member_name) as f:
                    with Image.open(f) as pil_image:
                        return cls._bound_pixels(pil_image.convert("RGB").copy())
        if isinstance(image, str):
            raise ValueError(f"unsupported image reference: {image}")
        return image


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
    parser.add_argument(
        "--image_mode",
        default=os.environ.get("IMAGE_MODE", "auto"),
        choices=("auto", "text_only", "strict"),
        help="auto uses available image archives and falls back to text-only for LLaVA-CoT without LLAVA_COT_IMAGE_ZIP.",
    )
    parser.add_argument("--image_zip", default=os.environ.get("IMAGE_ZIP"))

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=1024)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--report_to", default=os.environ.get("REPORT_TO", "none"), choices=("none", "wandb"))
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "rlpt_baselines"))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb_tags", default=os.environ.get("WANDB_TAGS", ""))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable_tqdm", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def load_model(transformers: Any, args: argparse.Namespace, torch: Any) -> Any:
    dtype = None
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    last_error: Exception | None = None
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            return model_class.from_pretrained(args.base_model, **model_kwargs)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not load model {args.base_model}") from last_error


def maybe_apply_lora(model: Any, args: argparse.Namespace) -> Any:
    if not args.use_lora:
        return model
    peft = import_or_raise("peft", "Install peft or remove --use_lora.")
    target_modules = [part.strip() for part in args.lora_target_modules.split(",") if part.strip()]
    config = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft.get_peft_model(model, config)


def make_step_progress_callback(transformers: Any, total_steps: int) -> Any:
    tqdm = get_tqdm()

    class StepProgressCallback(transformers.TrainerCallback):
        def __init__(self) -> None:
            self.progress = None
            self.last_step = 0

        def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            if args.disable_tqdm:
                return
            self.last_step = int(state.global_step)
            self.progress = tqdm(
                total=total_steps,
                initial=self.last_step,
                desc="SFT optimizer steps",
                unit="step",
                dynamic_ncols=True,
            )

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            if self.progress is None:
                return
            current_step = int(state.global_step)
            delta = current_step - self.last_step
            if delta > 0:
                self.progress.update(delta)
                self.last_step = current_step

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> None:
            if self.progress is None or not logs:
                return
            display = {
                key: logs[key]
                for key in ("loss", "learning_rate", "grad_norm")
                if key in logs
            }
            if display:
                self.progress.set_postfix(display)

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
            if self.progress is not None:
                self.progress.close()
                self.progress = None

    return StepProgressCallback()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_output_layout(args.output_dir, force=args.force)
    write_json(output_dir / "configs" / "args.json", vars(args))

    planned_max_steps = args.max_steps
    if planned_max_steps <= 0:
        planned_max_steps = infer_max_steps(
            token_budget=args.token_budget,
            max_seq_len=args.max_seq_len,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    run_plan = {
        "mode": "sft",
        "base_model": args.base_model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "data_ratio": args.data_ratio,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "token_budget": args.token_budget,
        "max_steps": planned_max_steps,
        "output_dir": str(output_dir),
    }
    write_json(
        output_dir / "configs" / "planned_sft.json",
        run_plan,
    )
    if args.config_only:
        print(f"Config-only SFT plan written to {output_dir}")
        return

    print("Loading training dataset", flush=True)
    raw_dataset = load_hf_dataset(args)
    print("Selecting deterministic training subset", flush=True)
    selected_dataset = scaled_dataset(raw_dataset, args.data_ratio, args.seed, args.max_train_samples)
    print("Preparing image references", flush=True)
    selected_dataset, image_manifest = prepare_images_for_sft(selected_dataset, args)
    data_manifest = {
        "train_rows": len(selected_dataset),
        "data_ratio": args.data_ratio,
        "seed": args.seed,
        **image_manifest,
    }
    write_json(
        output_dir / "configs" / "data_manifest.json",
        data_manifest,
    )
    train_dataset = SupervisedDataset(selected_dataset)

    if args.prepare_data_only:
        print(f"Prepared SFT data metadata under {output_dir}")
        return

    torch = import_or_raise("torch", "Activate the pinned RLPT environment before running SFT.")
    transformers = import_or_raise("transformers", "Activate the pinned RLPT environment before running SFT.")

    setup_wandb(args, output_dir, {**run_plan, **data_manifest})

    print(f"Loading processor: {args.base_model}", flush=True)
    processor = transformers.AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Loading model: {args.base_model}", flush=True)
    model = load_model(transformers, args, torch)
    model = maybe_apply_lora(model, args)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=planned_max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=args.disable_tqdm,
    )
    print(
        "Training plan: "
        f"rows={len(train_dataset)} "
        f"max_steps={planned_max_steps} "
        f"token_budget={args.token_budget} "
        f"max_seq_len={args.max_seq_len} "
        f"per_device_batch={args.per_device_train_batch_size} "
        f"grad_accum={args.gradient_accumulation_steps}",
        flush=True,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VLMDataCollator(processor=processor, max_seq_len=args.max_seq_len),
        callbacks=[make_step_progress_callback(transformers, planned_max_steps)],
    )
    print("Starting SFT training", flush=True)
    # RLPT_RESUME=1: continue from the latest checkpoint-* under checkpoints/
    # (opt-in so a plain rerun keeps the original fresh-start behaviour).
    resume_ckpt = None
    if os.environ.get("RLPT_RESUME") == "1":
        from transformers.trainer_utils import get_last_checkpoint

        resume_ckpt = get_last_checkpoint(str(output_dir / "checkpoints"))
        print(f"Resume requested; latest checkpoint: {resume_ckpt}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir / "checkpoints" / "final"))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(output_dir / "checkpoints" / "final"))
    metrics = dict(result.metrics)
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    write_json(output_dir / "metrics" / "train_metrics.json", metrics)
    print(f"SFT complete: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
