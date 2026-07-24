#!/usr/bin/env python
"""DeBERTaV3 pair classifier supporting LoRA and full fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import parse_csv, print_config


DEFAULT_BASE_MODEL = "microsoft/deberta-v3-large"
TRAINING_MODES = ("lora", "full")

# "all-linear" is resolved to every Linear layer inside the encoder only. This
# includes Q/K/V, relative-position Q/K, attention-output, and FFN projections,
# while deliberately excluding the task classifier, which is trained normally.
DEFAULT_LORA_TARGETS = "all-linear"


def model_configuration(
    base_model: str,
    lora_target_modules: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    mlp_hidden_dim: int,
    mlp_dropout: float,
    training_mode: str = "lora",
) -> dict[str, Any]:
    if training_mode not in TRAINING_MODES:
        raise ValueError(
            f"Unsupported training mode {training_mode!r}; "
            f"choose one of {', '.join(TRAINING_MODES)}"
        )
    if training_mode == "lora" and not lora_target_modules:
        raise ValueError("LoRA training requires at least one target module")
    return {
        "base_model": base_model,
        "training_mode": training_mode,
        "lora_target_modules": lora_target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "mlp_hidden_dim": mlp_hidden_dim,
        "mlp_dropout": mlp_dropout,
        "num_labels": 2,
        "reward_class_index": 1,
        "architecture": f"deberta_v3_{training_mode}_mlp_pair_classifier",
    }


def _pair_classifier_class(torch: Any, transformers: Any) -> type:
    class DebertaPairClassifier(transformers.PreTrainedModel):
        base_model_prefix = "encoder"

        def __init__(self, encoder: Any, mlp_hidden_dim: int, mlp_dropout: float) -> None:
            super().__init__(encoder.config)
            self.encoder = encoder
            self.num_labels = 2
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(encoder.config.hidden_size, mlp_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(mlp_dropout),
                torch.nn.Linear(mlp_hidden_dim, self.num_labels),
            )

        def get_input_embeddings(self) -> Any:
            return self.encoder.get_input_embeddings()

        def set_input_embeddings(self, value: Any) -> None:
            self.encoder.set_input_embeddings(value)

        def enable_input_require_grads(self) -> None:
            if hasattr(self.encoder, "enable_input_require_grads"):
                self.encoder.enable_input_require_grads()

        def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable(**kwargs)

        def gradient_checkpointing_disable(self) -> None:
            if hasattr(self.encoder, "gradient_checkpointing_disable"):
                self.encoder.gradient_checkpointing_disable()

        def forward(
            self,
            input_ids: Any = None,
            attention_mask: Any = None,
            token_type_ids: Any = None,
            labels: Any = None,
            **kwargs: Any,
        ) -> Any:
            encoder_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            if token_type_ids is not None:
                encoder_kwargs["token_type_ids"] = token_type_ids
            outputs = self.encoder(**encoder_kwargs)
            logits = self.classifier(outputs.last_hidden_state[:, 0])
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(logits, labels.long())
            return transformers.modeling_outputs.SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return DebertaPairClassifier


def _normalize_lora_targets(value: Any) -> list[str]:
    if isinstance(value, str):
        targets = parse_csv(value)
    else:
        targets = [str(item).strip() for item in value if str(item).strip()]
    if not targets:
        raise ValueError("LoRA training requires at least one target module")
    return targets


def _resolve_lora_targets(model: Any, requested_targets: list[str], torch: Any) -> list[str]:
    if requested_targets == ["all-linear"]:
        targets = [
            f"encoder.{name}"
            for name, module in model.encoder.named_modules()
            if name and isinstance(module, torch.nn.Linear)
        ]
        if not targets:
            raise ValueError("The encoder contains no Linear layers that can receive LoRA adapters")
        return targets

    available_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in model.encoder.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    missing = sorted(
        target
        for target in requested_targets
        if "." not in target and target not in available_suffixes
    )
    if missing:
        raise ValueError(
            "Requested LoRA target modules were not found in the encoder: "
            f"{missing}. Available Linear module names: {sorted(available_suffixes)}"
        )
    return requested_targets


def build_pair_classifier(
    config: dict[str, Any],
    *,
    apply_lora: bool | None = None,
    torch_dtype: Any = None,
    gradient_checkpointing: bool = False,
) -> Any:
    import torch
    import transformers

    training_mode = str(config.get("training_mode", "lora"))
    if training_mode not in TRAINING_MODES:
        raise ValueError(
            f"Unsupported training mode {training_mode!r}; "
            f"choose one of {', '.join(TRAINING_MODES)}"
        )
    use_lora = training_mode == "lora" if apply_lora is None else apply_lora

    model_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    encoder = transformers.AutoModel.from_pretrained(config["base_model"], **model_kwargs)
    classifier_type = _pair_classifier_class(torch, transformers)
    model = classifier_type(
        encoder,
        int(config["mlp_hidden_dim"]),
        float(config["mlp_dropout"]),
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if not use_lora:
        # AutoModel and the classifier head are trainable by default.
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model

    from peft import LoraConfig, get_peft_model

    requested_targets = _normalize_lora_targets(config["lora_target_modules"])
    resolved_targets = _resolve_lora_targets(model, requested_targets, torch)
    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=resolved_targets,
        modules_to_save=["classifier"],
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    if gradient_checkpointing:
        peft_model.enable_input_require_grads()
    return peft_model


def _load_full_checkpoint(model: Any, checkpoint_path: Path) -> Any:
    import transformers

    safe_index = checkpoint_path / "model.safetensors.index.json"
    bin_index = checkpoint_path / "pytorch_model.bin.index.json"
    index_path = safe_index if safe_index.is_file() else bin_index
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid checkpoint index: {index_path}")
        expected_keys = set(model.state_dict())
        loaded_keys: set[str] = set()
        unexpected_keys: set[str] = set()
        for shard_name in sorted(set(weight_map.values())):
            shard_path = checkpoint_path / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
            shard = transformers.modeling_utils.load_state_dict(str(shard_path))
            unexpected_keys.update(set(shard) - expected_keys)
            loaded_keys.update(shard)
            model.load_state_dict(shard, strict=False)
        missing_keys = expected_keys - loaded_keys
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Sharded full-model checkpoint does not match the configured architecture: "
                f"missing={sorted(missing_keys)}, "
                f"unexpected={sorted(unexpected_keys)}"
            )
        return model

    candidates = (
        checkpoint_path / "model.safetensors",
        checkpoint_path / "pytorch_model.bin",
    )
    weights_path = next((path for path in candidates if path.is_file()), None)
    if weights_path is None:
        raise FileNotFoundError(
            "Missing full-model weights. Expected one of: "
            + ", ".join(str(path) for path in candidates)
        )
    state_dict = transformers.modeling_utils.load_state_dict(str(weights_path))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Full-model checkpoint does not match the configured architecture: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model


def load_pair_classifier(
    checkpoint: str | Path,
    torch_dtype: Any = None,
) -> tuple[Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    config_path = checkpoint_path / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing process-reward model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_mode = str(config.get("training_mode", "lora"))

    if training_mode == "lora":
        from peft import PeftModel

        base = build_pair_classifier(
            config,
            apply_lora=False,
            torch_dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(base, str(checkpoint_path))
        return model, config

    if training_mode == "full":
        model = build_pair_classifier(
            config,
            apply_lora=False,
            torch_dtype=torch_dtype,
        )
        return _load_full_checkpoint(model, checkpoint_path), config

    raise ValueError(
        f"Unsupported training mode {training_mode!r} in {config_path}; "
        f"choose one of {', '.join(TRAINING_MODES)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--training_mode",
        choices=TRAINING_MODES,
        default="lora",
        help="'lora' trains adapters plus the classifier; 'full' trains every parameter.",
    )
    parser.add_argument(
        "--lora_target_modules",
        default=DEFAULT_LORA_TARGETS,
        help=(
            "Comma-separated encoder Linear module names. The default "
            "'all-linear' attaches LoRA to every encoder Linear layer while "
            "excluding the task classifier."
        ),
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--config_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = parse_csv(args.lora_target_modules)
    config = model_configuration(
        args.base_model,
        targets,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.mlp_hidden_dim,
        args.mlp_dropout,
        training_mode=args.training_mode,
    )
    if args.config_only:
        print_config(config)
        return
    model = build_pair_classifier(config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"trainable params: {trainable:,} || all params: {total:,} || "
            f"trainable%: {100 * trainable / total:.4f}"
        )


if __name__ == "__main__":
    main()
