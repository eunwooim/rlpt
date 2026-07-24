#!/usr/bin/env python
"""DeBERTaV3 pair classifier with configurable LoRA and an MLP head."""

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
DEFAULT_LORA_TARGETS = "query_proj,value_proj"


def model_configuration(
    base_model: str,
    lora_target_modules: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    mlp_hidden_dim: int,
    mlp_dropout: float,
) -> dict[str, Any]:
    return {
        "base_model": base_model,
        "lora_target_modules": lora_target_modules,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "mlp_hidden_dim": mlp_hidden_dim,
        "mlp_dropout": mlp_dropout,
        "num_labels": 2,
        "reward_class_index": 1,
        "architecture": "deberta_v3_lora_mlp_pair_classifier",
    }


def _pair_classifier_class(torch: Any, transformers: Any) -> type:
    class DebertaPairClassifier(torch.nn.Module):
        def __init__(self, encoder: Any, mlp_hidden_dim: int, mlp_dropout: float) -> None:
            super().__init__()
            self.encoder = encoder
            self.config = encoder.config
            self.num_labels = 2
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(encoder.config.hidden_size, mlp_hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(mlp_dropout),
                torch.nn.Linear(mlp_hidden_dim, self.num_labels),
            )

        def get_input_embeddings(self) -> Any:
            return self.encoder.get_input_embeddings()

        def enable_input_require_grads(self) -> None:
            if hasattr(self.encoder, "enable_input_require_grads"):
                self.encoder.enable_input_require_grads()

        def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable(**kwargs)

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


def build_pair_classifier(
    config: dict[str, Any],
    *,
    apply_lora: bool = True,
    torch_dtype: Any = None,
    gradient_checkpointing: bool = False,
) -> Any:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    model_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    encoder = transformers.AutoModel.from_pretrained(config["base_model"], **model_kwargs)
    classifier_type = _pair_classifier_class(torch, transformers)
    model = classifier_type(encoder, int(config["mlp_hidden_dim"]), float(config["mlp_dropout"]))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if not apply_lora:
        return model
    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["lora_target_modules"]),
        modules_to_save=["classifier"],
        bias="none",
    )
    return get_peft_model(model, lora_config)


def load_pair_classifier(checkpoint: str | Path, torch_dtype: Any = None) -> tuple[Any, dict[str, Any]]:
    from peft import PeftModel

    checkpoint_path = Path(checkpoint)
    config_path = checkpoint_path / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing process-reward model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = build_pair_classifier(config, apply_lora=False, torch_dtype=torch_dtype)
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    return model, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora_target_modules", default=DEFAULT_LORA_TARGETS)
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
    if not targets:
        raise ValueError("--lora_target_modules must contain at least one module")
    config = model_configuration(
        args.base_model,
        targets,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.mlp_hidden_dim,
        args.mlp_dropout,
    )
    if args.config_only:
        print_config(config)
        return
    model = build_pair_classifier(config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print(model)


if __name__ == "__main__":
    main()
