#!/usr/bin/env python
"""Dataset projection helpers for reward-model training."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import iter_jsonl, print_config


@dataclass(frozen=True)
class RewardExample:
    anchor: str
    generated: str
    compatibility_label: int
    source_sample_id: str


def load_reward_examples(path: str | Path) -> list[RewardExample]:
    """Load only the fields used to train the judge.

    The input may be a generated-pairs split or an enriched NLI-cache split.
    Any provenance or frozen-NLI feature fields are intentionally discarded.
    """
    examples: list[RewardExample] = []
    for line_index, row in enumerate(iter_jsonl(path), start=1):
        anchor = row.get("anchor")
        generated = row.get("generated")
        label = row.get("compatibility_label")
        source_sample_id = row.get("source_sample_id")
        if not isinstance(anchor, str) or not anchor.strip():
            raise ValueError(f"Invalid anchor at line {line_index}")
        if not isinstance(generated, str) or not generated.strip():
            raise ValueError(f"Invalid generated text at line {line_index}")
        if label not in {0, 1}:
            raise ValueError(f"Invalid compatibility_label at line {line_index}: {label!r}")
        if not isinstance(source_sample_id, (str, int)) or not str(source_sample_id).strip():
            raise ValueError(f"Invalid source_sample_id at line {line_index}")
        examples.append(
            RewardExample(
                anchor=anchor.strip(),
                generated=generated.strip(),
                compatibility_label=int(label),
                source_sample_id=str(source_sample_id),
            )
        )
    return examples


def split_reward_examples(
    examples: Sequence[RewardExample], validation_ratio: float, seed: int
) -> tuple[list[RewardExample], list[RewardExample]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    groups = sorted({example.source_sample_id for example in examples})
    random.Random(seed).shuffle(groups)
    if validation_ratio == 0 or len(groups) < 2:
        validation_groups: set[str] = set()
    else:
        validation_count = max(1, int(round(len(groups) * validation_ratio)))
        validation_count = min(validation_count, len(groups) - 1)
        validation_groups = set(groups[:validation_count])
    train = [example for example in examples if example.source_sample_id not in validation_groups]
    validation = [example for example in examples if example.source_sample_id in validation_groups]
    return train, validation


class TokenizedPairDataset:
    def __init__(self, examples: Sequence[RewardExample], tokenizer: Any, max_length: int) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.anchor,
            example.generated,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        encoded["labels"] = example.compatibility_label
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config_only:
        print_config(vars(args))
        return
    examples = load_reward_examples(args.input_jsonl)
    train, validation = split_reward_examples(examples, args.validation_ratio, args.seed)
    print_config(
        {
            "input_examples": len(examples),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_source_samples": len({item.source_sample_id for item in train}),
            "validation_source_samples": len({item.source_sample_id for item in validation}),
        }
    )


if __name__ == "__main__":
    main()
