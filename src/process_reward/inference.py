#!/usr/bin/env python
"""Score text pairs with a trained process-reward checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import iter_jsonl, print_config, write_jsonl
from process_reward.model import load_pair_classifier


class ProcessRewardScorer:
    """Inference wrapper returning binary compatibility probabilities."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", max_length: int = 256) -> None:
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.checkpoint = Path(checkpoint)
        self.max_length = max_length
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
            self.device = torch.device("cuda:0")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")
        self.model, self.config = load_pair_classifier(self.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint))
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls, checkpoint: str | Path, device: str = "auto", max_length: int = 256
    ) -> "ProcessRewardScorer":
        return cls(checkpoint=checkpoint, device=device, max_length=max_length)

    def score_pairs(self, pairs: Sequence[tuple[str, str]], batch_size: int = 32) -> list[float]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [anchor for anchor, _ in batch],
                [generated for _, generated in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = self.torch.softmax(logits, dim=-1)[:, int(self.config["reward_class_index"])]
            scores.extend(float(value) for value in probabilities.cpu().tolist())
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor", default=None)
    parser.add_argument("--generated", default=None)
    parser.add_argument("--pairs_jsonl", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--config_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config_only:
        print_config(vars(args))
        return
    single_pair = args.anchor is not None or args.generated is not None
    if single_pair and (args.anchor is None or args.generated is None):
        raise ValueError("--anchor and --generated must be provided together")
    if single_pair == bool(args.pairs_jsonl):
        raise ValueError("Provide either one --anchor/--generated pair or --pairs_jsonl")

    rows: list[dict[str, Any]]
    if args.pairs_jsonl:
        rows = list(iter_jsonl(args.pairs_jsonl))
        for line_index, row in enumerate(rows, start=1):
            if not isinstance(row.get("anchor"), str) or not isinstance(row.get("generated"), str):
                raise ValueError(f"Invalid pair at line {line_index}")
    else:
        rows = [{"anchor": args.anchor, "generated": args.generated}]
    scorer = ProcessRewardScorer.from_pretrained(args.checkpoint, args.device, args.max_length)
    scores = scorer.score_pairs(
        [(str(row["anchor"]), str(row["generated"])) for row in rows],
        batch_size=args.batch_size,
    )
    output_rows = [{**row, "process_reward": score} for row, score in zip(rows, scores)]
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, output_rows)
    else:
        for row in output_rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

"""
python src/process_reward/visualprocessbench.py \
  --validate_data_only \
  --data_path /mnt/data1/eunwooim/VisualProcessBench/test.jsonl \
  --model_path src/outputs/process_reward_judge/v1_1_0/checkpoints/final
"""