from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass
class ScorerError:
    scorer: str
    error_type: str
    error_message: str

    @classmethod
    def from_exception(cls, scorer: str, exc: BaseException) -> "ScorerError":
        return cls(scorer=scorer, error_type=type(exc).__name__, error_message=str(exc))


class Scorer(Protocol):
    name: str

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int) -> list[dict[str, Any]]:
        ...


def make_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

