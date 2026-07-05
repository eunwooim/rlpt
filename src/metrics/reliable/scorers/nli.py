from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm

from .base import clamp01
from .transformer_utils import load_classifier


class NLIScorer:
    name = "nli"
    model_name = "microsoft/deberta-large-mnli"

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer, self.model = load_classifier(self.model_name, device)
        self.label_map = self._label_map()

    def _label_map(self) -> dict[str, int]:
        id2label = self.model.config.id2label
        labels = {str(label).lower(): int(i) for i, label in id2label.items()}
        mapped: dict[str, int] = {}
        for key in ["entailment", "contradiction", "neutral"]:
            for label, idx in labels.items():
                if key in label:
                    mapped[key] = idx
                    break
        missing = sorted({"entailment", "contradiction", "neutral"} - set(mapped))
        if missing:
            raise ValueError(f"Could not infer NLI labels {missing} from {id2label}")
        return mapped

    @torch.inference_mode()
    def _directional(self, pairs: list[tuple[str, str]], batch_size: int, desc: str) -> list[dict[str, float]]:
        outputs: list[dict[str, float]] = []
        batch_starts = range(0, len(pairs), batch_size)
        for start in tqdm(
            batch_starts,
            total=(len(pairs) + batch_size - 1) // batch_size,
            desc=desc,
            unit="batch",
            leave=False,
        ):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [a for a, _ in batch],
                [b for _, b in batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu()
            for row in probs:
                outputs.append(
                    {
                        "E": float(row[self.label_map["entailment"]]),
                        "C": float(row[self.label_map["contradiction"]]),
                        "N": float(row[self.label_map["neutral"]]),
                    }
                )
        return outputs

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int) -> list[dict[str, Any]]:
        if not pairs:
            return []
        ab = self._directional(pairs, batch_size, "nli forward")
        ba = self._directional([(b, a) for a, b in pairs], batch_size, "nli reverse")
        rows: list[dict[str, Any]] = []
        for forward, reverse in zip(ab, ba):
            e_ab = forward["E"]
            c_ab = forward["C"]
            n_ab = forward["N"]
            e_ba = reverse["E"]
            c_ba = reverse["C"]
            n_ba = reverse["N"]
            coverage = clamp01(0.7 * e_ab + 0.3 * e_ba - 0.5 * max(c_ab, c_ba))
            equiv = clamp01(0.5 * e_ab + 0.5 * e_ba - 0.5 * max(c_ab, c_ba))
            rows.append(
                {
                    "raw_score": coverage,
                    "E_ab": e_ab,
                    "E_ba": e_ba,
                    "C_ab": c_ab,
                    "C_ba": c_ba,
                    "N_ab": n_ab,
                    "N_ba": n_ba,
                    "nli_score_coverage": coverage,
                    "nli_score_equiv": equiv,
                }
            )
        return rows
