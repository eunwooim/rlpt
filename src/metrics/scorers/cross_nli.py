from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm

from .transformer_utils import load_classifier


class CrossNLIScorer:
    name = "cross_nli"
    model_name = "cross-encoder/nli-deberta-v3-large"

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer, self.model = load_classifier(self.model_name, device)
        self.entailment_index = self._entailment_index()

    def _entailment_index(self) -> int:
        id2label = self.model.config.id2label
        for index, label in id2label.items():
            if "entail" in str(label).lower():
                return int(index)
        raise ValueError(f"Could not infer entailment label from {id2label}")

    def _label_indices(self) -> dict[str, int]:
        labels: dict[str, int] = {}
        for index, label in self.model.config.id2label.items():
            normalized = str(label).lower()
            if "contrad" in normalized:
                labels["contradiction"] = int(index)
            elif "neutral" in normalized:
                labels["neutral"] = int(index)
            elif "entail" in normalized:
                labels["entailment"] = int(index)
        return labels

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        label_indices = self._label_indices()
        batch_starts = range(0, len(pairs), batch_size)
        for start in tqdm(
            batch_starts,
            total=(len(pairs) + batch_size - 1) // batch_size,
            desc="cross_nli batches",
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
            logits_cpu = logits.cpu()
            for prob_row, logit_row in zip(probs, logits_cpu):
                output = {
                    "raw_score": float(prob_row[self.entailment_index]),
                    "cross_nli_logits": [float(value) for value in logit_row.tolist()],
                    "cross_nli_probs": [float(value) for value in prob_row.tolist()],
                }
                for label_name, label_index in label_indices.items():
                    output[f"cross_nli_{label_name}"] = float(prob_row[label_index])
                    output[f"cross_nli_{label_name}_logit"] = float(logit_row[label_index])
                rows.append(output)
        return rows
