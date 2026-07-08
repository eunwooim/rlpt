from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm

from .transformer_utils import load_classifier


class BGERerankerScorer:
    name = "bge_reranker"
    model_name = "BAAI/bge-reranker-large"

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer, self.model = load_classifier(self.model_name, device)

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        batch_starts = range(0, len(pairs), batch_size)
        for start in tqdm(
            batch_starts,
            total=(len(pairs) + batch_size - 1) // batch_size,
            desc="bge_reranker batches",
            unit="batch",
            leave=False,
        ):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [a for a, _ in batch],
                [b for _, b in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits.detach().cpu()
            flat = logits.reshape(logits.shape[0], -1)
            for row in flat:
                raw_score = float(row[0])
                rows.append(
                    {
                        "raw_score": raw_score,
                        "bge_reranker_logit": raw_score,
                        "bge_reranker_logits": [float(value) for value in row.tolist()],
                    }
                )
        return rows
