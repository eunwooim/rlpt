from __future__ import annotations

from typing import Any

import torch

from .transformer_utils import load_encoder, mean_pool, normalize_embeddings


class SbertCosineScorer:
    name = "sbert"
    model_name = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer, self.model = load_encoder(self.model_name, device)

    @torch.inference_mode()
    def _encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            chunks.append(normalize_embeddings(pooled).cpu())
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0)

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int) -> list[dict[str, Any]]:
        if not pairs:
            return []
        text_a = [pair[0] for pair in pairs]
        text_b = [pair[1] for pair in pairs]
        emb_a = self._encode(text_a, batch_size)
        emb_b = self._encode(text_b, batch_size)
        scores = (emb_a * emb_b).sum(dim=1).tolist()
        return [{"raw_score": float(score)} for score in scores]

