from __future__ import annotations

from typing import Any

import torch

from .transformer_utils import load_encoder, normalize_embeddings


class BertScoreScorer:
    name = "bertscore"
    model_name = "microsoft/deberta-xlarge-mnli"

    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer, self.model = load_encoder(self.model_name, device)

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            all_texts = [text for pair in batch for text in pair]
            encoded = self.tokenizer(
                all_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded).last_hidden_state
            hidden = normalize_embeddings(output)
            mask = encoded["attention_mask"].bool()
            for i in range(0, len(all_texts), 2):
                emb_a = hidden[i][mask[i]]
                emb_b = hidden[i + 1][mask[i + 1]]
                if emb_a.numel() == 0 or emb_b.numel() == 0:
                    precision = recall = f1 = 0.0
                else:
                    sim = emb_a @ emb_b.T
                    precision = float(sim.max(dim=0).values.mean().cpu())
                    recall = float(sim.max(dim=1).values.mean().cpu())
                    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                rows.append(
                    {
                        "raw_score": f1,
                        "bertscore_precision": precision,
                        "bertscore_recall": recall,
                        "bertscore_f1": f1,
                    }
                )
        return rows

