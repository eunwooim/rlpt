#!/usr/bin/env python
"""Bidirectional-NLI match scoring for the six-arm GRPO campaign.

Pair score, for rollout segment i and gold step j:
    s(i,j) = 0.5*E(i->j) + 0.5*E(j->i) - 1.0*max(C(i->j), C(j->i)), clipped to [0,1]
where E(a->b) = P(entailment | premise=a, hypothesis=b) and C likewise for
contradiction, from microsoft/deberta-xlarge-mnli. Label indices are read from
config.id2label — never hardcoded. The model runs fp32 with TF32 matmul
(fp16 NaNs on this checkpoint are a documented trap).

Match term: Hungarian one-to-one assignment on the s matrix, pairs kept iff
s >= tau, precision-dominant F_beta (beta=0.5) on matched mass:
    P = mass / n_segments, R = mass / n_gold, F = (1+b^2)PR / (b^2 P + R).
"""
import numpy as np
import torch

NLI_MODEL = "microsoft/deberta-xlarge-mnli"
BETA = 0.5


class NLIScorer:
    def __init__(self, model_name=NLI_MODEL, device=None, max_length=512, batch_size=64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=torch.float32
        ).to(self.device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.id2label = id2label
        try:
            self.ent_idx = next(k for k, v in id2label.items() if "entail" in v)
            self.con_idx = next(k for k, v in id2label.items() if "contra" in v)
        except StopIteration:
            raise RuntimeError(f"id2label lacks entailment/contradiction: {id2label}")
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.inference_mode()
    def ec_probs(self, premises, hypotheses):
        """P(entailment), P(contradiction) for each (premise, hypothesis) pair."""
        ent, con = [], []
        for a in range(0, len(premises), self.batch_size):
            enc = self.tok(
                premises[a : a + self.batch_size],
                hypotheses[a : a + self.batch_size],
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
            ent.extend(probs[:, self.ent_idx].cpu().tolist())
            con.extend(probs[:, self.con_idx].cpu().tolist())
        return ent, con

    def pair_scores(self, segments, gold_steps):
        """Full s-matrix (n_seg x n_gold) via one batched forward per direction."""
        n_s, n_g = len(segments), len(gold_steps)
        if n_s == 0 or n_g == 0:
            return np.zeros((n_s, n_g), dtype=np.float64)
        prem_f, hyp_f = [], []
        for s in segments:
            for g in gold_steps:
                prem_f.append(s)
                hyp_f.append(g)
        e_fwd, c_fwd = self.ec_probs(prem_f, hyp_f)   # i -> j
        e_rev, c_rev = self.ec_probs(hyp_f, prem_f)   # j -> i
        s_mat = (
            0.5 * np.array(e_fwd) + 0.5 * np.array(e_rev)
            - np.maximum(np.array(c_fwd), np.array(c_rev))
        ).clip(0.0, 1.0)
        return s_mat.reshape(n_s, n_g)


def match_fbeta(s_mat: np.ndarray, tau: float, beta: float = BETA):
    """Hungarian assignment + threshold + precision-dominant F_beta on matched mass."""
    from scipy.optimize import linear_sum_assignment

    n_s, n_g = s_mat.shape
    if n_s == 0 or n_g == 0:
        return {"match": 0.0, "n_matched": 0, "n_segments": n_s, "n_gold": n_g,
                "matched_mass": 0.0, "precision": 0.0, "recall": 0.0}
    rows, cols = linear_sum_assignment(-s_mat)
    kept = [(i, j, float(s_mat[i, j])) for i, j in zip(rows, cols) if s_mat[i, j] >= tau]
    mass = sum(v for _, _, v in kept)
    precision = mass / n_s
    recall = mass / n_g
    b2 = beta * beta
    denom = b2 * precision + recall
    f = 0.0 if denom <= 0 else (1 + b2) * precision * recall / denom
    return {"match": float(f), "n_matched": len(kept), "n_segments": n_s, "n_gold": n_g,
            "matched_mass": float(mass), "precision": float(precision), "recall": float(recall)}
