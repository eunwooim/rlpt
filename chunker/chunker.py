#!/usr/bin/env python
"""Inference: segment a raw reasoning trajectory into step-like chunks.

chunk(text, min_tokens=8, max_tokens=220) -> {
    "chunks": [str, ...],
    "split_token_indices": [int, ...],     # core-token index of each internal boundary
    "split_char_offsets": [int, ...],      # char offset (into text) where each chunk ends
}

Boundaries are chosen by DP constrained decoding over CANDIDATE tokens (the same
expanded rule used in training, imported from chunker_common), maximizing total
(P(split) - threshold) subject to every segment being within [min_tokens,
max_tokens]. If no candidate is reachable within max_tokens, a split is forced at
the least-bad token and a warning is emitted.
"""
import argparse, json, os, sys, warnings
import numpy as np

os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import chunker_common as cc

DEFAULT_MODEL_DIR = "/scratch/sghos104/rlpt/chunker/runs/deberta_chunker/best"
_PEN = 8.0        # penalty for a forced split at a non-candidate token
_MINPEN = 2.0     # soft penalty for a segment shorter than min_tokens

_tok = _model = _thr = _device = None


def load(model_dir=DEFAULT_MODEL_DIR, threshold=None):
    global _tok, _model, _thr, _device
    _tok = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForTokenClassification.from_pretrained(model_dir)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device).eval()
    if threshold is not None:
        _thr = threshold
    else:
        meta = os.path.join(model_dir, "chunker_meta.json")
        _thr = json.load(open(meta)).get("best_thr", 0.5) if os.path.exists(meta) else 0.5
    return _thr


@torch.no_grad()
def _token_probs(text):
    """P(split) per CORE token (no specials), averaging overlapping windows."""
    enc = _tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    T = len(ids)
    prob_sum = np.zeros(T); prob_cnt = np.zeros(T)
    for w in cc.make_windows(ids, offs, [cc.LABEL_IGNORE] * T, _tok.cls_token_id, _tok.sep_token_id):
        wi = torch.tensor([w["input_ids"]], device=_device)
        logits = _model(wi).logits[0].float().cpu().numpy()      # (win_len, 2)
        p1 = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p1 = p1[:, 1] / p1.sum(axis=-1)
        core = p1[1:-1]                                          # drop CLS/SEP
        prob_sum[w["core_start"]:w["core_end"]] += core
        prob_cnt[w["core_start"]:w["core_end"]] += 1.0
    prob_cnt[prob_cnt == 0] = 1.0
    return ids, offs, prob_sum / prob_cnt


def _dp_boundaries(prob, cand, min_t, max_t, thr, hard_min):
    """Max-score boundary set ending at the forced final token F=T-1.

    max_tokens is ALWAYS a hard constraint (non-candidate tokens carry -_PEN so a
    forced split within max is always reachable). min_tokens: when hard_min=True,
    segments shorter than min are forbidden outright; when False, they cost a soft
    _MINPEN. Returns (internal_boundary_indices, forced_flag, feasible)."""
    T = len(prob)
    F = T - 1
    score = np.where(cand, prob - thr, -_PEN).astype(np.float64)
    score[F] = 0.0                                              # end of text is forced, neutral
    NEG = -1e18
    best = np.full(T, NEG); bp = np.full(T, -2, dtype=np.int64)
    for j in range(T):
        sj = score[j]
        seglen0 = j + 1                                         # segment from virtual start (-1)
        if hard_min:
            ok0, pen0 = (min_t <= seglen0 <= max_t), 0.0
        else:
            ok0, pen0 = (1 <= seglen0 <= max_t), (_MINPEN if seglen0 < min_t else 0.0)
        if ok0 and (sj - pen0) > best[j]:
            best[j] = sj - pen0; bp[j] = -1
        lo = max(0, j - max_t)
        hi = (j - min_t) if hard_min else (j - 1)               # gap>=min (hard) or gap>=1 (soft)
        for i in range(lo, hi + 1):
            if i < 0 or best[i] <= NEG:
                continue
            gap = j - i
            pen = 0.0 if hard_min else (_MINPEN if gap < min_t else 0.0)
            val = best[i] + sj - pen
            if val > best[j]:
                best[j] = val; bp[j] = i
    feasible = best[F] > NEG / 2
    bnds = []; j = F
    while j >= 0:
        bnds.append(j); j = bp[j]
        if j == -1 or j == -2:
            break
    bnds = sorted(b for b in bnds if b != F)                    # internal boundaries only
    forced = any(not cand[b] for b in bnds)
    return bnds, forced, feasible


def chunk(text, min_tokens=8, max_tokens=220):
    if _model is None:
        load()
    ids, offs, prob = _token_probs(text)
    T = len(ids)
    if T <= 1:
        return {"chunks": [text] if text else [], "split_token_indices": [],
                "split_char_offsets": [], "segment_token_lengths": [T] if text else []}
    cand = np.array(cc.candidate_token_mask(text, offs), dtype=bool)
    # try HARD min-8 first; fall back to soft only if the text is too short to satisfy it
    bnds, forced, feasible = _dp_boundaries(prob, cand, min_tokens, max_tokens, _thr, hard_min=True)
    if not feasible:
        bnds, forced, _ = _dp_boundaries(prob, cand, min_tokens, max_tokens, _thr, hard_min=False)
        warnings.warn("chunk(): text too short for a hard min_tokens solution; used soft-min fallback.")
    if forced:
        warnings.warn("chunk(): forced a split at a non-candidate token (no candidate within max_tokens).")
    cuts = [offs[b][1] for b in bnds]                           # char offset where each chunk ends
    # segment lengths in TOKENS (for downstream min-fragment auditing)
    seg_tok, prev = [], -1
    for b in bnds + [T - 1]:
        seg_tok.append(b - prev); prev = b
    chunks, pc = [], 0
    for c in cuts + [len(text)]:
        seg = text[pc:c].strip("\n")
        if seg:
            chunks.append(seg)
        pc = c
    return {"chunks": chunks, "split_token_indices": bnds, "split_char_offsets": cuts,
            "segment_token_lengths": seg_tok}


def _demo(model_dir, traj_path, n=5):
    load(model_dir)
    rows = [json.loads(l) for l in open(traj_path)]
    multi = [r for r in rows if r["kind"] == "multi"][:n]
    for r in multi:
        steps = r["steps"]
        text = cc.join_steps(steps)
        gold, _ = cc.gold_boundary_char_positions(steps)
        out = chunk(text)
        print("=" * 78)
        print(f"lang={r['lang']}  n_steps={len(steps)}  gold boundaries(char)={gold}")
        print(f"  predicted char cuts = {out['split_char_offsets']}  (#chunks={len(out['chunks'])})")
        # align: for each gold, nearest predicted cut delta
        for g in gold:
            near = min(out["split_char_offsets"], key=lambda c: abs(c - g)) if out["split_char_offsets"] else None
            d = (near - g) if near is not None else None
            print(f"    gold@{g:<5d} nearest_pred@{near}  delta={d}   …{text[max(0,g-22):g+1]!r}")
        print(f"  predicted chunk[0]: {out['chunks'][0][:80]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--traj", default="/scratch/sghos104/rlpt/chunker/data/val_traj.jsonl")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    _demo(args.model_dir, args.traj, args.n)
