#!/usr/bin/env python
"""EXP 2 (score stage, GPU) — boundary statistics on the four populations.

Statistics (per boundary / per chunk):
  s1  cross-boundary similarity: SBERT (all-MiniLM-L6-v2) cosine between
      adjacent chunks.
  s2  within-chunk dispersion: mean pairwise cosine of a chunk's sentences
      (regex split on [.!?] + newline); single-sentence chunks skipped and
      counted.
  s3  boundary NLI profile: microsoft/deberta-xlarge-mnli
      P(entail/neutral/contradiction) for (chunk_k premise, chunk_k+1
      hypothesis), batch 128, fp16, truncation 512.

Deliverables: per-statistic distribution table (mean/p25/p50/p75) x 4
populations + Cohen's d separation summary (GOLD vs MERGED / FRAGMENTED /
PRODUCED). Statistics that fail to separate GOLD from the corruptions are
marked uninformative.
"""
import argparse
import json
import os
import re
from collections import Counter

import numpy as np

POPS = ("gold", "merged", "fragmented", "produced")
SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(chunk):
    return [s.strip() for s in SENT_RE.split(chunk) if s.strip()]


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp else None


def dist(a):
    a = np.asarray(a, dtype=np.float64)
    if not len(a):
        return None
    return {"n": int(len(a)), "mean": float(a.mean()),
            "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--popdir", default="/scratch/sghos104/rlpt/chunk_eval/exp2_populations")
    ap.add_argument("--outdir", default="/scratch/sghos104/rlpt/chunk_eval")
    ap.add_argument("--nli_batch", type=int, default=128)
    ap.add_argument("--emb_batch", type=int, default=512)
    ap.add_argument("--limit_records", type=int, default=0, help="smoke")
    args = ap.parse_args()

    import orjson
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda"
    assert torch.cuda.is_available()
    sbert = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device)
    nli_name = "microsoft/deberta-xlarge-mnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_name)
    # Plain fp32 + TF32 matmul. This DeBERTa-v1 implementation hard-codes
    # masked_fill(finfo(fp32).min), which overflows under fp16 AND bf16
    # autocast (fp32 min just exceeds bf16 range), and half weights trip a
    # dtype mismatch — reduced precision is unusable here; TF32 recovers
    # most of the throughput on A100.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    nli = (AutoModelForSequenceClassification.from_pretrained(nli_name)
           .to(device).eval())
    id2label = {int(k): v.lower() for k, v in nli.config.id2label.items()}
    print(f"[exp2-score] NLI labels: {id2label}", flush=True)

    stats = {}
    per_pop_values = {}
    for pop in POPS:
        path = os.path.join(args.popdir, f"{pop}.jsonl")
        records = []
        with open(path, "rb") as f:
            for line in f:
                records.append(orjson.loads(line))
                if args.limit_records and len(records) >= args.limit_records:
                    break
        print(f"[exp2-score] {pop}: {len(records)} records", flush=True)

        # ---- s1: cross-boundary SBERT cosine ----
        all_chunks, spans = [], []
        for r in records:
            spans.append((len(all_chunks), len(r["chunks"])))
            all_chunks.extend(r["chunks"])
        emb = sbert.encode(all_chunks, batch_size=args.emb_batch,
                           convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False)
        s1 = []
        pair_premise, pair_hypo = [], []
        for start, n in spans:
            for k in range(n - 1):
                s1.append(float(emb[start + k] @ emb[start + k + 1]))
                pair_premise.append(all_chunks[start + k])
                pair_hypo.append(all_chunks[start + k + 1])

        # ---- s2: within-chunk dispersion ----
        sent_lists = [sentences(c) for c in all_chunks]
        flat_sents, sent_spans = [], []
        for sl in sent_lists:
            sent_spans.append((len(flat_sents), len(sl)))
            flat_sents.extend(sl)
        semb = sbert.encode(flat_sents, batch_size=args.emb_batch,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False)
        s2, single_sent = [], 0
        for start, n in sent_spans:
            if n < 2:
                single_sent += 1
                continue
            e = semb[start:start + n]
            gram = e @ e.T
            iu = np.triu_indices(n, k=1)
            s2.append(float(gram[iu].mean()))

        # ---- s3: NLI profile over boundaries ----
        probs = np.zeros((len(pair_premise), 3), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(pair_premise), args.nli_batch):
                enc = nli_tok(pair_premise[i:i + args.nli_batch],
                              pair_hypo[i:i + args.nli_batch],
                              padding=True, truncation=True, max_length=512,
                              return_tensors="pt").to(device)
                lg = nli(**enc).logits.float()
                probs[i:i + args.nli_batch] = torch.softmax(lg, -1).cpu().numpy()
                if (i // args.nli_batch) % 200 == 0:
                    print(f"[exp2-score] {pop} NLI {i}/{len(pair_premise)}",
                          flush=True)
        by_label = {}
        for j, lab in id2label.items():
            key = ("entail" if "entail" in lab else
                   "contra" if "contra" in lab else "neutral")
            by_label[key] = probs[:, j]

        per_pop_values[pop] = {
            "s1_xboundary_cos": s1, "s2_within_disp": s2,
            "s3_entail": list(by_label["entail"]),
            "s3_neutral": list(by_label["neutral"]),
            "s3_contra": list(by_label["contra"])}
        stats[pop] = {
            "records": len(records), "chunks": len(all_chunks),
            "boundaries": len(s1), "single_sentence_chunks": single_sent,
            "s1_xboundary_cos": dist(s1), "s2_within_disp": dist(s2),
            "s3_entail": dist(by_label["entail"]),
            "s3_neutral": dist(by_label["neutral"]),
            "s3_contra": dist(by_label["contra"])}

    # ---- separation summary ----
    sep = {}
    for statname in ("s1_xboundary_cos", "s2_within_disp",
                     "s3_entail", "s3_neutral", "s3_contra"):
        row = {}
        for other in ("merged", "fragmented", "produced"):
            row[f"gold_vs_{other}"] = cohens_d(
                per_pop_values["gold"][statname], per_pop_values[other][statname])
        d_m = abs(row["gold_vs_merged"] or 0)
        d_f = abs(row["gold_vs_fragmented"] or 0)
        row["informative"] = bool(max(d_m, d_f) >= 0.2)
        sep[statname] = row

    out = {"populations": stats, "separation": sep}
    with open(os.path.join(args.outdir, "exp2_stats.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'statistic':20s} " + " ".join(f"{p:>12s}" for p in POPS))
    for statname in ("s1_xboundary_cos", "s2_within_disp",
                     "s3_entail", "s3_neutral", "s3_contra"):
        cells = " ".join(
            f"{stats[p][statname]['mean']:12.4f}" if stats[p][statname] else
            f"{'—':>12s}" for p in POPS)
        s = sep[statname]
        print(f"{statname:20s} {cells}   d(m/f/p)="
              f"{s['gold_vs_merged']:.2f}/{s['gold_vs_fragmented']:.2f}/"
              f"{s['gold_vs_produced']:.2f}"
              f"{'' if s['informative'] else '  [UNINFORMATIVE]'}")
    print("[exp2-score] DONE")


if __name__ == "__main__":
    main()
