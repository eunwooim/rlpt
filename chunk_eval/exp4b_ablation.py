#!/usr/bin/env python
"""EXP 4b — list-tier ablation on the STRUCTURED subset (gold records whose
steps contain internal marker typography; n = 334,411 in the full run).

Arms, all on the same "\n\n" join, everything else identical to V1
(guards + release model as fallback):
  A  step > para > model   (list tier OFF)
  B  step > model          (list AND para OFF)
Reference rows (from exp4_stats_full.json): V1 full cascade (list on)
0.392 and V2 model-only 0.755 on the same subset.

Reports the standard table (macroF1, coverage, density ratio, WindowDiff,
perfect%) + which tier/path resolved each record, per arm.
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/scratch/sghos104/rlpt/chunk_canonical")
sys.path.insert(0, HERE)
import cascade as CAS  # noqa: E402
from fence_guard import fence_regions, fenced_char_share  # noqa: E402
from marker_variants import marker_split_tiers  # noqa: E402
from exp4_resegment import (  # noqa: E402
    has_internal_markers, gold_boundaries_nonws, chunk_boundaries_nonws,
    prf, window_diff, run_model_batch)

loads = CAS.loads
ARMS = {"A_step_para_model": ("step", "para"), "B_step_model": ("step",)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl")
    ap.add_argument("--outdir", default="/scratch/sghos104/rlpt/chunk_eval")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--batch_tokens", type=int, default=65_536)
    ap.add_argument("--dp_workers", type=int, default=12)
    ap.add_argument("--flush_every", type=int, default=20_000)
    ap.add_argument("--limit_records", type=int, default=0, help="smoke")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    assert torch.cuda.is_available()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(CAS.MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(CAS.MODEL_DIR).to(device).eval()
    CAS._W["cls"], CAS._W["sep"], CAS._W["pad"] = (
        tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(args.dp_workers,
                                        initializer=CAS._worker_init,
                                        initargs=(CAS.CODE,))

    rows = {arm: [] for arm in ARMS}          # per-record metric dicts
    mixes = {arm: Counter() for arm in ARMS}
    queues = {arm: [] for arm in ARMS}        # (steps, joined, fences)
    n_struct = 0
    t0 = time.time()

    def score(arm, steps, chunks, method):
        pred = chunk_boundaries_nonws(chunks)
        gold = gold_boundaries_nonws(steps)
        prec, rec, f1, tp = prf(pred, gold)
        rows[arm].append({
            "prec": prec, "rec": rec, "f1": f1, "tp": tp,
            "n_pred": len(pred), "n_gold": len(gold),
            "density": (len(pred) / len(gold)) if gold else None,
            "wdiff": window_diff(steps, pred)})
        mixes[arm][method] += 1

    def flush(final=False):
        for arm in ARMS:
            q = queues[arm]
            if not q or (len(q) < args.flush_every and not final):
                continue
            batch = [(i, j) for i, (_s, j, _f) in enumerate(q)]
            fences = {i: f for i, (_s, _j, f) in enumerate(q)}
            out, _t = run_model_batch(model, tok, device, pool, batch, args, fences)
            for i, (steps, _j, _f) in enumerate(q):
                score(arm, steps, out[i], "model")
            queues[arm] = []

    with open(args.input, "rb") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0:
                print(f"[exp4b] row {i:,} structured={n_struct:,} "
                      f"t={time.time()-t0:.0f}s", flush=True)
            rec = loads(line)
            prov = rec["metadata"]["chunk_provenance"]
            if prov["method"] != "passthrough" or len(rec["steps"]) < 2:
                continue
            steps = rec["steps"]
            if not has_internal_markers(steps):
                continue
            n_struct += 1
            if args.limit_records and n_struct > args.limit_records:
                break
            joined = "\n\n".join(steps)
            for arm, tiers in ARMS.items():
                tier, chunks = marker_split_tiers(joined, tiers)
                if tier is not None:
                    score(arm, steps, chunks, "marker/" + tier)
                    continue
                nw = sum(1 for c in joined if not c.isspace())
                na = sum(1 for c in joined if not c.isspace() and not c.isalpha())
                if nw and na / nw > 0.5:
                    score(arm, steps, [joined], "skip/nonprose")
                    continue
                fences = fence_regions(joined)
                if fences and fenced_char_share(joined, fences) > 0.3:
                    score(arm, steps, [joined], "skip/code_fence")
                    continue
                queues[arm].append((steps, joined, fences))
            flush()
    flush(final=True)
    pool.close()
    pool.join()

    def agg(sel):
        micro_tp = sum(r["tp"] for r in sel)
        micro_p = sum(r["n_pred"] for r in sel)
        micro_g = sum(r["n_gold"] for r in sel)
        wds = [r["wdiff"] for r in sel if r["wdiff"] is not None]
        dens = [r["density"] for r in sel if r["density"]]
        return {"records": len(sel),
                "macro_f1": float(np.mean([r["f1"] for r in sel])),
                "coverage": micro_tp / micro_g if micro_g else None,
                "micro_prec": micro_tp / micro_p if micro_p else None,
                "density_ratio_mean": float(np.mean(dens)) if dens else None,
                "wdiff_mean": float(np.mean(wds)) if wds else None,
                "perfect_pct": 100 * sum(r["f1"] == 1.0 for r in sel) / max(1, len(sel))}

    report = {"n_structured": n_struct, "arms": {}}
    for arm in ARMS:
        report["arms"][arm] = {**agg(rows[arm]), "method_mix": dict(mixes[arm])}
        a = report["arms"][arm]
        print(f"[exp4b] {arm}: n={a['records']:,} macroF1={a['macro_f1']:.3f} "
              f"cov={a['coverage']:.3f} dens={a['density_ratio_mean']:.2f} "
              f"wdiff={a['wdiff_mean']:.3f} perfect={a['perfect_pct']:.1f}% "
              f"mix={a['method_mix']}", flush=True)
    with open(os.path.join(args.outdir, "exp4b_ablation.json"), "w") as fo:
        json.dump(report, fo, indent=2)
    print("[exp4b] DONE")


if __name__ == "__main__":
    main()
