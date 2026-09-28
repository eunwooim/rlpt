#!/usr/bin/env python
"""Cross-dataset strip-and-resegment, V2 protocol ONLY (the fair
cross-corpus test): join gold steps with a single space (kills every
marker tier / typography cue), force the model path (release chunker,
threshold 0.35, min/max 8/220), score predicted vs gold boundaries.

Datasets:
  prm800k       xdataset/prm800k.jsonl        (>=3-step solutions)
  processbench  xdataset/processbench.jsonl   (>=3-step solutions)
  visualprm     canonical_chunked_v2.jsonl passthrough >=2-step records —
                run through the IDENTICAL code path in the same job as the
                reference row (harness consistency check vs exp4 V2 0.796).

Reuses exp4_resegment / cascade verbatim: gold_boundaries_nonws,
chunk_boundaries_nonws, prf, window_diff, run_model_batch.
Per-record rows (incl. predicted boundary positions, for the example dumps)
are written for prm800k/processbench; visualprm keeps aggregates only.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK_EVAL = os.path.dirname(HERE)
CHUNK_CANON = "/scratch/sghos104/rlpt/chunk_canonical"
sys.path.insert(0, CHUNK_EVAL)
sys.path.insert(0, CHUNK_CANON)
import cascade as CAS                                    # noqa: E402
from exp4_resegment import (                             # noqa: E402
    gold_boundaries_nonws, chunk_boundaries_nonws, prf, window_diff,
    run_model_batch)

loads = CAS.loads
CANONICAL = "/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl"


def bucket(n):
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 7:
        return "5-7"
    return "8+"


def _self_test():
    assert gold_boundaries_nonws(["ab c", "de", "f"]) == [3, 5]
    assert chunk_boundaries_nonws(["ab", "cde f"]) == [2]
    p, r, f1, tp = prf([3, 5], [3, 5])
    assert (p, r, f1, tp) == (1.0, 1.0, 1.0, 2)
    print("[xreseg] self-test OK", flush=True)


def iter_harness(path, every=1):
    with open(path, "rb") as f:
        k = 0
        for line in f:
            r = loads(line)
            if k % every == 0:
                yield (r["sid"], r.get("subset", ""), r["steps"])
            k += 1


def iter_visualprm(every=1):
    seen = 0
    with open(CANONICAL, "rb") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0:
                print(f"[xreseg] visualprm row {i:,} gold_seen={seen:,}",
                      flush=True)
            rec = loads(line)
            prov = rec["metadata"]["chunk_provenance"]
            if prov["method"] != "passthrough" or len(rec["steps"]) < 2:
                continue
            seen += 1
            if (seen - 1) % every:
                continue
            key = (rec["metadata"]["source_sample_id"],
                   rec["metadata"]["source_index"])
            yield (f"{key[0]}#{key[1]}", "", rec["steps"])


def agg(rows):
    n = len(rows)
    if not n:
        return {}
    micro_tp = sum(r["tp"] for r in rows)
    micro_p = sum(r["n_pred"] - 1 for r in rows)
    micro_g = sum(r["n_gold"] - 1 for r in rows)
    wds = [r["wdiff"] for r in rows if r["wdiff"] is not None]
    dens = [r["density"] for r in rows if r["density"]]
    return {
        "records": n,
        "macro_prec": float(np.mean([r["prec"] for r in rows])),
        "macro_rec": float(np.mean([r["rec"] for r in rows])),
        "macro_f1": float(np.mean([r["f1"] for r in rows])),
        "micro_prec": micro_tp / micro_p if micro_p else None,
        "coverage": micro_tp / micro_g if micro_g else None,
        "density_ratio_mean": float(np.mean(dens)) if dens else None,
        "wdiff_mean": float(np.mean(wds)) if wds else None,
        "perfect_pct": 100 * sum(r["f1"] == 1.0 for r in rows) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--prm_every", type=int, default=1)
    ap.add_argument("--pb_every", type=int, default=1)
    ap.add_argument("--vp_every", type=int, default=1)
    ap.add_argument("--model_dir", default=None,
                    help="override CAS.MODEL_DIR (default: shipped release model)")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--batch_tokens", type=int, default=65_536)
    ap.add_argument("--dp_workers", type=int, default=12)
    ap.add_argument("--flush_every", type=int, default=20_000)
    args = ap.parse_args()
    _self_test()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    assert torch.cuda.is_available(), "GPU required"
    device = "cuda"
    if args.model_dir:
        CAS.MODEL_DIR = args.model_dir
    print(f"[xreseg] MODEL_DIR={CAS.MODEL_DIR} threshold={args.threshold}", flush=True)
    tok = AutoTokenizer.from_pretrained(CAS.MODEL_DIR)
    model = (AutoModelForTokenClassification.from_pretrained(CAS.MODEL_DIR)
             .to(device).eval())
    CAS._W["cls"], CAS._W["sep"], CAS._W["pad"] = (
        tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(
        args.dp_workers, initializer=CAS._worker_init, initargs=(CAS.CODE,))

    datasets = [
        ("prm800k", iter_harness(os.path.join(HERE, "prm800k.jsonl"),
                                 args.prm_every), True),
        ("processbench", iter_harness(os.path.join(HERE, "processbench.jsonl"),
                                      args.pb_every), True),
        ("visualprm", iter_visualprm(args.vp_every), False),
    ]

    report = {"config": vars(args), "datasets": {}}
    t0 = time.time()
    total_gpu = 0.0
    for name, it, keep_records in datasets:
        t_ds = time.time()
        rows = []
        rec_file = None
        if keep_records:
            rec_file = open(os.path.join(
                HERE, f"{args.tag}_records_{name}.jsonl"), "w")
        queue, meta = [], {}
        gpu_secs = Counter()
        n_ws_tokens = 0

        def flush(final=False):
            nonlocal queue
            if len(queue) < args.flush_every and not final:
                return
            out, timers = run_model_batch(
                model, tok, device, pool, queue, args, {})
            gpu_secs.update(timers)
            for key, chunks in out.items():
                steps = meta[key]["steps"]
                pred = chunk_boundaries_nonws(chunks)
                gold = gold_boundaries_nonws(steps)
                prec, rec_, f1, tp = prf(pred, gold)
                row = {"sid": key, "subset": meta[key]["subset"],
                       "bucket": bucket(len(steps)),
                       "n_gold": len(gold) + 1, "n_pred": len(pred) + 1,
                       "prec": prec, "rec": rec_, "f1": f1, "tp": tp,
                       "density": (len(pred) / len(gold)) if gold else None,
                       "wdiff": window_diff(steps, pred)}
                rows.append(row)
                if rec_file is not None:
                    rec_file.write(json.dumps({**row, "pred_nonws": pred}) + "\n")
                del meta[key]
            queue = []

        for sid, subset, steps in it:
            joined = " ".join(steps)
            n_ws_tokens += len(joined.split())
            meta[sid] = {"subset": subset, "steps": steps}
            queue.append((sid, joined))
            flush()
        flush(final=True)
        if rec_file is not None:
            rec_file.close()

        entry = {"overall": agg(rows), "by_bucket": {}, "by_subset": {},
                 "ws_tokens": n_ws_tokens,
                 "wall_s": time.time() - t_ds, "gpu_s": gpu_secs.get("gpu", 0.0)}
        for b in ("2", "3-4", "5-7", "8+"):
            sub = [r for r in rows if r["bucket"] == b]
            if sub:
                entry["by_bucket"][b] = agg(sub)
        for s in sorted(set(r["subset"] for r in rows)):
            if s:
                entry["by_subset"][s] = agg([r for r in rows if r["subset"] == s])
        report["datasets"][name] = entry
        total_gpu += entry["gpu_s"]
        o = entry["overall"]
        print(f"[xreseg] {name}: n={o.get('records',0):,} "
              f"macroF1={o.get('macro_f1',0):.3f} cov={o.get('coverage',0):.3f} "
              f"dens={o.get('density_ratio_mean',0):.2f} "
              f"wdiff={o.get('wdiff_mean',0) or 0:.3f} "
              f"perfect={o.get('perfect_pct',0):.1f}% "
              f"wall={entry['wall_s']:.0f}s gpu={entry['gpu_s']:.0f}s "
              f"ws_tokens={n_ws_tokens:,}", flush=True)

    report["wall_s"] = time.time() - t0
    report["gpu_s_total"] = total_gpu
    out_path = os.path.join(HERE, f"xdataset_stats_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    pool.close()
    pool.join()
    print(f"[xreseg] wall={report['wall_s']:.0f}s gpu={total_gpu:.0f}s "
          f"-> {out_path}")
    print("[xreseg] DONE")


if __name__ == "__main__":
    main()
