#!/usr/bin/env python
"""EXP 4 — strip-and-resegment over the gold (passthrough multi-step) records.

Two variants per record:
  V1 (cascade test):    join gold steps with "\\n\\n", run the FULL production
                        cascade (marker tiers + digit/fence guards + model).
                        CAVEAT (encoded in the report): the "\\n\\n" join
                        partially feeds the answer to the para tier — V1 is an
                        upper bound, V2 is the uncontaminated model number.
  V2 (model-only test): join with a single space (kills every marker tier and
                        any fence/line-start structure), force the model path
                        (no marker, no guards).

Scoring per record (and aggregated micro/macro, by n_steps bucket and by
source family):
  - boundary positions are canonicalized to "number of non-whitespace chars
    before the cut" — whitespace-insensitive, so it is immaterial which side
    of the consumed whitespace a cut lands on. Exact-position match on these.
  - precision / recall / F1 (recall == gold-boundary coverage).
  - boundary-density ratio: n_predicted / n_gold boundaries.
  - WindowDiff over whitespace tokens, k = max(2, round(mean gold segment
    token count / 2)) per record; boundaries snapped to the nearest token gap.
V1-vs-V2 confusion: records where V1 is perfect (F1 == 1) but V2 isn't
(marker-carried) vs both perfect vs only-V2 vs neither.

Runs in chunker/env (GPU). Reuses the production implementation verbatim by
importing chunk_canonical/cascade.py (batched forward, release DP decode,
marker_split, guards). --sample_every N takes every Nth gold record for the
timing/estimate run (spec gate: ~5K records first).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK_CANON = "/scratch/sghos104/rlpt/chunk_canonical"
sys.path.insert(0, CHUNK_CANON)
sys.path.insert(0, HERE)
import cascade as CAS                      # noqa: E402  production machinery
from marker_split import marker_split, STEP_RE, LIST_RE, PARA_RE  # noqa: E402
from fence_guard import fence_regions, fenced_char_share  # noqa: E402
from source_map import TierMap             # noqa: E402


def has_internal_markers(steps):
    """True iff any gold step contains marker typography INSIDE it: a \\n\\n
    paragraph break, or a line-start list/step marker at position > 0 (a
    marker at position 0 coincides with the gold boundary and is therefore
    unambiguous for the joined-text cascade). Uses the production regexes
    verbatim. Clean subset = False: the join separator is the only marker
    the cascade can split on, so V1 there is the fair cascade score."""
    for s in steps:
        if PARA_RE.search(s):
            return True
        for rx in (LIST_RE, STEP_RE):
            for m in rx.finditer(s):
                if m.start() > 0:
                    return True
    return False

loads = CAS.loads


# ---------------- boundary canonicalization ----------------------------------
def nonws_prefix(text, pos):
    return sum(1 for c in text[:pos] if not c.isspace())


def gold_boundaries_nonws(steps):
    """Canonical positions of the n-1 gold boundaries (independent of joiner)."""
    out, acc = [], 0
    for st in steps[:-1]:
        acc += sum(1 for c in st if not c.isspace())
        out.append(acc)
    return out


def chunk_boundaries_nonws(chunks):
    """Canonical positions of predicted boundaries from produced chunks."""
    out, acc = [], 0
    for ch in chunks[:-1]:
        acc += sum(1 for c in ch if not c.isspace())
        out.append(acc)
    return out


def prf(pred, gold):
    p, g = set(pred), set(gold)
    tp = len(p & g)
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp


def window_diff(steps, pred_nonws):
    """WindowDiff over whitespace tokens of the (conceptual) full text."""
    tok_nonws = []          # cumulative non-ws chars after each token
    acc = 0
    for st in steps:
        for t in st.split():
            acc += len(t)
            tok_nonws.append(acc)
    n = len(tok_nonws)
    if n < 2:
        return None
    gaps = n - 1

    def to_gapset(positions):
        s = set()
        for pos in positions:
            # nearest gap: first token j with cumulative >= pos
            j = int(np.searchsorted(np.asarray(tok_nonws), pos))
            j = min(max(j, 1), gaps)   # gap after token j-1 is index j-1... snap
            s.add(j - 1 if abs(tok_nonws[j - 1] - pos) <= abs(
                tok_nonws[min(j, n - 1)] - pos) else min(j, gaps - 1))
        return s

    gold = to_gapset(gold_boundaries_nonws(steps))
    pred = to_gapset(pred_nonws)
    seg_len = n / (len(gold) + 1)
    k = max(2, round(seg_len / 2))
    if gaps <= k:
        return None
    err = 0
    cnt = 0
    garr = np.zeros(gaps, dtype=np.int32)
    parr = np.zeros(gaps, dtype=np.int32)
    for b in gold:
        garr[b] = 1
    for b in pred:
        parr[b] = 1
    gcum = np.concatenate([[0], np.cumsum(garr)])
    pcum = np.concatenate([[0], np.cumsum(parr)])
    for i in range(gaps - k + 1):
        if (gcum[i + k] - gcum[i]) != (pcum[i + k] - pcum[i]):
            err += 1
        cnt += 1
    return err / cnt if cnt else None


# ---------------- record processing -------------------------------------------
def marker_or_queue(text):
    """V1 front half: returns ('marker', tier, chunks) or ('skip', reason, None)
    or ('model', fences, None)."""
    tier, chunks = marker_split(text)
    if tier is not None:
        return ("marker", tier, chunks)
    nw = sum(1 for c in text if not c.isspace())
    nonalpha = sum(1 for c in text if not c.isspace() and not c.isalpha())
    if nw and nonalpha / nw > 0.5:
        return ("skip", "nonprose", None)
    fences = fence_regions(text)
    if fences and fenced_char_share(text, fences) > 0.3:
        return ("skip", "code_fence", None)
    return ("model", fences, None)


def run_model_batch(model, tok, device, pool, queue, args, variant_fences):
    """queue: list of (key, text). Returns key -> chunks (model split)."""
    if not queue:
        return {}, {}
    recs = []
    B = 512
    for i in range(0, len(queue), B):
        texts = [t for _, t in queue[i:i + B]]
        enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        for j in range(len(texts)):
            recs.append({"rid": i + j, "text": texts[j],
                         "ids": enc["input_ids"][j],
                         "offsets": enc["offset_mapping"][j]})
    timers = Counter()
    probs = CAS._token_probs_batched(model, device, recs, args.batch_tokens, timers)
    jobs = [(r["text"], r["offsets"], probs.get(r["rid"], np.zeros(0)),
             args.min_tokens, args.max_tokens, args.threshold,
             variant_fences.get(queue[r["rid"]][0]) or [])
            for r in recs]
    results = pool.map(CAS._decode_one, jobs, chunksize=32)
    out = {}
    for r, (cuts, _forced, _soft, _fd) in zip(recs, results):
        key = queue[r["rid"]][0]
        chunks = CAS._slice_model_chunks(r["text"], cuts) if cuts else [r["text"]]
        if not chunks:
            chunks = [r["text"]]
        out[key] = chunks
    return out, dict(timers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl")
    ap.add_argument("--outdir", default="/scratch/sghos104/rlpt/chunk_eval")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--sample_every", type=int, default=1,
                    help="take every Nth gold record (timing run: ~100)")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--batch_tokens", type=int, default=65_536)
    ap.add_argument("--dp_workers", type=int, default=14)
    ap.add_argument("--flush_every", type=int, default=20_000,
                    help="model-queue flush size (records)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU required"
    tok = AutoTokenizer.from_pretrained(CAS.MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(CAS.MODEL_DIR).to(device).eval()
    CAS._W["cls"], CAS._W["sep"], CAS._W["pad"] = (
        tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(args.dp_workers,
                                        initializer=CAS._worker_init,
                                        initargs=(CAS.CODE,))
    tm = TierMap()
    t_start = time.time()

    # per-record result rows: keyed (sid, idx)
    results = {}     # key -> dict per variant
    meta = {}        # key -> {steps_n, bucket, family, steps}
    v1_queue, v2_queue = [], []
    v1_fences = {}
    n_gold = seen_gold = 0
    gpu_secs = Counter()

    def bucket(n):
        if n == 2:
            return "2"
        if n <= 4:
            return "3-4"
        if n <= 7:
            return "5-7"
        return "8+"

    def family_of(rec):
        tier, label = tm.classify(rec.get("images") or [])
        if label:
            return label
        imgs = rec.get("images") or []
        return ("pfx:" + imgs[0].split("/")[0]) if imgs else "<no-image>"

    def score(key, variant, chunks, steps):
        pred = chunk_boundaries_nonws(chunks)
        gold = gold_boundaries_nonws(steps)
        prec, rec_, f1, tp = prf(pred, gold)
        wd = window_diff(steps, pred)
        results.setdefault(key, {})[variant] = {
            "n_pred": len(pred) + 1, "n_gold": len(gold) + 1,
            "prec": prec, "rec": rec_, "f1": f1, "tp": tp,
            "density": (len(pred) / len(gold)) if gold else None,
            "wdiff": wd}

    def flush(final=False):
        nonlocal v1_queue, v2_queue
        if len(v1_queue) + len(v2_queue) < args.flush_every and not final:
            return
        out1, t1 = run_model_batch(model, tok, device, pool, v1_queue, args, v1_fences)
        out2, t2 = run_model_batch(model, tok, device, pool, v2_queue, args, {})
        for t in (t1, t2):
            gpu_secs.update(t)
        for key, chunks in out1.items():
            score(key, "v1", chunks, meta[key]["steps"])
            results[key]["v1"]["method"] = "model"
        for key, chunks in out2.items():
            score(key, "v2", chunks, meta[key]["steps"])
        # free steps for fully-scored records
        for key in list(meta):
            if len(results.get(key, {})) == 2:
                meta[key].pop("steps", None)
        v1_queue, v2_queue = [], []
        v1_fences.clear()

    with open(args.input, "rb") as f:
        for i, line in enumerate(f):
            if i % 2_000_000 == 0:
                print(f"[exp4] row {i:,} gold_seen={seen_gold:,} t={time.time()-t_start:.0f}s",
                      flush=True)
            rec = loads(line)
            prov = rec["metadata"]["chunk_provenance"]
            if prov["method"] != "passthrough" or len(rec["steps"]) < 2:
                continue
            seen_gold += 1
            if (seen_gold - 1) % args.sample_every:
                continue
            n_gold += 1
            key = (rec["metadata"]["source_sample_id"], rec["metadata"]["source_index"])
            steps = rec["steps"]
            meta[key] = {"n": len(steps), "bucket": bucket(len(steps)),
                         "family": family_of(rec), "steps": steps,
                         "internal": has_internal_markers(steps)}
            # ---- V1: full cascade on \n\n join ----
            j1 = "\n\n".join(steps)
            kind, x, chunks = marker_or_queue(j1)
            if kind == "marker":
                score(key, "v1", chunks, steps)
                results[key]["v1"]["method"] = "marker/" + x
            elif kind == "skip":
                score(key, "v1", [j1], steps)
                results[key]["v1"]["method"] = "skip/" + x
            else:
                v1_queue.append((key, j1))
                v1_fences[key] = x
            # ---- V2: model-only on space join ----
            v2_queue.append((key, " ".join(steps)))
            flush()
    flush(final=True)
    pool.close()
    pool.join()
    wall = time.time() - t_start
    print(f"[exp4] scored {n_gold:,} records (of {seen_gold:,} gold) "
          f"wall={wall:.0f}s gpu={gpu_secs['gpu']:.0f}s", flush=True)

    # ---------------- aggregate ----------------
    def agg(rows):
        n = len(rows)
        if not n:
            return {}
        out = {"records": n}
        for v in ("v1", "v2"):
            sel = [r[v] for r in rows if v in r]
            micro_tp = sum(r["tp"] for r in sel)
            micro_p = sum(r["n_pred"] - 1 for r in sel)
            micro_g = sum(r["n_gold"] - 1 for r in sel)
            wds = [r["wdiff"] for r in sel if r["wdiff"] is not None]
            dens = [r["density"] for r in sel if r["density"]]
            out[v] = {
                "macro_prec": float(np.mean([r["prec"] for r in sel])),
                "macro_rec": float(np.mean([r["rec"] for r in sel])),
                "macro_f1": float(np.mean([r["f1"] for r in sel])),
                "micro_prec": micro_tp / micro_p if micro_p else None,
                "micro_rec": micro_tp / micro_g if micro_g else None,
                "coverage": micro_tp / micro_g if micro_g else None,
                "density_ratio_mean": float(np.mean(dens)) if dens else None,
                "wdiff_mean": float(np.mean(wds)) if wds else None,
                "perfect_pct": 100 * sum(r["f1"] == 1.0 for r in sel) / max(1, len(sel)),
            }
        return out

    rows_all = [{**results[k], "bucket": meta[k]["bucket"],
                 "family": meta[k]["family"], "internal": meta[k]["internal"]}
                for k in results]
    report = {"config": vars(args), "n_records": n_gold,
              "gold_seen": seen_gold, "wall_s": wall,
              "gpu_s": gpu_secs.get("gpu", 0.0),
              "overall": agg(rows_all), "by_bucket": {}, "by_family": {},
              "v1_method_mix": dict(Counter(
                  r["v1"]["method"] for r in rows_all if "v1" in r))}
    for b in ("2", "3-4", "5-7", "8+"):
        report["by_bucket"][b] = agg([r for r in rows_all if r["bucket"] == b])
    fams = Counter(r["family"] for r in rows_all)
    for fam, _ in fams.most_common(20):
        report["by_family"][fam] = agg([r for r in rows_all if r["family"] == fam])

    # V1 stratification: clean (no internal markers in gold steps -> join
    # separator unambiguous, fair cascade score) vs structured (internal
    # markers -> quantified granularity disagreement). V2 shown as control.
    report["v1_marker_stratification"] = {}
    for name, flag in (("clean", False), ("structured", True)):
        sub = [r for r in rows_all if r["internal"] == flag]
        entry = agg(sub)
        entry["v1_method_mix"] = dict(Counter(
            r["v1"]["method"] for r in sub if "v1" in r))
        report["v1_marker_stratification"][name] = entry

    # confusion
    conf = Counter()
    for r in rows_all:
        if "v1" not in r or "v2" not in r:
            continue
        a = r["v1"]["f1"] == 1.0
        b = r["v2"]["f1"] == 1.0
        conf["both_perfect" if a and b else
             "v1_only_marker_carried" if a else
             "v2_only" if b else "neither"] += 1
    report["v1_v2_confusion"] = dict(conf)

    out_path = os.path.join(args.outdir, f"exp4_stats_{args.tag}.json")
    with open(out_path, "w") as fo:
        json.dump(report, fo, indent=2)
    o = report["overall"]
    for v in ("v1", "v2"):
        if v in o:
            s = o[v]
            print(f"[exp4] {v}: macroF1={s['macro_f1']:.3f} "
                  f"cov={s['coverage']:.3f} dens={s['density_ratio_mean']:.2f} "
                  f"wdiff={s['wdiff_mean']:.3f} perfect={s['perfect_pct']:.1f}%")
    print(f"[exp4] confusion: {report['v1_v2_confusion']}")
    print(f"[exp4] v1 method mix: {report['v1_method_mix']}")
    for name, entry in report["v1_marker_stratification"].items():
        if "v1" in entry:
            s = entry["v1"]
            print(f"[exp4] V1 {name} subset (n={entry['records']:,}): "
                  f"macroF1={s['macro_f1']:.3f} cov={s['coverage']:.3f} "
                  f"dens={s['density_ratio_mean']:.2f} wdiff={s['wdiff_mean']:.3f} "
                  f"perfect={s['perfect_pct']:.1f}% mix={entry['v1_method_mix']}")
    if args.sample_every > 1:
        scale = seen_gold / max(1, n_gold)
        print(f"[exp4] ESTIMATE full run: wall ~{wall*scale/3600:.2f} h "
              f"(gpu ~{gpu_secs.get('gpu',0.0)*scale/3600:.2f} h) — streaming "
              f"overhead already included once")
    print("[exp4] DONE")


if __name__ == "__main__":
    main()
