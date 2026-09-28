#!/usr/bin/env python
"""Operating-threshold selection on VALIDATION, with an explicit decision rule.

The threshold shipped in chunker_meta.json (0.35) was chosen to maximise
token-level split-F1 during training. That objective ignores the DP decoder
entirely and rewards recall, which is why test showed 2.43 false splits per
single-step trajectory. This sweeps the threshold on VAL (never test -- test was
spent once already) and reports, per candidate:

  token P/R/F1        raw classifier signal (what training selected on)
  boundary P/R/F1     DECODED output quality -- what the chunker actually emits
  zero_cut_frac       % of MULTI-STEP trajectories receiving ZERO cuts
  oversplit_mean      false splits per single-step trajectory
  frac_under_min      predicted segments below min_tokens

DECISION RULE (explicit, auditable):
  1. DISQUALIFY a threshold if zero_cut_frac > 0.05, OR if zero_cut_frac exceeds
     2x the zero_cut_frac at the 0.35 baseline. A zero-cut multi-step trajectory
     reaches the downstream reward model as one undivided blob -- a worse failure
     than an imprecise boundary -- so this is a hard gate, not a tie-break.
  2. Among survivors, MAXIMISE boundary F1 (not token F1): the chunker's product
     is decoded boundaries. Token F1 is reported so the alternative choice stays
     visible and the selection can be overridden from the table.

Probabilities are computed ONCE per trajectory and reused across all thresholds;
only the (cheap, CPU) DP re-runs per threshold.
"""
import argparse, json, os
import numpy as np
import chunker as ck
import chunker_common as cc

TOL = 2
GRID = [round(x, 2) for x in np.arange(0.20, 0.905, 0.05)]
BASELINE = 0.35


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def decode(prob, cand, thr, min_t, max_t):
    """Replicates chunker.chunk()'s hard-then-soft min policy."""
    bnds, _, feasible = ck._dp_boundaries(prob, cand, min_t, max_t, thr, hard_min=True)
    if not feasible:
        bnds, _, _ = ck._dp_boundaries(prob, cand, min_t, max_t, thr, hard_min=False)
    return bnds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=ck.DEFAULT_MODEL_DIR)
    ap.add_argument("--val_traj", default="/scratch/sghos104/rlpt/chunker/data/val_traj.jsonl")
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/chunker/runs/full_plain/threshold_sweep.json")
    ap.add_argument("--segment_stats", default="/scratch/sghos104/rlpt/chunker/runs/full_plain/segment_stats.json")
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--max_zero_cut", type=float, default=0.05)
    ap.add_argument("--zero_cut_ratio_cap", type=float, default=2.0)
    ap.add_argument("--limit_multi", type=int, default=0, help="debug: cap multi trajectories")
    args = ap.parse_args()

    ck.load(args.model_dir)
    tok = ck._tok
    rows = [json.loads(l) for l in open(args.val_traj)]
    multi = [r for r in rows if r["kind"] == "multi"]
    single = [r for r in rows if r["kind"] == "single"]
    if args.limit_multi:
        multi = multi[:args.limit_multi]
    print(f"[sweep] {len(multi):,} multi / {len(single):,} single; grid={GRID}", flush=True)

    T = len(GRID)
    tok_tp = np.zeros(T); tok_fp = np.zeros(T); tok_fn = np.zeros(T)
    bnd_tp = np.zeros(T); bnd_fp = np.zeros(T); bnd_fn = np.zeros(T)
    zero_cut = np.zeros(T); n_seg = np.zeros(T); n_short = np.zeros(T)
    over_splits = np.zeros(T)

    # ---- multi-step: token F1, boundary F1, zero-cut, short fragments ----
    for n, r in enumerate(multi):
        ex = cc.build_token_labels(r["steps"], tok)
        labels = np.array(ex["labels"])
        ids, offs, prob = ck._token_probs(ex["text"])
        m = min(len(labels), len(prob))
        lab, pr = labels[:m], prob[:m]
        msk = lab != cc.LABEL_IGNORE
        y = (lab[msk] == cc.LABEL_SPLIT).astype(int)
        pm = pr[msk]
        cand = np.array(cc.candidate_token_mask(ex["text"], offs), dtype=bool)
        gold, _ = cc.gold_boundary_char_positions(r["steps"])
        Tn = len(prob)
        for ti, thr in enumerate(GRID):
            pred = (pm >= thr).astype(int)
            tok_tp[ti] += int((pred & y).sum()); tok_fp[ti] += int((pred & (1 - y)).sum())
            tok_fn[ti] += int(((1 - pred) & y).sum())
            bnds = decode(prob, cand, thr, args.min_tokens, args.max_tokens)
            cuts = [offs[b][1] for b in bnds]
            if not cuts:
                zero_cut[ti] += 1
            used = set(); matched = 0
            for g in gold:
                for ci, c in enumerate(cuts):
                    if ci in used:
                        continue
                    if abs(c - (g + 1)) <= TOL:
                        matched += 1; used.add(ci); break
            bnd_tp[ti] += matched; bnd_fp[ti] += len(cuts) - matched; bnd_fn[ti] += len(gold) - matched
            prev = -1
            for b in bnds + [Tn - 1]:
                n_seg[ti] += 1
                if (b - prev) < args.min_tokens:
                    n_short[ti] += 1
                prev = b
        if (n + 1) % 2000 == 0:
            print(f"  ...multi {n+1:,}/{len(multi):,}", flush=True)

    # ---- single-step: over-split ----
    for n, r in enumerate(single):
        text = r["text"]
        ids, offs, prob = ck._token_probs(text)
        cand = np.array(cc.candidate_token_mask(text, offs), dtype=bool)
        for ti, thr in enumerate(GRID):
            bnds = decode(prob, cand, thr, args.min_tokens, args.max_tokens)
            over_splits[ti] += len(bnds)
        if (n + 1) % 2000 == 0:
            print(f"  ...single {n+1:,}/{len(single):,}", flush=True)

    table = []
    for ti, thr in enumerate(GRID):
        tp_, tr_, tf_ = prf(tok_tp[ti], tok_fp[ti], tok_fn[ti])
        bp_, br_, bf_ = prf(bnd_tp[ti], bnd_fp[ti], bnd_fn[ti])
        table.append({
            "threshold": thr,
            "token_precision": tp_, "token_recall": tr_, "token_f1": tf_,
            "boundary_precision": bp_, "boundary_recall": br_, "boundary_f1": bf_,
            "zero_cut_frac": round(float(zero_cut[ti] / max(1, len(multi))), 4),
            "zero_cut_n": int(zero_cut[ti]),
            "oversplit_mean_single": round(float(over_splits[ti] / max(1, len(single))), 4),
            "frac_under_min": round(float(n_short[ti] / max(1, n_seg[ti])), 6),
        })

    base = next((r for r in table if abs(r["threshold"] - BASELINE) < 1e-9), None)
    base_zero = base["zero_cut_frac"] if base else None
    cap = min(args.max_zero_cut,
              args.zero_cut_ratio_cap * base_zero) if base_zero is not None else args.max_zero_cut

    for r in table:
        reasons = []
        if r["zero_cut_frac"] > args.max_zero_cut:
            reasons.append(f"zero_cut {r['zero_cut_frac']:.4f} > {args.max_zero_cut}")
        if base_zero is not None and r["zero_cut_frac"] > args.zero_cut_ratio_cap * base_zero:
            reasons.append(f"zero_cut {r['zero_cut_frac']:.4f} > {args.zero_cut_ratio_cap}x baseline "
                           f"({base_zero:.4f})")
        r["eligible"] = not reasons
        r["disqualified_because"] = reasons

    eligible = [r for r in table if r["eligible"]]
    chosen = max(eligible, key=lambda r: r["boundary_f1"]) if eligible else None
    by_token = max(eligible, key=lambda r: r["token_f1"]) if eligible else None

    segstats = {}
    if os.path.exists(args.segment_stats):
        try:
            segstats = json.load(open(args.segment_stats))
        except Exception as e:
            segstats = {"error": str(e)}

    out = {
        "val_multi": len(multi), "val_single": len(single),
        "grid": GRID, "baseline_threshold": BASELINE, "baseline_zero_cut_frac": base_zero,
        "decision_rule": {
            "disqualify_if_zero_cut_gt": args.max_zero_cut,
            "disqualify_if_zero_cut_gt_ratio_of_baseline": args.zero_cut_ratio_cap,
            "effective_zero_cut_cap": cap,
            "objective": "maximise boundary_f1 among eligible thresholds",
        },
        "table": table,
        "n_eligible": len(eligible),
        "chosen": chosen,
        "would_choose_by_token_f1": by_token,
        "segment_stats_at_0.35_test": segstats,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\nthr   tokF1   bndF1   zero%   over/1step  under_min  eligible")
    for r in table:
        print(f"{r['threshold']:<5} {r['token_f1']:<7} {r['boundary_f1']:<7} "
              f"{100*r['zero_cut_frac']:<7.2f} {r['oversplit_mean_single']:<11} "
              f"{r['frac_under_min']:<10} {'yes' if r['eligible'] else 'NO'}")
    print(f"\n[chosen] {chosen['threshold'] if chosen else None} "
          f"(boundary_f1={chosen['boundary_f1'] if chosen else None})")
    if by_token and chosen and by_token["threshold"] != chosen["threshold"]:
        print(f"[note] token-F1 objective would have chosen {by_token['threshold']}")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
