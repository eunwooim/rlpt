#!/usr/bin/env python
"""Extended metrics pack for the chunker. VALIDATION ONLY, shipped threshold.

Two stages:
  A (needs model, slow): one forward pass per val trajectory -> per-trajectory
    prediction cache on disk (gzip json). Skipped when the cache exists.
  B (cache only, fast): every metric below computes from the cache -- no model,
    no GPU, no retraining, and re-runs are cheap.

Metrics:
  1. Boundary edit distance / boundary similarity (segeval, n_t=2 near-miss
     window). Normalized BED is reported as 1 - B; B is DEFINED in Fournier
     (2013) as the boundary-edit-distance-based similarity 1 - nBED, so this is
     the package's own normalization, stated not improvised.
  2. Segment length stats, predicted vs gold + overlaid log-count histograms.
  3. Trajectory token-length distributions + boundary F1 by length bucket
     (<128, 128-256, 256-512, >512) -- the >512 bucket exercises the
     sliding-window path.
  4. Detection-style segment mAP over 1D token intervals (AP@0.5, AP@0.75,
     mAP@[0.5:0.95:0.05]).  Confidence of a predicted segment:
         middle segment : mean(P(split) of its LEFT and RIGHT boundary tokens)
         first segment  : P(split) of its RIGHT boundary
         last segment   : P(split) of its LEFT boundary
         zero-cut traj  : the trajectory's MAX candidate-token probability
                          (its single segment has no defining boundary)
     Greedy matching per trajectory in descending confidence; a gold segment
     matches at most one prediction; a pair counts TP at IoU >= tau.
     AP = all-point interpolated area under the precision envelope (VOC-style).
  5. Bootstrap 95% CIs (resample trajectories, 1000 iters, seed 1234) for
     normalized BED, boundary F1, mAP@0.5, mean fragmentation ratio.
  6. Integration of the over-splitting analysis + structural validity checks
     into a single METRICS_REPORT.md.

Boundary F1 convention: char-position match with TOL=2, computed over
MULTI-STEP trajectories only -- identical to threshold_sweep.py, so numbers are
directly comparable.

Anything that fails or needed an improvised definition lands in the
`failures_and_improvisations` list, in the JSON, in METRICS_REPORT.md, and in
the MORNING_REPORT.md appendix. Nothing is silent.
"""
import argparse, gzip, json, os
from collections import defaultdict
import numpy as np

REPO = "/scratch/sghos104/rlpt/chunker"
TOL = 2                 # char tolerance for boundary F1 (same as threshold_sweep)
NT = 2                  # segeval near-miss (transposition) window
IOU_GRID = [round(0.5 + 0.05 * i, 2) for i in range(10)]     # 0.5 .. 0.95
BOOT_N = 1000
BOOT_SEED = 1234
LEN_BUCKETS = [(0, 128, "<128"), (128, 256, "128-256"), (256, 512, "256-512"),
               (512, 10 ** 9, ">512")]
LANGS = ("en", "cjk")   # per-table buckets; 'other' (n=9) goes to JSON only

NOTES = []              # failures + improvised definitions ledger


def note(msg):
    NOTES.append(msg)
    print(f"[note] {msg}", flush=True)


# ---------------------------------------------------------------- stage A ----
def build_cache(model_dir, val_traj, cache_path, min_tokens, max_tokens, limit,
                threshold=None):
    """One forward pass per trajectory -> cached per-trajectory predictions."""
    import chunker as ck
    import chunker_common as cc

    thr = ck.load(model_dir, threshold=threshold)
    print(f"[cache] threshold={thr} model={model_dir}", flush=True)

    def decode(prob, cand):
        bnds, _, feasible = ck._dp_boundaries(prob, cand, min_tokens, max_tokens, thr, hard_min=True)
        if not feasible:
            bnds, _, _ = ck._dp_boundaries(prob, cand, min_tokens, max_tokens, thr, hard_min=False)
        return bnds

    rows = [json.loads(l) for l in open(val_traj)]
    if limit:
        m = [r for r in rows if r["kind"] == "multi"][:limit]
        s = [r for r in rows if r["kind"] == "single"][:limit]
        rows = m + s
    recs = []
    for i, r in enumerate(rows):
        text = r["text"]
        ids, offs, prob = ck._token_probs(text)
        T = len(prob)
        cand = np.array(cc.candidate_token_mask(text, offs), dtype=bool)
        bnds = decode(prob, cand)
        gold_char, _ = cc.gold_boundary_char_positions(r["steps"]) if r["kind"] == "multi" else ([], None)
        # gold boundary TOKEN indices (dedup while preserving order)
        gold_tok, seen = [], set()
        for g in gold_char:
            t = cc._token_covering(offs, g)
            if t is not None and t not in seen:
                seen.add(t)
                gold_tok.append(t)
        if len(gold_tok) != len(gold_char):
            pass  # counted globally below
        cuts_char = [offs[b][1] for b in bnds]
        # boundary tp/fp/fn, char TOL=2 (threshold_sweep convention)
        used, matched = set(), 0
        for g in gold_char:
            for ci, c in enumerate(cuts_char):
                if ci in used:
                    continue
                if abs(c - (g + 1)) <= TOL:
                    matched += 1
                    used.add(ci)
                    break
        mcp = float(prob[cand].max()) if cand.any() else float(prob.max()) if T else 0.0
        recs.append({
            "lang": r["lang"], "kind": r["kind"], "T": int(T),
            "gold_tok": gold_tok, "cut_tok": [int(b) for b in bnds],
            "gold_char": gold_char, "cut_char": cuts_char,
            "cut_prob": [round(float(prob[b]), 4) for b in bnds],
            "max_cand_prob": round(mcp, 4),
            "no_candidates": bool(not cand.any()),
            "gold_tok_merged": len(gold_char) - len(gold_tok),
            "tp": matched, "fp": len(cuts_char) - matched, "fn": len(gold_char) - matched,
        })
        if (i + 1) % 2000 == 0:
            print(f"  ...cache {i+1:,}/{len(rows):,}", flush=True)
    payload = {"threshold": thr, "min_tokens": min_tokens, "max_tokens": max_tokens,
               "tol": TOL, "n": len(recs), "records": recs}
    with gzip.open(cache_path, "wt") as f:
        json.dump(payload, f)
    print(f"[cache] wrote {cache_path} ({len(recs):,} records)", flush=True)
    return payload


# ------------------------------------------------------------- primitives ----
def masses_from_bounds(bounds, T):
    """Segment token lengths from internal boundary token indices (last-token-
    of-segment convention). Always sums to T."""
    out, prev = [], -1
    for b in list(bounds) + [T - 1]:
        out.append(b - prev)
        prev = b
    return tuple(out)


def seg_intervals(bounds, T):
    """Half-open [start, end) token intervals for each segment."""
    iv, prev = [], -1
    for b in list(bounds) + [T - 1]:
        iv.append((prev + 1, b + 1))
        prev = b
    return iv


def iou_1d(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def seg_confidences(rec):
    """Per-predicted-segment confidence, per the documented formula."""
    cp = rec["cut_prob"]
    n_seg = len(cp) + 1
    if n_seg == 1:
        return [rec["max_cand_prob"]]
    conf = []
    for si in range(n_seg):
        if si == 0:
            conf.append(cp[0])
        elif si == n_seg - 1:
            conf.append(cp[-1])
        else:
            conf.append(0.5 * (cp[si - 1] + cp[si]))
    return conf


def ap_from_ranked(tp_flags, confs, n_gold):
    """All-point interpolated AP from pooled detections."""
    if n_gold == 0:
        return None
    if len(confs) == 0:
        return 0.0
    order = np.argsort(-np.asarray(confs), kind="stable")
    tp = np.asarray(tp_flags, dtype=np.float64)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / n_gold
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    # precision envelope
    penv = np.maximum.accumulate(prec[::-1])[::-1]
    r_prev = 0.0
    ap = 0.0
    for i in range(len(rec)):
        if rec[i] > r_prev:
            ap += (rec[i] - r_prev) * penv[i]
            r_prev = rec[i]
    return float(ap)


def match_detections(rec, tau):
    """Greedy per-trajectory matching at IoU >= tau.
    Returns list of (conf, is_tp) for every predicted segment."""
    pred_iv = seg_intervals(rec["cut_tok"], rec["T"])
    gold_iv = seg_intervals(rec["gold_tok"], rec["T"])
    confs = seg_confidences(rec)
    order = sorted(range(len(pred_iv)), key=lambda i: -confs[i])
    taken = set()
    out = []
    for pi in order:
        best_j, best_iou = None, 0.0
        for gj, giv in enumerate(gold_iv):
            if gj in taken:
                continue
            v = iou_1d(pred_iv[pi], giv)
            if v > best_iou:
                best_iou, best_j = v, gj
        if best_j is not None and best_iou >= tau:
            taken.add(best_j)
            out.append((confs[pi], 1))
        else:
            out.append((confs[pi], 0))
    return out


def pct(x, p):
    return float(np.percentile(np.asarray(x, dtype=np.float64), p)) if len(x) else None


def boot_ci(fn, n_items, rng, iters=BOOT_N):
    """Percentile 95% CI of statistic fn(index_array) under trajectory resampling."""
    vals = []
    for _ in range(iters):
        idx = rng.integers(0, n_items, n_items)
        v = fn(idx)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def micro_f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


# ---------------------------------------------------------------- stage B ----
def compute(payload, out_json, plots_dir, report_md, morning_md, oversplit_json,
            sweep_json, do_plots=True, tag=""):
    import segeval

    thr = payload["threshold"]
    recs = payload["records"]
    multi = [r for r in recs if r["kind"] == "multi"]
    single = [r for r in recs if r["kind"] == "single"]
    print(f"[pack] {len(multi):,} multi / {len(single):,} single @ thr={thr}", flush=True)

    merged = sum(r["gold_tok_merged"] for r in recs)
    if merged:
        note(f"{merged} gold boundaries merged into a neighbour's token during "
             f"char->token mapping (adjacent gold chars inside one token); "
             f"token-space metrics use the deduped set.")
    no_cand = sum(1 for r in recs if r["no_candidates"])
    if no_cand:
        note(f"IMPROVISED: {no_cand} trajectories have NO candidate tokens; their "
             f"zero-cut mAP confidence falls back to max RAW probability.")

    # ---------- 1. segeval boundary similarity / normalized BED ----------
    both_empty = 0
    for r in recs:
        gm = masses_from_bounds(r["gold_tok"], r["T"])
        pm = masses_from_bounds(r["cut_tok"], r["T"])
        if len(gm) == 1 and len(pm) == 1:
            b = 1.0           # segeval raises on two boundary-less segmentations
            both_empty += 1
        else:
            b = float(segeval.boundary_similarity(pm, gm, n_t=NT))
        r["_B"] = b
        r["_masses_pred"] = pm
        r["_masses_gold"] = gm
        r["_frag"] = len(pm) / len(gm)
    note(f"IMPROVISED (edge case): {both_empty} trajectories where BOTH gold and "
         f"prediction have zero boundaries scored B=1.0 (segeval raises "
         f"ValueError on empty boundary sets); they are perfect agreements.")

    def bed_block(rs):
        if not rs:
            return {"n": 0}
        B = np.array([r["_B"] for r in rs])
        return {"n": len(rs),
                "boundary_similarity_mean": round(float(B.mean()), 4),
                "normalized_boundary_edit_distance_mean": round(float((1 - B).mean()), 4)}

    seg_bed = {"ALL": bed_block(recs)}
    for lg in LANGS + ("other",):
        seg_bed[lg] = bed_block([r for r in recs if r["lang"] == lg])
    seg_bed_multi = {"ALL": bed_block(multi)}
    for lg in LANGS + ("other",):
        seg_bed_multi[lg] = bed_block([r for r in multi if r["lang"] == lg])

    # ---------- 2. segment length statistics ----------
    def len_stats(rs, key):
        alls = [m for r in rs for m in r[key]]
        if not alls:
            return {"n_segments": 0}
        return {"n_segments": len(alls), "mean": round(float(np.mean(alls)), 2),
                "median": pct(alls, 50), "p90": pct(alls, 90), "p99": pct(alls, 99)}

    seg_len = {}
    for scope, rs in [("ALL", recs)] + [(lg, [r for r in recs if r["lang"] == lg]) for lg in LANGS]:
        seg_len[scope] = {"predicted": len_stats(rs, "_masses_pred"),
                          "gold": len_stats(rs, "_masses_gold")}
    # "average distance between splits" == mean predicted segment length
    seg_len["average_distance_between_splits_tokens"] = seg_len["ALL"]["predicted"].get("mean")

    # ---------- 3. token-length distributions + F1 by length bucket ----------
    lenbuck = []
    for lo, hi, name in LEN_BUCKETS:
        rs = [r for r in multi if lo <= r["T"] < hi]
        tp = sum(r["tp"] for r in rs); fp = sum(r["fp"] for r in rs); fn = sum(r["fn"] for r in rs)
        lenbuck.append({"bucket": name, "n_traj": len(rs),
                        "boundary_f1": round(micro_f1(tp, fp, fn), 4) if rs else None,
                        "gold_boundaries": tp + fn})

    # ---------- 4. detection-style segment mAP ----------
    print("[pack] mAP matching...", flush=True)
    det = {}
    for tau in IOU_GRID:
        det[tau] = [match_detections(r, tau) for r in recs]      # per-traj lists

    def ap_scope(idx_list, tau):
        pool = [d for i in idx_list for d in det[tau][i]]
        n_gold = sum(len(recs[i]["gold_tok"]) + 1 for i in idx_list)
        return ap_from_ranked([t for _, t in pool], [c for c, _ in pool], n_gold)

    all_idx = list(range(len(recs)))
    lang_idx = {lg: [i for i, r in enumerate(recs) if r["lang"] == lg] for lg in LANGS}
    map_tbl = {}
    for scope, idxs in [("ALL", all_idx)] + list(lang_idx.items()):
        if not idxs:
            note(f"segment mAP: scope '{scope}' has no trajectories in this slice; skipped.")
            continue
        aps = {tau: ap_scope(idxs, tau) for tau in IOU_GRID}
        if any(v is None for v in aps.values()):
            note(f"segment mAP: scope '{scope}' has zero gold segments; skipped.")
            continue
        map_tbl[scope] = {"AP@0.5": round(aps[0.5], 4), "AP@0.75": round(aps[0.75], 4),
                          "mAP@[0.5:0.95]": round(float(np.mean([aps[t] for t in IOU_GRID])), 4)}

    # ---------- 5. bootstrap CIs ----------
    print("[pack] bootstrap...", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    Ball = np.array([1 - r["_B"] for r in recs])
    tp_m = np.array([r["tp"] for r in multi]); fp_m = np.array([r["fp"] for r in multi])
    fn_m = np.array([r["fn"] for r in multi])
    frag_m = np.array([r["_frag"] for r in multi])
    det05 = det[0.5]
    conf_pool = np.array([c for d in det05 for c, _ in d])
    tp_pool = np.array([t for d in det05 for _, t in d], dtype=np.float64)
    traj_of = np.array([i for i, d in enumerate(det05) for _ in d])
    gold_per = np.array([len(r["gold_tok"]) + 1 for r in recs], dtype=np.float64)
    order = np.argsort(-conf_pool, kind="stable")
    tp_sorted = tp_pool[order]; traj_sorted = traj_of[order]

    def bed_stat(idx):
        return float(Ball[idx].mean())

    def f1_stat(idx):
        return micro_f1(tp_m[idx].sum(), fp_m[idx].sum(), fn_m[idx].sum())

    def frag_stat(idx):
        return float(frag_m[idx].mean())

    def map_stat(idx):
        m = np.bincount(idx, minlength=len(recs)).astype(np.float64)
        w = m[traj_sorted]
        ng = float((gold_per * m).sum())
        if ng == 0:
            return None
        ctp = np.cumsum(tp_sorted * w); cfp = np.cumsum((1 - tp_sorted) * w)
        rec_ = ctp / ng
        prec = ctp / np.maximum(ctp + cfp, 1e-12)
        penv = np.maximum.accumulate(prec[::-1])[::-1]
        dr = np.diff(np.concatenate([[0.0], rec_]))
        return float((dr * penv).sum())

    cis = {}
    cis["normalized_BED_mean"] = {"point": round(float(Ball.mean()), 4),
                                  "ci95": boot_ci(bed_stat, len(recs), rng)}
    cis["boundary_f1_multi"] = {"point": round(micro_f1(tp_m.sum(), fp_m.sum(), fn_m.sum()), 4),
                                "ci95": boot_ci(f1_stat, len(multi), rng)}
    cis["mAP@0.5_ALL"] = {"point": map_tbl["ALL"]["AP@0.5"],
                          "ci95": boot_ci(map_stat, len(recs), rng)}
    cis["fragmentation_ratio_mean_multi"] = {"point": round(float(frag_m.mean()), 4),
                                             "ci95": boot_ci(frag_stat, len(multi), rng)}
    note(f"Bootstrap: {BOOT_N} iterations, trajectory-level resampling, seed {BOOT_SEED}, "
         f"percentile 95% CIs. mAP@0.5 bootstrap reuses the global confidence ordering "
         f"with multiplicity weights (exactly equivalent to re-pooling).")

    # ---------- structural validity ----------
    min_t, max_t = payload["min_tokens"], payload["max_tokens"]
    pred_seg_all = [m for r in recs for m in r["_masses_pred"]]
    n_under = sum(1 for m in pred_seg_all if m < min_t)
    n_over = sum(1 for m in pred_seg_all if m > max_t)
    bad_sum = sum(1 for r in recs if sum(r["_masses_pred"]) != r["T"] or sum(r["_masses_gold"]) != r["T"])
    non_mono = sum(1 for r in recs if any(b2 <= b1 for b1, b2 in zip(r["cut_tok"], r["cut_tok"][1:])))
    zero_cut_multi = sum(1 for r in multi if not r["cut_tok"])
    structural = {
        "segments_total_predicted": len(pred_seg_all),
        "segments_under_min_tokens": n_under,
        "frac_under_min": round(n_under / max(1, len(pred_seg_all)), 6),
        "segments_over_max_tokens": n_over,
        "mass_sum_mismatches": bad_sum,
        "non_monotonic_cut_lists": non_mono,
        "zero_cut_multi_count": zero_cut_multi,
        "zero_cut_multi_frac": round(zero_cut_multi / max(1, len(multi)), 4),
    }
    if n_over or bad_sum or non_mono:
        note(f"STRUCTURAL VIOLATIONS: over_max={n_over} mass_mismatch={bad_sum} "
             f"non_monotonic={non_mono} -- investigate before trusting the decoder.")

    # ---------- integration: over-splitting ----------
    ov = {}
    if not oversplit_json:
        note("over-splitting cross-link skipped: oversplit_analysis.json was "
             "computed at the SHIPPED threshold and does not apply to this "
             "tagged-threshold run (fragmentation CI above IS at this threshold).")
    elif os.path.exists(oversplit_json):
        o = json.load(open(oversplit_json))
        ov = {"false_splits_per_1step_mean":
                  {lg: (o["single_step_false_splits"].get(lg) or {}).get("false_splits_per_traj_mean")
                   for lg in ("ALL",) + LANGS},
              "fragmentation_ratio_mean_multi":
                  {lg: (o["multi_step_oversplit"].get(lg) or {}).get("frag_ratio_mean")
                   for lg in ("ALL",) + LANGS},
              "source": "reports/oversplit_analysis.json"}
    else:
        note(f"oversplit_analysis.json not found at {oversplit_json}; over-splitting "
             f"section cross-links nothing.")

    # ---------- plots ----------
    plots = []
    if do_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(plots_dir, exist_ok=True)

            def seg_hist(rs, tag):
                pl = [m for r in rs for m in r["_masses_pred"]]
                gl = [m for r in rs for m in r["_masses_gold"]]
                if not pl or not gl:
                    return
                bins = np.linspace(0, max(np.percentile(pl + gl, 99.5), 50), 60)
                fig, ax = plt.subplots(figsize=(7, 4.2))
                ax.hist(gl, bins=bins, alpha=0.55, label=f"gold (n={len(gl):,})")
                ax.hist(pl, bins=bins, alpha=0.55, label=f"predicted (n={len(pl):,})")
                ax.set_yscale("log")
                ax.set_xlabel("segment length (tokens)"); ax.set_ylabel("count (log)")
                ax.set_title(f"Segment lengths, predicted vs gold — {tag} (val @ thr {thr})")
                ax.legend(); fig.tight_layout()
                p = os.path.join(plots_dir, f"segment_lengths_{tag}.png")
                fig.savefig(p, dpi=120); plt.close(fig); plots.append(p)

            seg_hist(recs, "ALL")
            for lg in LANGS:
                seg_hist([r for r in recs if r["lang"] == lg], lg)

            # trajectory token lengths
            fig, ax = plt.subplots(figsize=(7, 4.2))
            Tm = [r["T"] for r in multi]; Ts = [r["T"] for r in single]
            bins = np.linspace(0, np.percentile(Tm + Ts, 99.5), 60)
            ax.hist(Tm, bins=bins, alpha=0.55, label=f"multi-step (n={len(Tm):,})")
            ax.hist(Ts, bins=bins, alpha=0.55, label=f"single-step (n={len(Ts):,})")
            ax.set_yscale("log")
            ax.set_xlabel("trajectory length (tokens)"); ax.set_ylabel("count (log)")
            ax.set_title("Val trajectory token lengths")
            ax.legend(); fig.tight_layout()
            p = os.path.join(plots_dir, "trajectory_token_lengths.png")
            fig.savefig(p, dpi=120); plt.close(fig); plots.append(p)

            # boundary F1 by length bucket
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            names = [b["bucket"] for b in lenbuck]
            vals = [b["boundary_f1"] or 0 for b in lenbuck]
            ax.bar(names, vals)
            for i, b in enumerate(lenbuck):
                ax.text(i, (b["boundary_f1"] or 0) + 0.01,
                        f"{b['boundary_f1']}\n(n={b['n_traj']:,})", ha="center", fontsize=8)
            ax.set_ylim(0, 1.0); ax.set_ylabel("boundary F1 (char TOL=2)")
            ax.set_title(f"Boundary F1 by trajectory length (val multi @ thr {thr})")
            fig.tight_layout()
            p = os.path.join(plots_dir, "boundary_f1_by_length.png")
            fig.savefig(p, dpi=120); plt.close(fig); plots.append(p)
        except Exception as e:
            note(f"FAILED: plotting raised {type(e).__name__}: {e}. Metrics JSON/report "
                 f"are unaffected.")

    # ---------- assemble JSON ----------
    out = {
        "split": "val", "threshold": thr, "tag": tag,
        "n_multi": len(multi), "n_single": len(single),
        "conventions": {
            "boundary_f1": f"char-position greedy match, TOL={TOL}, multi-step only "
                           f"(identical to threshold_sweep.py)",
            "segeval": f"boundary_similarity with n_t={NT} near-miss window; "
                       f"normalized BED reported as 1 - B (Fournier 2013)",
            "map_confidence": "middle: mean(P of left,right boundary); first: P(right); "
                              "last: P(left); zero-cut: max candidate probability",
            "ap": "all-point interpolated area under precision envelope (VOC-style)",
        },
        "boundary_edit_distance": {"all_trajectories": seg_bed, "multi_only": seg_bed_multi},
        "segment_lengths": seg_len,
        "boundary_f1_by_length_bucket": lenbuck,
        "segment_map": map_tbl,
        "bootstrap_ci95": cis,
        "structural_validity": structural,
        "oversplitting": ov,
        "plots": [os.path.basename(p) for p in plots],
        "failures_and_improvisations": NOTES,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[pack] wrote {out_json}", flush=True)

    write_report(out, report_md, morning_md, sweep_json, tag=tag,
                 plots_name=os.path.basename(plots_dir))
    return out


# ----------------------------------------------------------------- report ----
def fmt_ci(c):
    if not c or c.get("ci95") in (None, (None, None)):
        return "—"
    lo, hi = c["ci95"]
    return f"{c['point']} [{lo}, {hi}]"


def write_report(out, report_md, morning_md, sweep_json, tag="", plots_name="plots"):
    thr = out["threshold"]
    cis = out["bootstrap_ci95"]
    L = []
    A = L.append
    if tag:
        A(f"# Chunker extended metrics — validation @ threshold {thr} (tag `{tag}`)\n")
        A("_Candidate operating point re-run. The shipped release threshold and "
          "METRICS_REPORT.md are unchanged._\n")
    else:
        A(f"# Chunker extended metrics — validation @ shipped threshold {thr}\n")
    A(f"_All numbers: VALIDATION split only ({out['n_multi']:,} multi / "
      f"{out['n_single']:,} single). Test untouched. Bootstrap CIs: {BOOT_N} "
      f"trajectory resamples, 95% percentile._\n")

    A("## Headline (with 95% CIs)\n")
    A("| metric | value [95% CI] |")
    A("|---|---|")
    A(f"| boundary F1 (multi, char TOL=2) | {fmt_ci(cis['boundary_f1_multi'])} |")
    A(f"| normalized boundary edit distance (all, segeval n_t={NT}) | {fmt_ci(cis['normalized_BED_mean'])} |")
    A(f"| segment mAP@0.5 | {fmt_ci(cis['mAP@0.5_ALL'])} |")
    A(f"| fragmentation ratio mean (multi) | {fmt_ci(cis['fragmentation_ratio_mean_multi'])} |")

    A(f"\n## 1. Boundary edit distance (segeval, near-miss window n_t={NT})\n")
    A("Normalized BED = 1 − boundary similarity (B); B is segeval's boundary-"
      "edit-distance-based similarity (Fournier 2013), so this is the package's "
      "own normalization.\n")
    A("| scope | n | B similarity | normalized BED |")
    A("|---|---|---|---|")
    for tag, blk in [("all trajectories", out["boundary_edit_distance"]["all_trajectories"]),
                     ("multi-step only", out["boundary_edit_distance"]["multi_only"])]:
        for lg in ("ALL",) + LANGS:
            d = blk.get(lg, {})
            if d.get("n"):
                A(f"| {tag} / {lg} | {d['n']:,} | {d['boundary_similarity_mean']} | "
                  f"{d['normalized_boundary_edit_distance_mean']} |")

    A("\n## 2. Segment lengths (tokens)\n")
    A("| scope | which | n segs | mean | median | p90 | p99 |")
    A("|---|---|---|---|---|---|---|")
    for lg in ("ALL",) + LANGS:
        d = out["segment_lengths"].get(lg, {})
        for which in ("gold", "predicted"):
            s = d.get(which, {})
            if s.get("n_segments"):
                A(f"| {lg} | {which} | {s['n_segments']:,} | {s['mean']} | "
                  f"{s['median']} | {s['p90']} | {s['p99']} |")
    A(f"\n- **Average distance between splits** (= mean predicted segment length): "
      f"**{out['segment_lengths']['average_distance_between_splits_tokens']} tokens**.")
    A(f"- Plots: `{plots_name}/segment_lengths_ALL.png` (+ per-language), log count "
      f"axis, shared bins.")

    A("\n## 3. Trajectory lengths & the sliding-window regime\n")
    A("| length bucket | n multi traj | gold boundaries | boundary F1 |")
    A("|---|---|---|---|")
    for b in out["boundary_f1_by_length_bucket"]:
        A(f"| {b['bucket']} | {b['n_traj']:,} | {b['gold_boundaries']:,} | {b['boundary_f1']} |")
    A("\nThe >512 bucket runs entirely through the sliding-window merge path "
      f"(window 510, stride 384). Plots: `{plots_name}/trajectory_token_lengths.png`, "
      f"`{plots_name}/boundary_f1_by_length.png`.")

    A("\n## 4. Detection-style segment mAP (1D IoU in token space)\n")
    A("Confidence: middle segment = mean(P of its two defining boundaries); "
      "first/last = P of its only defining boundary; zero-cut trajectory = max "
      "candidate probability. Greedy per-trajectory matching by descending "
      "confidence; each gold matches ≤1 prediction; AP = all-point interpolated.\n")
    A("| scope | AP@0.5 | AP@0.75 | mAP@[0.5:0.95] |")
    A("|---|---|---|---|")
    for lg in ("ALL",) + LANGS:
        d = out["segment_map"].get(lg)
        if d:
            A(f"| {lg} | {d['AP@0.5']} | {d['AP@0.75']} | {d['mAP@[0.5:0.95]']} |")

    A("\n## 5. Structural validity\n")
    s = out["structural_validity"]
    A(f"- predicted segments: {s['segments_total_predicted']:,}; "
      f"under min-8: {s['segments_under_min_tokens']} (frac {s['frac_under_min']}); "
      f"over max-220: **{s['segments_over_max_tokens']}** (must be 0)")
    A(f"- mass-sum mismatches: **{s['mass_sum_mismatches']}** (must be 0); "
      f"non-monotonic cut lists: **{s['non_monotonic_cut_lists']}** (must be 0)")
    A(f"- zero-cut multi-step: {s['zero_cut_multi_count']} "
      f"({100*s['zero_cut_multi_frac']:.2f}%)")

    A("\n## 6. Over-splitting (cross-linked)\n")
    ov = out.get("oversplitting") or {}
    if ov:
        A("| scope | false splits / 1-step traj | fragmentation ratio (multi) |")
        A("|---|---|---|")
        fs = ov.get("false_splits_per_1step_mean", {})
        fr = ov.get("fragmentation_ratio_mean_multi", {})
        for lg in ("ALL",) + LANGS:
            A(f"| {lg} | {fs.get(lg)} | {fr.get(lg)} |")
        A(f"\nFull analysis: `{ov.get('source')}`, worst offenders annotated in "
          f"`reports/oversplit_examples.md`. Note ~0.97 of the 1-step mean is the "
          f"forced max_tokens=220 floor, not classifier error (see README).")
    else:
        A("_over-splitting artifacts not found — see failures below._")

    A("\n## Failures & improvised definitions\n")
    if out["failures_and_improvisations"]:
        for n_ in out["failures_and_improvisations"]:
            A(f"- {n_}")
    else:
        A("- none")

    A("\n## What these metrics add beyond token F1\n")
    A("Token F1 scores each candidate token independently, so it cannot see "
      "*structure*: it treats a boundary missed by one token the same as one "
      "missed by a paragraph, ignores how errors cluster into over- or "
      "under-segmented trajectories, and says nothing about whether the decoded "
      "segments are usable units. Boundary edit distance credits near-misses "
      "and prices boundary errors as edits; segment mAP asks the downstream "
      "question directly — does each *segment*, as an interval, line up with a "
      "human step, weighted by the model's own confidence; the length-bucket F1 "
      "isolates the long-text sliding-window regime; segment-length and "
      "fragmentation distributions expose systematic granularity bias (the "
      "list-item/LaTeX over-splitting); and the bootstrap CIs say which "
      "differences are real rather than sampling noise. Together they describe "
      "not just how often the classifier is right per token, but whether the "
      "chunker emits the *right pieces* — which is what the reward model "
      "actually consumes.")

    txt = "\n".join(L) + "\n"
    with open(report_md, "w") as f:
        f.write(txt)
    print(f"[pack] wrote {report_md}", flush=True)

    # ---- MORNING_REPORT append (idempotent between sentinels) ----
    try:
        suf = f"-{tag}" if tag else ""
        s0, e0 = f"<!-- metrics-pack{suf}:start -->", f"<!-- metrics-pack{suf}:end -->"
        block = (f"{s0}\n## Extended metrics pack (val @ thr {thr}"
                 f"{', tag ' + tag if tag else ''})\n\n"
                 f"- boundary F1 (multi) {fmt_ci(cis['boundary_f1_multi'])}; "
                 f"normalized BED {fmt_ci(cis['normalized_BED_mean'])}; "
                 f"segment mAP@0.5 {fmt_ci(cis['mAP@0.5_ALL'])}; "
                 f"fragmentation {fmt_ci(cis['fragmentation_ratio_mean_multi'])}\n"
                 f"- full report: `release/chunker-deberta-v3-small-v1/"
                 f"{os.path.basename(report_md)}`\n"
                 f"- failures / improvised definitions: "
                 f"{len(out['failures_and_improvisations'])} (listed in the report)\n"
                 f"{e0}")
        if os.path.exists(morning_md):
            t = open(morning_md).read()
            if s0 in t and e0 in t:
                t = t[:t.index(s0)] + block + t[t.index(e0) + len(e0):]
            else:
                t = t.rstrip() + "\n\n" + block + "\n"
            with open(morning_md, "w") as f:
                f.write(t)
            print(f"[pack] appended to {morning_md}", flush=True)
    except Exception as e:
        print(f"[pack] MORNING_REPORT append failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",
                    default=os.path.join(REPO, "release/chunker-deberta-v3-small-v1/model"))
    ap.add_argument("--val_traj", default=os.path.join(REPO, "data/val_traj.jsonl"))
    ap.add_argument("--cache", default=None,
                    help="prediction cache path (default depends on --tag)")
    ap.add_argument("--release", default=os.path.join(REPO, "release/chunker-deberta-v3-small-v1"))
    ap.add_argument("--min_tokens", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap trajectories per kind")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the model's chunker_meta.json threshold")
    ap.add_argument("--tag", default="",
                    help="suffix for all outputs (e.g. thr035); keeps the shipped-"
                         "threshold artifacts untouched")
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""

    if args.limit:
        # smoke mode: separate cache + outputs under logs/, never touching release
        if args.cache is None:
            args.cache = os.path.join(REPO, "logs", f"metrics_cache_smoke{args.limit}{suf}.json.gz")
        out_json = os.path.join(REPO, "logs", f"metrics_pack_smoke{suf}.json")
        report_md = os.path.join(REPO, "logs", f"METRICS_REPORT_smoke{suf}.md")
        plots_dir = os.path.join(REPO, "logs", f"plots_smoke{suf}")
        morning_md = "/dev/null"
    else:
        if args.cache is None:
            args.cache = os.path.join(REPO, "runs/full_plain", f"metrics_cache_val{suf}.json.gz")
        out_json = os.path.join(args.release, "reports", f"metrics_pack{suf}.json")
        report_md = os.path.join(args.release, f"METRICS_REPORT{suf}.md")
        plots_dir = os.path.join(args.release, f"plots{suf}")
        morning_md = os.path.join(REPO, "MORNING_REPORT.md")

    if os.path.exists(args.cache):
        print(f"[pack] reusing cache {args.cache}", flush=True)
        payload = json.load(gzip.open(args.cache, "rt"))
        if args.threshold is not None and abs(payload["threshold"] - args.threshold) > 1e-9:
            raise SystemExit(f"[pack] cache {args.cache} was built at threshold "
                             f"{payload['threshold']}, but --threshold {args.threshold} "
                             f"was requested. Delete the cache or use a different --tag.")
    else:
        payload = build_cache(args.model_dir, args.val_traj, args.cache,
                              args.min_tokens, args.max_tokens, args.limit,
                              threshold=args.threshold)

    # over-splitting artifacts were computed at the shipped threshold; a tagged
    # (overridden-threshold) run must not cross-link them as its own numbers
    oversplit_json = "" if args.tag else os.path.join(args.release, "reports",
                                                      "oversplit_analysis.json")
    compute(payload, out_json, plots_dir, report_md, morning_md,
            oversplit_json=oversplit_json,
            sweep_json=os.path.join(args.release, "reports", "threshold_sweep.json"),
            do_plots=not args.no_plots, tag=args.tag)


if __name__ == "__main__":
    main()
