#!/usr/bin/env python
"""Numerical sensitivity analysis (AUC + tau sweep) over saved reliability scores.

Reads `scores_top3.jsonl` from a reliability run and computes, per scorer/score_field:
  - ROC-AUC where both classes exist (sugarcrepe++ ranking; and a pooled match-detector AUC)
  - tau sweep: FPR(tau) for separation datasets; FPR/TPR(sensitivity)/specificity(tau) for ranking
  - operating thresholds tau* achieving target FPR (1%/5%/10%) from the negative score quantiles

Pure-Python, CPU, streams the file once. No sklearn.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

NLI_FIELDS = ["nli_score_coverage", "nli_score_equiv", "E_ab", "E_ba", "C_ab", "C_ba"]
BERTSCORE_FIELDS = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]

# Primary "similarity-oriented" field per scorer (higher = more match) for headline tables.
PRIMARY = {"sbert": "raw_score", "nli": "nli_score_coverage", "bertscore": "bertscore_f1"}
# Fields where higher = more match (oriented so positive should outscore negative).
# NLI contradiction fields (C_ab/C_ba) are the opposite; flagged below.
CONTRADICTION_FIELDS = {"C_ab", "C_ba"}


def fields_for(scorer: str) -> list[str]:
    if scorer == "nli":
        return NLI_FIELDS
    if scorer == "bertscore":
        return BERTSCORE_FIELDS
    return ["raw_score"]


def auc_rank(pos: list[float], neg: list[float]) -> float | None:
    """Exact ROC-AUC via Mann-Whitney rank-sum, with tie handling. AUC = P(pos > neg)."""
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    combined = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    combined.sort(key=lambda t: t[0])
    # assign average ranks (1-based), handling ties
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # average of ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos_ranks = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def quantile(sorted_vals: list[float], q: float) -> float:
    """q-quantile of an ascending-sorted list (linear interpolation)."""
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_vals):
        return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac
    return sorted_vals[lo]


def frac_ge(vals: list[float], tau: float) -> float:
    return sum(1 for v in vals) and sum(1 for v in vals if v >= tau) / len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="src/outputs/reliable_full_negbench")
    ap.add_argument("--out_md", default="docs/sensitivity_results.md")
    ap.add_argument("--out_csv_dir", default="src/outputs/reliable_full_negbench")
    args = ap.parse_args()

    in_path = Path(args.input_dir) / "scores_top3.jsonl"
    # Accumulate score lists per (dataset, scorer, field, pair_type)
    pos: dict[tuple, list[float]] = defaultdict(list)  # (dataset,scorer,field) -> positive scores
    neg: dict[tuple, list[float]] = defaultdict(list)  # (dataset,scorer,field) -> negative scores

    n_rows = 0
    with in_path.open() as f:
        for line in f:
            d = json.loads(line)
            scorer = d["scorer"]
            ds = d["dataset"]
            ptype = d["pair_type"]
            for field in fields_for(scorer):
                v = d.get(field)
                if v is None or v == "":
                    continue
                v = float(v)
                key = (ds, scorer, field)
                if ptype == "positive_pair":
                    pos[key].append(v)
                elif ptype == "negative_pair":
                    neg[key].append(v)
            n_rows += 1

    datasets = sorted({k[0] for k in neg})
    scorers = ["sbert", "nli", "bertscore"]
    taus = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
    targets = [0.01, 0.05, 0.10]

    # ---- 1. ROC-AUC ----
    # (a) sugarcrepe++ ranking: within-dataset pos vs neg
    # (b) pooled match-detector: positives = sugarcrepe++ positives; negatives = ALL hard negatives
    auc_rows = []
    for scorer in scorers:
        for field in fields_for(scorer):
            # per-dataset AUC where positives exist
            for ds in datasets:
                p = pos.get((ds, scorer, field), [])
                n = neg.get((ds, scorer, field), [])
                if p and n:
                    auc_rows.append({"scope": ds, "scorer": scorer, "field": field,
                                     "n_pos": len(p), "n_neg": len(n), "auc": auc_rank(p, n)})
            # pooled match-detector AUC
            pooled_pos = []
            pooled_neg = []
            for ds in datasets:
                pooled_pos += pos.get((ds, scorer, field), [])
                pooled_neg += neg.get((ds, scorer, field), [])
            if pooled_pos and pooled_neg:
                auc_rows.append({"scope": "POOLED_match_detector", "scorer": scorer, "field": field,
                                 "n_pos": len(pooled_pos), "n_neg": len(pooled_neg),
                                 "auc": auc_rank(pooled_pos, pooled_neg)})

    # ---- 2. tau sweep ----
    sweep_rows = []
    for ds in datasets:
        for scorer in scorers:
            for field in fields_for(scorer):
                n = neg.get((ds, scorer, field), [])
                p = pos.get((ds, scorer, field), [])
                if not n:
                    continue
                for tau in taus:
                    fpr = sum(1 for v in n if v >= tau) / len(n)
                    row = {"dataset": ds, "scorer": scorer, "field": field, "tau": tau,
                           "fpr": fpr, "specificity": 1 - fpr}
                    if p:
                        tpr = sum(1 for v in p if v >= tau) / len(p)
                        row["sensitivity_tpr"] = tpr
                    sweep_rows.append(row)

    # ---- 3. operating thresholds tau* for target FPR (from negative quantiles) ----
    opp_rows = []
    for ds in datasets:
        for scorer in scorers:
            for field in fields_for(scorer):
                n = sorted(neg.get((ds, scorer, field), []))
                if not n:
                    continue
                row = {"dataset": ds, "scorer": scorer, "field": field}
                for q in targets:
                    # tau achieving FPR=q  ->  (1-q) quantile of negatives
                    row[f"tau_for_fpr_{int(q*100)}pct"] = round(quantile(n, 1 - q), 4)
                opp_rows.append(row)

    # ---- write CSVs ----
    out_dir = Path(args.out_csv_dir)
    _write_csv(out_dir / "sensitivity_auc.csv", auc_rows,
               ["scope", "scorer", "field", "n_pos", "n_neg", "auc"])
    _write_csv(out_dir / "sensitivity_tau_sweep.csv", sweep_rows,
               ["dataset", "scorer", "field", "tau", "fpr", "specificity", "sensitivity_tpr"])
    _write_csv(out_dir / "sensitivity_operating_tau.csv", opp_rows,
               ["dataset", "scorer", "field", "tau_for_fpr_1pct", "tau_for_fpr_5pct", "tau_for_fpr_10pct"])

    # ---- write markdown ----
    _write_md(Path(args.out_md), auc_rows, sweep_rows, opp_rows, datasets, n_rows)
    print(f"rows read: {n_rows}")
    print(f"wrote: sensitivity_auc.csv, sensitivity_tau_sweep.csv, sensitivity_operating_tau.csv, {args.out_md}")


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _fmt(x) -> str:
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _write_md(path: Path, auc_rows, sweep_rows, opp_rows, datasets, n_rows) -> None:
    lines = []
    lines.append("# Numerical Sensitivity Analysis — AUC + tau sweep\n")
    lines.append(f"Computed from saved reliability scores ({n_rows:,} score rows). "
                 "Pure-CPU re-aggregation; no models re-run.\n")
    lines.append("**Metrics.** ROC-AUC = P(a true match scores above a hard negative); "
                 "threshold-free, 1.0 perfect, 0.5 chance, <0.5 inverted. "
                 "FPR(tau) = fraction of hard negatives scoring >= tau (lower better). "
                 "Sensitivity (TPR) = fraction of true matches scoring >= tau. "
                 "Specificity = 1 - FPR.\n")

    # AUC headline (primary fields)
    lines.append("## 1. ROC-AUC (threshold-free)\n")
    lines.append("Primary similarity field per scorer. "
                 "`sugarcrepe++` = within-dataset (genuine positives vs hard negatives). "
                 "`POOLED_match_detector` = positives from SugarCrepe++ vs ALL hard negatives.\n")
    lines.append("| scope | scorer | field | n_pos | n_neg | AUC |")
    lines.append("|---|---|---|--:|--:|--:|")
    prim = {"sbert": "raw_score", "nli": "nli_score_coverage", "bertscore": "bertscore_f1"}
    for r in auc_rows:
        if r["field"] == prim.get(r["scorer"]):
            lines.append(f"| {r['scope']} | {r['scorer']} | {r['field']} | "
                         f"{r['n_pos']} | {r['n_neg']} | {_fmt(r['auc'])} |")
    lines.append("\n<details><summary>All score fields (incl. NLI entailment/contradiction)</summary>\n")
    lines.append("| scope | scorer | field | n_pos | n_neg | AUC |")
    lines.append("|---|---|---|--:|--:|--:|")
    for r in auc_rows:
        lines.append(f"| {r['scope']} | {r['scorer']} | {r['field']} | "
                     f"{r['n_pos']} | {r['n_neg']} | {_fmt(r['auc'])} |")
    lines.append("\n</details>\n")

    # tau sweep — primary fields, FPR table per dataset
    lines.append("## 2. tau sweep — FPR(tau) on hard negatives (lower better)\n")
    prim_taus = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    for ds in datasets:
        lines.append(f"### {ds}\n")
        lines.append("| scorer (primary field) | " + " | ".join(f"tau={t:.2f}" for t in prim_taus) + " |")
        lines.append("|---|" + "|".join("--:" for _ in prim_taus) + "|")
        for scorer in ["nli", "sbert", "bertscore"]:
            field = prim[scorer]
            cells = []
            for t in prim_taus:
                match = [r for r in sweep_rows if r["dataset"] == ds and r["scorer"] == scorer
                         and r["field"] == field and abs(r["tau"] - t) < 1e-6]
                cells.append(_fmt(match[0]["fpr"]) if match else "")
            lines.append(f"| {scorer} `{field}` | " + " | ".join(cells) + " |")
        lines.append("")

    # ranking sensitivity/specificity for sugarcrepe++
    if any(r.get("sensitivity_tpr") is not None for r in sweep_rows):
        lines.append("## 3. Full ROC points — sugarcrepe++ (has both classes)\n")
        lines.append("Sensitivity (TPR, want high) and Specificity (1-FPR, want high) vs tau.\n")
        for scorer in ["nli", "sbert", "bertscore"]:
            field = prim[scorer]
            rows = [r for r in sweep_rows if r["dataset"] == "sugarcrepepp" and r["scorer"] == scorer
                    and r["field"] == field and "sensitivity_tpr" in r]
            if not rows:
                continue
            lines.append(f"**{scorer} `{field}`**\n")
            lines.append("| tau | sensitivity (TPR) | specificity | FPR |")
            lines.append("|--:|--:|--:|--:|")
            for r in sorted(rows, key=lambda x: x["tau"]):
                if abs((r["tau"] * 100) % 10) < 1e-6:  # every 0.10
                    lines.append(f"| {r['tau']:.2f} | {_fmt(r['sensitivity_tpr'])} | "
                                 f"{_fmt(r['specificity'])} | {_fmt(r['fpr'])} |")
            lines.append("")

    # operating thresholds
    lines.append("## 4. Operating threshold tau* for a target FPR\n")
    lines.append("The tau you'd set to cap false-accepts at 1% / 5% / 10% of hard negatives "
                 "(from the negative-score quantiles). A scorer whose tau* is unreachable/“too high” "
                 "has no usable operating point.\n")
    lines.append("| dataset | scorer | field | tau*@FPR=1% | tau*@FPR=5% | tau*@FPR=10% |")
    lines.append("|---|---|---|--:|--:|--:|")
    for r in opp_rows:
        if r["field"] == prim.get(r["scorer"]):
            lines.append(f"| {r['dataset']} | {r['scorer']} | {r['field']} | "
                         f"{_fmt(r['tau_for_fpr_1pct'])} | {_fmt(r['tau_for_fpr_5pct'])} | "
                         f"{_fmt(r['tau_for_fpr_10pct'])} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
