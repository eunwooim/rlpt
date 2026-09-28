#!/usr/bin/env python
"""jb_aggregate.py — CPU aggregation of the judge benchmark shards -> reward_redesign/JUDGE_BENCH.md, judge_pairs_300.{jsonl,md}, and the
E2_STATUS.md "Task 3" section. Missing shards are reported as MISSING (with counts), never estimated."""
import collections
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = "/scratch/sghos104/rlpt"
OUT = f"{REPO}/reward_redesign/judge_bench/out"
CONFIGS = ["whole", "clause-min", "clause-hung-F05", "clause-hung-mean", "asym-mean"]
MODELS = {"cur": "deberta-xlarge-mnli (current)", "v3": "DeBERTa-v3-large-mnli-fever-anli-ling-wanli"}
EXPECT = {"3a": 10, "3b": 10, "3c_nb": 5, "3c_para": 5, "3d": 1, "3e_score": 1}


def load(task, model):
    rows, missing = [], []
    for s in range(EXPECT[task]):
        p = f"{OUT}/{task}_{model}_{s}.jsonl"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            rows += [json.loads(l) for l in open(p)]
        else:
            missing.append(s)
    return rows, missing


def f(x, nd=3):
    return "MISSING" if x is None else f"{x:.{nd}f}"


def auroc(pos, neg):
    if not pos or not neg:
        return None
    ns = sorted(neg)
    import bisect
    tot = 0.0
    for x in pos:
        lo = bisect.bisect_left(ns, x); hi = bisect.bisect_right(ns, x)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def throughput():
    """pairs/s per model from the job logs: whole-step jobs (3c) = the reward's batch of whole steps; all jobs = mixed clause/step pairs."""
    tp = collections.defaultdict(list)
    for p in glob.glob(f"{OUT}/logs/jb-*.log"):
        txt = open(p, errors="replace").read()
        m = re.search(r"DONE task=(\S+) model=(\S+) shard=\d+: (\d+) rows, NLI calls=(\d+), nli_secs=([\d.]+), throughput=([\d.]+)", txt)
        g = re.search(r"gpu=([^\n]+?) TASK", txt)
        if m:
            tp[(m.group(2), "whole-step (3c/3d)" if m.group(1).startswith(("3c", "3d")) else "mixed (3a/3b/3e)")].append((float(m.group(6)), int(m.group(4)), (g.group(1).strip() if g else "?")))
    return tp


def main():
    L = [f"# JUDGE BENCH — current judge (microsoft/deberta-xlarge-mnli) vs MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli "
         f"(generated {datetime.now():%Y-%m-%d %H:%M} by reward_redesign/judge_bench/jb_aggregate.py)", "",
         "Every pair is scored with the reward's formula s = clip(0.5·E(a→b) + 0.5·E(b→a) − max C, 0, 1) at 512-token truncation; raw 3-class",
         "probabilities are kept in the shard files (reward_redesign/judge_bench/out/). Configs: whole (the reward today), clause-min (June's split",
         "+ SBERT-mpnet Hungarian alignment, tau 0.15, min-pool), clause-hung-F05 / clause-hung-mean (clauses both sides, SBERT top-3 candidate",
         "pairs → NLI → Hungarian on the sparse s matrix; F_0.5 of clause precision/recall, resp. mean s over assigned pairs), asym-mean (whole a vs",
         "each b-clause and whole b vs each a-clause, forward direction each; s = clip(0.5·mean E + 0.5·mean E − max C)). Single-clause pairs fall",
         "back to `whole` in the three clause configs. Samples: 3a = 4,999 rows of data/visualprm400k/pairs.jsonl (seed 0, stratified by source;",
         "3 rows without exactly one differing blank-line step dropped); 3b = 5,000 of the 9,254 faithful==True paraphrase rows (seed 0; whole",
         "solution as built); 3c = the same rows through reward_v2.score_new's Hungarian matching (whole steps, soft gates); 3d = EQUATE (9,702",
         "pairs, five subsets); 3e = 300 real (rollout step, reference step) pairs from run M's dumps.", ""]
    status = []
    # ---------------- 3a
    rows3a, calls = {}, {}
    L += ["## 3a. VisualPRM hard negatives, step level (edited step of neg vs the same step of pos → gold contradict; FPR = fraction with s ≥ 0.5)", "",
          "| model | config | n | FPR (s ≥ 0.5) ↓ | mean s (edited) ↓ | baseline mean s (unrelated step) | fallback (single clause) | NLI calls / pair |",
          "|---|---|---|---|---|---|---|---|"]
    cat_tab = collections.defaultdict(dict)
    for m in MODELS:
        rows, miss = load("3a", m)
        rows3a[m] = rows
        if miss:
            status.append(f"3a {m}: missing shards {miss}")
        for c in CONFIGS:
            if not rows:
                L.append(f"| {m} | {c} | 0 | MISSING | MISSING | MISSING | - | - |"); continue
            e = [r["edited"][c] for r in rows]; b = [r["baseline"][c] for r in rows]
            fb = st.mean(r["edited"]["fallback"] for r in rows) if c != "whole" else 0.0
            nc = st.mean(r["edited"]["calls"][c] for r in rows)
            calls[(m, c)] = nc
            L.append(f"| {m} | {c} | {len(rows)} | **{st.mean(1.0 if x >= 0.5 else 0.0 for x in e):.3f}** | {st.mean(e):.3f} | {st.mean(b):.3f} | {fb:.3f} | {nc:.1f} |")
            for cat in ("bare equation", "prose with number", "mixed"):
                sub = [r["edited"][c] for r in rows if r["category"] == cat]
                cat_tab[(m, c)][cat] = (st.mean(1.0 if x >= 0.5 else 0.0 for x in sub) if sub else None, len(sub))
    L += ["", "June reference (docs/visualprm_negation_results.md, docs/negation_clausesplit_results.md; June's equiv formula used −0.5·max C and 256-token truncation, on whole solutions):",
          "paragraph NLI equiv FPR@0.5 on neg = 0.594 (FPR@0.6 = 0.530, mean 0.569); clause-split min-equiv FPR@0.5 = 0.015 (FPR@0.6 = 0.008, mean 0.015).", "",
          "3a by edited-step category (FPR at s ≥ 0.5; n in the header):", ""]
    cats = ["bare equation", "prose with number", "mixed"]
    ns = {cat: sum(1 for r in rows3a.get("cur") or rows3a.get("v3") or [] if r["category"] == cat) for cat in cats}
    L += ["| model | config | " + " | ".join(f"{c} (n={ns[c]})" for c in cats) + " |", "|---|---|" + "---|" * 3]
    for m in MODELS:
        for c in CONFIGS:
            d = cat_tab.get((m, c), {})
            L.append(f"| {m} | {c} | " + " | ".join(f(d.get(cat, (None,))[0]) for cat in cats) + " |")
    # ---------------- 3b
    L += ["", "## 3b. Paraphrase set (paraphrase vs pos, whole solution → gold entail; FNR = fraction with s < 0.5)", "",
          "| model | config | n | FNR (s < 0.5) ↓ | mean s ↑ | fallback | NLI calls / pair |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        rows, miss = load("3b", m)
        if miss:
            status.append(f"3b {m}: missing shards {miss}")
        for c in CONFIGS:
            if not rows:
                L.append(f"| {m} | {c} | 0 | MISSING | MISSING | - | - |"); continue
            v = [r["para"][c] for r in rows]
            L.append(f"| {m} | {c} | {len(rows)} | **{st.mean(1.0 if x < 0.5 else 0.0 for x in v):.3f}** | {st.mean(v):.3f} | "
                     f"{st.mean(r['para']['fallback'] for r in rows) if c != 'whole' else 0.0:.3f} | {st.mean(r['para']['calls'][c] for r in rows):.1f} |")
    L += ["", "June reference: clause-split min-pooling raised paraphrase FNR (paraphrase precision FPR 0.60 in the memory note; see docs) — the gap this table tests."]
    # ---------------- 3c
    L += ["", "## 3c. Full-solution level through the reward (reward_v2.score_new matching, whole steps only, soft gates)", "",
          "| model | n (neg/base) | mean match pos→neg ↓ | mean match pos→baseline | n (para) | mean match pos→paraphrase ↑ | edited step credited (s ≥ 0.5 in the assignment) ↓ | edited step assigned to its own index | edited step deduped |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        nb, miss1 = load("3c_nb", m); pa, miss2 = load("3c_para", m)
        if miss1:
            status.append(f"3c_nb {m}: missing shards {miss1}")
        if miss2:
            status.append(f"3c_para {m}: missing shards {miss2}")
        L.append(f"| {m} | {len(nb)} | {f(st.mean(r['match_neg'] for r in nb) if nb else None)} | {f(st.mean(r['match_base'] for r in nb) if nb else None)} | {len(pa)} | "
                 f"{f(st.mean(r['match_para'] for r in pa) if pa else None)} | **{f(st.mean(r['edited_credited'] for r in nb) if nb else None)}** | "
                 f"{f(st.mean(r['edited_assigned_to_k'] for r in nb) if nb else None)} | {f(st.mean(r['edited_in_dup'] for r in nb) if nb else None)} |")
    # ---------------- 3d
    L += ["", "## 3d. EQUATE (whole-step config; 3-class accuracy = argmax of the forward p→h probabilities; AUROC of s for entail vs non-entail; FNR = gold-entail with s < 0.5)", "",
          "| subset | n | " + " | ".join(f"{m} acc | {m} AUROC | {m} FNR" for m in MODELS) + " |", "|---|---|" + "---|" * 6]
    d3 = {}
    for m in MODELS:
        rows, miss = load("3d", m)
        if miss:
            status.append(f"3d {m}: missing")
        d3[m] = rows
    subsets = ["RTE-Quant", "NewsNLI", "RedditNLI", "AWP-NLI", "StressTest"]
    for sub in subsets:
        cells = []
        n = 0
        for m in MODELS:
            rs = [r for r in d3[m] if r["subset"] == sub]
            n = max(n, len(rs))
            if not rs:
                cells += ["MISSING"] * 3; continue
            acc = st.mean(1.0 if r["pred"] == r["gold"] else 0.0 for r in rs)
            pos = [r["s"] for r in rs if r["gold"] == "entailment"]; neg = [r["s"] for r in rs if r["gold"] != "entailment"]
            au = auroc(pos, neg)
            fnr = st.mean(1.0 if x < 0.5 else 0.0 for x in pos) if pos else None
            cells += [f"{acc:.3f}", f(au), f(fnr)]
        L.append(f"| {sub} | {n} | " + " | ".join(cells) + " |")
    # ---------------- throughput
    L += ["", "## Throughput (one GPU, batch 32 = the reward's Scorers batch; pairs/s = NLI forward passes per second)", "", "| model | pair type | jobs | mean pairs/s | min–max | GPUs seen |", "|---|---|---|---|---|---|"]
    for (m, kind), v in sorted(throughput().items()):
        L.append(f"| {m} | {kind} | {len(v)} | {st.mean(x[0] for x in v):.0f} | {min(x[0] for x in v):.0f}–{max(x[0] for x in v):.0f} | {', '.join(sorted(set(x[2] for x in v)))} |")
    # ---------------- 3e dump
    pairs = [json.loads(l) for l in open(f"{OUT}/3e_pairs.jsonl")] if os.path.exists(f"{OUT}/3e_pairs.jsonl") else []
    sc = {m: {r["pair_id"]: r["scores"] for r in load("3e_score", m)[0]} for m in MODELS}
    if pairs:
        with open(f"{REPO}/reward_redesign/judge_pairs_300.jsonl", "w") as fj, open(f"{REPO}/reward_redesign/judge_pairs_300.md", "w") as fm:
            fm.write("# 300 matched (rollout step, reference step) pairs from run M — for hand labeling (label: entail / partial / contradict / unrelated)\n\n"
                     "Columns s_<model>_<config> are the judge scores under both models; `label` is blank.\n\n")
            for p in pairs:
                row = dict(p); row["label"] = ""
                for m in MODELS:
                    for c in CONFIGS:
                        row[f"s_{m}_{c}"] = sc[m].get(p["pair_id"], {}).get(c)
                fj.write(json.dumps(row, ensure_ascii=False) + "\n")
                fm.write(f"## pair {p['pair_id']} — step {p['step']} ({p['bucket']}), rollout {'correct' if p['acc'] else 'wrong'}, assigned s (cur, whole) = {p['s_assign_cur']:.3f}\n\n")
                fm.write("**rollout step:**\n\n```\n" + p["rollout_step"] + "\n```\n\n**reference step:**\n\n```\n" + p["ref_step"] + "\n```\n\n")
                fm.write("| " + " | ".join(f"{m}/{c}" for m in MODELS for c in CONFIGS) + " |\n|" + "---|" * 10 + "\n| " +
                         " | ".join(f(sc[m].get(p["pair_id"], {}).get(c)) for m in MODELS for c in CONFIGS) + " |\n\nlabel: ______\n\n")
        L += ["", f"## 3e. {len(pairs)} real matched pairs dumped to reward_redesign/judge_pairs_300.jsonl / .md (blank `label` field) — cells: "
              + ", ".join(f"{k[0]} acc={k[1]}: {v}" for k, v in sorted(collections.Counter((p['bucket'], p['acc']) for p in pairs).items()))]
    else:
        L += ["", "## 3e. MISSING (3e_pairs.jsonl absent)"]
    # ---------------- read + winner
    L += ["", "## Read (selection rule: among configs with 3a FPR < 10 %, lowest 3b FNR wins; throughput as tiebreak)", ""]
    cands = []
    for m in MODELS:
        ra, _ = load("3a", m); rb, _ = load("3b", m)
        if not ra or not rb:
            continue
        for c in CONFIGS:
            fpr = st.mean(1.0 if r["edited"][c] >= 0.5 else 0.0 for r in ra); fnr = st.mean(1.0 if r["para"][c] < 0.5 else 0.0 for r in rb)
            cands.append((m, c, fpr, fnr, calls.get((m, c), 0)))
    ok = [x for x in cands if x[2] < 0.10]
    if not cands:
        L.append("INCOMPLETE — 3a/3b rows missing.")
    else:
        L.append("| model | config | 3a FPR | 3b FNR | calls/pair | eligible (FPR < 0.10) |"); L.append("|---|---|---|---|---|---|")
        for m, c, fpr, fnr, nc in cands:
            L.append(f"| {m} | {c} | {fpr:.3f} | {fnr:.3f} | {nc:.1f} | {'yes' if fpr < 0.10 else 'no'} |")
        if ok:
            w = sorted(ok, key=lambda x: (x[3], x[4]))[0]
            cur_whole = next((x for x in cands if x[0] == "cur" and x[1] == "whole"), None)
            kind = ("both a model change and a pooling change" if (w[0] != "cur" and w[1] != "whole") else ("a model change only" if w[0] != "cur" else "a pooling change only"))
            L.append(f"\n**Winner: {w[0]} / {w[1]}** — 3a FPR {w[2]:.3f}, 3b FNR {w[3]:.3f}, {w[4]:.1f} NLI calls per pair; versus today's judge (cur / whole: FPR "
                     f"{cur_whole[2]:.3f}, FNR {cur_whole[3]:.3f}). This is {kind}. It is a **candidate for arms 2–6 after a probe rerun** "
                     f"(reward_redesign/redteam_reward.py). Epoch 2 (M2/R2) stays on the current judge and the `whole` config regardless.")
        else:
            L.append("\n**No config reaches 3a FPR < 10 %** — no winner under the rule; the corruption is a one-number edit inside a ~20-clause step, and neither judge "
                     "drives the reward's s below 0.5 reliably. Epoch 2 stays on the current judge and the `whole` config.")
    if status:
        L += ["", "## Missing / incomplete", ""] + [f"- {s}" for s in status]
    open(f"{REPO}/reward_redesign/JUDGE_BENCH.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
