#!/usr/bin/env python
"""report_v3.py — write grpo_arms/V3_REPORT.md from the vpb_test generations + the three training logs (chunker env, CPU).

1. VPB test table (2,456 q, 2048 tokens, greedy): base3b, cs25, arm1v2 step183, ctrl 60/90, v3a 60/90, v3b 60/90 — accuracy, delta vs cs25,
   finish_reason=length rate, per-source accuracy. Scorer = grpo_arms/score_vpb.py's extraction + correctness (arm_reward).
2. Training curves as tables (train-val acc, vpb_dev acc, match, entropy, mixed-group fraction at steps 10..90) via monitor_v3's parsers.
3. The decision against the tree in the brief.
4. Failures / resubmits (sacct on the chain ids).
Every number is copied from a file; missing rows are marked MISSING rather than estimated.
"""
import collections
import glob
import json
import os
import statistics as st
import subprocess
import sys
from datetime import datetime

REPO = "/scratch/sghos104/rlpt"
sys.path.insert(0, f"{REPO}/grpo_arms")
from arm_reward import answer_correct, extract_answer  # noqa: E402
import monitor_v3 as mv  # noqa: E402

E = f"{REPO}/grpo_arms/evals"
TEST_SHA = open(f"{REPO}/grpo_arms/data/vpb_test.sha256").read().strip()
ROWS = [("base3b_2ktest", "base3b", "-"), ("cs25_2ktest", "cs25 (init)", "25"), ("arm1v2_step183_2ktest", "arm1v2 (v2 reward, std-norm)", "183"),
        ("ctrl_step60_2ktest", "ctrl answer-only", "60"), ("ctrl_step90_2ktest", "ctrl answer-only", "90"),
        ("v3a_step60_2ktest", "v3a match, no std", "60"), ("v3a_step90_2ktest", "v3a match, no std", "90"),
        ("v3b_step40_2ktest", "v3b match, no std, n8, filter (stopped at step 46 by user; last checkpoint)", "40"),
        ("ctrl_base_step60_2ktest", "ctrl_base answer-only from BASE model (2026-09-12 follow-up)", "60"),
        ("ctrl_base_step90_2ktest", "ctrl_base answer-only from BASE model (2026-09-12 follow-up)", "90")]


def score(tag):
    p = f"{E}/vpb_gen_{tag}.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    if not rows or rows[0].get("eval_set_sha256") != TEST_SHA:
        return {"n": len(rows), "bad_sha": rows[0].get("eval_set_sha256") if rows else None}
    tot, cor = collections.Counter(), collections.Counter()
    n_len = 0
    for r in rows:
        ok = answer_correct(extract_answer(r["output"])[0], r["answer"])
        tot[r["data_source"]] += 1; cor[r["data_source"]] += ok
        tot["ALL"] += 1; cor["ALL"] += ok
        n_len += r.get("finish_reason") == "length"
    return {"n": tot["ALL"], "acc": cor["ALL"] / tot["ALL"], "len_rate": n_len / tot["ALL"],
            "per_src": {s: cor[s] / tot[s] for s in tot if s != "ALL"}}


def main():
    out = [f"# V3 REPORT — arm-1 v3 runs (generated {datetime.now():%Y-%m-%d %H:%M} by grpo_arms/report_v3.py)", "",
           "Reward spec frozen (reward_v2.score_new match path unchanged); tonight varied the optimizer only: answer-only control (ctrl),",
           "GRPO without std-normalisation (v3a), and no-std + rollout n=8 + mixed-correctness group filter (v3b). Init = cold-start checkpoint-25,",
           "90 steps, batch 32, MAXRESP 2048. Decisions on vpb_dev (400), numbers here on vpb_test (2,456, sha " + TEST_SHA[:16] + "…).", ""]
    # 1. VPB test table
    scores = {t: score(t) for t, _, _ in ROWS}
    cs = scores.get("cs25_2ktest")
    cs_acc = cs["acc"] if cs and "acc" in cs else None
    out += ["## 1. VPB test (2,456 questions, greedy, max_tokens 2048)", "",
            "| model | step | n | accuracy | Δ vs cs25 | finish=length rate |", "|---|---|---|---|---|---|"]
    for t, label, step in ROWS:
        r = scores[t]
        if r is None:
            out.append(f"| {label} | {step} | - | MISSING (vpb_gen_{t}.jsonl absent) | - | - |")
        elif "acc" not in r:
            out.append(f"| {label} | {step} | {r['n']} | WRONG EVAL SET (sha {r['bad_sha']}) | - | - |")
        else:
            d = f"{r['acc'] - cs_acc:+.4f}" if cs_acc is not None else "-"
            out.append(f"| {label} | {step} | {r['n']} | {r['acc']:.4f} | {d} | {r['len_rate']:.3f} |")
    srcs = sorted({s for r in scores.values() if r and "per_src" in r for s in r["per_src"]})
    if srcs:
        out += ["", "per-source accuracy:", "", "| model | step | " + " | ".join(srcs) + " |", "|---|---|" + "---|" * len(srcs)]
        for t, label, step in ROWS:
            r = scores[t]
            if r and "per_src" in r:
                out.append(f"| {label} | {step} | " + " | ".join(f"{r['per_src'].get(s, float('nan')):.4f}" for s in srcs) + " |")
    # 2. training curves
    out += ["", "## 2. Training curves (train-val = 120 VisualPRM prompts source-avg; vpb_dev = 400; match/answer/mixed from rollout dumps)", ""]
    curves = {}
    for name, run in mv.RUNS.items():
        if not os.path.isdir(run):
            out.append(f"### {name}: no run directory"); continue
        train, val, prog, filt, _ = mv.parse_logs(mv.logs_for(run))
        roll = mv.parse_rollouts(run)
        curves[name] = (train, val, roll, filt, prog)
        hdr = "| step | trainval acc | vpb_dev acc | match | answer | entropy | resp_len | clip | mixed grp frac |" + (" gen batches | trained | partial |" if name == "v3b" else "")
        out += [f"### {name} (progress {prog}/90)", "", hdr, "|---" * (hdr.count("|") - 1) + "|"]
        v0 = val.get(0, {})
        out.append(f"| 0 | {mv.fmt(v0.get('trainval_acc'))} | {mv.fmt(v0.get('vpb_dev_acc'))} | - | - | - | - | - | - |" + (" - | - | - |" if name == "v3b" else ""))
        for s in [x for x in range(10, 91, 10) if x <= max([0] + list(train.keys()) + list(roll.keys()))]:
            t, v, r = train.get(s, {}), val.get(s, {}), roll.get(s, {})
            row = (f"| {s} | {mv.fmt(v.get('trainval_acc'))} | {mv.fmt(v.get('vpb_dev_acc'))} | {mv.fmt(r.get('match'))} | {mv.fmt(r.get('answer'))} | "
                   f"{mv.fmt(t.get('entropy'))} | {mv.fmt(t.get('resp_len'), 0)} | {mv.fmt(t.get('clip'))} | {mv.fmt(r.get('mixed_frac'))} |")
            if name == "v3b":
                f_ = filt.get(s, {})
                row += f" {f_.get('gen_batches', '-')} | {f_.get('trained', '-')} | {f_.get('partial', '-')} |"
            out.append(row)
        out.append("")
    # 3. decision
    out += ["## 3. Decision (tree from the brief; vpb_test accuracy at 2048 tokens, final step 90; step 60 in brackets)", ""]
    def acc(t):
        r = scores.get(t); return r["acc"] if r and "acc" in r else None
    ctrl90, v3a90, v3b90 = acc("ctrl_step90_2ktest"), acc("v3a_step90_2ktest"), acc("v3b_step40_2ktest")   # v3b: final = step 40 (run stopped at 46)
    ctrl60, v3a60, v3b60 = acc("ctrl_step60_2ktest"), acc("v3a_step60_2ktest"), None
    def s_(x): return "MISSING" if x is None else f"{x:.4f}"
    out.append(f"cs25 = {s_(cs_acc)}; ctrl90 = {s_(ctrl90)} [60: {s_(ctrl60)}]; v3a90 = {s_(v3a90)} [60: {s_(v3a60)}]; v3b final = step 40 = {s_(v3b90)} (run stopped at step 46 on the user's instruction after vpb_dev fell to 0.0425)")
    if None in (cs_acc, ctrl90) or (v3a90 is None and v3b90 is None):
        out.append("**Decision: INCOMPLETE** — a required row is missing (see table); check V3_STATUS.md for the failed job.")
    else:
        ctrl_ok = ctrl90 >= cs_acc
        v3_ok = any(x is not None and x >= cs_acc for x in (v3a90, v3b90))
        if ctrl_ok and v3_ok:
            best = max([(v3a90 or -1, "v3a"), (v3b90 or -1, "v3b")])[1]
            out.append(f"**Decision: optimizer fix works** — ctrl ≥ cs25 and {best} ≥ cs25. Recommend the v3b configuration (no std-normalisation, n=8, "
                       "mixed-correctness filter) for arms 2–6" + ("" if best == "v3b" else f" (note: {best} scored higher than v3b at step 90; see table)") + ".")
        elif ctrl_ok and not v3_ok:
            out.append("**Decision: the reward or the references are the problem** — the answer-only control holds cs25 accuracy but neither match arm does, "
                       "even without std-normalisation / with mixed groups. See §7 re-aggregation table (grpo_arms/REWARD_REAGG.md) for which reward variant "
                       "keeps the answer share of the advantage.")
        else:
            out.append("**Decision: the training setup degrades regardless of reward** — ctrl < cs25. First suspects (from V3_STATUS.md step 0.3): "
                       "lr 1e-05, KL loss coef 0.001, entropy_coeff 0, temperature 1.0, rollout n 5, ppo_mini_batch 8 / micro 1 (dynamic bsz, 8192 tok cap).")
    # 3b. follow-up: answer-only control from the base model (tests the cold-start-init confound)
    cb90, cb60 = acc("ctrl_base_step90_2ktest"), acc("ctrl_base_step60_2ktest")
    base_acc = acc("base3b_2ktest")
    out += ["", "### 3b. Follow-up: answer-only control from the base model",
            f"base3b = {s_(base_acc)}; ctrl_base 60 = {s_(cb60)}; ctrl_base 90 = {s_(cb90)}"]
    if cb90 is not None and base_acc is not None:
        if cb90 >= base_acc:
            out.append("**ctrl_base ≥ base3b**: RLVR from the base model holds/improves accuracy, so the cold-start SFT init is the confound behind the v3 "
                       "degradation — rerun the reward comparison from the base model.")
        else:
            out.append("**ctrl_base < base3b**: even from the base model the answer-only RLVR degrades, so the optimizer config (lr 1e-05, KL 0.001, "
                       "entropy 0) or the VisualPRM prompt mix is the problem independent of the init.")
    # 4. failures
    out += ["", "## 4. Failures and resubmits", ""]
    for name, run in mv.RUNS.items():
        cj = os.path.join(run, "chain.json")
        if not os.path.exists(cj):
            out.append(f"- {name}: no chain.json"); continue
        c = json.load(open(cj))
        ids = [c.get("train")] + list(c.get("resub", [])) + str(c.get("eval_split_2ktest", "")).split(":")
        ids = [i for i in ids if i]
        try:
            sa = subprocess.run(["sacct", "-j", ",".join(ids), "-o", "JobID,JobName%20,State,Elapsed,ExitCode", "-P"], capture_output=True, text=True, timeout=60).stdout
            sa = "\n".join(l for l in sa.splitlines() if ".batch" not in l and ".extern" not in l)
        except Exception as exc:  # noqa: BLE001
            sa = f"sacct failed: {exc}"
        flags = [l for l in open(os.path.join(run, "STOP_WATCHDOG")).read().splitlines()] if os.path.exists(os.path.join(run, "STOP_WATCHDOG")) else []
        out += [f"- {name}: DONE={os.path.exists(os.path.join(run, 'DONE'))}" + (f"; WATCHDOG STOP: {flags[0]}" if flags else ""), "```", sa, "```"]
    open(f"{REPO}/grpo_arms/V3_REPORT.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
