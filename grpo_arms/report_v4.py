#!/usr/bin/env python
"""report_v4.py — write grpo_arms/V4_REPORT.md (derived from report_v3.py) from the vpb_test generations + the three training logs (chunker env, CPU).

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
import monitor_v4 as mv  # noqa: E402

E = f"{REPO}/grpo_arms/evals"
TEST_SHA = open(f"{REPO}/grpo_arms/data/vpb_test.sha256").read().strip()
ROWS = [("base3b_2ktest", "base3b", "-"), ("cs25_2ktest", "cs25 (init)", "25"), ("arm1v2_step183_2ktest", "arm1v2 (v2 reward, std-norm)", "183"),
        ("ctrl_step60_2ktest", "ctrl answer-only", "60"), ("ctrl_step90_2ktest", "ctrl answer-only", "90"),
        ("v3a_step60_2ktest", "v3a match, no std", "60"), ("v3a_step90_2ktest", "v3a match, no std", "90"),
        ("v3b_step40_2ktest", "v3b match, no std, n8, filter (stopped at 46)", "40"),
        ("ctrl_base_step60_2ktest", "ctrl_base answer-only, base init", "60"), ("ctrl_base_step90_2ktest", "ctrl_base answer-only, base init", "90"),
        ("A_ctrl_v4_step30_2ktest", "**A** answer-only, lr 1e-6 / KL 0.01 / clip 0.2-0.28", "30"), ("A_ctrl_v4_step60_2ktest", "**A** answer-only, lr 1e-6 / KL 0.01 / clip 0.2-0.28", "60"),
        ("B_matchv4_softgate_step30_2ktest", "**B** match + soft gates, same optimizer", "30"), ("B_matchv4_softgate_step60_2ktest", "**B** match + soft gates, same optimizer", "60")]


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
    out = [f"# V4 REPORT — answer-only control (A) vs match + soft gates (B) at lr 1e-6 / KL 0.01 / clip 0.2–0.28 (generated {datetime.now():%Y-%m-%d %H:%M} by grpo_arms/report_v4.py)", "",
           "Account grp_bshettah. Init = cold-start checkpoint-25, 60 steps, batch 32, n=5, MAXRESP 2048, std-normalisation on, no filter_groups, no entropy bonus.",
           "Reward matching unchanged; the only reward change is GATE_MODE=soft on run B (truncation still R=0; missing/misplaced answer costs only the answer term).",
           "Decisions on vpb_dev (400), numbers here on vpb_test (2,456, sha " + TEST_SHA[:16] + "…).", ""]
    # 1. VPB test table
    scores = {t: score(t) for t, _, _ in ROWS}
    cs = scores.get("cs25_2ktest")
    cs_acc = cs["acc"] if cs and "acc" in cs else None
    out += ["## 1. VPB test (2,456 questions, greedy, max_tokens 2048)", "",
            "| model | step | n | accuracy | Δ vs cs25 | Δ vs base3b | finish=length rate |", "|---|---|---|---|---|---|---|"]
    base_r = scores.get("base3b_2ktest"); base_acc0 = base_r["acc"] if base_r and "acc" in base_r else None
    for t, label, step in ROWS:
        r = scores[t]
        if r is None:
            out.append(f"| {label} | {step} | - | MISSING (vpb_gen_{t}.jsonl absent) | - | - | - |")
        elif "acc" not in r:
            out.append(f"| {label} | {step} | {r['n']} | WRONG EVAL SET (sha {r['bad_sha']}) | - | - | - |")
        else:
            d = f"{r['acc'] - cs_acc:+.4f}" if cs_acc is not None else "-"
            db = f"{r['acc'] - base_acc0:+.4f}" if base_acc0 is not None else "-"
            out.append(f"| {label} | {step} | {r['n']} | {r['acc']:.4f} | {d} | {db} | {r['len_rate']:.3f} |")
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
        hdr = "| step | trainval acc | vpb_dev acc | match | answer | entropy | KL | resp_len | clip | mixed grp frac |" + (" soft-gate fired |" if name.startswith("B_") else "")
        out += [f"### {name} (progress {prog}/60)", "", hdr, "|---" * (hdr.count("|") - 1) + "|"]
        v0 = val.get(0, {})
        out.append(f"| 0 | {mv.fmt(v0.get('trainval_acc'))} | {mv.fmt(v0.get('vpb_dev_acc'))} | - | - | - | - | - | - | - |" + (" - |" if name.startswith("B_") else ""))
        for s in [x for x in range(10, 61, 10) if x <= max([0] + list(train.keys()) + list(roll.keys()))]:
            t, v, r = train.get(s, {}), val.get(s, {}), roll.get(s, {})
            row = (f"| {s} | {mv.fmt(v.get('trainval_acc'))} | {mv.fmt(v.get('vpb_dev_acc'))} | {mv.fmt(r.get('match'))} | {mv.fmt(r.get('answer'))} | "
                   f"{mv.fmt(t.get('entropy'))} | {mv.fmt(t.get('kl'), 4)} | {mv.fmt(t.get('resp_len'), 0)} | {mv.fmt(t.get('clip'))} | {mv.fmt(r.get('mixed_frac'))} |")
            if name.startswith("B_"):
                row += f" {mv.fmt(r.get('soft_gated'))} |"
            out.append(row)
        out.append("")
    # 3. probe table (hard vs soft) + verdict tree
    out += ["", "## 3. Probe (reward_redesign/probe.jsonl + attackers, 1,080 rows incl. the v4 answer-first attacker): hard vs soft gates", ""]
    try:
        import statistics as _st
        hard = [json.loads(l) for l in open(f"{REPO}/reward_redesign/probe_scored_v4atk_hard.jsonl")]
        soft = [json.loads(l) for l in open(f"{REPO}/reward_redesign/probe_scored_v4atk_soft.jsonl")]
        out += ["| population | hard R median | soft R median | soft gated |", "|---|---|---|---|"]
        for pop in ("gold", "clean", "attack_giant", "attack_repeat6", "attack_answer_first", "degenerate"):
            h = [r["new"]["R"] for r in hard if r["population"] == pop]; sf = [r["new"] for r in soft if r["population"] == pop]
            out.append(f"| {pop} | {_st.median(h):.2f} | {_st.median(x['R'] for x in sf):.2f} | {sum(x['gated'] for x in sf)}/{len(sf)} |")
    except Exception as exc:  # noqa: BLE001
        out.append(f"(probe files missing: {exc})")
    out += ["", "## 4. Verdict (tree from the brief)", ""]
    def acc(t):
        r = scores.get(t); return r["acc"] if r and "acc" in r else None
    a60, b60, a30, b30 = acc("A_ctrl_v4_step60_2ktest"), acc("B_matchv4_softgate_step60_2ktest"), acc("A_ctrl_v4_step30_2ktest"), acc("B_matchv4_softgate_step30_2ktest")
    def s_(x): return "MISSING" if x is None else f"{x:.4f}"
    ca = curves.get("A_ctrl_v4")
    dev60 = ent_min = tv0 = tv60 = None
    if ca:
        train, val, roll, filt, prog = ca
        dev60 = val.get(60, {}).get("vpb_dev_acc")
        ents = [t.get("entropy") for st_, t in train.items() if st_ % 10 == 0 and t.get("entropy") is not None]
        ent_min = min(ents) if ents else None
        tv0, tv60 = val.get(0, {}).get("trainval_acc"), val.get(60, {}).get("trainval_acc")
    out.append(f"A: vpb_test 30/60 = {s_(a30)}/{s_(a60)}; vpb_dev@60 = {s_(dev60)}; min entropy at logged steps = {s_(ent_min)}; train-val 0→60 = {s_(tv0)} → {s_(tv60)}")
    out.append(f"B: vpb_test 30/60 = {s_(b30)}/{s_(b60)}")
    if None in (a60, dev60, ent_min):
        out.append("**Verdict: INCOMPLETE** — a required A number is missing (see above / V4_STATUS.md).")
    else:
        a_holds = dev60 >= 0.29 and ent_min >= 0.4
        stagnant = (tv0 is not None and tv60 is not None and (tv60 - tv0) < 0.05)
        if not a_holds:
            out.append("**Verdict: A does not hold** (needs vpb_dev ≥ 0.29 at step 60 and entropy ≥ 0.4 throughout) — lr 1e-6 / KL 0.01 / clip-high 0.28 "
                       "is insufficient; recommend an adaptive entropy bonus and/or lr 5e-7. Entropy curve: see §2 table for A.")
        elif b60 is None:
            out.append("**Verdict: A holds; B missing** (run B not evaluated — see V4_STATUS.md for whether the probe go-condition blocked it).")
        elif b60 >= a60:
            out.append("**Verdict: optimizer fixed and match + soft gates ≥ control** — this is the configuration for arms 2–6.")
        else:
            out.append("**Verdict: optimizer fixed; match + soft gates < control on vpb_test** — the match term or the soft gate is costing accuracy; "
                       "recommend the one-line ablation: match with HARD gates at the same optimizer (GATE_MODE=hard).")
        if a_holds and stagnant:
            out.append("**Also:** A is stable but stagnant (train-val moved < +0.05 by step 60) — recommend 120 steps or lr 2e-6.")
    # fairshare cost
    out += ["", "## 5. Fairshare cost (grp_bshettah)", ""]
    try:
        fs = subprocess.run(["myfairshare"], capture_output=True, text=True, timeout=60).stdout
        out += ["after (myfairshare):", "```", fs.strip(), "```", "before: see V4_STATUS.md step 0 (RawUsage_CHE 1262.7, RealFairShare 0.492369)."]
    except Exception as exc:  # noqa: BLE001
        out.append(f"(myfairshare failed: {exc})")
    # 4. failures
    out += ["", "## 6. Failures and resubmits", ""]
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
    open(f"{REPO}/grpo_arms/V4_REPORT.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
