#!/usr/bin/env python
"""report_e2.py — write grpo_arms/E2_REPORT.md for the EPOCH-2 runs (M2: match resumed from M@183 -> 366; R2: RLVR + adaptive entropy resumed from R@183 -> 366); derived from report_full.py.
1. vpb_test table (2,456 q, greedy, 2048 tok): base3b, cs25, v4 A@30/60, v4 B@30/60, M@90/120/150/183, R@30..183 — accuracy, Δ vs cs25 (0.3192)
   and base (0.3025), finish=length rate, per-source; paired McNemar (exact binomial on discordant pairs) vs cs25, vs v4 B@60 (M rows) /
   v4 A@60 (R rows), and M vs R at matching steps. Scorer = arm_reward extract_answer/answer_correct (same as score_vpb.py).
2. Curves 0..183 for both runs (M with B's 0-60 prepended via the v4 B logs) from monitor_full's parsers.
3. Verdict: does M hold/rise to 183 with entropy >= 0.4? does R avoid the post-step-30 decline of v4 A, where does its entropy settle and what
   coefficient did it need? best checkpoint each; M vs R at those steps with p-values.
4. Fairshare before/after on grp_vgupt140; failures/resubmits (sacct).
Every number is copied from a file; missing rows are MISSING, never estimated.
"""
import collections
import json
import math
import os
import statistics as st
import subprocess
import sys
from datetime import datetime

REPO = "/scratch/sghos104/rlpt"
sys.path.insert(0, f"{REPO}/grpo_arms")
from arm_reward import answer_correct, extract_answer  # noqa: E402
import monitor_e2 as mf  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

E = f"{REPO}/grpo_arms/evals"
TEST_SHA = open(f"{REPO}/grpo_arms/data/vpb_test.sha256").read().strip()
FS_BEFORE = "grp_bshettah / sghos104 RawUsage_CHE 1518.0, RealFairShare 0.4679 (2026-09-19 ~04:00, E2_STATUS.md preflight)"
ROWS = [("base3b_2ktest", "base3b (Qwen2.5-VL-3B-Instruct)", "-", "ref"), ("cs25_2ktest", "cs25 (cold-start init)", "25", "ref"),
        ("A_ctrl_v4_step30_2ktest", "v4 A answer-only", "30", "A"), ("A_ctrl_v4_step60_2ktest", "v4 A answer-only", "60", "A"),
        ("B_matchv4_softgate_step30_2ktest", "v4 B match + soft gates", "30", "B"), ("B_matchv4_softgate_step60_2ktest", "v4 B match + soft gates", "60", "B")]
ROWS += [(f"B_matchv4_softgate_step{s}_2ktest", "epoch-1 M match + soft gates", str(s), "M1") for s in (90, 120, 150, 183)]
ROWS += [(f"R_rlvr_entropy_step{s}_2ktest", "epoch-1 R RLVR + adaptive entropy", str(s), "R1") for s in (30, 60, 90, 120, 150, 183)]
E2_STEPS = (210, 240, 270, 300, 330, 360, 366)
ROWS += [(f"M2_match_e2_step{s}_2ktest", "**M2** epoch 2 of M", str(s), "M") for s in E2_STEPS]
ROWS += [(f"R2_rlvr_e2_step{s}_2ktest", "**R2** epoch 2 of R", str(s), "R") for s in E2_STEPS]
SE = math.sqrt(0.33 * 0.67 / 2456)   # ~0.0095, one binomial SE at the accuracies in play


def score(tag):
    p = f"{E}/vpb_gen_{tag}.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    if not rows or rows[0].get("eval_set_sha256") != TEST_SHA:
        return {"n": len(rows), "bad_sha": rows[0].get("eval_set_sha256") if rows else None}
    tot, cor, ok_by = collections.Counter(), collections.Counter(), {}
    n_len = 0
    for r in rows:
        ok = int(answer_correct(extract_answer(r["output"])[0], r["answer"]))
        ok_by[r["qid"]] = ok
        tot[r["data_source"]] += 1; cor[r["data_source"]] += ok
        tot["ALL"] += 1; cor["ALL"] += ok
        n_len += r.get("finish_reason") == "length"
    return {"n": tot["ALL"], "acc": cor["ALL"] / tot["ALL"], "len_rate": n_len / tot["ALL"], "ok": ok_by,
            "per_src": {s: cor[s] / tot[s] for s in tot if s != "ALL"}}


def mcnemar(x, y):
    """x, y: score dicts. Returns (b, c, p): b = x right & y wrong, c = x wrong & y right; exact two-sided binomial p."""
    if not (x and y and "ok" in x and "ok" in y):
        return None
    qs = set(x["ok"]) & set(y["ok"])
    b = sum(1 for q in qs if x["ok"][q] and not y["ok"][q])
    c = sum(1 for q in qs if not x["ok"][q] and y["ok"][q])
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def pstr(m):
    return "-" if m is None else f"b={m[0]} c={m[1]} p={m[2]:.3f}"


def acc_of(scores, tag):
    r = scores.get(tag)
    return r["acc"] if r and "acc" in r else None


def s_(x, nd=4):
    return "MISSING" if x is None else f"{x:.{nd}f}"


def main():
    scores = {t: score(t) for t, _, _, _ in ROWS}
    cs, base = scores.get("cs25_2ktest"), scores.get("base3b_2ktest")
    cs_acc, base_acc = acc_of(scores, "cs25_2ktest"), acc_of(scores, "base3b_2ktest")
    a60, b60 = scores.get("R_rlvr_entropy_step183_2ktest"), scores.get("B_matchv4_softgate_step183_2ktest")   # reference = each run's own step 183
    out = [f"# E2 REPORT — epoch 2: M2 (match + soft gates, M resumed 183→366) vs R2 (RLVR + adaptive entropy, R resumed 183→366) on grp_bshettah "
           f"(generated {datetime.now():%Y-%m-%d %H:%M} by grpo_arms/report_full.py)", "",
           "Common config: lr 1e-6, kl_loss_coef 0.01 (low_var_kl), clip 0.2/0.28, n=5, batch 32, MAXRESP 2048, 8192-token caps, temperature 1.0, seed 42,",
           "std-normalised GRPO, no filter_groups, 2×A100 public, 183 steps, save every 30. M = v4 B's process continued (same dataloader from batch 61,",
           "REWARD_MODE=match GATE_MODE=soft, no entropy bonus). R = fresh from cs25, REWARD_MODE=answer_only GATE_MODE=hard, ENTROPY_MODE=adaptive",
           "(target 0.6, coeff <- clip(coeff + 0.002·(0.6 − entropy), 0, 0.01) after every step; grpo_arms/adaptive_entropy.py).",
           f"Numbers on vpb_test (2,456, sha {TEST_SHA[:16]}…), greedy, 2048 tokens; McNemar = exact binomial test on the discordant pairs (b = row right/reference wrong, c = reverse).", ""]
    # 1. table
    out += ["## 1. vpb_test", "", "| model | step | n | accuracy | Δ vs cs25 | Δ vs base3b | finish=length | McNemar vs cs25 | McNemar vs own step 183 (M2 vs M@183, R2 vs R@183) | McNemar M2 vs R2 (same step) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for t, label, step, fam in ROWS:
        r = scores[t]
        if r is None:
            out.append(f"| {label} | {step} | - | MISSING (vpb_gen_{t}.jsonl absent) | - | - | - | - | - | - |"); continue
        if "acc" not in r:
            out.append(f"| {label} | {step} | {r['n']} | WRONG EVAL SET (sha {r['bad_sha']}) | - | - | - | - | - | - |"); continue
        d = f"{r['acc'] - cs_acc:+.4f}" if cs_acc is not None else "-"
        db = f"{r['acc'] - base_acc:+.4f}" if base_acc is not None else "-"
        ref = b60 if fam == "M" else (a60 if fam == "R" else None)
        other = scores.get(f"R2_rlvr_e2_step{step}_2ktest") if fam == "M" else (scores.get(f"M2_match_e2_step{step}_2ktest") if fam == "R" else None)
        out.append(f"| {label} | {step} | {r['n']} | {r['acc']:.4f} | {d} | {db} | {r['len_rate']:.3f} | {pstr(mcnemar(r, cs))} | "
                   f"{pstr(mcnemar(r, ref)) if ref else '-'} | {pstr(mcnemar(r, other)) if other else '-'} |")
    srcs = sorted({s for r in scores.values() if r and "per_src" in r for s in r["per_src"]})
    if srcs:
        out += ["", "per-source accuracy:", "", "| model | step | " + " | ".join(srcs) + " |", "|---|---|" + "---|" * len(srcs)]
        for t, label, step, _ in ROWS:
            r = scores[t]
            if r and "per_src" in r:
                out.append(f"| {label} | {step} | " + " | ".join(f"{r['per_src'].get(s, float('nan')):.4f}" for s in srcs) + " |")
    # 2. curves
    out += ["", "## 2. Training curves 0..183 (train-val = 120 VisualPRM prompts source-avg gated acc; vpb_dev = 400; match/answer/format/mixed/n_seg from rollout dumps; M's 0-60 = v4 B)", ""]
    curves = {}
    for name, cfg in mf.RUNS.items():
        if not os.path.isdir(cfg["dir"]):
            out.append(f"### {name}: no run directory"); continue
        train, val, prog, _ = mf.parse_logs(mf.logs_for(cfg["dir"]))
        roll = mf.parse_rollouts(cfg["dir"])
        curves[name] = (train, val, roll, prog)
        hdr = "| step | trainval acc | vpb_dev acc | match | answer | format | entropy | ent coeff | KL | resp_len | trunc | mixed grp frac | n_seg | soft-gate fired |"
        out += [f"### {name} — {cfg['label']} (progress {prog}/366; rows ≤ 183 = epoch 1)", "", hdr, "|---" * (hdr.count("|") - 1) + "|"]
        v0 = val.get(0, {})
        out.append(f"| 0 | {mf.fmt(v0.get('trainval_acc'))} | {mf.fmt(v0.get('vpb_dev_acc'))} | - | - | - | - | - | - | - | - | - | - | - |")
        top = max([0] + list(train.keys()) + list(roll.keys()))
        for s in [x for x in list(range(10, 361, 10)) + [183, 366] if x <= top]:
            t, v, r = train.get(s, {}), val.get(s, {}), roll.get(s, {})
            out.append(f"| {s} | {mf.fmt(v.get('trainval_acc'))} | {mf.fmt(v.get('vpb_dev_acc'))} | {mf.fmt(r.get('match')) if cfg['match'] else 'n/a'} | "
                       f"{mf.fmt(r.get('answer'))} | {mf.fmt(r.get('format'))} | {mf.fmt(t.get('entropy'))} | {mf.fmt(t.get('ent_coeff'), 5) if not cfg['match'] else 'off'} | "
                       f"{mf.fmt(t.get('kl'), 4)} | {mf.fmt(t.get('resp_len'), 0)} | {mf.fmt(t.get('clip'))} | {mf.fmt(r.get('mixed_frac'))} | {mf.fmt(r.get('n_seg'), 1)} | "
                       f"{mf.fmt(r.get('soft_gated')) if cfg['match'] else 'n/a'} |")
        out.append("")
    # 3. verdict (epoch 2)
    out += ["## 3. Verdict", ""]
    m_tags = {s: f"M2_match_e2_step{s}_2ktest" for s in E2_STEPS}
    r_tags = {s: f"R2_rlvr_e2_step{s}_2ktest" for s in E2_STEPS}
    m_acc = {s: acc_of(scores, t) for s, t in m_tags.items()}
    r_acc = {s: acc_of(scores, t) for s, t in r_tags.items()}
    m183, r183 = acc_of(scores, "B_matchv4_softgate_step183_2ktest"), acc_of(scores, "R_rlvr_entropy_step183_2ktest")
    def ent_stats(name):
        if name not in curves:
            return None
        tr = curves[name][0]
        ents = [(s, t["entropy"], t.get("ent_coeff")) for s, t in sorted(tr.items()) if s > 183 and t.get("entropy") is not None]
        if not ents:
            return None
        coeffs = [c for _, _, c in ents if c is not None]
        return dict(min=min(ents, key=lambda x: x[1]), mean_last30=st.mean(e for _, e, _ in ents[-30:]), n=len(ents),
                    coeff_max=max(coeffs) if coeffs else None, coeff_mean=st.mean(coeffs) if coeffs else None)
    em, er = ent_stats("M2"), ent_stats("R2")
    out.append(f"**M2** (own reference M@183 = {s_(m183)}): vpb_test " + " / ".join(f"{s}={s_(m_acc[s])}" for s in E2_STEPS)
               + (f"; entropy 184-366: min {em['min'][1]:.3f} at {em['min'][0]}, last-30 mean {em['mean_last30']:.3f}" if em else "; entropy: MISSING"))
    def trend(acc, ref, label):
        have = [(s, a) for s, a in acc.items() if a is not None]
        if not have or ref is None:
            return f"- {label}: **INCOMPLETE**"
        best = max(have, key=lambda x: x[1]); last = have[-1]
        mm = mcnemar(scores[m_tags[last[0]] if label.startswith("M") else r_tags[last[0]]], b60 if label.startswith("M") else a60)
        d = last[1] - ref
        if d > SE and mm and mm[2] < 0.05:
            v = f"**keeps rising**: step {last[0]} = {last[1]:.4f} vs 183 = {ref:.4f} ({d:+.4f}, p={mm[2]:.3f})"
        elif abs(d) <= SE:
            v = f"**plateaus**: step {last[0]} within one SE ({SE:.4f}) of 183 ({d:+.4f}, p={mm[2]:.3f})"
        elif d > 0:
            v = f"**rises slightly (not significant)**: {d:+.4f}, p={mm[2]:.3f}"
        else:
            v = f"**declines**: step {last[0]} is {d:+.4f} vs 183 (p={mm[2]:.3f})"
        return f"- {label}: {v}; best epoch-2 checkpoint step {best[0]} ({best[1]:.4f})"
    out.append(trend(m_acc, m183, "M2 verdict"))
    out.append(f"**R2** (own reference R@183 = {s_(r183)}): vpb_test " + " / ".join(f"{s}={s_(r_acc[s])}" for s in E2_STEPS)
               + (f"; entropy 184-366: min {er['min'][1]:.3f} at {er['min'][0]}, last-30 mean {er['mean_last30']:.3f}; coefficient max {s_(er['coeff_max'], 5)}, mean {s_(er['coeff_mean'], 5)}" if er else "; entropy: MISSING"))
    out.append(trend(r_acc, r183, "R2 verdict"))
    for s in E2_STEPS:
        mr = mcnemar(scores.get(m_tags[s]), scores.get(r_tags[s]))
        if mr:
            out.append(f"- M2 vs R2 at step {s}: {s_(m_acc[s])} vs {s_(r_acc[s])} ({m_acc[s] - r_acc[s]:+.4f}), McNemar {pstr(mr)}")
    # 4. fairshare + failures
    out += ["", "## 4. Fairshare (grp_bshettah)", "", f"before: {FS_BEFORE}"]
    try:
        fs = subprocess.run(["myfairshare"], capture_output=True, text=True, timeout=60).stdout
        out += ["after (myfairshare):", "```", fs.strip(), "```"]
    except Exception as exc:  # noqa: BLE001
        out.append(f"(myfairshare failed: {exc})")
    out += ["", "## 5. Jobs, failures and resubmits", ""]
    state = mf.load_state()
    for name, cfg in mf.RUNS.items():
        c = mf.chain(cfg["dir"])
        ids = mf.train_job_ids(c) + [j for k, e in state.get("evals", {}).items() if k.startswith(name + ":") for j in e.get("jobs", [])]
        if not ids:
            out.append(f"- {name}: no jobs"); continue
        try:
            sa = subprocess.run(["sacct", "-j", ",".join(ids), "-o", "JobID,JobName%22,State,Elapsed,ExitCode", "-P", "-X"], capture_output=True, text=True, timeout=60).stdout
        except Exception as exc:  # noqa: BLE001
            sa = f"sacct failed: {exc}"
        out += [f"- {name}: DONE={os.path.exists(os.path.join(cfg['dir'], 'DONE'))}; pruned FSDP: {[p for p in state.get('pruned', []) if p.startswith(name + ':')]}", "```", sa.strip(), "```"]
    open(f"{REPO}/grpo_arms/E2_REPORT.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
