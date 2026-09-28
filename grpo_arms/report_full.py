#!/usr/bin/env python
"""report_full.py — write grpo_arms/FULL_REPORT.md for the FULL runs (M: match resumed from v4 B@60 -> 183; R: RLVR + adaptive entropy).
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
import monitor_full as mf  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

E = f"{REPO}/grpo_arms/evals"
TEST_SHA = open(f"{REPO}/grpo_arms/data/vpb_test.sha256").read().strip()
FS_BEFORE = "grp_vgupt140 / sghos104 RawUsage_CHE 3832.4, RealFairShare 0.164363 (2026-09-18 02:28, FULL_STATUS.md preflight)"
ROWS = [("base3b_2ktest", "base3b (Qwen2.5-VL-3B-Instruct)", "-", "ref"), ("cs25_2ktest", "cs25 (cold-start init)", "25", "ref"),
        ("A_ctrl_v4_step30_2ktest", "v4 A answer-only", "30", "A"), ("A_ctrl_v4_step60_2ktest", "v4 A answer-only", "60", "A"),
        ("B_matchv4_softgate_step30_2ktest", "v4 B match + soft gates", "30", "B"), ("B_matchv4_softgate_step60_2ktest", "v4 B match + soft gates", "60", "B")]
ROWS += [(f"B_matchv4_softgate_step{s}_2ktest", "**M** match + soft gates (B resumed)", str(s), "M") for s in (90, 120, 150, 183)]
ROWS += [(f"R_rlvr_entropy_step{s}_2ktest", "**R** RLVR + adaptive entropy", str(s), "R") for s in (30, 60, 90, 120, 150, 183)]
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
    a60, b60 = scores.get("A_ctrl_v4_step60_2ktest"), scores.get("B_matchv4_softgate_step60_2ktest")
    out = [f"# FULL REPORT — M (match + soft gates, v4 B resumed 60→183) vs R (RLVR answer-only + adaptive entropy, 0→183) on grp_vgupt140 "
           f"(generated {datetime.now():%Y-%m-%d %H:%M} by grpo_arms/report_full.py)", "",
           "Common config: lr 1e-6, kl_loss_coef 0.01 (low_var_kl), clip 0.2/0.28, n=5, batch 32, MAXRESP 2048, 8192-token caps, temperature 1.0, seed 42,",
           "std-normalised GRPO, no filter_groups, 2×A100 public, 183 steps, save every 30. M = v4 B's process continued (same dataloader from batch 61,",
           "REWARD_MODE=match GATE_MODE=soft, no entropy bonus). R = fresh from cs25, REWARD_MODE=answer_only GATE_MODE=hard, ENTROPY_MODE=adaptive",
           "(target 0.6, coeff <- clip(coeff + 0.002·(0.6 − entropy), 0, 0.01) after every step; grpo_arms/adaptive_entropy.py).",
           f"Numbers on vpb_test (2,456, sha {TEST_SHA[:16]}…), greedy, 2048 tokens; McNemar = exact binomial test on the discordant pairs (b = row right/reference wrong, c = reverse).", ""]
    # 1. table
    out += ["## 1. vpb_test", "", "| model | step | n | accuracy | Δ vs cs25 | Δ vs base3b | finish=length | McNemar vs cs25 | McNemar vs v4 @60 (B for M, A for R) | McNemar M vs R (same step) |",
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
        other = scores.get(f"R_rlvr_entropy_step{step}_2ktest") if fam == "M" else (scores.get(f"B_matchv4_softgate_step{step}_2ktest") if fam == "R" else None)
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
        out += [f"### {name} — {cfg['label']} (progress {prog}/183)", "", hdr, "|---" * (hdr.count("|") - 1) + "|"]
        v0 = val.get(0, {})
        out.append(f"| 0 | {mf.fmt(v0.get('trainval_acc'))} | {mf.fmt(v0.get('vpb_dev_acc'))} | - | - | - | - | - | - | - | - | - | - | - |")
        top = max([0] + list(train.keys()) + list(roll.keys()))
        for s in [x for x in list(range(10, 181, 10)) + [183] if x <= top]:
            t, v, r = train.get(s, {}), val.get(s, {}), roll.get(s, {})
            out.append(f"| {s} | {mf.fmt(v.get('trainval_acc'))} | {mf.fmt(v.get('vpb_dev_acc'))} | {mf.fmt(r.get('match')) if cfg['match'] else 'n/a'} | "
                       f"{mf.fmt(r.get('answer'))} | {mf.fmt(r.get('format'))} | {mf.fmt(t.get('entropy'))} | {mf.fmt(t.get('ent_coeff'), 5) if not cfg['match'] else 'off'} | "
                       f"{mf.fmt(t.get('kl'), 4)} | {mf.fmt(t.get('resp_len'), 0)} | {mf.fmt(t.get('clip'))} | {mf.fmt(r.get('mixed_frac'))} | {mf.fmt(r.get('n_seg'), 1)} | "
                       f"{mf.fmt(r.get('soft_gated')) if cfg['match'] else 'n/a'} |")
        out.append("")
    # 3. verdict
    out += ["## 3. Verdict", ""]
    m_tags = {s: f"B_matchv4_softgate_step{s}_2ktest" for s in (90, 120, 150, 183)}
    r_tags = {s: f"R_rlvr_entropy_step{s}_2ktest" for s in (30, 60, 90, 120, 150, 183)}
    m_acc = {s: acc_of(scores, t) for s, t in m_tags.items()}
    r_acc = {s: acc_of(scores, t) for s, t in r_tags.items()}
    b60_acc, a60_acc, a30_acc = acc_of(scores, "B_matchv4_softgate_step60_2ktest"), acc_of(scores, "A_ctrl_v4_step60_2ktest"), acc_of(scores, "A_ctrl_v4_step30_2ktest")
    # M
    ent_m = None
    if "M" in curves:
        tr = curves["M"][0]
        ents = [(s, t["entropy"]) for s, t in sorted(tr.items()) if s > 60 and t.get("entropy") is not None]
        ent_m = (min(ents, key=lambda x: x[1]) if ents else None, st.mean(e for _, e in ents[-30:]) if ents else None, len(ents))
    out.append(f"**M** (reference v4 B@60 = {s_(b60_acc)}): vpb_test 90/120/150/183 = " + " / ".join(s_(m_acc[s]) for s in (90, 120, 150, 183)) +
               (f"; entropy over steps 61-183: min {ent_m[0][1]:.3f} at step {ent_m[0][0]}, mean of the last 30 logged steps {ent_m[1]:.3f} ({ent_m[2]} steps logged)" if ent_m and ent_m[0] else "; entropy: MISSING"))
    if m_acc[183] is None or b60_acc is None:
        out.append("- M verdict: **INCOMPLETE** (step-183 row or the B@60 reference is missing).")
    else:
        mm = mcnemar(scores[m_tags[183]], b60)
        ent_ok = bool(ent_m and ent_m[0] and ent_m[0][1] >= 0.4)
        if m_acc[183] > b60_acc and mm and mm[2] < 0.05:
            verdict = f"**RISES**: 183 > B@60 by {m_acc[183] - b60_acc:+.4f} (McNemar p={mm[2]:.3f})"
        elif m_acc[183] >= b60_acc - SE:
            verdict = f"**HOLDS**: 183 within one SE ({SE:.4f}) of B@60 ({m_acc[183] - b60_acc:+.4f}, p={mm[2]:.3f})"
        else:
            verdict = f"**DECLINES**: 183 is {m_acc[183] - b60_acc:+.4f} vs B@60 (p={mm[2]:.3f})"
        out.append(f"- M verdict: {verdict}; entropy ≥ 0.4 throughout 61-183: **{'yes' if ent_ok else 'NO'}**.")
    # R
    ent_r = None
    if "R" in curves:
        tr = curves["R"][0]
        ents = [(s, t["entropy"], t.get("ent_coeff")) for s, t in sorted(tr.items()) if t.get("entropy") is not None]
        if ents:
            last = ents[-30:]
            coeffs = [c for _, _, c in ents if c is not None]
            first_pos = next((s for s, _, c in ents if c is not None and c > 0), None)
            ent_r = dict(min=min(ents, key=lambda x: x[1]), mean_last30=st.mean(e for _, e, _ in last), coeff_mean_last30=st.mean(c for _, _, c in last if c is not None) if any(c is not None for _, _, c in last) else None,
                         coeff_max=max(coeffs) if coeffs else None, first_pos=first_pos, n=len(ents))
    out.append(f"**R** (reference v4 A: 30 = {s_(a30_acc)}, 60 = {s_(a60_acc)}, i.e. the post-30 decline of {s_((a60_acc - a30_acc) if None not in (a30_acc, a60_acc) else None)}): vpb_test 30/60/90/120/150/183 = " +
               " / ".join(s_(r_acc[s]) for s in (30, 60, 90, 120, 150, 183)) +
               (f"; entropy: min {ent_r['min'][1]:.3f} at step {ent_r['min'][0]}, mean of the last 30 steps {ent_r['mean_last30']:.3f}; coefficient first > 0 at step {ent_r['first_pos']}, "
                f"max {s_(ent_r['coeff_max'], 5)}, mean over the last 30 steps {s_(ent_r['coeff_mean_last30'], 5)}" if ent_r else "; entropy: MISSING"))
    if r_acc[30] is None or r_acc[60] is None:
        out.append("- R verdict: **INCOMPLETE** (30 or 60 row missing).")
    else:
        later = [(s, r_acc[s]) for s in (60, 90, 120, 150, 183) if r_acc[s] is not None]
        worst = min(later, key=lambda x: x[1])
        drop = worst[1] - r_acc[30]
        m60 = mcnemar(scores[r_tags[60]], scores[r_tags[30]])
        if r_acc[60] >= r_acc[30] - SE and all(a >= r_acc[30] - SE for _, a in later):
            v = f"**avoids the decline**: no later checkpoint falls more than one SE ({SE:.4f}) below step 30 (worst: step {worst[0]} {drop:+.4f}); 60 vs 30 McNemar p={m60[2]:.3f}"
        else:
            v = f"**declines**: step {worst[0]} is {drop:+.4f} vs step 30 (60 vs 30 McNemar p={m60[2]:.3f}); v4 A fell {s_((a60_acc - a30_acc) if None not in (a30_acc, a60_acc) else None)} over 30→60"
        out.append(f"- R verdict: {v}.")
    # best checkpoints + M vs R
    best_m = max(((s, a) for s, a in m_acc.items() if a is not None), key=lambda x: x[1], default=None)
    best_r = max(((s, a) for s, a in r_acc.items() if a is not None), key=lambda x: x[1], default=None)
    out.append(f"- best checkpoints: M = {f'step {best_m[0]} ({best_m[1]:.4f})' if best_m else 'MISSING'}; R = {f'step {best_r[0]} ({best_r[1]:.4f})' if best_r else 'MISSING'}"
               + (f"; incl. v4 B@60 {s_(b60_acc)} the best match checkpoint is {'B@60' if best_m and b60_acc is not None and b60_acc > best_m[1] else ('M@' + str(best_m[0]) if best_m else '?')}" if b60_acc is not None else ""))
    for s in (90, 120, 150, 183):
        mr = mcnemar(scores.get(m_tags[s]), scores.get(r_tags[s]))
        if mr:
            tag = " ← best M" if best_m and best_m[0] == s else ""
            tag += " ← best R" if best_r and best_r[0] == s else ""
            out.append(f"- M vs R at step {s}: {s_(m_acc[s])} vs {s_(r_acc[s])} ({m_acc[s] - r_acc[s]:+.4f}), McNemar {pstr(mr)}{tag}")
    # 4. fairshare + failures
    out += ["", "## 4. Fairshare (grp_vgupt140)", "", f"before: {FS_BEFORE}"]
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
    open(f"{REPO}/grpo_arms/FULL_REPORT.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
