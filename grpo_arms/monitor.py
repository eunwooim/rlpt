#!/usr/bin/env python
"""monitor.py — training monitor for a GRPO arm run (arm-1 v2 campaign).

Reads two sources under OUTDIR:
  * logs/verl_stdout_stderr.log  — verl console metrics ("step:N - key:value - ...")
      train steps : critic/score/mean, response_length/mean, response_length/clip_ratio,
                    actor/entropy, timing_s/step
      val steps   : val-aux/<source>/<metric>/mean@1 -> unweighted source-average of
                    score, answer, match, format, pun, gated, n_dup, n_segments, inv_frac
  * rollouts/<step>.jsonl        — per-sample reward components dumped by the reward fn
      -> per-step means of answer, match, format, pun, gated, n_dup, n_segments, inv_frac,
         match_precision, match_recall, plus within-step Pearson r(match, answer) and
         match | answer=1 / match | answer=0 strata (gate-2 evidence).

Writes OUTDIR/monitor.csv (one row per train step) and OUTDIR/monitor.png, prints a snapshot
table every --every steps, the gate-2 correlation of match/mean vs answer/mean over steps, and
RED-FLAG lines.  With --check it prints only the flag verdict (used by the in-job watchdog):
last line is "STOP" when an unmistakable degeneracy signal is confirmed, else "OK".

Every number here is copied from the log / rollout dumps — nothing is estimated.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st
import sys

STEP_RE = re.compile(r"step:(\d+) - (.*)$")
VAL_KEYS = ("score", "answer", "match", "format", "pun", "gated", "n_dup", "n_segments", "inv_frac")
ROLL_KEYS = ("score", "answer", "match", "format", "pun", "gated", "n_dup", "n_segments", "inv_frac",
             "match_precision", "match_recall")
TRAIN_KEYS = {"critic/score/mean": "score_mean", "response_length/mean": "resp_len",
              "response_length/clip_ratio": "clip_ratio", "actor/entropy": "entropy",
              "timing_s/step": "step_s", "actor/kl_loss": "kl_loss"}


def console_logs(run):
    """OUTDIR/logs/verl_stdout_stderr.log is block-buffered until the job exits; the Slurm logs
    grpo_arms/logs/arm-<jobid>.log (job ids from OUTDIR/chain.json, or any arm-*.log naming this OUTDIR) are live."""
    paths = [os.path.join(run, "logs", "verl_stdout_stderr.log")]
    logdir = os.path.join(os.path.dirname(os.path.abspath(run.rstrip("/"))), "..", "logs")
    ids = []
    cj = os.path.join(run, "chain.json")
    if os.path.exists(cj):
        try:
            c = json.load(open(cj)); ids = [c.get("train")] + list(c.get("resub", []))
        except Exception:  # noqa: BLE001
            ids = []
    for j in ids:
        if j:
            paths.append(os.path.normpath(os.path.join(logdir, f"arm-{j}.log")))
    return [p for p in paths if os.path.exists(p)]


def parse_console(paths):
    train, val = {}, {}
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        _parse_one(path, train, val)
    return train, val


def _parse_one(path, train, val):
    for line in open(path, errors="replace"):
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        kv = {}
        for tok in m.group(2).split(" - "):
            if ":" not in tok:
                continue
            k, _, v = tok.rpartition(":")
            try:
                kv[k] = float(v)
            except ValueError:
                pass
        # at test steps verl merges val-aux/* and the train metrics into ONE line: parse both from every line
        agg = {}
        for k, v in kv.items():
            mm = re.match(r"val-aux/(.+)/([^/]+)/mean@1$", k)
            if mm and mm.group(2) in VAL_KEYS:
                agg.setdefault(mm.group(2), []).append(v)
        if agg:
            val[step] = {f"val_{k}": st.mean(vs) for k, vs in agg.items()}
        row = {name: kv[k] for k, name in TRAIN_KEYS.items() if k in kv}
        if row:
            train.setdefault(step, {}).update(row)


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def parse_rollouts(rdir):
    out = {}
    for f in glob.glob(os.path.join(rdir, "*.jsonl")):
        try:
            step = int(os.path.basename(f).split(".")[0])
        except ValueError:
            continue
        rows = []
        for line in open(f, errors="replace"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not rows:
            continue
        d = {"n": len(rows)}
        for k in ROLL_KEYS:
            vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
            if vals:
                d[f"r_{k}"] = st.mean(vals)
        if all(k in rows[0] for k in ("match", "answer")):
            ms = [float(r["match"]) for r in rows]; an = [float(r["answer"]) for r in rows]
            d["r_corr_match_answer"] = pearson(ms, an)
            m1 = [m for m, a in zip(ms, an) if a >= 0.5]; m0 = [m for m, a in zip(ms, an) if a < 0.5]
            d["r_match_given_ans1"] = st.mean(m1) if m1 else float("nan")
            d["r_match_given_ans0"] = st.mean(m0) if m0 else float("nan")
        if "output" in rows[0]:
            d["r_words"] = st.mean(len(str(r["output"]).split()) for r in rows)
        out[step] = d
    return out


def build_table(run):
    train, val = parse_console(console_logs(run))
    roll = parse_rollouts(os.path.join(run, "rollouts"))
    steps = sorted(set(train) | set(roll))
    cols = ["step", "score_mean", "resp_len", "clip_ratio", "entropy", "kl_loss", "step_s",
            "n", "r_score", "r_answer", "r_match", "r_format", "r_pun", "r_gated", "r_n_dup",
            "r_n_segments", "r_inv_frac", "r_match_precision", "r_match_recall", "r_words",
            "r_corr_match_answer", "r_match_given_ans1", "r_match_given_ans0"] + [f"val_{k}" for k in VAL_KEYS]
    rows = []
    for s in steps:
        r = {"step": s}
        r.update(train.get(s, {})); r.update(roll.get(s, {})); r.update(val.get(s, {}))
        rows.append(r)
    val0 = val.get(0, {})
    return cols, rows, val0


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "   -  "
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def window_mean(rows, key, lo, hi):
    vals = [r[key] for r in rows if lo <= r["step"] <= hi and key in r and not math.isnan(r[key])]
    return st.mean(vals) if vals else float("nan")


def red_flags(rows):
    """Returns (flags, stop). stop only on unmistakable degeneracy confirmed over the last 20 steps."""
    flags, stop = [], False
    if not rows:
        return flags, stop
    last = rows[-1]["step"]
    if last < 10:
        return flags, stop
    w = max(10, min(20, last // 2))
    lo_a, hi_a = max(1, last - 2 * w + 1), last - w          # earlier window
    lo_b, hi_b = last - w + 1, last                          # latest window
    def cmp(key, name, thresh_rise, hard=None, direction=+1):
        nonlocal stop
        a, b = window_mean(rows, key, lo_a, hi_a), window_mean(rows, key, lo_b, hi_b)
        if math.isnan(a) or math.isnan(b):
            return
        rising = (b - a) * direction > thresh_rise
        if rising:
            flags.append(f"WARN {name}: {a:.3f} (steps {lo_a}-{hi_a}) -> {b:.3f} (steps {lo_b}-{hi_b})")
        if hard is not None and ((direction > 0 and b > hard) or (direction < 0 and b < hard)) and rising:
            flags.append(f"STOP-CRITERION {name}: latest-window mean {b:.3f} beyond hard limit {hard} and moving the wrong way")
            stop = True
    cmp("r_gated", "gated fraction rising", 0.05, hard=0.5)
    cmp("r_n_dup", "n_dup/mean rising", 0.3, hard=3.0)
    cmp("r_n_segments", "n_segments collapsing", 0.5, hard=1.5, direction=-1)
    # length-vs-score: warn only (normal early in a cold-started run), never auto-stop
    a_len, b_len = window_mean(rows, "resp_len", lo_a, hi_a), window_mean(rows, "resp_len", lo_b, hi_b)
    a_sc, b_sc = window_mean(rows, "score_mean", lo_a, hi_a), window_mean(rows, "score_mean", lo_b, hi_b)
    if not any(math.isnan(x) for x in (a_len, b_len, a_sc, b_sc)) and b_len > 1.15 * a_len and b_sc > a_sc:
        flags.append(f"WARN response_length/mean up {a_len:.0f}->{b_len:.0f} while score up {a_sc:.2f}->{b_sc:.2f} (steps {lo_a}-{hi_a} vs {lo_b}-{hi_b})")
    return flags, stop


def gate2(rows):
    xs = [(r["r_match"], r["r_answer"]) for r in rows if "r_match" in r and "r_answer" in r]
    if len(xs) < 3:
        return "gate-2: fewer than 3 steps with rollout dumps — no correlation yet"
    m, a = zip(*xs)
    r_all = pearson(list(m), list(a))
    n = len(xs); q = max(1, n // 4)
    parts = [f"gate-2: over {n} steps, Pearson r(match/mean, answer/mean) = {r_all:+.3f}"]
    for name, sl in (("first quarter", xs[:q]), ("last quarter", xs[-q:])):
        mm, aa = zip(*sl)
        parts.append(f"  {name}: match {st.mean(mm):.3f} answer {st.mean(aa):.3f} r={pearson(list(mm), list(aa)):+.3f}")
    s1 = [r["r_match_given_ans1"] for r in rows if "r_match_given_ans1" in r and not math.isnan(r["r_match_given_ans1"])]
    s0 = [r["r_match_given_ans0"] for r in rows if "r_match_given_ans0" in r and not math.isnan(r["r_match_given_ans0"])]
    if s1 and s0:
        parts.append(f"  within-step strata (all steps): mean match|answer=1 = {st.mean(s1):.3f}, match|answer=0 = {st.mean(s0):.3f}")
    # rule-based judgment (the numbers above are the evidence; this paragraph only reads them)
    fm, fa = st.mean([x for x, _ in xs[:q]]), st.mean([y for _, y in xs[:q]])
    lm, la = st.mean([x for x, _ in xs[-q:]]), st.mean([y for _, y in xs[-q:]])
    dm, da = lm - fm, la - fa
    gap = (st.mean(s1) - st.mean(s0)) if (s1 and s0) else float("nan")
    if abs(dm) < 0.03 and abs(da) < 0.03:
        verdict = "neither match nor answer moved materially (both < 0.03 between first and last quarter) — no information about independence yet"
    elif abs(dm) >= 0.03 and abs(da) < 0.03:
        verdict = f"match moved ({dm:+.3f}) while answer stayed flat ({da:+.3f}) — match carries a signal the answer term does not"
    elif abs(dm) < 0.03 and abs(da) >= 0.03:
        verdict = f"answer moved ({da:+.3f}) while match stayed flat ({dm:+.3f}) — the match term is not what training optimised"
    elif dm * da > 0:
        verdict = (f"match ({dm:+.3f}) and answer ({da:+.3f}) moved together; the across-step Pearson r = {r_all:+.3f}. "
                   + ("They track each other closely — match adds little beyond answer over the run." if r_all > 0.7 else
                      "They share direction but are not locked (r <= 0.7) — match still varies independently of answer."))
    else:
        verdict = (f"match ({dm:+.3f}) and answer ({da:+.3f}) moved in OPPOSITE directions (r = {r_all:+.3f}) — the arm-1 v1 hacking "
                   "signature; check response_length / gated / n_dup before trusting the match term")
    strata = ("" if math.isnan(gap) else
              f" Within steps, correct-answer rollouts score {gap:+.3f} higher on match than wrong-answer ones"
              + (" (match is answer-sensitive at the sample level)." if gap > 0.05 else " (match is essentially answer-blind at the sample level)."))
    verdict = verdict.rstrip(".")
    parts.append(f"  JUDGMENT (auto, rule-based): {verdict}.{strata}")
    return "\n".join(parts)


def plot(rows, png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[monitor] matplotlib unavailable ({exc}); skipping png")
        return
    def series(key):
        pts = [(r["step"], r[key]) for r in rows if key in r and r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return [p[0] for p in pts], [p[1] for p in pts]
    fig, ax = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    ax = ax.ravel()
    ax[0].plot(*series("score_mean"), label="critic/score/mean"); ax[0].plot(*series("r_score"), ":", label="rollout score mean"); ax[0].set_title("score"); ax[0].legend(fontsize=8)
    ax[1].plot(*series("resp_len"), label="response_length/mean"); ax[1].set_title("response length (tokens)"); a2 = ax[1].twinx(); a2.plot(*series("clip_ratio"), "r--", label="clip_ratio"); a2.set_ylim(0, 1); a2.legend(loc="upper left", fontsize=8); ax[1].legend(fontsize=8)
    for k in ("r_answer", "r_match", "r_format"):
        ax[2].plot(*series(k), label=k[2:])
    for k in ("val_answer", "val_match"):
        x, y = series(k); ax[2].plot(x, y, "o", label=k)
    ax[2].set_title("reward components (train rollouts; val = dots)"); ax[2].set_ylim(0, 1.05); ax[2].legend(fontsize=8)
    for k in ("r_pun", "r_gated", "r_n_dup"):
        ax[3].plot(*series(k), label=k[2:])
    ax[3].set_title("pun / gated / n_dup"); ax[3].legend(fontsize=8)
    ax[4].plot(*series("r_n_segments"), label="n_segments"); ax[4].plot(*series("r_inv_frac"), label="offset (inv_frac)"); ax[4].set_title("n_segments / offset"); ax[4].legend(fontsize=8)
    ax[5].plot(*series("r_corr_match_answer"), label="within-step r(match,answer)"); ax[5].plot(*series("r_match_given_ans1"), label="match|ans=1"); ax[5].plot(*series("r_match_given_ans0"), label="match|ans=0"); ax[5].axhline(0, color="k", lw=0.5); ax[5].set_title("gate-2: match vs answer"); ax[5].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
    print(f"[monitor] wrote {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--every", type=int, default=50, help="snapshot rows every N steps (plus last)")
    ap.add_argument("--check", action="store_true", help="watchdog mode: print flags and OK/STOP only")
    ap.add_argument("--no_png", action="store_true")
    a = ap.parse_args()
    cols, rows, val0 = build_table(a.run)
    if a.check:
        flags, stop = red_flags(rows)
        for f in flags:
            print(f)
        print(f"last_step={rows[-1]['step'] if rows else 0}")
        print("STOP" if stop else "OK")
        return
    csv_path = os.path.join(a.run, "monitor.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    print(f"[monitor] {len(rows)} steps -> {csv_path}")
    if val0:
        print("step-0 val (source-avg): " + "  ".join(f"{k[4:]}={fmt(v)}" for k, v in sorted(val0.items())))
    hdr = ["step", "score", "resp_len", "clip", "gated", "n_dup", "n_seg", "pun", "match", "answer", "format", "offset", "r(m,a)", "m|a1", "m|a0", "val_ans", "val_match", "step_s"]
    print(" ".join(f"{h:>8s}" for h in hdr))
    keys = ["score_mean", "resp_len", "clip_ratio", "r_gated", "r_n_dup", "r_n_segments", "r_pun", "r_match", "r_answer", "r_format", "r_inv_frac", "r_corr_match_answer", "r_match_given_ans1", "r_match_given_ans0", "val_answer", "val_match", "step_s"]
    for r in rows:
        if r["step"] % a.every == 0 or r is rows[-1] or r["step"] in (1, 20, 40, 60, 80, 100, 120, 140, 160, 180):
            print(f"{r['step']:>8d} " + " ".join(f"{fmt(r.get(k), 1 if k in ('resp_len', 'step_s') else 3):>8s}" for k in keys))
    print(gate2(rows))
    flags, stop = red_flags(rows)
    for f in flags:
        print("RED-FLAG " + f)
    if not flags:
        print("RED-FLAG none")
    if not a.no_png:
        plot(rows, os.path.join(a.run, "monitor.png"))


if __name__ == "__main__":
    main()
