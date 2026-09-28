#!/usr/bin/env python
"""monitor_v4.py — CPU-only monitor for the v4 pair (A answer-only control, B match + soft gates); derived from monitor_v3.py. Run under chunker/env on the login node:
    python grpo_arms/monitor_v3.py            # one pass: rewrite the "Live monitor" section of grpo_arms/V3_STATUS.md
    python grpo_arms/monitor_v3.py --loop     # every 30 min

Per run, per logged step it reports: real step (from the "Training Progress: N/90" tqdm bar, not the `step:` metric prefix),
s/step, train-val answer acc (source-avg over the VisualPRM val sources), vpb_dev acc, mean response length, mean match, mean answer,
actor entropy, fraction of prompt groups with mixed `acc` in that step (from rollouts/<step>.jsonl). For v3b it adds the number of
generation batches the step needed, the number of prompt groups trained on, and whether the partial-batch fallback fired.
Flags: truncation (clip_ratio) > 0, mean response length > 800, mixed-group fraction < 0.1 for three consecutive logged steps.
All numbers are copied from the logs / dumps.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import statistics as st
import time
from datetime import datetime

REPO = "/scratch/sghos104/rlpt"
RUNS = {
    "A_ctrl_v4": f"{REPO}/grpo_arms/runs/arm1_3b_ctrl_v4",
    "B_matchv4_softgate": f"{REPO}/grpo_arms/runs/arm1_3b_matchv4_softgate",
}
STATUS = f"{REPO}/grpo_arms/V4_STATUS.md"
TOTAL_STEPS = 60
STEP_RE = re.compile(r"(?<![\w/])step:(\d+) - (.*)$")
PROG_RE = re.compile(r"Training Progress:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")
FILTER_RE = re.compile(r"\[filter_groups\] step=(\d+) gen_batches=(\d+) mixed=(\d+)/(\d+) trained=(\d+) partial=(\d)")
TRAIN_KEYS = {"critic/score/mean": "score", "response_length/mean": "resp_len", "response_length/clip_ratio": "clip",
              "actor/entropy": "entropy", "timing_s/step": "step_s", "actor/kl_loss": "kl", "filter/num_gen_batches": "gen_batches",
              "filter/num_prompts_trained": "trained", "filter/partial": "partial", "filter/mixed_frac": "mixed_frac_gen"}


def logs_for(run):
    paths = [os.path.join(run, "logs", "verl_stdout_stderr.log")]
    cj = os.path.join(run, "chain.json")
    if os.path.exists(cj):
        try:
            c = json.load(open(cj))
            for j in [c.get("train")] + list(c.get("resub", [])):
                if j:
                    paths.append(f"{REPO}/grpo_arms/logs/arm-{j}.log")
        except Exception:  # noqa: BLE001
            pass
    return [p for p in paths if os.path.exists(p)]


def parse_logs(paths):
    train, val, prog, filt, jobinfo = {}, {}, 0, {}, []
    for path in paths:
        for line in open(path, errors="replace"):
            m = PROG_RE.search(line)
            if m:
                prog = max(prog, int(m.group(1)))
            m = FILTER_RE.search(line)
            if m:
                filt[int(m.group(1))] = dict(gen_batches=int(m.group(2)), mixed=int(m.group(3)), generated=int(m.group(4)),
                                             trained=int(m.group(5)), partial=int(m.group(6)))
            if line.startswith("[arm1] host=") or line.startswith("[arm1] done=") or "TERM received" in line or "FATAL" in line:
                jobinfo.append(line.strip()[:160])
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
            agg = collections.defaultdict(dict)
            for k, v in kv.items():
                mm = re.match(r"val-aux/(.+)/(answer|acc|match|format|gated|score)/mean@1$", k)
                if mm:
                    agg[mm.group(1)][mm.group(2)] = v
            if agg:
                vpb = agg.pop("vpb_dev", {})
                d = {}
                if vpb:
                    d["vpb_dev_acc"] = vpb.get("acc", vpb.get("answer"))
                    d["vpb_dev_gated"] = vpb.get("gated")
                if agg:
                    d["trainval_acc"] = st.mean(v.get("acc", v.get("answer", float("nan"))) for v in agg.values())
                    d["trainval_match"] = st.mean(v.get("match", float("nan")) for v in agg.values())
                val.setdefault(step, {}).update(d)
            row = {name: kv[k] for k, name in TRAIN_KEYS.items() if k in kv}
            if row:
                train.setdefault(step, {}).update(row)
    return train, val, prog, filt, jobinfo


def parse_rollouts(run):
    out = {}
    for f in glob.glob(os.path.join(run, "rollouts", "*.jsonl")):
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
        key = "acc" if "acc" in rows[0] else "answer"
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r.get("input")].append(float(r.get(key, 0.0)))
        mixed = sum(1 for g in groups.values() if len(g) > 1 and max(g) != min(g))
        out[step] = {
            "mixed_frac": mixed / max(1, len(groups)), "n_groups": len(groups),
            "match": st.mean(float(r.get("match", 0.0)) for r in rows),
            "answer": st.mean(float(r.get("answer", 0.0)) for r in rows),
            "gated": st.mean(float(r.get("gated", 0.0)) for r in rows),
            "n_seg": st.mean(float(r.get("n_segments", 0.0)) for r in rows),
            "soft_gated": st.mean(float(r.get("soft_gated", 0.0)) for r in rows),
        }
    return out


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render_run(name, run):
    lines = [f"### {name} — {run}"]
    if not os.path.isdir(run):
        lines.append("(no run dir yet)")
        return lines, []
    train, val, prog, filt, jobinfo = parse_logs(logs_for(run))
    roll = parse_rollouts(run)
    steps = sorted(set(train) | set(roll))
    last = steps[-1] if steps else 0
    ckpts = sorted(int(d.rsplit("_", 1)[1]) for d in glob.glob(os.path.join(run, "checkpoints", "global_step_*"))
                   if os.path.isdir(os.path.join(d, "actor")))
    lines.append(f"progress bar: {prog}/{TOTAL_STEPS} · last metric step: {last} · checkpoints with actor/: {ckpts} · "
                 f"DONE={os.path.exists(os.path.join(run, 'DONE'))} · STOP_WATCHDOG={os.path.exists(os.path.join(run, 'STOP_WATCHDOG'))}")
    for j in jobinfo[-4:]:
        lines.append(f"  {j}")
    v0 = val.get(0, {})
    if v0:
        lines.append(f"step-0 val: train-val acc {fmt(v0.get('trainval_acc'))}, vpb_dev acc {fmt(v0.get('vpb_dev_acc'))}")
    hdr = "| step | s/step | trainval acc | vpb_dev acc | resp_len | clip | match | answer | entropy | KL | mixed grp frac | n_seg |"
    if name.startswith("B_"):
        hdr += " soft-gate fired |"
    lines.append(hdr)
    lines.append("|---" * (hdr.count("|") - 1) + "|")
    flags = []
    mixed_low_run = 0
    show = [s for s in steps if s % 10 == 0 or s == last or s in (1, 5)]
    for s in steps:
        t, v, r = train.get(s, {}), val.get(s, {}), roll.get(s, {})
        if t.get("clip", 0) and t["clip"] > 0.05:
            flags.append(f"step {s}: truncation clip_ratio={t['clip']:.3f} > 5%")
        if t.get("resp_len", 0) and t["resp_len"] > 700:
            flags.append(f"step {s}: response_length/mean={t['resp_len']:.0f} > 700")
        if t.get("entropy") is not None and t["entropy"] < 0.4:
            flags.append(f"step {s}: entropy={t['entropy']:.3f} < 0.4")
        if v.get("vpb_dev_acc") is not None and v["vpb_dev_acc"] < 0.27:
            flags.append(f"step {s}: vpb_dev={v['vpb_dev_acc']:.4f} < 0.27")
        if s not in show:
            continue
        row = (f"| {s} | {fmt(t.get('step_s'), 0)} | {fmt(v.get('trainval_acc'))} | {fmt(v.get('vpb_dev_acc'))} | {fmt(t.get('resp_len'), 0)} | "
               f"{fmt(t.get('clip'))} | {fmt(r.get('match'))} | {fmt(r.get('answer'))} | {fmt(t.get('entropy'))} | {fmt(t.get('kl'), 4)} | {fmt(r.get('mixed_frac'))} | {fmt(r.get('n_seg'), 1)} |")
        if name.startswith("B_"):
            row += f" {fmt(r.get('soft_gated'))} |"
        lines.append(row)
    if False and filt:
        gb = [f["gen_batches"] for f in filt.values()]; tr = [f["trained"] for f in filt.values()]; pa = sum(f["partial"] for f in filt.values())
        lines.append(f"filter summary: {len(filt)} steps; gen batches/step mean {st.mean(gb):.2f} (max {max(gb)}); groups trained/step mean "
                     f"{st.mean(tr):.1f} (min {min(tr)}); partial-batch fallback fired {pa}x")
    # dedupe flags, keep last 8
    seen, uniq = set(), []
    for f in flags:
        if f not in seen:
            seen.add(f); uniq.append(f)
    if uniq:
        lines.append("**FLAGS:** " + " · ".join(uniq[-8:]))
    else:
        lines.append("flags: none")
    return lines, uniq


def render():
    out = [f"## Live monitor (monitor_v4.py, {datetime.now().strftime('%Y-%m-%d %H:%M')})", ""]
    for name, run in RUNS.items():
        lines, _ = render_run(name, run)
        out.extend(lines); out.append("")
    return "\n".join(out)


def write_status(section):
    s = open(STATUS).read() if os.path.exists(STATUS) else "# V3 STATUS\n"
    start = s.find("## Live monitor")
    if start < 0:
        s = s.rstrip("\n") + "\n\n" + section + "\n"
    else:
        end = s.find("\n## ", start + 5)
        s = s[:start] + section + ("\n" + s[end:] if end >= 0 else "\n")
    open(STATUS, "w").write(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    while True:
        sec = render()
        write_status(sec)
        if a.print or not a.loop:
            print(sec)
        if not a.loop:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
