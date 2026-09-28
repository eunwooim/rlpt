#!/usr/bin/env python
"""monitor_full.py — CPU-only monitor + orchestrator for the FULL runs on grp_vgupt140 (2026-09-18):
    M = bipartite match resumed in place from v4 B global_step_60 to 183 (runs/arm1_3b_matchv4_softgate; eval tags B_matchv4_softgate_stepN_2ktest)
    R = RLVR answer-only from cs25 with the adaptive entropy coefficient, 183 steps (runs/arm1_3b_rlvr_entropy; tags R_rlvr_entropy_stepN_2ktest)
Every 30 min (grpo_arms/monitor_full.sbatch, public CPU job):
  1. rewrites the "## Job table" and "## Live monitor" sections of grpo_arms/FULL_STATUS.md — per run, per logged step: real step, s/step,
     train-val gated acc, vpb_dev acc, match (M), answer, format, entropy, entropy coefficient (R), KL, resp_len, n_segments, mixed-group
     fraction, soft-gate fired (M), truncation (clip ratio). M's rows 0-60 come from the v4 B Slurm logs (chain.json prev_train_logs).
     Flags: entropy < 0.4; entropy > 1.8 after step 20; vpb_dev < 0.30; resp_len > 600; n_segments < 3.5; truncation > 5 %.
  2. submits one 1-hour htc gpu:1 vpb_test eval (2048 tokens, greedy) per new checkpoint (M: 90/120/150/183; R: 30/60/90/120/150/183),
     resubmitting a failed/timed-out eval up to 3 attempts; the eval job merges FSDP -> hf_step_N and generates.
  3. after a VERIFIED merge (config.json + every safetensors shard in the index + processor bundle) deletes checkpoints/global_step_N
     unless N in {60, 120, 183}; never the newest checkpoint.
  4. verifies M's resume once rollouts/61.jsonl exists (first resumed step == 61; its prompts absent from steps 1-60) -> status section.
  5. when both chains are finished and every eval generation exists (or is exhausted), runs report_full.py -> grpo_arms/FULL_REPORT.md and
     writes grpo_arms/full/ALL_DONE. It also submits report_full.sbatch with afterany on all training + eval ids once every eval is queued.
All numbers are copied from logs / dumps; state in grpo_arms/full/monitor_state.json; events in grpo_arms/full/monitor_events.log.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import shutil
import statistics as st
import subprocess
import sys
import time
from datetime import datetime

REPO = "/scratch/sghos104/rlpt"
PY = f"{REPO}/chunker/env/bin/python"
ACCOUNT = "grp_vgupt140"
STATUS = f"{REPO}/grpo_arms/FULL_STATUS.md"
STATE_PATH = f"{REPO}/grpo_arms/full/monitor_state.json"
EVENTS = f"{REPO}/grpo_arms/full/monitor_events.log"
ALL_DONE = f"{REPO}/grpo_arms/full/ALL_DONE"
EVAL_SET = f"{REPO}/grpo_arms/data/vpb_test.jsonl"
EVALS = f"{REPO}/grpo_arms/evals"
EXCLUDE = "scg001,scg002,scg003,scg004,sg048,sg049,sg050"   # MIG-slice nodes: 20/40 GB slices are too slow for a 1-h eval
TOTAL_STEPS = 183
KEEP_FSDP = {60, 120, 183}
BUNDLE = ["added_tokens.json", "chat_template.jinja", "merges.txt", "preprocessor_config.json", "special_tokens_map.json",
          "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json", "generation_config.json"]
RUNS = {
    "M": dict(dir=f"{REPO}/grpo_arms/runs/arm1_3b_matchv4_softgate", tagpfx="B_matchv4_softgate", eval_steps=[90, 120, 150, 183], match=True,
              label="M: match + soft gates, resumed from v4 B@60"),
    "R": dict(dir=f"{REPO}/grpo_arms/runs/arm1_3b_rlvr_entropy", tagpfx="R_rlvr_entropy", eval_steps=[30, 60, 90, 120, 150, 183], match=False,
              label="R: RLVR answer-only + adaptive entropy (target 0.6)"),
}
STEP_RE = re.compile(r"(?<![\w/])step:(\d+) - (.*)$")
PROG_RE = re.compile(r"Training Progress:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")
TRAIN_KEYS = {"critic/score/mean": "score", "response_length/mean": "resp_len", "response_length/clip_ratio": "clip",
              "actor/entropy": "entropy", "timing_s/step": "step_s", "actor/kl_loss": "kl", "actor/entropy_coeff": "ent_coeff",
              "actor/entropy_coeff_applied": "ent_coeff_applied", "actor/entropy_loss": "entropy_loss"}
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}


# ------------------------------------------------------------------------------------------------------------------ parsing
def chain(run_dir):
    cj = os.path.join(run_dir, "chain.json")
    if not os.path.exists(cj):
        return {}
    try:
        return json.load(open(cj))
    except Exception:  # noqa: BLE001
        return {}


def train_job_ids(c):
    return [j for j in [c.get("train")] + list(c.get("resub", [])) if j]


def logs_for(run_dir):
    c = chain(run_dir)
    paths = [os.path.join(run_dir, "logs", "verl_stdout_stderr.log")]
    for j in list(c.get("prev_train_logs", [])) + train_job_ids(c):
        paths.append(f"{REPO}/grpo_arms/logs/arm-{j}.log")
    return [p for p in paths if os.path.exists(p)]


_LOG_CACHE = {}   # path -> (size, (train, val, prog, jobinfo)); a log whose size has not changed is not re-read
_ROLL_CACHE = {}  # run_dir -> {step: row}


def parse_logs(paths):
    train, val, prog, jobinfo = {}, {}, 0, []
    for path in paths:
        size = os.path.getsize(path)
        if path in _LOG_CACHE and _LOG_CACHE[path][0] == size:
            t_, v_, p_, j_ = _LOG_CACHE[path][1]
        else:
            t_, v_, p_, j_ = _parse_one_log(path)
            _LOG_CACHE[path] = (size, (t_, v_, p_, j_))
        for k, v in t_.items():
            train.setdefault(k, {}).update(v)
        for k, v in v_.items():
            val.setdefault(k, {}).update(v)
        prog = max(prog, p_)
        jobinfo.extend(j_)
    return train, val, prog, jobinfo


def _parse_one_log(path):
    train, val, prog, jobinfo = {}, {}, 0, []
    for path in [path]:
        for line in open(path, errors="replace"):
            m = PROG_RE.search(line)
            if m:
                prog = max(prog, int(m.group(1)))
            if line.startswith("[arm1] host=") or line.startswith("[arm1] done=") or "TERM received" in line or "FATAL" in line \
                    or "Resuming from" in line or "Setting global step" in line or "[adaptive_entropy] resumed" in line:
                jobinfo.append(line.strip()[:200])
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
                if agg:
                    d["trainval_acc"] = st.mean(v.get("acc", v.get("answer", float("nan"))) for v in agg.values())
                val.setdefault(step, {}).update(d)
            row = {name: kv[k] for k, name in TRAIN_KEYS.items() if k in kv}
            if row:
                train.setdefault(step, {}).update(row)
    return train, val, prog, jobinfo


def parse_rollouts(run_dir, cache=None):
    out = _ROLL_CACHE.setdefault(run_dir, {}) if cache is None else cache
    for f in glob.glob(os.path.join(run_dir, "rollouts", "*.jsonl")):
        try:
            step = int(os.path.basename(f).split(".")[0])
        except ValueError:
            continue
        if step in out:
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
        mean = lambda k: st.mean(float(r.get(k, 0.0)) for r in rows)  # noqa: E731
        out[step] = {"mixed_frac": mixed / max(1, len(groups)), "n_groups": len(groups), "match": mean("match"), "answer": mean("answer"),
                     "format": mean("format"), "gated": mean("gated"), "n_seg": mean("n_segments"), "soft_gated": mean("soft_gated")}
    return out


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def ckpt_steps(run_dir):
    return sorted(int(d.rsplit("_", 1)[1]) for d in glob.glob(os.path.join(run_dir, "checkpoints", "global_step_*"))
                  if os.path.isdir(os.path.join(d, "actor")))


def latest_step(run_dir):
    p = os.path.join(run_dir, "checkpoints", "latest_checkpointed_iteration.txt")
    try:
        return int(open(p).read().strip())
    except Exception:  # noqa: BLE001
        return 0


# ------------------------------------------------------------------------------------------------------------------ slurm
def sacct_state(jobid):
    try:
        out = subprocess.run(["sacct", "-j", str(jobid), "-o", "State", "-P", "-n", "-X"], capture_output=True, text=True, timeout=60).stdout
        s = out.strip().splitlines()
        return s[0].split()[0] if s else "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def sbatch(args):
    r = subprocess.run(["sbatch", "--parsable", f"--account={ACCOUNT}"] + args, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip().split(";")[0]


def event(state, msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M} {msg}"
    state.setdefault("events", []).append(line)
    with open(EVENTS, "a") as f:
        f.write(line + "\n")
    print("[monitor_full] " + msg, flush=True)


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH))
        except Exception:  # noqa: BLE001
            pass
    return {"evals": {}, "events": [], "pruned": [], "resume_verified": False, "report_job": None}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    json.dump(state, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE_PATH)


# ------------------------------------------------------------------------------------------------------------------ evals + pruning
def gen_path(cfg, step):
    return f"{EVALS}/vpb_gen_{cfg['tagpfx']}_step{step}_2ktest.jsonl"


def gen_ok(cfg, step):
    p = gen_path(cfg, step)
    return os.path.exists(p) and os.path.getsize(p) > 0


def submit_eval(name, cfg, step):
    return sbatch(["--partition=htc", "--qos=public", "--gres=gpu:1", "--cpus-per-task=4", "--mem=64G", "--time=01:00:00", "--no-requeue",
                   f"--exclude={EXCLUDE}", f"--job-name=full_{name}_s{step}", "--mail-type=FAIL", "--mail-user=sghos104@asu.edu",
                   f"--export=ALL,RUN={cfg['dir']},DO_CS25=0,DO_BASE=0,STEP_LIST={step},DO_SCORE=0,MAXTOK=2048,TAGSFX=_2ktest,"
                   f"TAGPFX={cfg['tagpfx']},EVAL_SET={EVAL_SET}", f"{REPO}/grpo_arms/eval_arm1v2.sbatch"])


def manage_evals(state):
    for name, cfg in RUNS.items():
        if not os.path.isdir(cfg["dir"]):
            continue
        have, latest = set(ckpt_steps(cfg["dir"])), latest_step(cfg["dir"])
        for step in cfg["eval_steps"]:
            key = f"{name}:{step}"
            e = state["evals"].setdefault(key, {"jobs": [], "done": False})
            if e["done"]:
                continue
            if gen_ok(cfg, step):
                e["done"] = True
                event(state, f"eval {key}: generation present ({os.path.basename(gen_path(cfg, step))})")
                continue
            merged = os.path.exists(os.path.join(cfg["dir"], f"hf_step_{step}", "config.json"))
            if not ((step in have and step <= latest) or merged):
                continue
            if e["jobs"]:
                stt = sacct_state(e["jobs"][-1])
                if stt not in TERMINAL:
                    continue  # pending/running
                if len(e["jobs"]) >= 3:
                    if not e.get("exhausted"):
                        e["exhausted"] = True
                        event(state, f"eval {key}: 3 attempts, last state {stt}; giving up (see grpo_arms/logs/eval_arm1v2-{e['jobs'][-1]}.log)")
                    continue
                event(state, f"eval {key}: job {e['jobs'][-1]} ended {stt} without a generation -> resubmitting")
            try:
                j = submit_eval(name, cfg, step)
                e["jobs"].append(j)
                event(state, f"eval {key}: submitted job {j} (htc, gpu:1, 1 h, vpb_test 2048 tok)")
            except Exception as exc:  # noqa: BLE001
                event(state, f"eval {key}: submit FAILED: {exc}")


def merge_verified(run_dir, step):
    hf = os.path.join(run_dir, f"hf_step_{step}")
    idx = os.path.join(hf, "model.safetensors.index.json")
    if not (os.path.exists(os.path.join(hf, "config.json")) and os.path.exists(idx)):
        return False, "no config/index"
    try:
        shards = sorted(set(json.load(open(idx))["weight_map"].values()))
    except Exception as exc:  # noqa: BLE001
        return False, f"bad index: {exc}"
    missing = [s for s in shards if not (os.path.exists(os.path.join(hf, s)) and os.path.getsize(os.path.join(hf, s)) > 0)]
    missing += [b for b in BUNDLE if not os.path.exists(os.path.join(hf, b))]
    return (not missing), (f"{len(shards)} shards + bundle ok" if not missing else f"missing {missing}")


def prune(state):
    for name, cfg in RUNS.items():
        if not os.path.isdir(cfg["dir"]):
            continue
        steps = ckpt_steps(cfg["dir"])
        if not steps:
            continue
        newest = max(steps)
        for n in steps:
            if n in KEEP_FSDP or n == newest:
                continue
            ok, why = merge_verified(cfg["dir"], n)
            if not ok:
                continue
            target = os.path.join(cfg["dir"], "checkpoints", f"global_step_{n}")
            try:
                size = subprocess.run(["du", "-sh", target], capture_output=True, text=True, timeout=300).stdout.split()[0]
            except Exception:  # noqa: BLE001
                size = "?"
            shutil.rmtree(target, ignore_errors=True)
            state["pruned"].append(f"{name}:{n}")
            event(state, f"pruned FSDP checkpoint {name} global_step_{n} ({size}; merge verified: {why}; newest={newest}, keep={sorted(KEEP_FSDP)})")


# ------------------------------------------------------------------------------------------------------------------ resume check (M)
def verify_resume(state):
    if state.get("resume_verified"):
        return
    cfg = RUNS["M"]
    r61 = os.path.join(cfg["dir"], "rollouts", "61.jsonl")
    if not os.path.exists(r61):
        return
    c = chain(cfg["dir"])
    lines = []
    first_step, resume_lines = None, []
    for j in train_job_ids(c):
        p = f"{REPO}/grpo_arms/logs/arm-{j}.log"
        if not os.path.exists(p):
            continue
        for line in open(p, errors="replace"):
            if "Resuming from" in line or "Setting global step" in line:
                resume_lines.append(f"job {j}: {line.strip()[:160]}")
            m = STEP_RE.search(line)
            if m and first_step is None and ("actor/entropy" in line or "timing_s/step" in line):
                first_step = int(m.group(1))   # first TRAINING metric line (the resume-time validation is also logged as step:60)
        if first_step is not None:
            break
    def prompts(step):
        p = os.path.join(cfg["dir"], "rollouts", f"{step}.jsonl")
        s = set()
        if os.path.exists(p):
            for line in open(p, errors="replace"):
                try:
                    s.add(json.loads(line)["input"])
                except Exception:  # noqa: BLE001
                    pass
        return s
    p61, p1 = prompts(61), prompts(1)
    prev = set()
    for s_ in range(1, 61):
        prev |= prompts(s_)
    overlap = len(p61 & prev)
    # the prompt text is the question only (no image id) and the train subset repeats question texts across images ("Find x." on 75 rows),
    # so a small text overlap is expected even for a perfectly continued dataloader; the within-run baseline is step 60 vs steps 1-59.
    base_prev = set()
    for s_ in range(1, 60):
        base_prev |= prompts(s_)
    baseline = len(prompts(60) & base_prev)
    same_as_step1 = len(p61 & p1)
    ok = first_step == 61 and len(p61) > 0 and same_as_step1 < len(p61) // 2 and overlap <= max(baseline, 4)
    lines += [f"## Resume verification (M, {datetime.now():%Y-%m-%d %H:%M})", ""]
    lines += [f"- {l}" for l in resume_lines[:4]]
    lines += [f"- first TRAINING `step:N` metric line in the resumed job: **{first_step}** (expected 61; verl also logs the resume-time validation as step:60)",
              f"- rollouts/61.jsonl prompt groups: {len(p61)}; identical to step 1's batch: {same_as_step1}/{len(p61)} (a restart would give {len(p61)}/{len(p61)})",
              f"- text overlap of step 61 with the union of steps 1-60 ({len(prev)} distinct prompts): **{overlap}** — within-run baseline (v4 B step 60 vs 1-59): {baseline}; "
              f"the train subset has 171 question texts that occur on >1 row (different images), e.g. 'Find x.' ×75",
              f"- **{'PASS' if ok else 'FAIL'}** — the dataloader {'continued from batch 61' if ok else 'did NOT continue cleanly; see above'}", ""]
    upsert_section("## Resume verification (M", "\n".join(lines))
    state["resume_verified"] = True
    event(state, f"resume verification M: first_step={first_step} overlap={overlap} -> {'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------------------------------------------------------------------ rendering
def render_run(name, cfg):
    run = cfg["dir"]
    lines = [f"### {name} — {cfg['label']} — {run}"]
    if not os.path.isdir(run):
        lines.append("(no run dir yet)")
        return lines, []
    train, val, prog, jobinfo = parse_logs(logs_for(run))
    roll = parse_rollouts(run)
    steps = sorted(set(train) | set(roll))
    last = steps[-1] if steps else 0
    lines.append(f"progress bar: {prog}/{TOTAL_STEPS} · last metric step: {last} · FSDP checkpoints: {ckpt_steps(run)} (latest={latest_step(run)}) · "
                 f"merged: {sorted(int(d.rsplit('_', 1)[1]) for d in glob.glob(os.path.join(run, 'hf_step_*')))} · "
                 f"DONE={os.path.exists(os.path.join(run, 'DONE'))} · STOP_WATCHDOG={os.path.exists(os.path.join(run, 'STOP_WATCHDOG'))}")
    for j in jobinfo[-5:]:
        lines.append(f"  {j}")
    v0 = val.get(0, {})
    if v0:
        lines.append(f"step-0 val: train-val acc {fmt(v0.get('trainval_acc'))}, vpb_dev acc {fmt(v0.get('vpb_dev_acc'))}")
    hdr = "| step | s/step | trainval acc | vpb_dev acc | match | answer | format | entropy | ent coeff | KL | resp_len | n_seg | mixed grp frac | soft-gate fired | trunc (clip) |"
    lines += [hdr, "|---" * (hdr.count("|") - 1) + "|"]
    flags, show = [], set(s for s in steps if s % 10 == 0 or s == last or s in (1, 5, 61, 183))
    for s in steps:
        t, v, r = train.get(s, {}), val.get(s, {}), roll.get(s, {})
        if t.get("clip") is not None and t["clip"] > 0.05:
            flags.append(f"step {s}: truncation clip_ratio={t['clip']:.3f} > 5%")
        if t.get("resp_len") is not None and t["resp_len"] > 600:
            flags.append(f"step {s}: response_length/mean={t['resp_len']:.0f} > 600")
        if t.get("entropy") is not None and t["entropy"] < 0.4:
            flags.append(f"step {s}: entropy={t['entropy']:.3f} < 0.4")
        if t.get("entropy") is not None and s > 20 and t["entropy"] > 1.8:
            flags.append(f"step {s}: entropy={t['entropy']:.3f} > 1.8 after step 20")
        if v.get("vpb_dev_acc") is not None and v["vpb_dev_acc"] < 0.30:
            flags.append(f"step {s}: vpb_dev={v['vpb_dev_acc']:.4f} < 0.30")
        if r.get("n_seg") is not None and r["n_seg"] < 3.5:
            flags.append(f"step {s}: n_segments={r['n_seg']:.2f} < 3.5")
        if s not in show:
            continue
        lines.append(f"| {s} | {fmt(t.get('step_s'), 0)} | {fmt(v.get('trainval_acc'))} | {fmt(v.get('vpb_dev_acc'))} | "
                     f"{fmt(r.get('match')) if cfg['match'] else 'n/a'} | {fmt(r.get('answer'))} | {fmt(r.get('format'))} | {fmt(t.get('entropy'))} | "
                     f"{fmt(t.get('ent_coeff'), 5) if not cfg['match'] else 'off'} | {fmt(t.get('kl'), 4)} | {fmt(t.get('resp_len'), 0)} | {fmt(r.get('n_seg'), 1)} | "
                     f"{fmt(r.get('mixed_frac'))} | {fmt(r.get('soft_gated')) if cfg['match'] else 'n/a'} | {fmt(t.get('clip'))} |")
    seen, uniq = set(), []
    for f in flags:
        if f not in seen:
            seen.add(f); uniq.append(f)
    lines.append(("**FLAGS:** " + " · ".join(uniq[-10:])) if uniq else "flags: none")
    return lines, uniq


def render_jobs(state):
    out = ["## Job table", "", "| run | job | role | state | notes |", "|---|---|---|---|---|"]
    for name, cfg in RUNS.items():
        c = chain(cfg["dir"])
        if not c:
            out.append(f"| {name} | – | training | not launched | {cfg['label']} |")
            continue
        for i, j in enumerate(train_job_ids(c)):
            out.append(f"| {name} | {j} | {'training' if i == 0 else f'resubmit {i} (afternotok)'} | {sacct_state(j)} | wall {c.get('wall')}, launched {c.get('launched', '?')} |")
        for step in cfg["eval_steps"]:
            e = state["evals"].get(f"{name}:{step}", {})
            jobs = e.get("jobs", [])
            stt = "generation present" if e.get("done") else (sacct_state(jobs[-1]) if jobs else "waiting for checkpoint")
            out.append(f"| {name} | {', '.join(jobs) if jobs else '–'} | eval step {step} | {stt} | vpb_gen_{cfg['tagpfx']}_step{step}_2ktest.jsonl |")
    smoke = f"{REPO}/grpo_arms/full/smoke_jobid.txt"
    if os.path.exists(smoke):
        sj = open(smoke).read().strip()
        out.append(f"| smoke | {sj} | adaptive-entropy smoke (htc, 1 A100, 3 steps) | {sacct_state(sj)} | grpo_arms/full/smoke_entropy.sbatch |")
    if state.get("report_job"):
        out.append(f"| report | {state['report_job']} | FULL_REPORT.md post-job (afterany) | {sacct_state(state['report_job'])} | grpo_arms/report_full.sbatch |")
    out.append(f"| monitor | {os.environ.get('SLURM_JOB_ID', 'login')} | this monitor (public CPU) | running | every 30 min |")
    return "\n".join(out) + "\n"


def render_live(state):
    out = [f"## Live monitor (monitor_full.py, {datetime.now():%Y-%m-%d %H:%M}, job {os.environ.get('SLURM_JOB_ID', 'login')})", ""]
    for name, cfg in RUNS.items():
        lines, _ = render_run(name, cfg)
        out.extend(lines); out.append("")
    ev = state.get("events", [])[-25:]
    out += ["events (last 25; full log grpo_arms/full/monitor_events.log):", ""] + [f"- {e}" for e in ev] + [""]
    return "\n".join(out)


APPEND_LOG = "## Training job log (appended by run_arm.sbatch; keep this the LAST section)"


def upsert_section(title_prefix, body):
    """Replace the section whose header LINE starts with title_prefix (anchored at line start, so prose mentioning a header is never matched);
    a new section is inserted before the append-only training-job-log section (which stays last) or appended if that is absent."""
    s = open(STATUS).read() if os.path.exists(STATUS) else "# FULL STATUS\n"
    body = body.rstrip("\n") + "\n"
    m = re.search(r"^" + re.escape(title_prefix), s, flags=re.M)
    if m is None:
        anchor = re.search(r"^" + re.escape(APPEND_LOG), s, flags=re.M)
        if anchor:
            s = s[:anchor.start()] + body + "\n" + s[anchor.start():]
        else:
            s = s.rstrip("\n") + "\n\n" + body
    else:
        start = m.start()
        nxt = re.search(r"^## ", s[m.end():], flags=re.M)
        end = m.end() + nxt.start() if nxt else len(s)
        s = s[:start] + body + ("\n" if nxt else "") + s[end:]
    tmp = STATUS + ".tmp"
    open(tmp, "w").write(s)
    os.replace(tmp, STATUS)


# ------------------------------------------------------------------------------------------------------------------ completion
def chain_finished(cfg):
    if not os.path.isdir(cfg["dir"]):
        return False
    if os.path.exists(os.path.join(cfg["dir"], "DONE")):
        return True
    c = chain(cfg["dir"])
    ids = train_job_ids(c)
    return bool(ids) and all(sacct_state(j) in TERMINAL for j in ids)


def maybe_report(state):
    all_submitted = all(state["evals"].get(f"{n}:{s}", {}).get("jobs") or state["evals"].get(f"{n}:{s}", {}).get("done")
                        for n, c in RUNS.items() for s in c["eval_steps"])
    if all_submitted and not state.get("report_job"):
        ids = [j for c in RUNS.values() for j in train_job_ids(chain(c["dir"]))]
        ids += [j for e in state["evals"].values() for j in e.get("jobs", [])]
        try:
            j = sbatch(["--partition=htc", "--qos=public", "--time=00:30:00", f"--dependency=afterany:{':'.join(ids)}", "--no-requeue",
                        f"{REPO}/grpo_arms/report_full.sbatch"])
            state["report_job"] = j
            event(state, f"report post-job {j} submitted (afterany on {len(ids)} training + eval jobs)")
        except Exception as exc:  # noqa: BLE001
            event(state, f"report post-job submit FAILED: {exc}")
    finished = all(chain_finished(c) for c in RUNS.values())
    evals_settled = all(e.get("done") or e.get("exhausted") for n, c in RUNS.items() for e in [state["evals"].get(f"{n}:{s}", {}) for s in c["eval_steps"]])
    if finished and evals_settled and not os.path.exists(ALL_DONE):
        r = subprocess.run([PY, f"{REPO}/grpo_arms/report_full.py"], capture_output=True, text=True, timeout=1800)
        event(state, f"both chains finished and evals settled -> report_full.py rc={r.returncode} {r.stderr.strip()[-200:] if r.returncode else ''}")
        if r.returncode == 0:
            open(ALL_DONE, "w").write(f"{datetime.now()}\n")
    return os.path.exists(ALL_DONE)


def one_pass(do_actions=True):
    state = load_state()
    if do_actions:
        try:
            manage_evals(state)
        except Exception as exc:  # noqa: BLE001
            event(state, f"manage_evals error: {exc}")
        try:
            prune(state)
        except Exception as exc:  # noqa: BLE001
            event(state, f"prune error: {exc}")
        try:
            verify_resume(state)
        except Exception as exc:  # noqa: BLE001
            event(state, f"verify_resume error: {exc}")
    upsert_section("## Job table", render_jobs(state))
    upsert_section("## Live monitor", render_live(state))
    done = False
    if do_actions:
        try:
            done = maybe_report(state)
        except Exception as exc:  # noqa: BLE001
            event(state, f"maybe_report error: {exc}")
    save_state(state)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--budget_s", type=int, default=0, help="stop looping after this many seconds (self-chaining sbatch)")
    ap.add_argument("--no_actions", action="store_true", help="render only (no sbatch / rm)")
    a = ap.parse_args()
    t0 = time.time()
    while True:
        t1 = time.time()
        done = one_pass(do_actions=not a.no_actions)
        print(f"[monitor_full] pass finished in {time.time() - t1:.0f}s at {datetime.now():%H:%M}", flush=True)
        if not a.loop or done:
            print("[monitor_full] " + ("ALL_DONE" if done else "single pass done"), flush=True)
            break
        if a.budget_s and time.time() - t0 + a.interval > a.budget_s:
            print("[monitor_full] time budget reached; exiting for the self-chained follow-up", flush=True)
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
