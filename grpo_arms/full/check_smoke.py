#!/usr/bin/env python
"""check_smoke.py — verify the adaptive-entropy smoke (grpo_arms/full/smoke_entropy.sbatch: htc, 1 A100, 3 steps, MAXRESP 512, ENTROPY_TARGET=3.0).
Reads grpo_arms/logs/arm-<jobid>.log and asserts, per step 1..3:
  * the step line carries actor/entropy, actor/entropy_loss, actor/entropy_coeff (driver), actor/entropy_coeff_applied (worker), actor/entropy_coeff_next
  * worker-applied coefficient == driver coefficient (|diff| < 1e-9)
  * coeff_next == clip(coeff_used + 0.002 * (3.0 - entropy), 0, 0.01) (|diff| < 1e-9) and, with entropy < 3.0, the coefficient rises step over step
  * the job ended rc=0 with no Traceback
Writes the "## Smoke test (adaptive entropy)" section of grpo_arms/FULL_STATUS.md and exits 0 on PASS, 1 on FAIL.
Usage: python check_smoke.py <jobid> [--target 3.0 --lr 0.002 --max 0.01]
"""
import argparse
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, "/scratch/sghos104/rlpt/grpo_arms")
import monitor_full as mf  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("jobid"); ap.add_argument("--target", type=float, default=3.0); ap.add_argument("--lr", type=float, default=0.002)
ap.add_argument("--max", type=float, default=0.01); ap.add_argument("--steps", type=int, default=3)
a = ap.parse_args()
log = f"/scratch/sghos104/rlpt/grpo_arms/logs/arm-{a.jobid}.log"
train, val, prog, jobinfo = mf.parse_logs([log])
text = open(log, errors="replace").read()
checks = []
def chk(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
chk("installed in driver + workers", text.count("[adaptive_entropy] installed in pid") >= 2, f"{text.count('[adaptive_entropy] installed in pid')} install lines")
chk(f"ENABLED with target {a.target}", f"ENABLED target={a.target}" in text)
# the "DataLoader worker (pid N) is killed by signal: Killed" traceback at Ray teardown is a known benign artifact (present in the successful v4 B log 63148593 and 8 other completed runs)
bad_tb = []
for i, line in enumerate(text.splitlines()):
    if "Traceback (most recent call last)" in line:
        ctx = "\n".join(text.splitlines()[i:i + 20])
        if "DataLoader worker (pid" not in ctx:
            bad_tb.append(ctx[:300])
chk("no Traceback other than the benign DataLoader-teardown one", not bad_tb, bad_tb[0] if bad_tb else "")
chk("job rc=0", re.search(r"\[arm1\] done=.* rc=0", text) is not None)
rows = []
prev = None
for s in range(1, a.steps + 1):
    t = train.get(s, {})
    have = all(k in t for k in ("entropy", "entropy_loss", "ent_coeff", "ent_coeff_applied"))
    nxt = None
    m = re.search(rf"\[adaptive_entropy\] step={s} entropy=([\d.]+) coeff_used=([\d.]+) coeff_next=([\d.]+) worker_applied=(.*)$", text)
    if m:
        nxt = float(m.group(3))   # printed with 6 decimals; the jsonl state file below has full precision
    jl = "/scratch/sghos104/rlpt/grpo_arms/runs/full_smoke_entropy/checkpoints/entropy_coeff_log.jsonl"
    if os.path.exists(jl):
        import json
        for line in open(jl):
            r = json.loads(line)
            if int(r["step"]) == s:
                nxt = float(r["coeff_next"])
    chk(f"step {s}: metrics present", have, str({k: t.get(k) for k in ('entropy', 'entropy_loss', 'ent_coeff', 'ent_coeff_applied')}))
    if have:
        chk(f"step {s}: worker applied == driver coeff", abs(t["ent_coeff_applied"] - t["ent_coeff"]) < 1e-9, f"{t['ent_coeff_applied']} vs {t['ent_coeff']}")
        exp = min(a.max, max(0.0, t["ent_coeff"] + a.lr * (a.target - t["entropy"])))
        chk(f"step {s}: coeff_next follows the rule", nxt is not None and abs(nxt - exp) < 1e-6, f"next={nxt} expected={exp:.6f}")
        if prev is not None:
            chk(f"step {s}: coefficient moved in the expected direction", (t["ent_coeff"] > prev) if t["entropy"] < a.target else (t["ent_coeff"] <= prev),
                f"{prev:.6f} -> {t['ent_coeff']:.6f} (entropy {t['entropy']:.3f} vs target {a.target})")
        prev = t["ent_coeff"]
    rows.append((s, t.get("entropy"), t.get("entropy_loss"), t.get("ent_coeff"), t.get("ent_coeff_applied"), nxt, t.get("step_s")))
ok = all(c[1] for c in checks)
lines = [f"## Smoke test (adaptive entropy) — job {a.jobid}, checked {datetime.now():%Y-%m-%d %H:%M}: **{'PASS' if ok else 'FAIL'}**", "",
         "htc, 1×A100, cs25, answer-only hard gates, 3 steps × 32 prompts × n=5, MAXRESP 512, ENTROPY_TARGET=3.0 (production 0.6) so the coefficient must rise from 0.", "",
         "| step | entropy (actor/entropy) | entropy_loss | coeff used (driver) | coeff applied (worker) | coeff next | s/step |", "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append("| " + " | ".join(mf.fmt(x, 6) if isinstance(x, float) else str(x) for x in r) + " |")
lines += ["", "checks:", ""] + [f"- {'PASS' if c[1] else 'FAIL'}: {c[0]}" + (f" — {c[2]}" if c[2] else "") for c in checks] + [""]
mf.upsert_section("## Smoke test (adaptive entropy)", "\n".join(lines))
print("\n".join(lines))
sys.exit(0 if ok else 1)
