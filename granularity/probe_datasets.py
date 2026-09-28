#!/usr/bin/env python
"""Verify we have the expected VisualPRM and VisualProcessBench, and report
raw step-granularity stats. No assumptions about VPB's schema -- it prints
what it finds and falls back across plausible field names.
"""
import glob, json, os, sys, itertools, statistics as st
from collections import Counter

VPB_DIR = "/scratch/sghos104/rlpt/data/visualprocessbench"
CANON   = "/scratch/sghos104/rlpt/canonical.jsonl"

STEP_KEYS  = ("steps", "reasoning_steps", "solution_steps", "process", "response_steps")
LABEL_KEYS = ("step_labels", "labels", "correctness", "step_correctness", "annotations")
Q_KEYS     = ("question", "question_orig", "prompt", "query")


def first_key(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return k
    return None


def load_any(path, limit=None):
    """jsonl / json-array / parquet -> list of dicts"""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        rows = t.to_pylist()
        return rows[:limit] if limit else rows
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            rows = json.load(f)
            return rows[:limit] if limit else rows
        out = []
        for line in itertools.islice(f, limit) if limit else f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out


def describe(name, rows):
    print(f"\n{'='*72}\n{name}: {len(rows):,} records\n{'='*72}")
    if not rows:
        return None, None
    r0 = rows[0]
    print("top-level keys:", sorted(r0.keys()))
    sk = first_key(r0, STEP_KEYS)
    lk = first_key(r0, LABEL_KEYS)
    qk = first_key(r0, Q_KEYS)
    print(f"resolved -> steps={sk!r}  labels={lk!r}  question={qk!r}")
    if sk is None:
        print("!! no steps field found. Dump of record 0:")
        print(json.dumps(r0, indent=2, ensure_ascii=False)[:2000])
        return None, None

    nsteps, wlens = [], []
    lab = Counter()
    for r in rows:
        s = r.get(sk) or []
        if not isinstance(s, list):
            continue
        nsteps.append(len(s))
        for step in s:
            if isinstance(step, str):
                wlens.append(len(step.split()))
            elif isinstance(step, dict):
                txt = step.get("text") or step.get("content") or ""
                wlens.append(len(str(txt).split()))
        if lk:
            v = r.get(lk)
            if isinstance(v, list):
                for x in v:
                    lab[str(x)] += 1

    def q(xs, p):
        return st.quantiles(xs, n=100)[p-1] if len(xs) > 100 else (st.median(xs) if xs else 0)

    print(f"\nsteps per record : mean {st.mean(nsteps):.2f}  p50 {st.median(nsteps):.0f}  "
          f"p90 {q(nsteps,90):.0f}  max {max(nsteps)}")
    print(f"multi-step (>=2) : {sum(1 for n in nsteps if n>=2):,} "
          f"({100*sum(1 for n in nsteps if n>=2)/len(nsteps):.1f}%)")
    if wlens:
        print(f"WORDS PER STEP   : mean {st.mean(wlens):.1f}  p50 {st.median(wlens):.0f}  "
              f"p90 {q(wlens,90):.0f}  max {max(wlens)}")
    if lab:
        print("label values     :", dict(lab.most_common(8)))
    print("\nrecord 0, first 2 steps:")
    for i, s in enumerate((rows[0].get(sk) or [])[:2]):
        print(f"  [{i}] {str(s)[:220]}")
    return nsteps, wlens


# ---- VisualProcessBench ----
files = sorted(glob.glob(os.path.join(VPB_DIR, "**", "*"), recursive=True))
data_files = [f for f in files if f.endswith((".json", ".jsonl", ".parquet"))]
print("VPB files found:")
for f in data_files[:20]:
    print(f"  {os.path.getsize(f)/1e6:8.2f} MB  {f}")
if not data_files:
    print("!! no data files under", VPB_DIR)
    sys.exit(1)

vpb_rows = []
for f in data_files:
    if "config" in os.path.basename(f).lower():
        continue
    try:
        vpb_rows = load_any(f)
        print(f"\nloaded: {f}")
        break
    except Exception as e:
        print(f"  skip {f}: {e}")
vpb_n, vpb_w = describe("VisualProcessBench", vpb_rows)

# ---- VisualPRM (multi-step only, strided so we don't read the ordered head) ----
size = os.path.getsize(CANON)
vprm = []
with open(CANON, "rb") as f:
    for k in range(40):
        f.seek(int(size * k / 40)); f.readline()
        for line in itertools.islice(f, 3000):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if len(r.get("steps") or []) >= 2:
                vprm.append(r)
print(f"\n(VisualPRM: {len(vprm):,} multi-step records sampled across 40 blocks)")
vprm_n, vprm_w = describe("VisualPRM400K-v1.1 (multi-step)", vprm)

if vpb_w and vprm_w:
    print(f"\n{'='*72}\nGRANULARITY COMPARISON (words per step)\n{'='*72}")
    print(f"  VisualPRM        mean {st.mean(vprm_w):6.1f}   p50 {st.median(vprm_w):5.0f}")
    print(f"  VisualProcessB.  mean {st.mean(vpb_w):6.1f}   p50 {st.median(vpb_w):5.0f}")
    print(f"  ratio (VPB/VPRM) {st.mean(vpb_w)/st.mean(vprm_w):.2f}x")
    print(f"\n  steps/record     VPRM {st.mean(vprm_n):.2f}  vs  VPB {st.mean(vpb_n):.2f}")
