#!/usr/bin/env python3
"""MetaMathQA -> Step-N cold-start SFT data at a target granularity.

Subsamples (seed 0, stratified by augmentation 'type'), splits each
response into sentences, greedily packs sentences into steps targeting
TARGET_WPS words/step, renumbers as 'Step k: ...', appends the answer
as a final line. Emits jsonl + a stats json (verify grain BEFORE training).
"""
import json, random, re, argparse
from collections import defaultdict

TARGET_WPS = 24          # the independent variable of this experiment
MAX_WPS    = 30          # close a step before exceeding this
MIN_STEPS, MAX_STEPS = 3, 8
ANS_RE  = re.compile(r"The answer is:?\s*(.+?)\s*$", re.S)
# sentence split: end punctuation + space + capital/digit/$; protects decimals (3.5), \n kept as splitter
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\\])")

def to_steps(body):
    sents = [s.strip() for s in SENT_RE.split(body) if s.strip()]
    steps, cur, cw = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and cw + w > MAX_WPS and cw >= TARGET_WPS * 0.6:
            steps.append(" ".join(cur)); cur, cw = [], 0
        cur.append(s); cw += w
    if cur: steps.append(" ".join(cur))
    # merge tiny tail steps
    while len(steps) > 1 and len(steps[-1].split()) < 6:
        tail = steps.pop()
        steps[-1] += " " + tail
    return steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/metamath_coldstart/MetaMathQA-395K.json")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default="data/metamath_coldstart/coldstart_sft_30k.jsonl")
    a = ap.parse_args()
    random.seed(0)
    data = json.load(open(a.src))
    by_type = defaultdict(list)
    for r in data: by_type[r.get("type","?")].append(r)
    per = max(1, a.n // len(by_type))
    sample = []
    for t, rows in sorted(by_type.items()):
        random.shuffle(rows); sample += rows[:per]
    random.shuffle(sample); sample = sample[:a.n]

    kept, wps_all, nsteps_all, skipped = 0, [], [], 0
    with open(a.out, "w") as f:
        for r in sample:
            resp = r["response"].strip()
            m = ANS_RE.search(resp)
            ans  = m.group(1).strip() if m else None
            body = resp[:m.start()].strip() if m else resp
            body = re.sub(r"####\s*[^\s]+", "", body).strip()      # GSM8K residue, anywhere
            body = re.sub(r"\s+", " ", body)                        # collapse newlines/wraps
            if "cannot determine" in body.lower(): skipped += 1; continue
            if not ans or not body: skipped += 1; continue
            steps = to_steps(body)
            if not (MIN_STEPS <= len(steps) <= MAX_STEPS): skipped += 1; continue
            lines = [f"Step {i+1}: {s}" for i, s in enumerate(steps)]
            lines.append(f"The answer is {ans}")
            f.write(json.dumps({
                "prompt": r["query"].strip(),
                "images": [],
                "response": "\n".join(lines),
                "metadata": {"source": "MetaMathQA", "type": r.get("type"),
                             "n_steps": len(steps)}}, ensure_ascii=False) + "\n")
            kept += 1
            nsteps_all.append(len(steps))
            wps_all += [len(s.split()) for s in steps]
    wps_all.sort()
    stats = {"kept": kept, "skipped": skipped,
             "mean_wps": round(sum(wps_all)/len(wps_all), 2),
             "p50_wps": wps_all[len(wps_all)//2],
             "mean_steps": round(sum(nsteps_all)/len(nsteps_all), 2),
             "approx_train_tokens": int(sum(wps_all) * 1.35)}
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=2)
    print(stats)

if __name__ == "__main__":
    main()
