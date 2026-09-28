"""jb_build_samples.py — deterministic samples for tasks 3a/3b/3c/3d (CPU). Writes out/sample_3a.jsonl, out/sample_3b.jsonl, out/sample_3d.jsonl."""
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jb_lib as jb  # noqa: E402

os.makedirs(jb.OUT, exist_ok=True)
rows = [json.loads(l) for l in open(f"{jb.REPO}/data/visualprm400k/pairs.jsonl")]
random.seed(0)
by = collections.defaultdict(list)
for r in rows:
    by[r["source"]].append(r)
samp = []
for s, rs in sorted(by.items()):
    k = max(1, round(5000 * len(rs) / len(rows)))
    random.shuffle(rs)
    samp += rs[:k]
out3a, skipped = [], collections.Counter()
for r in samp:
    ps, _ = jb.rv.split_steps(r["pos"]); ns, _ = jb.rv.split_steps(r["neg"]); bs, _ = jb.rv.split_steps(r["baseline"])
    k = jb.edited_index(ps, ns)
    if k is None:
        skipped["no single edited step"] += 1; continue
    if not bs:
        skipped["empty baseline"] += 1; continue
    rnd = random.Random(r["idx"])
    out3a.append({"idx": r["idx"], "source": r["source"], "edit": r["edit"], "k": k, "n_steps": len(ps), "pos_step": ps[k], "neg_step": ns[k],
                  "base_step": rnd.choice(bs), "category": jb.step_category(ps[k], r["edit"]), "pos": r["pos"], "neg": r["neg"], "baseline": r["baseline"]})
with open(f"{jb.OUT}/sample_3a.jsonl", "w") as f:
    for r in out3a:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("3a:", len(out3a), "rows (sampled", len(samp), "; skipped", dict(skipped), "); categories", collections.Counter(r["category"] for r in out3a))
para = [json.loads(l) for l in open(f"{jb.REPO}/data/visualprm400k/paraphrase_pairs.jsonl")]
faithful = [p for p in para if p["faithful"]]
random.seed(0)
s3b = random.sample(faithful, 5000)
with open(f"{jb.OUT}/sample_3b.jsonl", "w") as f:
    for p in s3b:
        f.write(json.dumps({"idx": p["idx"], "pos": p["pos"], "paraphrase": p["paraphrase"]}, ensure_ascii=False) + "\n")
print("3b:", len(s3b), "rows from", len(faithful), "faithful==True of", len(para))
eq = f"{jb.REPO}/reward_redesign/judge_bench/EQUATE/ProcessedDatasets"
n = 0
with open(f"{jb.OUT}/sample_3d.jsonl", "w") as f:
    for name, fn in (("RTE-Quant", "RTE_Quant"), ("NewsNLI", "NewsNLI"), ("RedditNLI", "RedditNLI"), ("AWP-NLI", "AWPNLI"), ("StressTest", "StressTest")):
        for i, l in enumerate(open(f"{eq}/{fn}.jsonl")):
            if not l.strip():
                continue
            r = json.loads(l)
            f.write(json.dumps({"subset": name, "i": i, "premise": r["sentence1"], "hypothesis": r["sentence2"], "gold": r["gold_label"]}, ensure_ascii=False) + "\n"); n += 1
print("3d:", n, "EQUATE rows")
