#!/usr/bin/env python
"""Split the frozen VPB eval set (2,856 single-image questions, sha 7cd6d974...) into vpb_dev (~400, stratified by
source, seed 0) and vpb_test (the rest); write vpb_dev as a verl val parquet (same schema as the train-val parquet,
data_source = "vpb_dev", ground_truth without gold steps) plus jsonl copies of both splits for vpb_generate.py.

Runs as a CPU sbatch job (image bounding for ~400 images). Outputs under grpo_arms/data/:
  vpb_dev.jsonl / vpb_dev.sha256, vpb_test.jsonl / vpb_test.sha256, vpb_dev.parquet / vpb_dev.parquet.sha256,
  vpb_dev_images/<qid>.<ext>   (bounded to 640*28*28, same cap as training)
and appends the recipe + shas to grpo_arms/DATA_SPLITS.md.
"""
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/scratch/sghos104/rlpt")
sys.path.insert(0, str(REPO / "src/train"))
from train_common import write_bounded_image  # noqa: E402

SRC = REPO / "grpo_arms/data/vpb_eval.jsonl"
OUT = REPO / "grpo_arms/data"
N_DEV, SEED = 400, 0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    src_sha = sha(SRC)
    assert src_sha == (OUT / "vpb_eval.sha256").read_text().strip(), "vpb_eval.jsonl sha mismatch"
    rows = [json.loads(l) for l in SRC.open()]
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["data_source"]].append(r)
    rng = random.Random(SEED)
    dev_ids = set()
    # stratified: dev share per source = N_DEV * n_src / N, largest-remainder rounding to hit exactly N_DEV
    quotas = {s: N_DEV * len(v) / len(rows) for s, v in by_src.items()}
    base = {s: int(q) for s, q in quotas.items()}
    for s in sorted(quotas, key=lambda s: quotas[s] - base[s], reverse=True)[: N_DEV - sum(base.values())]:
        base[s] += 1
    for s, v in sorted(by_src.items()):
        ids = sorted(r["qid"] for r in v)
        rng.shuffle(ids)
        dev_ids.update(ids[: base[s]])
    dev = [r for r in rows if r["qid"] in dev_ids]
    test = [r for r in rows if r["qid"] not in dev_ids]
    assert len(dev) == N_DEV and len(dev) + len(test) == len(rows)

    for name, rs in (("vpb_dev", dev), ("vpb_test", test)):
        p = OUT / f"{name}.jsonl"
        with p.open("w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (OUT / f"{name}.sha256").write_text(sha(p) + "\n")

    # verl val parquet for vpb_dev
    img_root = OUT / "vpb_dev_images"
    img_root.mkdir(exist_ok=True)
    prows = []
    for i, r in enumerate(dev):
        src_img = REPO / r["image"]
        tgt = img_root / f"{r['qid']}{src_img.suffix.lower()}"
        if not tgt.exists():
            write_bounded_image(src_img.read_bytes(), tgt)
        gt = {"answer": r["answer"], "gold_steps": [], "question": r["prompt"], "image_path": str(tgt)}
        prows.append({
            "data_source": "vpb_dev",
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "images": [{"image": str(tgt)}],
            "ability": "vl_reasoning",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
            "extra_info": {"index": i, "source_file": f"vpb:{r['data_source']}", "line_index": int(r["qid"]), "n_gold_steps": 0},
        })
    import datasets
    pq = OUT / "vpb_dev.parquet"
    datasets.Dataset.from_list(prows).to_parquet(str(pq))
    (OUT / "vpb_dev.parquet.sha256").write_text(sha(pq) + "\n")

    per_src = {s: (sum(r["data_source"] == s for r in dev), sum(r["data_source"] == s for r in test)) for s in sorted(by_src)}
    md = f"""# Data splits

## VPB dev / test (created 2026-09-11 by grpo_arms/build_vpb_split.py)
Source: grpo_arms/data/vpb_eval.jsonl (2,856 single-image questions), sha256 {src_sha}.
Recipe: per source, qids sorted then shuffled with random.Random(0); dev quota = 400 * n_source / 2856 with largest-remainder rounding;
first `quota` qids of each source go to dev, the rest to test. Deterministic; rerunning reproduces the same files.

| source | dev | test |
|---|---|---|
""" + "\n".join(f"| {s} | {d} | {t} |" for s, (d, t) in per_src.items()) + f"""
| **total** | **{len(dev)}** | **{len(test)}** |

| file | rows | sha256 |
|---|---|---|
| grpo_arms/data/vpb_dev.jsonl | {len(dev)} | {sha(OUT / 'vpb_dev.jsonl')} |
| grpo_arms/data/vpb_test.jsonl | {len(test)} | {sha(OUT / 'vpb_test.jsonl')} |
| grpo_arms/data/vpb_dev.parquet (verl val file, data_source=vpb_dev, images bounded to 640*28*28 under vpb_dev_images/) | {len(dev)} | {sha(pq)} |

Usage: vpb_dev is the second entry of data.val_files in the v3 training runs (metrics logged as val-aux/vpb_dev/...; ground_truth has
no gold steps, so arm_reward scores it answer-only without calling the NLI server). vpb_test is the paper number:
`vpb_generate.py --eval_set grpo_arms/data/vpb_test.jsonl`. Decisions use vpb_dev; reported numbers use vpb_test.
"""
    (REPO / "grpo_arms/DATA_SPLITS.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
