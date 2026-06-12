"""Greedy vLLM generation pass over an eval parquet, for the mimicry-vs-reasoning
validation plan (CLAUDE.md). One JSONL line per row with the raw model output plus
everything the offline scorers need (GT solution/answer/choices, topic).

Modes:
  normal       — images as stored in the parquet.
  --image-swap — every row gets a DIFFERENT row's image (seeded cyclic derangement,
                 no fixed points). Real image analysis collapses under swap; a
                 text-prior/mimicry model barely notices (validation test 3).

The same script serves ScienceQA sweep256 and the A-OKVQA transfer parquet — any
parquet in the scienceqa_to_verl schema works.

Usage:
  python tools/gen_eval_outputs.py --model-path <hf-dir-or-id> \
      --parquet data/scienceqa_verl/test_sweep256.parquet \
      --out outputs/validation/gen_step176.jsonl [--image-swap] [--swap-seed 1234]
"""

import argparse
import io
import json
import os

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")

import numpy as np
import pandas as pd
from PIL import Image


def load_rows(parquet):
    df = pd.read_parquet(parquet)
    rows = []
    for i, r in df.iterrows():
        text = r["prompt"][0]["content"].replace("<image>", "").strip()
        gt = r["reward_model"]["ground_truth"]
        info = r["extra_info"]
        rows.append({
            "question_id": info.get("question_id", str(i)),
            "topic": info.get("topic", ""),
            "subject": info.get("subject", ""),
            "text": text,
            "image_bytes": r["images"][0]["bytes"],
            "answer": gt.get("answer"),
            "solution": gt.get("solution", ""),
            "choices": list(gt.get("choices", [])),
        })
    return rows


def derangement(n, seed):
    """Cyclic shift of a seeded shuffle — guaranteed no fixed points."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    perm = np.empty(n, dtype=int)
    for k in range(n):
        perm[order[k]] = order[(k + 1) % n]
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-swap", action="store_true")
    ap.add_argument("--swap-seed", type=int, default=1234)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    rows = load_rows(args.parquet)
    n = len(rows)
    img_src = list(range(n))
    if args.image_swap:
        img_src = derangement(n, args.swap_seed).tolist()
    print(f"[gen] {n} rows from {args.parquet}  swap={args.image_swap}", flush=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(model=args.model_path, trust_remote_code=True, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1})
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    requests = []
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": row["text"]},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        img = Image.open(io.BytesIO(rows[img_src[i]]["image_bytes"])).convert("RGB")
        requests.append({"prompt": prompt, "multi_modal_data": {"image": img}})

    outputs = llm.generate(requests, sampling)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for i, (row, out) in enumerate(zip(rows, outputs)):
            rec = {
                "question_id": row["question_id"],
                "topic": row["topic"],
                "subject": row["subject"],
                "output": out.outputs[0].text,
                "answer": row["answer"],
                "solution": row["solution"],
                "choices": row["choices"],
                "image_swapped_from": (rows[img_src[i]]["question_id"]
                                       if args.image_swap else None),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[gen] wrote {n} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
