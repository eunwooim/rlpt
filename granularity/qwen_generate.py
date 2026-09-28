#!/usr/bin/env python
"""Granularity-controllability generation: Qwen2.5-VL x 4 instruction conditions.

Adapted from src/eval/generate_eval.py (the validated Qwen-VL vLLM path used
for the ScienceQA/MMK12 checkpoint evals) — same processor / chat-template /
multi_modal_data plumbing and the same 1280*28*28 pixel cap as the SFT
training config. Changed vs that script: inputs come from the frozen
granularity/qwen_question_set.json (file paths, not parquet bytes) and
sampling is temperature 0.7 / top_p 0.9 / seed 0 rather than greedy.

The question set is READ, never re-sampled: every model x condition hits
byte-identical inputs. <imageN> tags were already stripped at build time.

Output: qwen_gen_{tag}.jsonl, one row per (question x condition).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

MAX_IMAGE_PIXELS = int(os.environ.get("EVAL_MAX_IMAGE_PIXELS", str(1280 * 28 * 28)))

CONDITIONS = {
    "A_unconstrained": "Give step by step reasoning before you answer.",
    "B_count": "Give step by step reasoning before you answer, in about 6 steps.",
    "C_length": ("Give step by step reasoning before you answer. "
                 "Keep each step to about 25 words."),
    "D_format": ("Give step by step reasoning before you answer. "
                 "Number each step: 1) 2) 3)"),
}


def bound_pixels(image):
    """Identical to src/eval/generate_eval.py — keeps the input distribution
    consistent with how the model was tuned."""
    total = image.width * image.height
    if total <= MAX_IMAGE_PIXELS:
        return image
    scale = (MAX_IMAGE_PIXELS / total) ** 0.5
    return image.resize((max(28, int(image.width * scale)),
                         max(28, int(image.height * scale))))


def build_prompt_text(question, condition):
    return f"{question}\n{CONDITIONS[condition]}"


def _self_test():
    t = build_prompt_text("Q?", "A_unconstrained")
    assert t == "Q?\nGive step by step reasoning before you answer.", t
    assert set(CONDITIONS) == {"A_unconstrained", "B_count", "C_length", "D_format"}
    print("[gen] self-test OK", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--question_set",
                   default="/scratch/sghos104/rlpt/granularity/qwen_question_set.json")
    p.add_argument("--tag", required=True, help="output suffix: qwen_gen_{tag}.jsonl")
    p.add_argument("--outdir", default="/scratch/sghos104/rlpt/granularity")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--limit", type=int, default=0, help="smoke: first N questions")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true",
                   help="build prompts and exit (no GPU, no model load)")
    args = p.parse_args()
    _self_test()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"[gen] unknown conditions: {bad}")

    with open(args.question_set) as f:
        qs = json.load(f)
    questions = qs["questions"]
    if args.limit:
        questions = questions[: args.limit]
    img_root = qs["image_root"]
    print(f"[gen] question set: {len(questions)} questions "
          f"(sha256={qs['question_set_sha256'][:16]}…) conditions={conds}",
          flush=True)

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    requests, meta = [], []
    for q in questions:
        path = os.path.join(img_root, q["image"])
        with Image.open(path) as im:
            image = bound_pixels(im.convert("RGB").copy())
        for cond in conds:
            text = build_prompt_text(q["question"], cond)
            messages = [{"role": "user",
                         "content": [{"type": "image"},
                                     {"type": "text", "text": text}]}]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            meta.append({"qid": q["qid"], "data_source": q["data_source"],
                         "condition": cond, "image": q["image"],
                         "prompt_text": text, "answer": q.get("answer")})
    print(f"[gen] built {len(requests)} requests "
          f"({len(questions)} questions x {len(conds)} conditions)", flush=True)

    if args.dry_run:
        print("\n[gen] --- sample prompt (question 0, first condition) ---")
        print(requests[0]["prompt"][:1500])
        print("[gen] dry run complete, no model loaded")
        return

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.model_path, trust_remote_code=True,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": 1}, seed=args.seed)
    print(f"[gen] model loaded in {time.time()-t0:.0f}s", flush=True)

    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens, seed=args.seed)
    t1 = time.time()
    outputs = llm.generate(requests, sampling)
    gen_s = time.time() - t1

    out_path = Path(args.outdir) / f"qwen_gen_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with out_path.open("w", encoding="utf-8") as f:
        for m, o in zip(meta, outputs):
            text = o.outputs[0].text
            n_empty += (not text.strip())
            f.write(json.dumps({
                **m, "output": text, "model": args.model_path,
                "n_gen_tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason,
            }, ensure_ascii=False) + "\n")
    print(f"[gen] generated {len(meta)} rows in {gen_s:.0f}s "
          f"({len(meta)/max(gen_s,1e-9):.2f} gen/s), empty={n_empty}", flush=True)
    print(f"[gen] -> {out_path}")
    print("[gen] DONE")


if __name__ == "__main__":
    main()
