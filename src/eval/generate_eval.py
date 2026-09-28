#!/usr/bin/env python
"""Greedy vLLM generation over an eval parquet (ScienceQA test / MMK12).

One model x one parquet per process invocation (clean GPU teardown between
models). Prompts are taken from the parquet verbatim (identical for every
model — the fairness control); images come from the parquet bytes, pixel-
bounded like the SFT training cap. Output: jsonl with one row per example:
{question_id, subject, output, answer, choices, model, parquet}.
"""
from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

MAX_IMAGE_PIXELS = int(os.environ.get("EVAL_MAX_IMAGE_PIXELS", str(1280 * 28 * 28)))


def bound_pixels(image):
    total = image.width * image.height
    if total <= MAX_IMAGE_PIXELS:
        return image
    scale = (MAX_IMAGE_PIXELS / total) ** 0.5
    return image.resize((max(28, int(image.width * scale)), max(28, int(image.height * scale))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--limit", type=int, default=0, help="debug: only first N rows")
    args = p.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    rows = pq.read_table(args.parquet).to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        seed=0,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)

    requests = []
    meta = []
    for row in rows:
        text = row["prompt"][0]["content"]
        if text.startswith("<image>"):
            text = text[len("<image>") :].lstrip("\n")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(BytesIO(row["images"][0]["bytes"])) as im:
            image = bound_pixels(im.convert("RGB").copy())
        requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        gt = row["reward_model"]["ground_truth"]
        info = row.get("extra_info") or {}
        meta.append({
            "question_id": info.get("question_id"),
            "subject": info.get("subject") or info.get("topic") or "_",
            "answer": gt["answer"],
            "choices": list(gt.get("choices") or []),
        })

    outputs = llm.generate(requests, sampling)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for m, o in zip(meta, outputs):
            f.write(json.dumps({
                **m,
                "output": o.outputs[0].text,
                "model": args.model_path,
                "parquet": args.parquet,
            }, ensure_ascii=False) + "\n")
    print(f"[generate_eval] wrote {len(meta)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
