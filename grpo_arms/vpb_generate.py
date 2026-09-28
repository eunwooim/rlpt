#!/usr/bin/env python
"""Generate answers on the frozen VPB eval set for one model/checkpoint.

Adapted from the validated Qwen-VL vLLM path (granularity/qwen_generate.py /
src/eval/generate_eval.py): AutoProcessor chat template, single-image
multi_modal_data, bounded pixels. Greedy decoding (temperature 0) — accuracy
eval, one sample per question, identical inputs for every arm.

Image cap defaults to 640*28*28 = the RLVR training materialization cap, so
the eval input distribution matches what the arms were trained on.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

REPO = "/scratch/sghos104/rlpt"
MAX_IMAGE_PIXELS = int(os.environ.get("EVAL_MAX_IMAGE_PIXELS", str(640 * 28 * 28)))


def bound_pixels(image):
    total = image.width * image.height
    if total <= MAX_IMAGE_PIXELS:
        return image
    scale = (MAX_IMAGE_PIXELS / total) ** 0.5
    return image.resize((max(28, int(image.width * scale)),
                         max(28, int(image.height * scale))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True, help="HF id or merged checkpoint dir")
    p.add_argument("--tag", required=True, help="output: vpb_gen_{tag}.jsonl")
    p.add_argument("--eval_set", default=os.path.join(REPO, "grpo_arms/data/vpb_eval.jsonl"))
    p.add_argument("--outdir", default=os.path.join(REPO, "grpo_arms/evals"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = p.parse_args()

    sha = hashlib.sha256(open(args.eval_set, "rb").read()).hexdigest()
    expected = open(args.eval_set.replace(".jsonl", ".sha256")).read().strip()
    assert sha == expected, f"vpb_eval sha mismatch: {sha} != {expected}"
    rows = [json.loads(l) for l in open(args.eval_set)]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[vpbgen] model={args.model_path} tag={args.tag} n={len(rows)} "
          f"eval_sha={sha[:16]}.. pixel_cap={MAX_IMAGE_PIXELS}", flush=True)

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    requests, meta = [], []
    for r in rows:
        with Image.open(os.path.join(REPO, r["image"])) as im:
            image = bound_pixels(im.convert("RGB").copy())
        text = r["prompt"].replace("<image>\n", "", 1)
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        meta.append({"qid": r["qid"], "data_source": r["data_source"],
                     "answer": r["answer"], "image": r["image"]})

    t0 = time.time()
    llm = LLM(model=args.model_path, trust_remote_code=True,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": 1}, seed=0)
    print(f"[vpbgen] model loaded in {time.time()-t0:.0f}s", flush=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t1 = time.time()
    outputs = llm.generate(requests, sampling)
    gen_s = time.time() - t1

    out_path = Path(args.outdir) / f"vpb_gen_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with out_path.open("w") as f:
        for m, o in zip(meta, outputs):
            text = o.outputs[0].text
            n_empty += (not text.strip())
            f.write(json.dumps({**m, "output": text, "model": args.model_path,
                                "n_gen_tokens": len(o.outputs[0].token_ids),
                                "finish_reason": o.outputs[0].finish_reason,
                                "eval_set_sha256": sha},
                               ensure_ascii=False) + "\n")
    print(f"[vpbgen] {len(meta)} rows in {gen_s:.0f}s, empty={n_empty} -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
