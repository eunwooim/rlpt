"""Build a small A-OKVQA eval parquet in the scienceqa_to_verl schema, for the
transfer-eval leg (test 4) of the mimicry-vs-reasoning validation plan.

A-OKVQA is differently styled from ScienceQA (COCO photos, commonsense VQA,
crowd-written rationales vs textbook solutions) — style mimicry of ScienceQA
solutions cannot transfer here, genuine visual-reasoning gains can.

Streams `HuggingFaceM4/A-OKVQA` validation (needs internet; run on a login/OOD
shell, NOT under HF_HUB_OFFLINE), takes the first N multiple-choice rows, embeds
images as JPEG bytes, and uses the same prompt template as the ScienceQA data so
the trained model's format transfers.

Usage:
  python src/data_factory/aokvqa_to_eval.py [--n 256] \
      [--out data/aokvqa_eval/aokvqa_val256.parquet]
"""

import argparse
import io
import os

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")

import pandas as pd

PROMPT_TEMPLATE = (
    "<image>\n"
    "Question: {question}\n"
    "Choices:\n{choices_block}\n\n"
    "Look at the image and reason step by step, then give the final answer as the "
    "exact text of the correct choice.\n"
    "Return exactly:\n"
    "<think> your step-by-step reasoning </think>\n"
    "<answer> the exact text of the correct choice </answer>"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default="/scratch/sghos104/rlpt/data/aokvqa_eval/aokvqa_val256.parquet")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation", streaming=True)

    rows = []
    for ex in ds:
        choices = list(ex["choices"])
        idx = ex["correct_choice_idx"]
        if idx is None or not choices:
            continue
        img = ex["image"].convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        content = PROMPT_TEMPLATE.format(
            question=ex["question"].strip(),
            choices_block="\n".join(f"- {c}" for c in choices))
        rows.append({
            "data_source": "aokvqa_composite",
            "prompt": [{"role": "user", "content": content}],
            "images": [{"bytes": buf.getvalue(), "path": None}],
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "answer": choices[idx],
                    "solution": " ".join(ex.get("rationales") or []),
                    "choices": choices,
                },
            },
            "ability": "image_grounded_reasoning",
            "extra_info": {
                "question_id": ex["question_id"],
                "index": len(rows),
                "split": "validation",
                "subject": "aokvqa",
                "topic": "aokvqa",
                "grade": "",
            },
        })
        if len(rows) >= args.n:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
