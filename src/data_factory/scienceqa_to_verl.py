"""Format ScienceQA into verl parquet for the image-description RL track.

One output row = one prompt verl rolls out on. Columns verl expects:
    data_source   -> routes to the reward fn (tools/graph_match_reward.compute_score)
    prompt        -> chat list [{"role":"user","content": "<image>\\n..."}]
    images        -> [abs path to the saved jpg]
    reward_model  -> {"style":"rule","ground_truth": {answer, solution, choices}}
    ability       -> task tag
    extra_info    -> {question_id, split, subject, topic, ...}

The model is prompted to emit  <think> reasoning </think><answer> exact choice text
</answer>  — choice TEXT (not a letter), since ScienceQA letters are positionally
ambiguous and `answer_correct` matches on text.

Default keeps only rows with BOTH an image and a non-empty solution (the multimodal
+ reasoning-match subset): 5678 train / 1922 val / 1836 test.

Run:
    python src/data_factory/scienceqa_to_verl.py --splits train validation test
"""

import argparse
import io
import os

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")

DATA_SOURCE = "scienceqa_composite"
ABILITY = "image_grounded_reasoning"

PROMPT_TEMPLATE = (
    "<image>\n"
    "Question: {question}\n"
    "{hint}"
    "Choices:\n{choices_block}\n\n"
    "Look at the image and reason step by step, then give the final answer as the "
    "exact text of the correct choice.\n"
    "Return exactly:\n"
    "<think> your step-by-step reasoning </think>\n"
    "<answer> the exact text of the correct choice </answer>"
)


def build_prompt(question, choices, hint):
    choices_block = "\n".join(f"- {c}" for c in choices)
    hint_line = f"Context: {hint.strip()}\n" if hint and hint.strip() else ""
    content = PROMPT_TEMPLATE.format(
        question=question.strip(), hint=hint_line,
        choices_block=choices_block,
    )
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--out-dir", default="/scratch/sghos104/rlpt/data/scienceqa_verl")
    ap.add_argument("--image-out-dir", default="/scratch/sghos104/rlpt/data/scienceqa_images")
    ap.add_argument("--require-image", type=int, default=1, help="keep only rows with an image")
    ap.add_argument("--require-solution", type=int, default=1, help="keep only rows with a solution")
    ap.add_argument("--max-records", type=int, default=0, help="0 = no cap (debug only)")
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.image_out_dir, exist_ok=True)

    for split in args.splits:
        ds = load_dataset("derek-thomas/ScienceQA", split=split)
        img_dir = os.path.join(args.image_out_dir, split)
        os.makedirs(img_dir, exist_ok=True)

        rows, kept, skipped = [], 0, 0
        for i, r in enumerate(ds):
            sol = r["solution"]
            img = r["image"]
            if args.require_image and img is None:
                skipped += 1
                continue
            if args.require_solution and not (sol and sol.strip()):
                skipped += 1
                continue

            qid = f"scienceqa_{split}_{i:06d}"
            images_field = []
            if img is not None:
                img_path = os.path.join(img_dir, f"{i:06d}.jpg")
                rgb = img.convert("RGB")
                if not os.path.exists(img_path):
                    rgb.save(img_path, "JPEG", quality=92)
                # verl wants each image as a dict (bytes/path) or PIL, NOT a bare path
                buf = io.BytesIO()
                rgb.save(buf, format="JPEG", quality=92)
                images_field = [{"bytes": buf.getvalue(), "path": img_path}]

            choices = list(r["choices"])
            gold = choices[r["answer"]]
            rows.append({
                "data_source": DATA_SOURCE,
                "prompt": build_prompt(r["question"], choices, r["hint"]),
                "images": images_field,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "answer": gold,
                        "solution": sol.strip(),
                        "choices": choices,
                    },
                },
                "ability": ABILITY,
                "extra_info": {
                    "question_id": qid,
                    "index": i,
                    "split": split,
                    "subject": r.get("subject", ""),
                    "topic": r.get("topic", ""),
                    "grade": r.get("grade", ""),
                },
            })
            kept += 1
            if args.max_records and kept >= args.max_records:
                break

        out_path = os.path.join(args.out_dir, f"{split}.parquet")
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"[{split}] kept={kept} skipped={skipped} -> {out_path}", flush=True)

    # tiny manifest for sanity
    print("[done] wrote splits:", args.splits, "to", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
