import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from tqdm.auto import tqdm

from prompts import SCIENCEQA_PROMPT_TEMPLATE
from utils import (
    extract_mc_answer,
    append_jsonl,
    save_json,
    set_seed,
)

CHOICES = ["A", "B", "C", "D", "E"]


def build_prompt(sample):
    question = sample["question"]

    if sample.get("hint"):
        question += f"\n\nContext:\n{sample['hint']}"

    choice_text = "\n".join(
        [
            f"{CHOICES[i]}. {choice}"
            for i, choice in enumerate(sample["choices"])
        ]
    )

    return SCIENCEQA_PROMPT_TEMPLATE.format(
        question=question,
        choices=choice_text,
    )


@torch.inference_mode()
def generate_answer(
    model,
    processor,
    prompt,
    image=None,
    max_new_tokens=128,
):
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text],
            return_tensors="pt",
        )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return response.strip()


from pathlib import Path


def resolve_output_dir(
    model_path: str,
    benchmark_name: str,
    output_dir: str | None = None,
):
    """
    Output structure:

    outputs/
        Qwen2.5-VL-3B-Instruct/
            scienceqa/
                eval_scienceqa_test.json
                predictions_scienceqa_test.jsonl

        2025-06-07-exp/
            scienceqa/
                ...
    """

    if output_dir is not None:
        exp_name = output_dir

    else:
        path_obj = Path(model_path)

        if path_obj.exists():
            exp_name = path_obj.name
        else:
            exp_name = model_path.split("/")[-1]

    save_dir = (
        Path("outputs")
        / exp_name
        / benchmark_name
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return save_dir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
    )

    parser.add_argument(
        "--split",
        default="test",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--image_only",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    output_dir = resolve_output_dir(
        model_path=args.model_path,
        benchmark_name="scienceqa",
        output_dir=args.output_dir,
    )

    pred_path = (
        output_dir
        / f"scienceqa_predictions_{args.split}.jsonl"
    )

    summary_path = (
        output_dir
        / "scienceqa.json"
    )

    print(f"Loading model: {args.model_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True,
    )

    dataset = load_dataset(
        "derek-thomas/ScienceQA",
        split=args.split,
    )

    if args.image_only:
        dataset = dataset.filter(
            lambda x: x["image"] is not None
        )

    if args.max_samples:
        dataset = dataset.select(
            range(min(args.max_samples, len(dataset)))
        )

    print(f"Evaluating {len(dataset)} samples")

    if pred_path.exists():
        pred_path.unlink()

    correct, total, parse_failures = 0, 0, 0

    pbar = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc="ScienceQA",
        ncols=79
    )
    for idx, sample in pbar:
        prompt = build_prompt(sample)

        image = sample["image"]

        response = generate_answer(
            model=model,
            processor=processor,
            prompt=prompt,
            image=image,
        )

        prediction = extract_mc_answer(response)
        if prediction is None:
            parse_failures += 1

        gold = CHOICES[sample["answer"]]

        is_correct = prediction == gold

        correct += int(is_correct)
        total += 1

        append_jsonl(
            {
                "idx": idx,
                "question": sample["question"],
                "gold": gold,
                "prediction": prediction,
                "correct": is_correct,
                "has_image": image is not None,
                "response": response,
            },
            pred_path,
        )

        if total % 50 == 0:
            print(
                f"{total}/{len(dataset)} "
                f"| acc={correct/total:.4f}"
            )

    results = {
        "benchmark": "scienceqa",
        "model_path": args.model_path,
        "split": args.split,
        "image_only": args.image_only,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }

    save_json(
        results,
        summary_path,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    set_seed()
    main()
