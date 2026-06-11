#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_dataset


CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def split_trace_segments(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(p) for p in pieces if clean_text(p)]


def normalize_answer(row: Dict[str, Any]) -> str:
    answer = row.get("answer")
    choices = row.get("choices", [])

    if isinstance(answer, int):
        if 0 <= answer < len(CHOICE_LABELS):
            return CHOICE_LABELS[answer]
        return str(answer)

    answer_str = clean_text(answer)

    if len(answer_str) == 1 and answer_str.upper() in CHOICE_LABELS:
        return answer_str.upper()

    if isinstance(choices, list):
        for idx, choice in enumerate(choices):
            if clean_text(choice).lower() == answer_str.lower():
                return CHOICE_LABELS[idx]

    return answer_str


def format_choices(choices: Any) -> str:
    if not isinstance(choices, list):
        return ""

    lines = []
    for idx, choice in enumerate(choices):
        label = CHOICE_LABELS[idx]
        lines.append(f"{label}. {clean_text(choice)}")
    return "\n".join(lines)


def get_image_path(
    row: Dict[str, Any],
    image_root: Optional[str],
    image_save_dir: Optional[str],
    split_name: str,
    idx: int,
) -> Optional[str]:
    image_obj = row.get("image")

    if image_obj is not None and hasattr(image_obj, "save"):
        if not image_save_dir:
            return None

        out_dir = Path(image_save_dir) / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        qid = clean_text(row.get("id", row.get("question_id", idx)))
        if not qid:
            qid = str(idx)

        safe_qid = re.sub(r"[^a-zA-Z0-9_.-]+", "_", qid)
        out_path = out_dir / f"{safe_qid}.jpg"

        if not out_path.exists():
            image_obj.convert("RGB").save(out_path, quality=95)

        return str(out_path)

    if not image_root:
        return None

    candidates = []

    for key in ["image", "image_path", "imagePath", "img", "picture"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    qid = clean_text(row.get("id", row.get("question_id", "")))
    split = clean_text(row.get("split", row.get("fold", split_name)))

    if qid:
        candidates.extend(
            [
                f"{qid}.png",
                f"{qid}.jpg",
                f"{qid}.jpeg",
                os.path.join(split, f"{qid}.png") if split else "",
                os.path.join(split, f"{qid}.jpg") if split else "",
                os.path.join("images", f"{qid}.png"),
                os.path.join("images", f"{qid}.jpg"),
                os.path.join("images", split, f"{qid}.png") if split else "",
                os.path.join("images", split, f"{qid}.jpg") if split else "",
            ]
        )

    root = Path(image_root)

    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)
        if path.is_absolute() and path.exists():
            return str(path)

        path = root / candidate
        if path.exists():
            return str(path)

    return None


def get_trace_text(row: Dict[str, Any], trace_source: str) -> str:
    lecture = clean_text(row.get("lecture", ""))
    solution = clean_text(row.get("solution", row.get("explanation", "")))

    if trace_source == "lecture":
        return lecture
    if trace_source == "solution":
        return solution
    if trace_source == "lecture_solution":
        return " ".join([x for x in [lecture, solution] if x]).strip()

    raise ValueError(f"Unsupported trace_source: {trace_source}")


def build_scienceqa_prompt(row: Dict[str, Any], mode: str, trace_tag: str) -> str:
    question = clean_text(row.get("question", ""))
    hint = clean_text(row.get("hint", ""))
    context = clean_text(row.get("context", ""))
    choices = format_choices(row.get("choices", []))

    parts = ["<image>"]

    if context:
        parts.append(f"Context: {context}")
    if hint:
        parts.append(f"Hint: {hint}")

    parts.append(f"Question: {question}")

    if choices:
        parts.append(f"Choices:\n{choices}")

    if mode == "sft":
        parts.append(
            "Return exactly:\n"
            f"<{trace_tag}>...</{trace_tag}>\n"
            "<answer>...</answer>\n"
            "The answer must be one of the choice letters."
        )
    elif mode == "rlvr":
        parts.append(
            "Solve the problem. You may reason step by step inside <think>...</think>.\n"
            "Then provide the final answer as one choice letter.\n"
            "Return exactly:\n"
            f"<{trace_tag}>...</{trace_tag}>\n"
            "<answer>...</answer>\n"
            "The answer must be one of the choice letters."
        )
    elif mode == "ours":
        parts.append(
            "First give concise visual evidence, then choose the correct answer.\n"
            "Return exactly:\n"
            f"<{trace_tag}>...</{trace_tag}>\n"
            "<answer>...</answer>\n"
            "The answer must be one of the choice letters."
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return "\n".join(parts)


def build_sft_response(row: Dict[str, Any], trace_source: str, trace_tag: str) -> str:
    trace_text = get_trace_text(row, trace_source)
    answer = normalize_answer(row)

    if not trace_text:
        trace_text = "The correct answer follows from the visual and textual information in the question."

    return f"<{trace_tag}>{trace_text}</{trace_tag}>\n<answer>{answer}</answer>"


def build_reward_model(
    row: Dict[str, Any],
    mode: str,
    trace_source: str,
    trace_tag: str,
    answer_weight: float,
    trace_weight: float,
) -> Dict[str, Any]:
    answer = normalize_answer(row)
    trace_text = get_trace_text(row, trace_source)
    trace_segments = split_trace_segments(trace_text)

    ground_truth: Dict[str, Any] = {
        "answer": answer,
        "choices": row.get("choices", []),
    }

    if mode in {"sft", "ours"}:
        ground_truth["trace_segments"] = trace_segments

    if mode == "rlvr":
        reward_config = {
            "type": "multiple_choice_answer_format",
            "answer_tag": "answer",
            "trace_tag": trace_tag,
            "answer_weight": answer_weight,
            "format_weight": trace_weight,
        }
    elif mode == "ours":
        reward_config = {
            "type": "answer_trace_matching",
            "answer_tag": "answer",
            "trace_tag": trace_tag,
            "similarity": "sbert",
            "matching": "bipartite",
            "trace_metric": "f1",
            "similarity_threshold": 0.55,
            "answer_weight": answer_weight,
            "trace_weight": trace_weight,
        }
    else:
        reward_config = {
            "type": "sft_reference",
            "answer_tag": "answer",
            "trace_tag": trace_tag,
        }

    ground_truth["reward_config"] = reward_config

    return {
        "style": "rule",
        "ground_truth": ground_truth,
    }


def build_record(
    row: Dict[str, Any],
    idx: int,
    split_name: str,
    dataset: str,
    mode: str,
    image_root: Optional[str],
    image_save_dir: Optional[str],
    require_image: bool,
    data_source: str,
    trace_source: str,
    trace_tag: str,
    answer_weight: float,
    trace_weight: float,
) -> Optional[Dict[str, Any]]:
    if dataset != "scienceqa":
        raise ValueError(f"Unsupported dataset: {dataset}")

    image_path = get_image_path(
        row=row,
        image_root=image_root,
        image_save_dir=image_save_dir,
        split_name=split_name,
        idx=idx,
    )

    if require_image and image_path is None:
        return None

    prompt = build_scienceqa_prompt(row=row, mode=mode, trace_tag=trace_tag)
    reward_model = build_reward_model(
        row=row,
        mode=mode,
        trace_source=trace_source,
        trace_tag=trace_tag,
        answer_weight=answer_weight,
        trace_weight=trace_weight,
    )

    answer = normalize_answer(row)
    trace_text = get_trace_text(row, trace_source)
    trace_segments = split_trace_segments(trace_text)

    ability_map = {
        "sft": "scienceqa_sft",
        "rlvr": "scienceqa_answer_rlvr",
        "ours": "evidence_matched_multimodal_reasoning",
    }

    record: Dict[str, Any] = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "images": [{"image": image_path}] if image_path is not None else [],
        "videos": [],
        "reward_model": reward_model,
        "ability": ability_map[mode],
        "extra_info": {
            "question_id": clean_text(row.get("id", row.get("question_id", idx))),
            "source_dataset": data_source,
            "split": split_name,
            "has_image": image_path is not None,
            "answer": answer,
            "num_trace_segments": len(trace_segments),
            "trace_source": trace_source,
            "mode": mode,
            "task": clean_text(row.get("task", "")),
            "grade": clean_text(row.get("grade", "")),
            "subject": clean_text(row.get("subject", "")),
            "topic": clean_text(row.get("topic", "")),
            "category": clean_text(row.get("category", "")),
            "skill": clean_text(row.get("skill", "")),
        },
    }

    if mode == "sft":
        response = build_sft_response(row=row, trace_source=trace_source, trace_tag=trace_tag)
        record["messages"] = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    return record


def dataset_to_records(
    ds: Dataset,
    split_name: str,
    dataset: str,
    mode: str,
    image_root: Optional[str],
    image_save_dir: Optional[str],
    require_image: bool,
    data_source: str,
    trace_source: str,
    trace_tag: str,
    answer_weight: float,
    trace_weight: float,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    records = []

    for idx, row in enumerate(ds):
        if max_samples is not None and len(records) >= max_samples:
            break

        record = build_record(
            row=dict(row),
            idx=idx,
            split_name=split_name,
            dataset=dataset,
            mode=mode,
            image_root=image_root,
            image_save_dir=image_save_dir,
            require_image=require_image,
            data_source=data_source,
            trace_source=trace_source,
            trace_tag=trace_tag,
            answer_weight=answer_weight,
            trace_weight=trace_weight,
        )

        if record is not None:
            records.append(record)

    return records


def save_records(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(str(path))


def resolve_default_dataset_name(dataset: str) -> str:
    if dataset == "scienceqa":
        return "derek-thomas/ScienceQA"
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="scienceqa", choices=["scienceqa"])
    parser.add_argument("--mode", required=True, choices=["sft", "rlvr", "ours"])

    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)

    parser.add_argument("--image_root", default=None)
    parser.add_argument("--image_save_dir", default=None)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--data_source", default=None)
    parser.add_argument("--require_image", action="store_true")

    parser.add_argument("--trace_source", default="lecture_solution", choices=["lecture", "solution", "lecture_solution"])
    parser.add_argument("--trace_tag", default="think", choices=["think", "evidence", "reasoning"])

    parser.add_argument("--answer_weight", type=float, default=0.5)
    parser.add_argument("--trace_weight", type=float, default=0.5)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--trust_remote_code", action="store_true")

    args = parser.parse_args()

    dataset_name = args.dataset_name or resolve_default_dataset_name(args.dataset)
    data_source = args.data_source or args.dataset

    load_kwargs = {}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    if args.dataset_config:
        raw = load_dataset(dataset_name, args.dataset_config, **load_kwargs)
    else:
        raw = load_dataset(dataset_name, **load_kwargs)

    if not isinstance(raw, DatasetDict):
        raise ValueError(f"Expected DatasetDict, got {type(raw)}")

    print(raw)
    print("Available splits:", list(raw.keys()))

    train_split = "train" if "train" in raw else list(raw.keys())[0]

    val_split = args.val_split if args.val_split in raw else None
    if val_split is None:
        for candidate in ["validation", "val", "test"]:
            if candidate in raw:
                val_split = candidate
                break

    if val_split is None:
        raise ValueError("Could not find validation/test split.")

    train_records = dataset_to_records(
        ds=raw[train_split],
        split_name=train_split,
        dataset=args.dataset,
        mode=args.mode,
        image_root=args.image_root,
        image_save_dir=args.image_save_dir,
        require_image=args.require_image,
        data_source=data_source,
        trace_source=args.trace_source,
        trace_tag=args.trace_tag,
        answer_weight=args.answer_weight,
        trace_weight=args.trace_weight,
        max_samples=args.max_train_samples,
    )

    val_records = dataset_to_records(
        ds=raw[val_split],
        split_name=val_split,
        dataset=args.dataset,
        mode=args.mode,
        image_root=args.image_root,
        image_save_dir=args.image_save_dir,
        require_image=args.require_image,
        data_source=data_source,
        trace_source=args.trace_source,
        trace_tag=args.trace_tag,
        answer_weight=args.answer_weight,
        trace_weight=args.trace_weight,
        max_samples=args.max_val_samples,
    )

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    meta_path = out_dir / "meta.json"

    save_records(train_records, train_path)
    save_records(val_records, val_path)

    meta = {
        "dataset": args.dataset,
        "mode": args.mode,
        "dataset_name": dataset_name,
        "dataset_config": args.dataset_config,
        "train_split": train_split,
        "val_split": val_split,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "require_image": args.require_image,
        "image_root": args.image_root,
        "image_save_dir": args.image_save_dir,
        "data_source": data_source,
        "trace_source": args.trace_source,
        "trace_tag": args.trace_tag,
        "answer_weight": args.answer_weight,
        "trace_weight": args.trace_weight,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))

    if train_records:
        print("Example record:")
        print(json.dumps(train_records[0], indent=2)[:5000])


if __name__ == "__main__":
    main()

'''
python data_factory/build_scienceqa_sft.py \
  --dataset_name derek-thomas/ScienceQA \
  --out_dir /mnt/data1/eunwooim/rlpt/data/scienceqa_sft_debug \
  --image_save_dir /mnt/data1/eunwooim/rlpt/data/scienceqa_sft_debug/images \
  --require_image \
  --max_train_samples 200 \
  --max_val_samples 50
'''