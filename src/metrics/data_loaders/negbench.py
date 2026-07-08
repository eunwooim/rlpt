from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import Case, LoadError, collect_texts, coerce_text, first_text, limit_cases


REQUESTED_SUBCATEGORIES = {
    "MSR-VTT MCQ-Neg": [
        "msr_vtt_mcq_rephrased_llama.csv",
        "msr-vtt_mcq-neg.csv",
    ],
    "HardNeg-Syn MCQ-Neg": [
        "synthetic_mcq_llama3.1_rephrased.csv",
        "hardneg_syn_mcq_neg.csv",
    ],
    "VOC2007 MCQ-Neg": [
        "VOC2007_mcq_llama3.1_rephrased.csv",
        "voc2007_mcq_neg.csv",
    ],
    "COCO-MCQ-Neg": [
        "COCO_val_mcq_llama3.1_rephrased.csv",
        "coco_mcq_neg.csv",
    ],
    "CheXpert control/negation task": [
        "chexpert_binary_mcq.csv",
        "chexpert_binary_mcq_control.csv",
        "chexpert_control_negation.csv",
        "CheXpert_control_negation.csv",
        "chexpert.csv",
    ],
}

SEARCH_ROOTS = [
    Path("data"),
    Path("src/metrics/data/negbench"),
    Path("src/data/negbench"),
    Path("data/negbench"),
    Path("benchmarks/data"),
]

POSITIVE_FIELDS = [
    "positive",
    "positive_caption",
    "affirmative_caption",
    "control_caption",
    "caption",
    "true_caption",
    "answer",
    "correct",
    "option_0",
    "choice_0",
    "A",
]

NEGATIVE_FIELDS = [
    "negative",
    "negative_caption",
    "negated_caption",
    "hard_negative",
    "false_caption",
    "option_1",
    "option_2",
    "option_3",
    "choice_1",
    "choice_2",
    "choice_3",
    "B",
    "C",
    "D",
]


def _candidate_paths(filenames: list[str]) -> list[Path]:
    paths: list[Path] = []
    for root in SEARCH_ROOTS:
        for filename in filenames:
            paths.append(root / filename)
            paths.append(root / "images" / filename)
            paths.append(root / "videos" / filename)
            paths.append(root / "medical" / filename)
    return paths


def _find_file(filenames: list[str]) -> Path:
    for path in _candidate_paths(filenames):
        if path.exists():
            return path
    raise FileNotFoundError(
        "No NegBench CSV found. Checked: "
        + ", ".join(str(path) for path in _candidate_paths(filenames))
    )


def _find_files(filenames: list[str], load_all: bool = False) -> list[Path]:
    found: list[Path] = []
    for path in _candidate_paths(filenames):
        if path.exists() and path not in found:
            found.append(path)
            if not load_all:
                return found
    if found:
        return found
    raise FileNotFoundError(
        "No NegBench CSV found. Checked: "
        + ", ".join(str(path) for path in _candidate_paths(filenames))
    )


def _mcq_case(row: dict[str, Any]) -> tuple[str, list[str]]:
    caption_keys = sorted(
        [key for key in row if key.startswith("caption_")],
        key=lambda key: int(key.rsplit("_", 1)[-1]) if key.rsplit("_", 1)[-1].isdigit() else 999,
    )
    if not caption_keys or "correct_answer" not in row:
        return "", []
    try:
        correct_idx = int(row["correct_answer"])
    except (TypeError, ValueError):
        return "", []
    positive_key = f"caption_{correct_idx}"
    positive = coerce_text(row.get(positive_key))
    negatives = [coerce_text(row.get(key)) for key in caption_keys if key != positive_key]
    negatives = [text for text in negatives if text]
    return positive, negatives


def _row_negatives(row: dict[str, Any]) -> list[str]:
    negatives = collect_texts(row, NEGATIVE_FIELDS)
    for key, value in row.items():
        lower = key.lower()
        if lower.startswith(("negative_", "neg_", "hard_negative_")):
            text = coerce_text(value)
            if text:
                negatives.append(text)
    deduped: list[str] = []
    for text in negatives:
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _load_csv(subcategory: str, path: Path, max_cases_per_subcategory: int | None) -> list[Case]:
    df = pd.read_csv(path)
    subcases: list[Case] = []
    for i, row_obj in df.iterrows():
        row = row_obj.to_dict()
        positive, negatives = _mcq_case(row)
        if not positive or not negatives:
            positive = first_text(row, POSITIVE_FIELDS)
            negatives = _row_negatives(row)
        if not positive or not negatives:
            continue
        case_id = coerce_text(row.get("id")) or coerce_text(row.get("image_id")) or str(i)
        subcases.append(
            Case(
                case_id=f"negbench::{subcategory}::{path.stem}::{case_id}",
                dataset="negbench",
                subcategory=subcategory,
                mode="separation",
                anchor=positive,
                positive=positive,
                negatives=negatives,
                metadata={
                    "source_path": str(path),
                    "source_id": case_id,
                },
            )
        )
    return limit_cases(subcases, max_cases_per_subcategory)


def load_negbench(max_cases_per_subcategory: int | None = None) -> tuple[list[Case], list[LoadError]]:
    cases: list[Case] = []
    errors: list[LoadError] = []
    for subcategory, filenames in REQUESTED_SUBCATEGORIES.items():
        try:
            paths = _find_files(filenames, load_all=subcategory == "CheXpert control/negation task")
            subcases: list[Case] = []
            for path in paths:
                subcases.extend(_load_csv(subcategory, path, None))
            cases.extend(limit_cases(subcases, max_cases_per_subcategory))
        except Exception as exc:  # noqa: BLE001
            errors.append(LoadError.from_exception("negbench", subcategory, exc))
    return cases, errors
