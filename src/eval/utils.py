import json
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformers


MULTIPLE_CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F"]


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_mc_answer(text: str) -> Optional[str]:
    """
    Extract a multiple-choice answer from model output.

    Examples:
        Answer: C
        The answer is C.
        (C)
        <answer>C</answer>
    """

    if text is None:
        return None

    text = text.upper()

    patterns = [
        r"<ANSWER>\s*([A-F])\s*</ANSWER>",
        r"ANSWER\s*[:：]\s*([A-F])",
        r"THE ANSWER IS\s*([A-F])",
        r"\(([A-F])\)",
        r"\b([A-F])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    """
    Extract the final number from a response.
    Useful for MathVista / ChartQA style tasks.
    """

    if text is None:
        return None

    matches = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?",
        text.replace(",", "")
    )

    if len(matches) == 0:
        return None

    return matches[-1]


def extract_free_form_answer(text: str) -> str:
    """
    Generic fallback.
    """

    return normalize_text(text)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    return
