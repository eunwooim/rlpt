"""Rule-based baseline rewards for RLVR/GRPO.

This file is intentionally limited to accuracy and format rewards. It is used by
veRL via custom_reward_function.path and custom_reward_function.name.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any


def _extract_answer(text: str) -> str:
    text = str(text or "").strip()
    patterns = (
        r"<answer>\s*(.*?)\s*</answer>",
        r"<final>\s*(.*?)\s*</final>",
        r"<conclusion>\s*(.*?)\s*</conclusion>",
        r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    return text


def _normalize(text: str) -> str:
    answer = _extract_answer(text).lower()
    answer = re.sub(r"[^a-z0-9.\-]+", " ", answer)
    return re.sub(r"\s+", " ", answer).strip()


def _number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", _extract_answer(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def accuracy_reward(solution_str: str, ground_truth: Any) -> float:
    prediction = _normalize(solution_str)
    target = _normalize(str(ground_truth or ""))
    if not prediction or not target:
        return 0.0
    if prediction == target:
        return 1.0

    pred_num = _number(solution_str)
    target_num = _number(str(ground_truth or ""))
    if pred_num is not None and target_num is not None:
        tolerance = max(1e-6, abs(target_num) * 1e-4)
        return 1.0 if math.isclose(pred_num, target_num, rel_tol=1e-4, abs_tol=tolerance) else 0.0
    return 0.0


def format_reward(solution_str: str) -> float:
    text = str(solution_str or "").strip()
    if not text:
        return 0.0
    structured_patterns = (
        r"<answer>.*?</answer>",
        r"<final>.*?</final>",
        r"<conclusion>.*?</conclusion>",
        r"<CONCLUSION>.*?</CONCLUSION>",
        r"####\s*\S+",
    )
    if any(re.search(pattern, text, flags=re.DOTALL) for pattern in structured_patterns):
        return 1.0
    return 0.5


def _components(extra_info: Any = None) -> list[str]:
    if isinstance(extra_info, dict):
        value = extra_info.get("reward_components")
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(part).strip() for part in value if str(part).strip()]
    value = os.environ.get("RLPT_REWARD_COMPONENTS", "accuracy,format")
    return [part.strip() for part in value.split(",") if part.strip()]


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: Any = None) -> float:
    """veRL-compatible reward function.

    Signature follows veRL's custom reward function contract:
    (data_source, solution_str, ground_truth, extra_info) -> float.
    """

    scores: list[float] = []
    for component in _components(extra_info):
        if component == "accuracy":
            scores.append(accuracy_reward(solution_str, ground_truth))
        elif component == "format":
            scores.append(format_reward(solution_str))
        else:
            raise ValueError(f"Unsupported baseline reward component: {component}")
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
