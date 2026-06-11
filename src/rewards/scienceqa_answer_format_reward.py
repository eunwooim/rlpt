#!/usr/bin/env python3

import re
from typing import Any, Dict, Optional


CHOICE_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def _extract_answer(text: str, answer_tag: str = "answer") -> str:
    text = _normalize_text(text)

    tagged = _extract_tag(text, answer_tag)
    if tagged:
        m = re.search(r"\b([A-Za-z])\b", tagged)
        if m:
            return m.group(1).upper()

    m = re.search(r"(?:answer|final answer)\s*[:：]\s*([A-Za-z])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.fullmatch(r"\s*\(?\s*([A-Za-z])\s*\)?\s*", text)
    if m:
        return m.group(1).upper()

    return ""


def _get_gt_answer(ground_truth: Any) -> str:
    if isinstance(ground_truth, dict):
        answer = ground_truth.get("answer", "")
    else:
        answer = ground_truth

    answer = _normalize_text(answer).upper()
    if len(answer) == 1 and answer in CHOICE_LABELS:
        return answer
    return answer


def _format_score(text: str, answer_tag: str, trace_tag: str) -> float:
    text = _normalize_text(text)

    trace = _extract_tag(text, trace_tag)
    answer = _extract_tag(text, answer_tag)
    pred = _extract_answer(text, answer_tag)

    if not trace:
        return 0.0
    if not answer:
        return 0.0
    if len(pred) != 1 or pred not in CHOICE_LABELS:
        return 0.0

    # Penalize multiple final answer tags because that is easy reward hacking.
    n_answer_tags = len(re.findall(rf"<{answer_tag}>", text, flags=re.IGNORECASE))
    if n_answer_tags != 1:
        return 0.0

    return 1.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    """
    ScienceQA vanilla RLVR reward.

    Allows free reasoning inside <think>...</think>, but rewards only:
      1. final multiple-choice answer correctness
      2. strict output format

    Expected output:
      <think>...</think>
      <answer>A</answer>
    """
    if isinstance(ground_truth, dict):
        reward_config = ground_truth.get("reward_config", {})
    else:
        reward_config = {}

    answer_tag = reward_config.get("answer_tag", "answer")
    trace_tag = reward_config.get("trace_tag", "think")

    answer_weight = float(reward_config.get("answer_weight", 0.9))
    format_weight = float(reward_config.get("format_weight", 0.1))

    pred = _extract_answer(solution_str, answer_tag=answer_tag)
    gt = _get_gt_answer(ground_truth)

    answer_score = 1.0 if pred and gt and pred == gt else 0.0
    fmt_score = _format_score(solution_str, answer_tag=answer_tag, trace_tag=trace_tag)

    denom = answer_weight + format_weight
    if denom <= 0:
        return answer_score

    return (answer_weight * answer_score + format_weight * fmt_score) / denom


# Some veRL reward managers look for `score`.
score = compute_score
