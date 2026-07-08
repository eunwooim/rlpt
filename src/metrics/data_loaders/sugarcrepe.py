from __future__ import annotations

import json
import urllib.request
from typing import Any

from .common import Case, LoadError, coerce_text, limit_cases


SUGARCREPE_SUBCATEGORIES = [
    "replace_obj",
    "replace_att",
    "replace_rel",
    "swap_obj",
    "swap_att",
    "add_obj",
    "add_att",
]

SUGARCREPE_URL = "https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data/{subcategory}.json"


def _download_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _iter_records(data: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict):
        return [(str(key), value) for key, value in data.items() if isinstance(value, dict)]
    if isinstance(data, list):
        return [(str(i), value) for i, value in enumerate(data) if isinstance(value, dict)]
    raise TypeError(f"Expected dict or list JSON payload, got {type(data).__name__}")


def load_sugarcrepe(max_cases_per_subcategory: int | None = None) -> tuple[list[Case], list[LoadError]]:
    cases: list[Case] = []
    errors: list[LoadError] = []
    for subcategory in SUGARCREPE_SUBCATEGORIES:
        try:
            data = _download_json(SUGARCREPE_URL.format(subcategory=subcategory))
            subcases: list[Case] = []
            for record_id, row in _iter_records(data):
                positive = coerce_text(row.get("caption"))
                negative = coerce_text(row.get("negative_caption"))
                if not positive or not negative:
                    continue
                subcases.append(
                    Case(
                        case_id=f"sugarcrepe::{subcategory}::{record_id}",
                        dataset="sugarcrepe",
                        subcategory=subcategory,
                        mode="separation",
                        anchor=positive,
                        positive=positive,
                        negatives=[negative],
                        metadata={
                            "source": "RAIVNLab/sugar-crepe",
                            "source_id": record_id,
                            "filename": row.get("filename"),
                        },
                    )
                )
            cases.extend(limit_cases(subcases, max_cases_per_subcategory))
        except Exception as exc:  # noqa: BLE001 - errors are an experiment artifact.
            errors.append(LoadError.from_exception("sugarcrepe", subcategory, exc))
    return cases, errors

