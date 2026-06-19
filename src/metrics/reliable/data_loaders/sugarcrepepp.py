from __future__ import annotations

from typing import Any

from datasets import get_dataset_config_names, load_dataset

from .common import Case, LoadError, coerce_text, limit_cases


DATASET_ID = "Aman-J/SugarCrepe_pp"
CONFIG_TO_SUBCATEGORY = {
    "swap_object": "swap_object",
    "swap_atribute": "swap_atribute",
    "replace_object": "replace_object",
    "replace_attribute": "replace_attribute",
    "replace_relation": "replace_relation",
}


def _load_config(config: str) -> Any:
    return load_dataset(DATASET_ID, config, split="train")


def load_sugarcrepepp(max_cases_per_subcategory: int | None = None) -> tuple[list[Case], list[LoadError]]:
    cases: list[Case] = []
    errors: list[LoadError] = []
    try:
        configs = get_dataset_config_names(DATASET_ID)
    except Exception as exc:  # noqa: BLE001
        return [], [LoadError.from_exception("sugarcrepepp", "__configs__", exc)]

    for config in configs:
        subcategory = CONFIG_TO_SUBCATEGORY.get(config, config)
        try:
            ds = _load_config(config)
            subcases: list[Case] = []
            for i, row in enumerate(ds):
                positive_1 = coerce_text(row.get("caption"))
                positive_2 = coerce_text(row.get("caption2"))
                negative = coerce_text(row.get("negative_caption"))
                if not positive_1 or not positive_2 or not negative:
                    continue
                row_id = coerce_text(row.get("id")) or str(i)
                subcases.append(
                    Case(
                        case_id=f"sugarcrepepp::{subcategory}::{row_id}",
                        dataset="sugarcrepepp",
                        subcategory=subcategory,
                        mode="ranking",
                        anchor=positive_1,
                        positive=positive_2,
                        negatives=[negative],
                        metadata={
                            "source": DATASET_ID,
                            "source_config": config,
                            "source_id": row_id,
                            "filename": row.get("filename"),
                        },
                    )
                )
            cases.extend(limit_cases(subcases, max_cases_per_subcategory))
        except Exception as exc:  # noqa: BLE001
            errors.append(LoadError.from_exception("sugarcrepepp", subcategory, exc))
    return cases, errors

