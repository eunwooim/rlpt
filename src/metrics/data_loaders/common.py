from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Case:
    case_id: str
    dataset: str
    subcategory: str
    mode: str
    anchor: str
    positive: str
    negatives: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("metadata", None)
        return data


@dataclass
class LoadError:
    dataset: str
    subcategory: str
    error_type: str
    error_message: str

    @classmethod
    def from_exception(cls, dataset: str, subcategory: str, exc: BaseException) -> "LoadError":
        return cls(
            dataset=dataset,
            subcategory=subcategory,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def first_text(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = coerce_text(row.get(name))
        if value:
            return value
    return ""


def collect_texts(row: dict[str, Any], names: Iterable[str]) -> list[str]:
    texts: list[str] = []
    for name in names:
        value = row.get(name)
        if isinstance(value, list):
            for item in value:
                text = coerce_text(item)
                if text:
                    texts.append(text)
        else:
            text = coerce_text(value)
            if text:
                texts.append(text)
    return texts


def limit_cases(cases: list[Case], max_cases: int | None) -> list[Case]:
    if max_cases is None or max_cases <= 0:
        return cases
    return cases[:max_cases]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

