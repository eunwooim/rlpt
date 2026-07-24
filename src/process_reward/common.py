"""Dependency-light utilities shared by process-reward stages."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_output_dir(stage: str) -> Path:
    return REPO_ROOT / "src" / "outputs" / f"process_reward_{stage}" / f"run_{timestamp()}"


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def print_config(payload: dict[str, Any]) -> None:
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))


def create_stage_layout(output_dir: str | Path, include_checkpoints: bool = False) -> Path:
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")
    names = ["configs", "data", "errors", "logs", "metrics"]
    if include_checkpoints:
        names.append("checkpoints")
    for name in names:
        (path / name).mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {source}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def append_log(output_dir: str | Path, message: str) -> None:
    path = Path(output_dir) / "logs" / "run.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]
