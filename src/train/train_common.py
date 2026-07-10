"""Shared utilities for RLPT training entrypoints."""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import zipfile
from importlib import import_module
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


def ratio(value: float, name: str = "data_ratio") -> float:
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]; got {value}")
    return value


def ensure_output_layout(output_dir: str | Path, force: bool = False) -> Path:
    path = Path(output_dir)
    if path.exists() and force:
        # Do not delete existing work; force means allow writing additional files.
        pass
    path.mkdir(parents=True, exist_ok=True)
    for child in ("configs", "logs", "checkpoints", "metrics", "data"):
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def command_to_text(command: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(command))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def setup_wandb(args: Any, output_dir: str | Path, run_config: dict[str, Any]) -> bool:
    report_to = getattr(args, "report_to", "none")
    if report_to != "wandb":
        return False

    project = getattr(args, "wandb_project", None) or "rlpt_baselines"
    mode = getattr(args, "wandb_mode", None) or os.environ.get("WANDB_MODE", "offline")
    run_name = getattr(args, "wandb_run_name", None) or Path(output_dir).parent.name + "/" + Path(output_dir).name
    group = getattr(args, "wandb_group", None)
    entity = getattr(args, "wandb_entity", None)
    tags_value = getattr(args, "wandb_tags", "")
    tags = [tag.strip() for tag in tags_value.split(",") if tag.strip()]

    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_MODE", mode)
    os.environ.setdefault("WANDB_DIR", str(Path(output_dir) / "logs" / "wandb"))
    os.environ.setdefault("WANDB_NAME", run_name)
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    if entity:
        os.environ.setdefault("WANDB_ENTITY", entity)
    if group:
        os.environ.setdefault("WANDB_RUN_GROUP", group)

    wandb = import_or_raise("wandb", "Install wandb or run with --report_to none.")
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags or None,
        config=to_jsonable(run_config),
        dir=os.environ["WANDB_DIR"],
        mode=mode,
        reinit=True,
    )
    print(f"Initialized W&B run: project={project} mode={mode} name={run_name}", flush=True)
    return True


def import_or_raise(module_name: str, install_hint: str | None = None) -> Any:
    try:
        return import_module(module_name)
    except ImportError as exc:
        hint = install_hint or f"Install or activate an environment with {module_name}."
        raise RuntimeError(f"Missing dependency '{module_name}'. {hint}") from exc


def get_tqdm() -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        class _NoOpTqdm:
            def __init__(self, iterable=None, **_: Any) -> None:
                self.iterable = iterable

            def __iter__(self):
                return iter(self.iterable or [])

            def __enter__(self):
                return self

            def __exit__(self, *_: Any) -> None:
                return None

            def update(self, _: int = 1) -> None:
                return None

        return _NoOpTqdm


def load_hf_dataset(args: Any) -> Any:
    if str(args.dataset_name).startswith("OpenGVLab/VisualPRM400K"):
        return load_visualprm_zip_source(args)

    datasets = import_or_raise(
        "datasets",
        "Activate the pinned RLPT environment before running training.",
    )
    kwargs: dict[str, Any] = {"split": args.dataset_split}
    if getattr(args, "dataset_config", None):
        kwargs["name"] = args.dataset_config
    return datasets.load_dataset(args.dataset_name, **kwargs)


class VisualPRMZipSource:
    def __init__(self, archive_path: Path, dataset_name: str, dataset_config: str | None, split: str) -> None:
        self.archive_path = archive_path
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.annotation_names = self._annotation_names()
        self.total_rows = self._count_rows()

    def _annotation_names(self) -> list[str]:
        with zipfile.ZipFile(self.archive_path) as archive:
            return sorted(name for name in archive.namelist() if name.endswith(".jsonl"))

    def _count_rows(self) -> int:
        total = 0
        tqdm = get_tqdm()
        with zipfile.ZipFile(self.archive_path) as archive:
            iterator = tqdm(self.annotation_names, desc="Counting VisualPRM annotations", unit="file")
            for annotation_name in iterator:
                with archive.open(annotation_name) as f:
                    total += sum(1 for line in f if line.strip())
        return total

    def __len__(self) -> int:
        return self.total_rows

    def select_indices(self, indices: list[int]) -> list[dict[str, Any]]:
        wanted = set(indices)
        rows_by_index: dict[int, dict[str, Any]] = {}
        row_index = 0
        max_index = max(wanted) if wanted else -1
        tqdm = get_tqdm()
        with zipfile.ZipFile(self.archive_path) as archive:
            with tqdm(total=len(wanted), desc="Selecting VisualPRM rows", unit="row") as progress:
                for annotation_name in self.annotation_names:
                    if row_index > max_index:
                        break
                    with archive.open(annotation_name) as f:
                        for raw_line in f:
                            if not raw_line.strip():
                                continue
                            if row_index in wanted:
                                row = json.loads(raw_line.decode("utf-8"))
                                if isinstance(row, dict):
                                    row.setdefault("source", annotation_name)
                                    rows_by_index[row_index] = row
                                    progress.update(1)
                            row_index += 1
                            if len(rows_by_index) == len(wanted):
                                return [rows_by_index[index] for index in indices if index in rows_by_index]
                            if row_index > max_index:
                                break
        return [rows_by_index[index] for index in indices if index in rows_by_index]


def load_visualprm_zip_source(args: Any) -> VisualPRMZipSource:
    if getattr(args, "dataset_split", "train") != "train":
        raise ValueError(f"{args.dataset_name} only exposes the train split, got: {args.dataset_split}")
    if getattr(args, "dataset_config", None) not in {None, "default"}:
        raise ValueError(f"{args.dataset_name} only exposes the default config, got: {args.dataset_config}")

    download_module = import_or_raise(
        "datasets.download.download_manager",
        "Activate the pinned RLPT environment before running training.",
    )
    url = f"https://huggingface.co/datasets/{args.dataset_name}/resolve/main/annotations.zip"
    print(f"Loading VisualPRM annotations from {url}", flush=True)
    archive_path = Path(download_module.DownloadManager(dataset_name=args.dataset_name).download(url))
    return VisualPRMZipSource(
        archive_path=archive_path,
        dataset_name=args.dataset_name,
        dataset_config=getattr(args, "dataset_config", None),
        split=getattr(args, "dataset_split", "train"),
    )


def download_visualprm_repo_file(dataset_name: str, filename: str) -> Path:
    download_module = import_or_raise(
        "datasets.download.download_manager",
        "Activate the pinned RLPT environment before running training.",
    )
    url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{filename}"
    print(f"Downloading/resolving VisualPRM asset: {url}", flush=True)
    return Path(download_module.DownloadManager(dataset_name=dataset_name).download(url))


def zip_uri(archive_path: str | Path, member_name: str) -> str:
    return f"zip://{archive_path}::{member_name}"


def row_image_values(row: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in ("image", "images"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            values.extend(item for item in value if item is not None)
        else:
            values.append(value)
    return values


def normalize_visualprm_image_member(image_name: str) -> str:
    candidates = [image_name]
    prefix = "VisualPRM400K-v1.1-Raw/"
    if image_name.startswith(prefix):
        stripped = image_name[len(prefix) :]
        candidates.append(stripped)
        candidates.append("images/" + stripped)
    if not image_name.startswith("images/"):
        candidates.append("images/" + image_name)
    return candidates[-1]


def resolve_zip_member(archive: zipfile.ZipFile, image_name: str, dataset_name: str) -> str | None:
    candidates = [image_name]
    if str(dataset_name).startswith("OpenGVLab/VisualPRM400K"):
        prefix = "VisualPRM400K-v1.1-Raw/"
        if image_name.startswith(prefix):
            stripped = image_name[len(prefix) :]
            candidates.extend([stripped, "images/" + stripped])
        if not image_name.startswith("images/"):
            candidates.append("images/" + image_name)
    elif str(dataset_name) == "Xkev/LLaVA-CoT-100k":
        candidates.extend([image_name.lstrip("/"), "images/" + image_name.lstrip("/")])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            archive.getinfo(candidate)
            return candidate
        except KeyError:
            continue
    return None


def materialize_visualprm_images(rows: list[dict[str, Any]], args: Any, output_dir: str | Path) -> int:
    if not str(args.dataset_name).startswith("OpenGVLab/VisualPRM400K"):
        return 0

    image_names = sorted(
        {
            image
            for row in rows
            for image in row_image_values(row)
            if isinstance(image, str)
            if image
        }
    )
    if not image_names:
        return 0

    print(f"Materializing {len(image_names)} selected VisualPRM images", flush=True)
    archive_path = download_visualprm_repo_file(args.dataset_name, "images.zip")
    target_root = Path(output_dir) / "data" / "images"
    target_root.mkdir(parents=True, exist_ok=True)

    materialized: dict[str, str] = {}
    tqdm = get_tqdm()
    with zipfile.ZipFile(archive_path) as archive:
        for image_name in tqdm(image_names, desc="Extracting VisualPRM images", unit="image"):
            member_name = resolve_zip_member(archive, image_name, args.dataset_name)
            if member_name is None:
                raise FileNotFoundError(f"VisualPRM image not found in images.zip: {image_name}")
            target = target_root / member_name
            target.resolve().relative_to(target_root.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with archive.open(member_name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            materialized[image_name] = str(target)

    for row in rows:
        materialized_images = [materialized[image] for image in row_image_values(row) if isinstance(image, str) and image in materialized]
        if materialized_images:
            row["image"] = materialized_images[0]
            row.pop("images", None)
    return len(materialized)


def attach_visualprm_image_zip(rows: list[dict[str, Any]], args: Any) -> int:
    if not str(args.dataset_name).startswith("OpenGVLab/VisualPRM400K"):
        return 0

    image_names = sorted(
        {
            image
            for row in rows
            for image in row_image_values(row)
            if isinstance(image, str)
            if image
        }
    )
    if not image_names:
        return 0

    print(f"Resolving VisualPRM images.zip for lazy image loading ({len(image_names)} selected images)", flush=True)
    archive_path = download_visualprm_repo_file(args.dataset_name, "images.zip")
    resolved: dict[str, str] = {}
    missing = 0
    with zipfile.ZipFile(archive_path) as archive:
        tqdm = get_tqdm()
        for image_name in tqdm(image_names, desc="Resolving VisualPRM image members", unit="image"):
            member_name = resolve_zip_member(archive, image_name, args.dataset_name)
            if member_name is None:
                if getattr(args, "image_mode", "auto") == "strict":
                    raise FileNotFoundError(f"VisualPRM image not found in images.zip: {image_name}")
                missing += 1
                continue
            resolved[image_name] = member_name
    if missing:
        print(f"Dropped {missing} VisualPRM image paths that were missing from images.zip", flush=True)

    for row in rows:
        attached_images = [
            zip_uri(archive_path, resolved[image])
            for image in row_image_values(row)
            if isinstance(image, str) and image in resolved
        ]
        if attached_images:
            row["image"] = attached_images[0]
            if len(attached_images) > 1:
                row["images"] = attached_images
            else:
                row.pop("images", None)
        else:
            row.pop("image", None)
            row.pop("images", None)
    return len(resolved)


def dataset_to_list(dataset: Any) -> list[dict[str, Any]]:
    if isinstance(dataset, list):
        return dataset
    if hasattr(dataset, "to_list"):
        return dataset.to_list()
    return [dataset[index] for index in range(len(dataset))]


def resolve_existing_zip(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.exists():
        return path
    return None


def attach_llava_cot_image_zip(rows: list[dict[str, Any]], args: Any) -> int:
    image_names = sorted(
        {
            image
            for row in rows
            for image in row_image_values(row)
            if isinstance(image, str)
            if image
        }
    )
    if not image_names:
        return 0

    archive_path = resolve_existing_zip(getattr(args, "image_zip", None) or os.environ.get("LLAVA_COT_IMAGE_ZIP"))
    if archive_path is None:
        print(
            "LLaVA-CoT image archive is not configured. Dropping image paths for text-only SFT. "
            "Set LLAVA_COT_IMAGE_ZIP or --image_zip to train with images.",
            flush=True,
        )
        for row in rows:
            row.pop("image", None)
            row.pop("images", None)
        return 0

    print(f"Resolving LLaVA-CoT image archive: {archive_path}", flush=True)
    resolved: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        tqdm = get_tqdm()
        for image_name in tqdm(image_names, desc="Resolving LLaVA-CoT image members", unit="image"):
            member_name = resolve_zip_member(archive, image_name, args.dataset_name)
            if member_name is None:
                if getattr(args, "image_mode", "auto") == "strict":
                    raise FileNotFoundError(f"LLaVA-CoT image not found in archive: {image_name}")
                continue
            resolved[image_name] = member_name

    missing = 0
    for row in rows:
        attached_images = [
            zip_uri(archive_path, resolved[image])
            for image in row_image_values(row)
            if isinstance(image, str) and image in resolved
        ]
        if attached_images:
            row["image"] = attached_images[0]
            if len(attached_images) > 1:
                row["images"] = attached_images
            else:
                row.pop("images", None)
        elif any(isinstance(image, str) for image in row_image_values(row)):
            row.pop("image", None)
            row.pop("images", None)
            missing += 1
    if missing:
        print(f"Dropped {missing} LLaVA-CoT image paths that were missing from the archive", flush=True)
    return len(resolved)


def prepare_images_for_sft(dataset: Any, args: Any) -> tuple[Any, dict[str, Any]]:
    dataset_name = str(args.dataset_name)
    image_mode = getattr(args, "image_mode", "auto")
    if image_mode == "text_only":
        rows = dataset_to_list(dataset)
        for row in rows:
            row.pop("image", None)
            row.pop("images", None)
        return rows, {"image_mode": image_mode, "attached_images": 0, "dropped_images": len(rows)}

    if dataset_name.startswith("OpenGVLab/VisualPRM400K"):
        rows = dataset_to_list(dataset)
        attached = attach_visualprm_image_zip(rows, args)
        return rows, {"image_mode": image_mode, "attached_images": attached, "image_archive": "visualprm_images_zip"}

    if dataset_name == "Xkev/LLaVA-CoT-100k":
        rows = dataset_to_list(dataset)
        attached = attach_llava_cot_image_zip(rows, args)
        return rows, {
            "image_mode": image_mode,
            "attached_images": attached,
            "image_archive": str(resolve_existing_zip(getattr(args, "image_zip", None) or os.environ.get("LLAVA_COT_IMAGE_ZIP")) or ""),
        }

    return dataset, {"image_mode": image_mode, "attached_images": 0}


def scaled_dataset(dataset: Any, data_ratio: float, seed: int, max_train_samples: int | None) -> Any:
    ratio(data_ratio)
    if isinstance(dataset, VisualPRMZipSource):
        total = len(dataset)
        if total == 0:
            raise ValueError("Loaded VisualPRM annotations are empty.")
        ratio_count = max(1, int(math.floor(total * data_ratio)))
        count = min(ratio_count, max_train_samples) if max_train_samples else ratio_count
        indices = random.Random(seed).sample(range(total), count)
        return dataset.select_indices(indices)

    if not hasattr(dataset, "__len__"):
        if max_train_samples is None:
            raise ValueError(
                "Iterable datasets require --max_train_samples because --data_ratio "
                "cannot be applied without a known length."
            )
        return dataset.take(max_train_samples)

    total = len(dataset)
    if total == 0:
        raise ValueError("Loaded dataset is empty.")
    ratio_count = max(1, int(math.floor(total * data_ratio)))
    count = min(ratio_count, max_train_samples) if max_train_samples else ratio_count
    return dataset.shuffle(seed=seed).select(range(count))


def infer_max_steps(
    token_budget: int,
    max_seq_len: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int | None = None,
) -> int:
    positive_int(token_budget, "token_budget")
    positive_int(max_seq_len, "max_seq_len")
    positive_int(per_device_train_batch_size, "per_device_train_batch_size")
    positive_int(gradient_accumulation_steps, "gradient_accumulation_steps")
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = max_seq_len * per_device_train_batch_size * gradient_accumulation_steps * max(1, world_size)
    return max(1, int(math.ceil(token_budget / tokens_per_step)))


def first_string(example: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def normalize_messages(messages: Any) -> tuple[str, str] | None:
    if not isinstance(messages, list):
        return None
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        content = message.get("content", message.get("value", ""))
        text = content_to_text(content)
        if not text:
            continue
        if role in {"assistant", "gpt", "model"}:
            assistant_parts.append(text)
        else:
            user_parts.append(text)
    if not assistant_parts:
        return None
    return "\n".join(user_parts).strip(), assistant_parts[-1].strip()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(stringify(item.get("text")))
                elif "text" in item:
                    parts.append(stringify(item.get("text")))
            else:
                parts.append(stringify(item))
        return "\n".join(part for part in parts if part).strip()
    return stringify(content)


def normalize_supervised_example(example: dict[str, Any]) -> dict[str, Any]:
    for key in ("messages", "conversations", "conversation"):
        parsed = normalize_messages(example.get(key))
        if parsed:
            prompt, response = parsed
            return {"prompt": prompt, "response": response, "images": extract_images(example)}

    prompt = first_string(
        example,
        (
            "prompt",
            "question",
            "query",
            "instruction",
            "problem",
            "input",
            "user",
        ),
    )
    response = first_string(
        example,
        (
            "response",
            "answer",
            "output",
            "solution",
            "chosen",
            "assistant",
            "label",
        ),
    )
    if prompt is None or response is None:
        raise ValueError(f"Could not infer prompt/response fields from keys: {sorted(example.keys())}")
    return {"prompt": prompt, "response": response, "images": extract_images(example)}


def extract_images(example: dict[str, Any]) -> list[Any]:
    images: list[Any] = []
    for key in ("image", "images", "pixel_values"):
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            images.extend([item for item in value if item is not None])
        else:
            images.append(value)
    deduped: list[Any] = []
    seen: set[str] = set()
    for image in images:
        key = repr(image)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(image)
    return deduped


def build_user_prompt(prompt: str, images: list[Any] | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images or []:
        content.append({"type": "image", "image": image})
    prompt = re.sub(r"^\s*<image>\s*", "", prompt, count=1)
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_sft_messages(prompt: str, response: str, images: list[Any] | None = None) -> list[dict[str, Any]]:
    messages = build_user_prompt(prompt, images)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


def extract_answer(text: str) -> str:
    text = text.strip()
    tag_patterns = (
        r"<answer>\s*(.*?)\s*</answer>",
        r"<final>\s*(.*?)\s*</final>",
        r"<conclusion>\s*(.*?)\s*</conclusion>",
        r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>",
    )
    for pattern in tag_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    return text.strip()


def normalize_answer(text: str) -> str:
    answer = extract_answer(text)
    answer = answer.lower()
    answer = re.sub(r"[^a-z0-9.\-]+", " ", answer)
    return re.sub(r"\s+", " ", answer).strip()
