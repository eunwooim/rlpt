#!/usr/bin/env python
"""Generate positive paraphrases and hard negatives from canonical steps.

Requests are streamed from the canonical JSONL. SQLite is the durable source of
truth while a run is active; the traditional JSONL files are atomic snapshots
materialized at clean shutdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.common import (
    append_log,
    create_stage_layout,
    default_output_dir,
    iter_jsonl,
    print_config,
    write_json,
)


BACKEND_DEFAULTS = {"vllm": "Qwen/Qwen3-32B", "openai": "gpt-4o-mini"}
GENERATION_TYPES = ("positive_paraphrase", "hard_negative")
OUTPUT_PROFILES = ("train", "audit", "debug")


@dataclass(frozen=True)
class GenerationRequest:
    pair_id: str
    anchor: str
    context_steps: list[str]
    compatibility_label: int
    generation_type: str
    source_sample_id: str
    step_index: int
    generation_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical_jsonl", required=True)
    parser.add_argument("--backend", choices=sorted(BACKEND_DEFAULTS), default="vllm")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--prompt_version", default="v2", choices=("v1", "v2"))
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--in_context_pool_json", default=None)
    parser.add_argument("--context_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--deberta_tokenizer_name", default="microsoft/deberta-v3-large")
    parser.add_argument("--max_deberta_anchor_tokens", type=int, default=224)
    parser.add_argument("--max_deberta_pair_tokens", type=int, default=512)
    parser.add_argument("--min_output_tokens", type=int, default=64)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--generation_batch_size", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--output_profile", choices=OUTPUT_PROFILES, default="train")
    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Deprecated compatibility alias: save raw generations as in debug profile.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--config_only", action="store_true")
    return parser.parse_args()


def validate_canonical(row: dict[str, Any], line_index: int) -> None:
    required = {"question", "images", "steps", "step_labels", "answer", "metadata"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Canonical row {line_index} is missing fields: {missing}")
    if not isinstance(row["question"], str) or not isinstance(row["steps"], list):
        raise TypeError(f"Canonical row {line_index} has invalid question or steps")
    labels = row["step_labels"]
    if not isinstance(labels, list) or len(row["steps"]) != len(labels):
        raise ValueError(f"Canonical row {line_index} has misaligned steps and step_labels")
    if any(not isinstance(step, str) or not step.strip() for step in row["steps"]):
        raise ValueError(f"Canonical row {line_index} contains an empty or non-string step")
    if any(label not in {-1, 0, 1} for label in labels):
        raise ValueError(f"Canonical row {line_index} contains a label outside -1, 0, 1")
    metadata = row["metadata"]
    if not isinstance(metadata, dict) or not str(metadata.get("source_sample_id", "")).strip():
        raise ValueError(f"Canonical row {line_index} is missing metadata.source_sample_id")


def _request_seed(seed: int, source_sample_id: str, step_index: int, generation_type: str) -> int:
    key = f"{seed}:{source_sample_id}:{step_index}:{generation_type}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def iter_requests(
    canonical_jsonl: str,
    seed: int,
    context_steps: int,
    completed_pair_ids: set[str] | None = None,
    on_discovered: Callable[[str, int], None] | None = None,
) -> Iterator[GenerationRequest]:
    """Validate lazily and yield only work which has not already completed."""
    completed = completed_pair_ids or set()
    for row_index, row in enumerate(iter_jsonl(canonical_jsonl), start=1):
        validate_canonical(row, row_index)
        source_sample_id = str(row["metadata"]["source_sample_id"])
        for step_index, (step, label) in enumerate(zip(row["steps"], row["step_labels"])):
            if label == 0:
                continue
            generation_type = "positive_paraphrase" if label == 1 else "hard_negative"
            pair_key = f"{source_sample_id}:{step_index}:{generation_type}"
            pair_id = "pair_" + hashlib.sha256(pair_key.encode()).hexdigest()[:20]
            if on_discovered is not None:
                on_discovered(pair_id, row_index)
            if pair_id in completed:
                continue
            yield GenerationRequest(
                pair_id=pair_id,
                anchor=step.strip(),
                context_steps=[
                    item.strip()
                    for item in row["steps"][max(0, step_index - context_steps) : step_index]
                ],
                compatibility_label=1 if label == 1 else 0,
                generation_type=generation_type,
                source_sample_id=source_sample_id,
                step_index=step_index,
                generation_seed=_request_seed(seed, source_sample_id, step_index, generation_type),
            )


def build_requests(rows: list[dict[str, Any]], seed: int, context_steps: int) -> list[GenerationRequest]:
    """Compatibility helper for callers; main generation intentionally does not use it."""
    requests: list[GenerationRequest] = []
    for row_index, row in enumerate(rows, start=1):
        validate_canonical(row, row_index)
        source_sample_id = str(row["metadata"]["source_sample_id"])
        for step_index, (step, label) in enumerate(zip(row["steps"], row["step_labels"])):
            if label == 0:
                continue
            kind = "positive_paraphrase" if label == 1 else "hard_negative"
            pair_id = "pair_" + hashlib.sha256(
                f"{source_sample_id}:{step_index}:{kind}".encode()
            ).hexdigest()[:20]
            requests.append(GenerationRequest(
                pair_id, step.strip(),
                [x.strip() for x in row["steps"][max(0, step_index-context_steps):step_index]],
                1 if label == 1 else 0, kind, source_sample_id, step_index,
                _request_seed(seed, source_sample_id, step_index, kind),
            ))
    return requests


def batched(items: Iterable[GenerationRequest], size: int) -> Iterator[list[GenerationRequest]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def load_in_context_pools(path: str | None) -> dict[str, list[dict[str, Any]]]:
    pools = {kind: [] for kind in GENERATION_TYPES}
    if path is None:
        return pools
    source = Path(path)
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"In-context pool JSON does not exist: {source}") from exc
    if not isinstance(records, list):
        raise ValueError("In-context pool JSON must contain a top-level array")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"In-context example at index {index} must be a JSON object")
        missing = [field for field in ("generation_type", "anchor", "generated") if field not in record]
        if missing:
            raise ValueError(f"In-context example at index {index} is missing fields: {missing}")
        kind = record["generation_type"]
        if kind not in GENERATION_TYPES:
            raise ValueError(f"Unsupported generation_type at index {index}: {kind!r}")
        if not isinstance(record["anchor"], str) or not record["anchor"].strip():
            raise ValueError(f"In-context example at index {index} has an invalid anchor")
        if not isinstance(record["generated"], str) or not record["generated"].strip():
            raise ValueError(f"In-context example at index {index} has invalid generated text")
        explicit_id = record.get("id", record.get("example_id"))
        example_id = explicit_id if isinstance(explicit_id, (str, int)) and str(explicit_id) else index
        pools[kind].append({
            "example_id": example_id, "pool_index": index, "generation_type": kind,
            "anchor": record["anchor"].strip(), "generated": record["generated"].strip(),
        })
    return pools


def output_token_budget(tokenizer: Any, anchor: str, max_model_len: int,
                        prompt_tokens: int, hard_cap: int = 256) -> int:
    anchor_tokens = len(tokenizer.encode(anchor, add_special_tokens=False))
    requested = max(64, int(anchor_tokens * 1.10) + 16)
    return max(0, min(requested, max_model_len - prompt_tokens, hard_cap))


def _few_shots(request: GenerationRequest, num_shots: int,
               pools: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    pool = pools[request.generation_type]
    if num_shots > len(pool):
        raise ValueError(f"Requested {num_shots} {request.generation_type} examples, but pool has {len(pool)}")
    return random.Random(request.generation_seed).sample(pool, num_shots)


def generation_prompt(request: GenerationRequest, shots: list[dict[str, Any]],
                      prompt_version: str) -> str:
    if prompt_version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported prompt version: {prompt_version}")
    if request.generation_type == "positive_paraphrase":
        instruction = (
            "Paraphrase the anchor faithfully. Preserve every claim, entity, value, relation, "
            "equation, inference, and conclusion. Keep similar detail, order, structure, and length; "
            "do not summarize, correct, add, or omit reasoning."
        )
    else:
        instruction = (
            "Create a fluent version that disagrees with the anchor through exactly one underlying "
            "semantic change. Preserve all unrelated content and keep similar detail, order, structure, "
            "and length. Update only direct consequences of that change. Do not explain the alteration."
        )
    examples = ""
    if shots:
        examples = "\n\nExamples:\n" + "\n\n".join(
            f"Anchor:\n{s['anchor']}\nGenerated:\n{s['generated']}" for s in shots
        )
    return f'{instruction}\nReturn only {{"generated_segment":"..."}}.{examples}\n\nAnchor:\n{request.anchor}'


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_generated_segment(text: str) -> str:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        segment = value.get("generated_segment") if isinstance(value, dict) else None
        if isinstance(segment, str) and segment.strip():
            return segment.strip()
    raise ValueError("Generation did not contain a non-empty generated_segment JSON field")


def _profile_record(request: GenerationRequest, args: argparse.Namespace, model_name: str,
                    prompt: str, shots: list[dict[str, Any]], *, generated: str | None = None,
                    metadata: dict[str, Any] | None = None, raw_output: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "pair_id": request.pair_id, "source_sample_id": request.source_sample_id,
        "step_index": request.step_index, "compatibility_label": request.compatibility_label,
        "anchor": request.anchor,
    }
    if generated is not None:
        record["generated"] = generated
    if args.output_profile in {"audit", "debug"}:
        record.update({
            "generation_type": request.generation_type, "generation_seed": request.generation_seed,
            "in_context_example_ids": [shot["example_id"] for shot in shots],
        })
        record.update(metadata or {})
    if args.output_profile == "debug":
        record.update({
            "generation_backend": args.backend, "generation_model": model_name,
            "prompt_version": args.prompt_version, "prompt": prompt,
        })
        if raw_output is not None:
            record["raw_output"] = raw_output
    return record


def _error_record(request: GenerationRequest, error_type: str, message: str,
                  details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"pair_id": request.pair_id, "error_type": error_type,
            "error_message": message, **(details or {})}


def _profile_error(request: GenerationRequest, args: argparse.Namespace, model_name: str,
                   prompt: str, shots: list[dict[str, Any]], error_type: str, message: str,
                   details: dict[str, Any] | None = None, raw_output: str | None = None) -> dict[str, Any]:
    error = _error_record(request, error_type, message, details)
    if args.output_profile in {"audit", "debug"}:
        error.update(_profile_record(request, args, model_name, prompt, shots,
                                     metadata=details, raw_output=raw_output))
    return error


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


class GenerationState:
    """Transactional, deduplicated persistence with compatibility JSONL snapshots."""

    def __init__(self, output_dir: Path, checkpoint_every: int, resume: bool,
                 save_raw: bool = False) -> None:
        self.output_dir = output_dir
        self.checkpoint_every = checkpoint_every
        self.save_raw = save_raw
        self.db_path = output_dir / "generation_state.sqlite3"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS results (pair_id TEXT PRIMARY KEY, status TEXT NOT NULL, "
            "pair_json TEXT, raw_json TEXT, error_json TEXT, updated_at TEXT NOT NULL)"
        )
        self._import_legacy_jsonl_if_needed(resume)
        self.completed_pair_ids = {row[0] for row in self.connection.execute("SELECT pair_id FROM results")}
        success, skipped = self.connection.execute(
            "SELECT SUM(status='ok'), SUM(status='error') FROM results"
        ).fetchone()
        self.success_count = int(success or 0)
        self.skipped_count = int(skipped or 0)
        self.pending: list[tuple[Any, ...]] = []
        self.progress: Any = None
        self.discovered_pair_ids: set[str] = set()
        self.canonical_samples = 0

    def _import_legacy_jsonl_if_needed(self, resume: bool) -> None:
        if not resume or self.connection.execute("SELECT 1 FROM results LIMIT 1").fetchone():
            return
        pairs = {str(row["pair_id"]): row for row in iter_jsonl(self.output_dir / "data" / "pairs.jsonl")} \
            if (self.output_dir / "data" / "pairs.jsonl").is_file() else {}
        errors = {str(row["pair_id"]): row for row in iter_jsonl(
            self.output_dir / "errors" / "generation_errors.jsonl")} \
            if (self.output_dir / "errors" / "generation_errors.jsonl").is_file() else {}
        raw = {str(row["pair_id"]): row for row in iter_jsonl(
            self.output_dir / "data" / "raw_generations.jsonl")} \
            if (self.output_dir / "data" / "raw_generations.jsonl").is_file() else {}
        now = datetime.now().astimezone().isoformat()
        for pair_id in pairs.keys() | errors.keys():
            self.connection.execute(
                "INSERT OR IGNORE INTO results VALUES (?, ?, ?, ?, ?, ?)",
                (pair_id, "ok" if pair_id in pairs else "error",
                 json.dumps(pairs.get(pair_id)) if pair_id in pairs else None,
                 json.dumps(raw.get(pair_id)) if pair_id in raw else None,
                 json.dumps(errors.get(pair_id)) if pair_id in errors else None, now),
            )
        self.connection.commit()

    def note_discovered(self, pair_id: str, row_index: int) -> None:
        self.discovered_pair_ids.add(pair_id)
        self.canonical_samples = max(self.canonical_samples, row_index)

    def record(self, request: GenerationRequest, *, pair: dict[str, Any] | None = None,
               raw: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> None:
        if (pair is None) == (error is None):
            raise ValueError("Exactly one of pair or error must be supplied")
        now = datetime.now().astimezone().isoformat()
        self.pending.append((request.pair_id, "ok" if pair is not None else "error",
                             json.dumps(pair, ensure_ascii=False) if pair is not None else None,
                             json.dumps(raw, ensure_ascii=False) if raw is not None else None,
                             json.dumps(error, ensure_ascii=False) if error is not None else None, now))
        self.completed_pair_ids.add(request.pair_id)
        if pair is not None:
            self.success_count += 1
        else:
            self.skipped_count += 1
        if self.progress is not None:
            self.progress.set_postfix(
                success=self.success_count,
                skipped=self.skipped_count,
                refresh=False,
            )
            self.progress.update(1)
        if len(self.pending) >= self.checkpoint_every:
            self.checkpoint()

    def checkpoint(self, export: bool = False) -> None:
        if self.pending:
            with self.connection:
                self.connection.executemany(
                    "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(pair_id) DO UPDATE SET status=excluded.status, "
                    "pair_json=excluded.pair_json, raw_json=excluded.raw_json, "
                    "error_json=excluded.error_json, updated_at=excluded.updated_at", self.pending)
            self.pending.clear()
        counts = self.counts()
        _atomic_write_json(self.output_dir / "configs" / "generation_checkpoint.json", {
            "completed_requests": counts["completed_requests"],
            "generated_pairs": counts["generated_pairs"],
            "generation_errors": counts["generation_errors"],
            "updated_at": datetime.now().astimezone().isoformat(),
        })
        if export:
            self.export_jsonl()

    def counts(self) -> dict[str, int]:
        ok, errors = self.connection.execute(
            "SELECT SUM(status='ok'), SUM(status='error') FROM results"
        ).fetchone()
        ok, errors = int(ok or 0), int(errors or 0)
        positive = negative = prompt_long = 0
        for pair_json, error_json in self.connection.execute("SELECT pair_json, error_json FROM results"):
            if pair_json:
                label = json.loads(pair_json).get("compatibility_label")
                positive += label == 1
                negative += label == 0
            if error_json and json.loads(error_json).get("error_type") in {"PromptTooLong", "QwenContextOverflow"}:
                prompt_long += 1
        return {"completed_requests": ok + errors, "generated_pairs": ok,
                "generation_errors": errors, "positive_pairs": positive,
                "negative_pairs": negative, "prompt_too_long_skipped": prompt_long}

    def export_jsonl(self) -> None:
        targets = [
            (self.output_dir / "data" / "pairs.jsonl", "pair_json", "pair_json IS NOT NULL"),
            (self.output_dir / "errors" / "generation_errors.jsonl", "error_json", "error_json IS NOT NULL"),
        ]
        if self.save_raw:
            targets.append((self.output_dir / "data" / "raw_generations.jsonl", "raw_json", "raw_json IS NOT NULL"))
        else:
            raw_path = self.output_dir / "data" / "raw_generations.jsonl"
            temporary = raw_path.with_name(raw_path.name + ".tmp")
            temporary.write_text("", encoding="utf-8")
            temporary.replace(raw_path)
        for path, column, predicate in targets:
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for (payload,) in self.connection.execute(
                    f"SELECT {column} FROM results WHERE {predicate} ORDER BY rowid"
                ):
                    handle.write(payload + "\n")
            temporary.replace(path)

    def close(self) -> None:
        self.connection.close()


def _validate_resume_config(output_dir: Path, config: dict[str, Any]) -> None:
    path = output_dir / "configs" / "args.json"
    if not path.is_file():
        raise FileNotFoundError(f"Resume requires the saved generation config: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    comparable = ("canonical_jsonl", "backend", "model_name", "prompt_version", "num_shots",
                  "in_context_pool_json", "context_steps", "seed", "temperature", "max_tokens",
                  "deberta_tokenizer_name", "max_deberta_anchor_tokens", "max_deberta_pair_tokens",
                  "min_output_tokens", "max_model_len", "output_profile")
    mismatches = [key for key in comparable if saved.get(key, "train" if key == "output_profile" else None)
                  != config.get(key)]
    if mismatches:
        details = {key: {"saved": saved.get(key), "requested": config.get(key)} for key in mismatches}
        raise ValueError(f"Resume configuration does not match the saved run: {details}")


def generate_vllm(requests: Iterable[GenerationRequest], args: argparse.Namespace,
                  model_name: str, pools: dict[str, list[dict[str, Any]]],
                  state: GenerationState) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    qwen_tokenizer = llm.get_tokenizer()
    deberta_tokenizer = AutoTokenizer.from_pretrained(args.deberta_tokenizer_name, use_fast=True)

    for batch in batched(requests, args.generation_batch_size):
        active: list[tuple[Any, ...]] = []
        for request in batch:
            shots = _few_shots(request, args.num_shots, pools)
            prompt = generation_prompt(request, shots, args.prompt_version)
            rendered = render_chat_prompt(qwen_tokenizer, prompt)
            anchor_tokens = len(deberta_tokenizer.encode(request.anchor, add_special_tokens=False))
            if anchor_tokens > args.max_deberta_anchor_tokens:
                state.record(request, error=_profile_error(
                    request, args, model_name, prompt, shots, "DebertaAnchorTooLong",
                    "Anchor exceeds the configured DeBERTa token limit",
                    {"observed_tokens": anchor_tokens, "limit": args.max_deberta_anchor_tokens}))
                continue
            prompt_tokens = len(qwen_tokenizer.encode(rendered, add_special_tokens=False))
            budget = output_token_budget(qwen_tokenizer, request.anchor, args.max_model_len,
                                         prompt_tokens, args.max_tokens)
            if budget < args.min_output_tokens or prompt_tokens + budget > args.max_model_len:
                state.record(request, error=_profile_error(
                    request, args, model_name, prompt, shots, "QwenContextOverflow",
                    "Prompt and output budget do not fit the context window",
                    {"prompt_tokens": prompt_tokens, "max_output_tokens": budget,
                     "min_output_tokens": args.min_output_tokens, "limit": args.max_model_len}))
                continue
            metadata = {"prompt_tokens": prompt_tokens, "anchor_tokens": anchor_tokens,
                        "max_output_tokens": budget}
            active.append((request, shots, prompt, rendered, metadata,
                           SamplingParams(temperature=args.temperature, max_tokens=budget,
                                          seed=request.generation_seed)))
        if not active:
            continue
        outputs = llm.generate(
            [x[3] for x in active],
            [x[5] for x in active],
            use_tqdm=False,
        )
        for (request, shots, prompt, _rendered, metadata, _params), output in zip(active, outputs):
            completion = output.outputs[0] if output.outputs else None
            text = completion.text if completion is not None else ""
            metadata["finish_reason"] = getattr(completion, "finish_reason", None)
            raw = _profile_record(request, args, model_name, prompt, shots,
                                  metadata=metadata, raw_output=text) if state.save_raw else None
            try:
                generated = parse_generated_segment(text)
                pair_tokens = len(deberta_tokenizer(request.anchor, generated, add_special_tokens=True,
                                                    truncation=False)["input_ids"])
                if pair_tokens > args.max_deberta_pair_tokens:
                    raise PairTooLong(pair_tokens)
                pair = _profile_record(request, args, model_name, prompt, shots, generated=generated,
                                       metadata={**metadata, "pair_tokens": pair_tokens})
                state.record(request, pair=pair, raw=raw)
            except Exception as exc:  # persisted and resumable
                details = {"observed_tokens": exc.tokens, "limit": args.max_deberta_pair_tokens} \
                    if isinstance(exc, PairTooLong) else {}
                error_type = "DebertaPairTooLong" if isinstance(exc, PairTooLong) else type(exc).__name__
                error = _profile_error(request, args, model_name, prompt, shots, error_type,
                                       str(exc), {**metadata, **details}, text)
                state.record(request, raw=raw, error=error)


class PairTooLong(ValueError):
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        super().__init__("Generated anchor/text pair exceeds the configured DeBERTa token limit")


def generate_openai(requests: Iterable[GenerationRequest], args: argparse.Namespace,
                    model_name: str, pools: dict[str, list[dict[str, Any]]],
                    state: GenerationState) -> None:
    import tiktoken
    from openai import OpenAI

    client = OpenAI()
    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
    except KeyError:
        tokenizer = tiktoken.get_encoding("o200k_base")
    for request in requests:
        shots = _few_shots(request, args.num_shots, pools)
        prompt = generation_prompt(request, shots, args.prompt_version)
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens + args.max_tokens > args.max_model_len:
            state.record(request, error=_profile_error(
                request, args, model_name, prompt, shots, "PromptTooLong",
                "Prompt and output budget do not fit the context window",
                {"prompt_tokens": prompt_tokens, "limit": args.max_model_len}))
            continue
        text = ""
        try:
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature, max_tokens=args.max_tokens,
                seed=request.generation_seed, response_format={"type": "json_object"})
            text = response.choices[0].message.content or ""
            generated = parse_generated_segment(text)
            metadata = {"prompt_tokens": prompt_tokens,
                        "finish_reason": response.choices[0].finish_reason}
            pair = _profile_record(request, args, model_name, prompt, shots,
                                   generated=generated, metadata=metadata)
            raw = _profile_record(request, args, model_name, prompt, shots,
                                  metadata=metadata, raw_output=text) if state.save_raw else None
            state.record(request, pair=pair, raw=raw)
        except Exception as exc:
            error = _profile_error(request, args, model_name, prompt, shots,
                                   type(exc).__name__, str(exc), raw_output=text)
            raw = _profile_record(request, args, model_name, prompt, shots,
                                  raw_output=text) if state.save_raw else None
            state.record(request, raw=raw, error=error)


def main() -> None:
    args = parse_args()
    if args.save_raw and args.output_profile != "debug":
        args.output_profile = "debug"
    model_name = args.model_name or BACKEND_DEFAULTS[args.backend]
    resolved_output = Path(args.output_dir) if args.output_dir else default_output_dir("generation")
    config = {**vars(args), "model_name": model_name, "output_dir": str(resolved_output)}
    if args.config_only:
        print_config(config)
        return
    if args.context_steps < 0 or args.generation_batch_size <= 0:
        raise ValueError("--context_steps must be non-negative and --generation_batch_size positive")
    if min(args.max_tokens, args.max_model_len, args.checkpoint_every) <= 0:
        raise ValueError("Token limits and --checkpoint_every must be positive")
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires an explicit --output_dir")
    if args.num_shots < 0 or (args.num_shots > 0 and not args.in_context_pool_json):
        raise ValueError("Non-negative --num_shots requires --in_context_pool_json when nonzero")
    pools = load_in_context_pools(args.in_context_pool_json)
    for kind in GENERATION_TYPES:
        if args.num_shots > len(pools[kind]):
            raise ValueError(f"Requested {args.num_shots} {kind} examples, but pool has {len(pools[kind])}")

    if args.resume:
        output_dir = resolved_output
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume output directory does not exist: {output_dir}")
        _validate_resume_config(output_dir, config)
    else:
        output_dir = create_stage_layout(resolved_output)
        write_json(output_dir / "configs" / "args.json", config)

    state = GenerationState(output_dir, args.checkpoint_every, args.resume,
                            save_raw=args.output_profile == "debug")
    requests = iter_requests(args.canonical_jsonl, args.seed, args.context_steps,
                             state.completed_pair_ids, state.note_discovered)
    from tqdm.auto import tqdm
    progress = tqdm(initial=len(state.completed_pair_ids), total=None,
                    desc="Generating process-reward pairs", unit="pair", dynamic_ncols=True)
    state.progress = progress
    progress.set_postfix(success=state.success_count, skipped=state.skipped_count)
    try:
        if args.backend == "vllm":
            generate_vllm(requests, args, model_name, pools, state)
        else:
            generate_openai(requests, args, model_name, pools, state)
    finally:
        state.checkpoint(export=True)
        progress.close()

    counts = state.counts()
    summary = {
        "canonical_samples": state.canonical_samples,
        "generation_requests": len(state.discovered_pair_ids),
        **counts,
        "in_context_pool_sizes": {key: len(value) for key, value in pools.items()},
        "output_profile": args.output_profile,
        "state_database": str(state.db_path),
        "output_dir": str(output_dir),
    }
    _atomic_write_json(output_dir / "metrics" / "summary.json", summary)
    append_log(output_dir, json.dumps(summary, sort_keys=True))
    state.close()
    print_config(summary)


if __name__ == "__main__":
    main()
