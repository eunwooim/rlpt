from __future__ import annotations

import json
import random
import re
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from metrics.data_loaders.common import coerce_text, read_jsonl
from tqdm.auto import tqdm


ID_FIELDS = ("id", "trace_id", "uid", "sample_id")
QUESTION_FIELDS = ("question", "problem", "prompt", "query")
REASONING_FIELDS = ("reasoning", "solution", "response", "analysis", "answer", "rationale", "steps")
SUBCATEGORY_FIELDS = ("subcategory", "category", "task", "domain")
SOURCE_FIELDS = ("source", "filename", "file", "data_source")
HF_SOURCE_FIELDS = ("source", "dataset", "data_source", "image", "image_path", "filename", "file")

NEGATIVE_KEY_TO_TYPE = {
    "negative_value_flip": "value_flip",
    "negative_entity_flip": "entity_flip",
    "negative_relation_flip": "relation_flip",
}

MARKER_ONLY = {
    "given",
    "therefore",
    "thus",
    "hence",
    "now",
    "final answer",
    "answer",
    "solution",
    "step",
}


@dataclass
class VisualPRMTrace:
    trace_id: str
    question: str
    reasoning: str
    subcategory: str
    source: str
    row_index: int
    metadata: dict[str, Any]


@dataclass
class SegmentRequest:
    case_id: str
    trace_id: str
    dataset: str
    subcategory: str
    source: str
    question: str
    segment_index: int
    reference: str


def _first_text(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = _coerce_reasoning_text(row.get(name))
        if value:
            return value
    return ""


def _coerce_reasoning_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        texts = [_coerce_reasoning_text(item) for item in value]
        return "\n".join(text for text in texts if text)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "response", "reasoning", "solution"):
            text = _coerce_reasoning_text(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _clean_label(value: str) -> str:
    text = coerce_text(value)
    if not text:
        return ""
    name = Path(text).name
    for suffix in (".jsonl", ".json", ".parquet", ".csv", ".txt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or "unknown"


def _source_subcategory(value: str) -> str:
    text = coerce_text(value)
    if not text:
        return ""
    parts = [part for part in re.split(r"[\\/]+", text) if part]
    for part in parts:
        if part.startswith("VisualPRM400K"):
            continue
        if part.lower() in {"images", "train", "val", "test", "raw"}:
            continue
        if re.search(r"\.(?:png|jpg|jpeg|webp|jsonl|json|parquet|csv|txt)$", part, re.IGNORECASE):
            continue
        return _clean_label(part)
    return _clean_label(text)


def _trace_id(row: dict[str, Any], row_index: int) -> str:
    value = _first_text(row, ID_FIELDS)
    if value:
        return value
    return f"visualprm_{row_index:06d}"


def load_visualprm_traces(input_jsonl: Path) -> tuple[list[VisualPRMTrace], list[dict[str, Any]]]:
    rows = read_jsonl(input_jsonl)
    traces: list[VisualPRMTrace] = []
    errors: list[dict[str, Any]] = []
    input_source = input_jsonl.name
    for row_index, row in enumerate(tqdm(rows, desc="Loading JSONL traces", unit="row"), start=1):
        if not isinstance(row, dict):
            errors.append(
                {
                    "row_index": row_index,
                    "error_type": "TypeError",
                    "error_message": f"Expected JSON object, got {type(row).__name__}",
                }
            )
            continue

        question = _first_text(row, QUESTION_FIELDS)
        reasoning = _first_text(row, REASONING_FIELDS)
        source = _first_text(row, SOURCE_FIELDS) or input_source
        explicit_subcategory = _first_text(row, SUBCATEGORY_FIELDS)
        subcategory = _clean_label(explicit_subcategory or source or "unknown")
        trace_id = _trace_id(row, row_index)

        if not reasoning:
            errors.append(
                {
                    "trace_id": trace_id,
                    "row_index": row_index,
                    "source": source,
                    "subcategory": subcategory,
                    "error_type": "ValueError",
                    "error_message": "Missing reasoning text",
                }
            )
            continue

        traces.append(
            VisualPRMTrace(
                trace_id=trace_id,
                question=question,
                reasoning=reasoning,
                subcategory=subcategory,
                source=source,
                row_index=row_index,
                metadata={
                    "input_jsonl": str(input_jsonl),
                    "source_id": trace_id,
                },
            )
        )
    return traces, errors


def load_visualprm_dataset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    streaming: bool = False,
    max_rows: int | None = None,
) -> tuple[list[VisualPRMTrace], list[dict[str, Any]]]:
    if dataset_name.startswith("OpenGVLab/VisualPRM400K"):
        return load_visualprm_annotations_zip(dataset_name, config_name, split, max_rows)

    from datasets import load_dataset

    traces: list[VisualPRMTrace] = []
    errors: list[dict[str, Any]] = []
    features = _dataset_features(dataset_name)
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming, features=features)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming, features=features)
    except Exception as exc:
        if streaming:
            raise
        errors.append(
            {
                "dataset_name": dataset_name,
                "dataset_config": config_name,
                "dataset_split": split,
                "error_type": type(exc).__name__,
                "error_message": f"Non-streaming load failed; retrying with streaming: {exc}",
            }
        )
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split, streaming=True, features=features)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True, features=features)

    dataset_iter = tqdm(dataset, desc="Loading dataset traces", total=max_rows, unit="row")
    for row_index, row in enumerate(dataset_iter, start=1):
        if max_rows is not None and max_rows > 0 and row_index > max_rows:
            break
        if not isinstance(row, dict):
            errors.append(
                {
                    "row_index": row_index,
                    "dataset_name": dataset_name,
                    "error_type": "TypeError",
                    "error_message": f"Expected dataset row dict, got {type(row).__name__}",
                }
            )
            continue

        question = _first_text(row, QUESTION_FIELDS)
        reasoning = _first_text(row, REASONING_FIELDS)
        source = _first_text(row, HF_SOURCE_FIELDS) or dataset_name
        explicit_subcategory = _first_text(row, SUBCATEGORY_FIELDS)
        subcategory = _clean_label(explicit_subcategory) or _source_subcategory(source) or "unknown"
        trace_id = _trace_id(row, row_index)

        if not reasoning:
            errors.append(
                {
                    "trace_id": trace_id,
                    "row_index": row_index,
                    "source": source,
                    "subcategory": subcategory,
                    "dataset_name": dataset_name,
                    "error_type": "ValueError",
                    "error_message": "Missing reasoning text",
                }
            )
            continue

        traces.append(
            VisualPRMTrace(
                trace_id=trace_id,
                question=question,
                reasoning=reasoning,
                subcategory=subcategory,
                source=source,
                row_index=row_index,
                metadata={
                    "dataset_name": dataset_name,
                    "dataset_config": config_name,
                    "dataset_split": split,
                    "source_id": trace_id,
                },
            )
        )
    return traces, errors


def load_visualprm_annotations_zip(
    dataset_name: str,
    config_name: str | None,
    split: str,
    max_rows: int | None = None,
) -> tuple[list[VisualPRMTrace], list[dict[str, Any]]]:
    if split != "train":
        raise ValueError(f"{dataset_name} only exposes the train split, got: {split}")
    if config_name not in {None, "default"}:
        raise ValueError(f"{dataset_name} only exposes the default config, got: {config_name}")

    from datasets.download.download_manager import DownloadManager

    url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/annotations.zip"
    archive_path = Path(DownloadManager(dataset_name=dataset_name).download(url))
    traces: list[VisualPRMTrace] = []
    errors: list[dict[str, Any]] = []
    row_index = 0
    with zipfile.ZipFile(archive_path) as archive:
        annotation_names = sorted(name for name in archive.namelist() if name.endswith(".jsonl"))
        with tqdm(total=max_rows, desc="Loading VisualPRM rows", unit="row") as row_progress:
            with tqdm(annotation_names, desc="VisualPRM annotation files", unit="file") as annotation_iter:
                for annotation_name in annotation_iter:
                    annotation_iter.set_postfix_str(Path(annotation_name).name[:40])
                    source = Path(annotation_name).name
                    subcategory = _clean_label(source)
                    with archive.open(annotation_name) as f:
                        for line_index, raw_line in enumerate(f, start=1):
                            if max_rows is not None and max_rows > 0 and row_index >= max_rows:
                                return traces, errors
                            line = raw_line.decode("utf-8").strip()
                            if not line:
                                continue
                            row_index += 1
                            row_progress.update(1)
                            try:
                                row = json.loads(line)
                            except Exception as exc:  # noqa: BLE001 - persisted as experiment artifact.
                                errors.append(
                                    {
                                        "row_index": row_index,
                                        "line_index": line_index,
                                        "source": source,
                                        "subcategory": subcategory,
                                        "dataset_name": dataset_name,
                                        "error_type": type(exc).__name__,
                                        "error_message": str(exc),
                                    }
                                )
                                continue
                            if not isinstance(row, dict):
                                errors.append(
                                    {
                                        "row_index": row_index,
                                        "line_index": line_index,
                                        "source": source,
                                        "subcategory": subcategory,
                                        "dataset_name": dataset_name,
                                        "error_type": "TypeError",
                                        "error_message": f"Expected dataset row dict, got {type(row).__name__}",
                                    }
                                )
                                continue

                            image = row.get("image")
                            if image is not None and not isinstance(image, str):
                                errors.append(
                                    {
                                        "row_index": row_index,
                                        "line_index": line_index,
                                        "source": source,
                                        "subcategory": subcategory,
                                        "dataset_name": dataset_name,
                                        "error_type": "ValueError",
                                        "error_message": f"Skipping row with non-string image field: {type(image).__name__}",
                                    }
                                )
                                continue

                            question = _first_text(row, QUESTION_FIELDS)
                            reasoning = _first_text(row, REASONING_FIELDS)
                            raw_trace_id = _clean_label(_trace_id(row, row_index))
                            trace_id = f"{subcategory}_{raw_trace_id}"
                            if not reasoning:
                                errors.append(
                                    {
                                        "trace_id": trace_id,
                                        "row_index": row_index,
                                        "line_index": line_index,
                                        "source": source,
                                        "subcategory": subcategory,
                                        "dataset_name": dataset_name,
                                        "error_type": "ValueError",
                                        "error_message": "Missing reasoning text",
                                    }
                                )
                                continue

                            traces.append(
                                VisualPRMTrace(
                                    trace_id=trace_id,
                                    question=question,
                                    reasoning=reasoning,
                                    subcategory=subcategory,
                                    source=source,
                                    row_index=row_index,
                                    metadata={
                                        "dataset_name": dataset_name,
                                        "dataset_config": config_name,
                                        "dataset_split": split,
                                        "annotation_name": annotation_name,
                                        "line_index": line_index,
                                        "source_id": raw_trace_id,
                                    },
                                )
                            )
    return traces, errors


def _dataset_features(dataset_name: str) -> Any:
    if not dataset_name.startswith("OpenGVLab/VisualPRM400K"):
        return None

    from datasets import Features, Value

    return Features(
        {
            "image": Value("string"),
            "id": Value("string"),
            "question": Value("string"),
            "question_zh": Value("string"),
            "answer": Value("string"),
            "answer_orig": Value("string"),
            "analysis": Value("string"),
            "analysis_zh": Value("string"),
            "response": Value("string"),
            "tikz": Value("string"),
            "steps_with_score": [
                {
                    "step": Value("string"),
                    "score": Value("float64"),
                    "num_mc_correct": Value("int64"),
                    "num_mc_total": Value("int64"),
                }
            ],
            "num_mc_sequences": Value("int64"),
            "question_orig": Value("string"),
        }
    )


def punctuation_sentence_segments(text: str) -> list[str]:
    segments: list[str] = []
    for block in re.split(r"[\r\n]+", text):
        block = block.strip()
        if not block:
            continue
        parts = re.split(r"(?<=[.!?;:])\s+", block)
        for part in parts:
            segment = re.sub(r"\s+", " ", part).strip()
            if segment and not _is_marker_only(segment):
                segments.append(segment)
    return segments


def _is_marker_only(segment: str) -> bool:
    text = segment.strip()
    text = re.sub(r"^[\s>*#-]+", "", text)
    normalized = re.sub(r"[^A-Za-z0-9 ]+", "", text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        return True
    if normalized in MARKER_ONLY:
        return True
    if re.fullmatch(r"(step\s*)?\d+[a-z]?", normalized):
        return True
    return False


def is_numeric_or_symbolic(segment: str) -> bool:
    if re.search(r"\d", segment):
        return True
    if re.search(r"[=<>+\-*/^%$(){}\[\]|√∑π∞≤≥±×÷]", segment):
        return True
    if re.search(r"\b[A-Za-z]\s*(?:=|<|>|≤|≥)\s*[A-Za-z0-9]", segment):
        return True
    return False


def eligible_segments(reasoning: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for index, segment in enumerate(punctuation_sentence_segments(reasoning), start=1):
        if is_numeric_or_symbolic(segment):
            out.append((index, segment))
    return out


def balanced_sample_traces(traces: list[VisualPRMTrace], max_traces: int | None, seed: int) -> list[VisualPRMTrace]:
    if max_traces is not None and max_traces <= 0:
        return []
    rng = random.Random(seed)
    buckets: dict[str, list[VisualPRMTrace]] = defaultdict(list)
    for trace in tqdm(traces, desc="Balancing traces", unit="trace"):
        buckets[trace.subcategory].append(trace)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[VisualPRMTrace] = []
    counts = {subcategory: 0 for subcategory in buckets}
    limit = max_traces if max_traces is not None else len(traces)
    progress = tqdm(total=min(limit, len(traces)), desc="Sampling traces", unit="trace")
    while len(selected) < limit:
        available = [subcategory for subcategory, bucket in buckets.items() if bucket]
        if not available:
            break
        min_count = min(counts[subcategory] for subcategory in available)
        candidates = [subcategory for subcategory in available if counts[subcategory] == min_count]
        subcategory = rng.choice(candidates)
        selected.append(buckets[subcategory].pop())
        counts[subcategory] += 1
        progress.update(1)
    progress.close()
    return selected


def build_segment_requests(traces: list[VisualPRMTrace]) -> tuple[list[SegmentRequest], list[dict[str, Any]]]:
    requests: list[SegmentRequest] = []
    errors: list[dict[str, Any]] = []
    for trace in tqdm(traces, desc="Segmenting traces", unit="trace"):
        segments = eligible_segments(trace.reasoning)
        if not segments:
            errors.append(
                {
                    "trace_id": trace.trace_id,
                    "subcategory": trace.subcategory,
                    "source": trace.source,
                    "error_type": "ValueError",
                    "error_message": "No eligible numeric/symbolic segments",
                }
            )
            continue
        for segment_index, segment in segments:
            requests.append(
                SegmentRequest(
                    case_id=f"{trace.trace_id}_seg_{segment_index:02d}",
                    trace_id=trace.trace_id,
                    dataset="visualprm",
                    subcategory=trace.subcategory,
                    source=trace.source,
                    question=trace.question,
                    segment_index=segment_index,
                    reference=segment,
                )
            )
    return requests, errors


def generation_prompt(segment: str) -> str:
    return f"""You are generating controlled adversarial examples for evaluating semantic similarity metrics.

Given one reference reasoning segment, generate:
1. A POSITIVE paraphrase that is logically equivalent to the reference.
2. A NEGATIVE_VALUE_FLIP that changes exactly one numerical value while preserving the entity/relation as much as possible.
3. A NEGATIVE_ENTITY_FLIP that changes exactly one variable, object, edge, point, or symbol while preserving the numerical value as much as possible.
4. A NEGATIVE_RELATION_FLIP that changes exactly one relation/operator/direction while preserving entities and numbers as much as possible.

Rules:
- Do not solve the problem.
- Do not add new context.
- Do not generate a full reasoning chain.
- Each output must be one standalone sentence or equation.
- The POSITIVE must preserve every number, variable, entity, unit, and mathematical relation.
- Each NEGATIVE must be minimally different and logically non-equivalent.
- If a requested negative type is impossible, output null.
- Return only valid JSON.

Reference segment:
{segment}

Return JSON:
{{
  "positive": "...",
  "negative_value_flip": "... or null",
  "negative_entity_flip": "... or null",
  "negative_relation_flip": "... or null",
  "notes": {{
    "changed_value": "... or null",
    "changed_entity": "... or null",
    "changed_relation": "... or null"
  }}
}}"""


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("No valid JSON object found in generation")


def _nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    text = coerce_text(value)
    return text or None


def case_from_generation(request: SegmentRequest, generation: dict[str, Any]) -> dict[str, Any]:
    positive = _nullable_text(generation.get("positive"))
    if positive is None:
        raise ValueError("Generation is missing positive")
    negatives: list[dict[str, str]] = []
    for key, negative_type in NEGATIVE_KEY_TO_TYPE.items():
        text = _nullable_text(generation.get(key))
        if text is not None:
            negatives.append({"type": negative_type, "text": text})
    if not negatives:
        raise ValueError("Generation has no valid negatives")
    return {
        "case_id": request.case_id,
        "trace_id": request.trace_id,
        "dataset": request.dataset,
        "subcategory": request.subcategory,
        "source": request.source,
        "question": request.question,
        "segment_index": request.segment_index,
        "reference": request.reference,
        "positive": positive,
        "negatives": negatives,
        "mode": "ranking",
    }


def mock_generation_for_request(request: SegmentRequest) -> dict[str, Any]:
    reference = request.reference
    return {
        "positive": reference,
        "negative_value_flip": _flip_first_number(reference),
        "negative_entity_flip": _flip_first_entity(reference),
        "negative_relation_flip": _flip_first_relation(reference),
        "notes": {
            "changed_value": "mock",
            "changed_entity": "mock",
            "changed_relation": "mock",
        },
    }


def _flip_first_number(text: str) -> str | None:
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = match.group(0)
    replacement = str(float(value) + 1) if "." in value else str(int(value) + 1)
    return text[: match.start()] + replacement + text[match.end() :]


def _flip_first_entity(text: str) -> str | None:
    match = re.search(r"\b([A-Za-z])\b", text)
    if not match:
        return None
    replacement = "y" if match.group(1) != "y" else "x"
    return text[: match.start()] + replacement + text[match.end() :]


def _flip_first_relation(text: str) -> str | None:
    replacements = [("<=", ">="), (">=", "<="), ("≤", "≥"), ("≥", "≤"), ("=", "!="), ("<", ">"), (">", "<"), ("+", "-"), ("-", "+")]
    for old, new in replacements:
        index = text.find(old)
        if index >= 0:
            return text[:index] + new + text[index + len(old) :]
    return None


def generate_cases_with_mock(
    requests: list[SegmentRequest],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    raw_generations: list[dict[str, Any]] = []
    for request in tqdm(requests, desc="Mock-generating cases", unit="segment"):
        generation = mock_generation_for_request(request)
        raw_generations.append(
            {
                **asdict(request),
                "generator_model": "mock",
                "prompt": generation_prompt(request.reference),
                "raw_output": json.dumps(generation, ensure_ascii=False),
            }
        )
        try:
            cases.append(case_from_generation(request, generation))
        except Exception as exc:  # noqa: BLE001 - persisted as experiment artifact.
            errors.append(
                {
                    **asdict(request),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "raw_output": json.dumps(generation, ensure_ascii=False),
                }
            )
    return cases, errors, raw_generations


def generate_cases_with_vllm(
    requests: list[SegmentRequest],
    model_name: str,
    tensor_parallel_size: int,
    generation_batch_size: int,
    temperature: float,
    max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    raw_generations: list[dict[str, Any]] = []
    batch_starts = range(0, len(requests), generation_batch_size)
    for start in tqdm(batch_starts, total=(len(requests) + generation_batch_size - 1) // generation_batch_size, desc="Generating cases", unit="batch"):
        batch = requests[start : start + generation_batch_size]
        prompts = [render_chat_prompt(tokenizer, generation_prompt(request.reference)) for request in batch]
        outputs = llm.generate(prompts, sampling_params)
        for request, output in zip(batch, outputs):
            text = output.outputs[0].text if output.outputs else ""
            raw_generations.append(
                {
                    **asdict(request),
                    "generator_model": model_name,
                    "prompt": generation_prompt(request.reference),
                    "raw_output": text,
                }
            )
            try:
                parsed = extract_json_object(text)
                cases.append(case_from_generation(request, parsed))
            except Exception as exc:  # noqa: BLE001 - persisted as experiment artifact.
                errors.append(
                    {
                        **asdict(request),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "raw_output": text,
                    }
                )
    return cases, errors, raw_generations


def score_pair_rows_from_cases(cases: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    pairs: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        reference = coerce_text(case.get("reference"))
        positive = coerce_text(case.get("positive"))
        if reference and positive:
            pairs.append((reference, positive))
            rows.append(_score_base_row(case, pair_type="positive", negative_type=None))
        for negative in case.get("negatives") or []:
            if not isinstance(negative, dict):
                continue
            negative_text = coerce_text(negative.get("text"))
            negative_type = coerce_text(negative.get("type")) or "unknown"
            if reference and negative_text:
                pairs.append((reference, negative_text))
                rows.append(_score_base_row(case, pair_type="negative", negative_type=negative_type))
    return pairs, rows


def _score_base_row(case: dict[str, Any], pair_type: str, negative_type: str | None) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "trace_id": case.get("trace_id"),
        "dataset": case.get("dataset", "visualprm"),
        "subcategory": case.get("subcategory", "unknown"),
        "negative_type": negative_type,
        "pair_type": pair_type,
        "mode": case.get("mode", "ranking"),
    }
