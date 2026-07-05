from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from metrics.reliable.aggregate_visualprm_numeric import aggregate
from metrics.reliable.data_loaders.common import read_jsonl, write_jsonl
from metrics.reliable.scorers import SCORER_REGISTRY, ScorerError, make_device
from metrics.reliable.visualprm_numeric import (
    balanced_sample_traces,
    build_segment_requests,
    generate_cases_with_mock,
    generate_cases_with_vllm,
    load_visualprm_dataset,
    load_visualprm_traces,
    score_pair_rows_from_cases,
)
from tqdm.auto import tqdm


SUMMARY_FIELDS = {
    "sbert": ["raw_score"],
    "bertscore": ["bertscore_f1"],
    "nli": ["nli_score_coverage", "nli_score_equiv", "E_ab", "E_ba"],
    "cross_nli": ["raw_score"],
    "bge_reranker": ["raw_score"],
}


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def set_cache_defaults() -> None:
    hf_home = os.environ.get("HF_HOME")
    if hf_home and not os.environ.get("DATASETS_CACHE"):
        os.environ["DATASETS_CACHE"] = str(Path(hf_home).expanduser() / "datasets")
    if hf_home and not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(Path(hf_home).expanduser() / "hub")
    if hf_home and not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_home).expanduser() / "hub")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("src/outputs/reliable/visualprm_numeric") / f"run_{stamp}"


def ensure_layout(output_dir: Path) -> None:
    for name in ("raw", "tables", "logs", "errors", "configs", "generations"):
        output_dir.joinpath(name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_once(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    write_jsonl(path, rows)


def append_log(output_dir: Path, message: str) -> None:
    log_path = output_dir / "logs" / "run.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def config_path(output_dir: Path, name: str, mode: str) -> Path:
    preferred = output_dir / "configs" / name
    if not preferred.exists():
        return preferred
    stem = preferred.stem
    suffix = preferred.suffix
    return output_dir / "configs" / f"{stem}_{mode}{suffix}"


def score_fields_for_scorer(scorer: str, values: dict[str, Any]) -> list[str]:
    fields = SUMMARY_FIELDS.get(scorer, ["raw_score"])
    return [field for field in fields if field in values]


def run_scorers(
    scorer_names: list[str],
    pair_inputs: list[tuple[str, str]],
    pair_rows: list[dict[str, Any]],
    batch_size: int,
    device_arg: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    all_scores: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    loaded: list[str] = []
    device = make_device(device_arg)
    for scorer_name in tqdm(scorer_names, desc="Running scorers", unit="scorer"):
        scorer_cls = SCORER_REGISTRY.get(scorer_name)
        if scorer_cls is None:
            errors.append(
                {
                    "scorer": scorer_name,
                    "error_type": "ValueError",
                    "error_message": f"Unknown scorer: {scorer_name}",
                }
            )
            continue
        try:
            scorer = scorer_cls(device)
            loaded.append(scorer_name)
            tqdm.write(f"Scoring {len(pair_inputs)} pairs with {scorer_name}")
            scorer_values = scorer.score_pairs(pair_inputs, batch_size)
            if len(scorer_values) != len(pair_rows):
                raise RuntimeError(f"{scorer_name} returned {len(scorer_values)} scores for {len(pair_rows)} pairs")
            for base, values in tqdm(
                zip(pair_rows, scorer_values),
                total=len(pair_rows),
                desc=f"Writing {scorer_name} score rows",
                unit="pair",
                leave=False,
            ):
                raw_values = dict(values)
                for field in score_fields_for_scorer(scorer_name, raw_values):
                    all_scores.append(
                        {
                            **base,
                            "scorer": scorer_name,
                            "score_field": field,
                            "score": raw_values[field],
                            "all_raw_values": raw_values,
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - persisted for inspection.
            errors.append(ScorerError.from_exception(scorer_name, exc).__dict__)
    return all_scores, errors, loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--dataset_name", default="OpenGVLab/VisualPRM400K-v1.1-Raw")
    parser.add_argument("--dataset_config", default="default")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--dataset_streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_source_rows", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cases_jsonl", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_traces", type=int, default=10000)
    parser.add_argument("--scorers", default="sbert,nli,cross_nli,bge_reranker")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--generator_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--generation_batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--mock_generation", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.generate_only and args.score_only:
        raise ValueError("Use only one of --generate_only or --score_only")
    if args.score_only:
        if not args.cases_jsonl:
            raise ValueError("--score_only requires --cases_jsonl")
        cases_jsonl = Path(args.cases_jsonl)
        if not cases_jsonl.is_file():
            raise FileNotFoundError(f"--cases_jsonl does not exist or is not a file: {cases_jsonl}")
        return
    if not args.input_jsonl:
        if not args.dataset_name:
            raise ValueError("--input_jsonl or --dataset_name is required unless --score_only is set")
        return
    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"--input_jsonl does not exist or is not a file: {input_jsonl}")


def generate_cases(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.input_jsonl:
        tqdm.write(f"Loading VisualPRM traces from JSONL: {args.input_jsonl}")
        print('loading traces')
        traces, load_errors = load_visualprm_traces(Path(args.input_jsonl))
        input_source = args.input_jsonl
    else:
        print('loading traces')
        config_name = args.dataset_config or None
        tqdm.write(
            f"Loading VisualPRM dataset {args.dataset_name}/{config_name or 'default'}:{args.dataset_split}"
        )
        traces, load_errors = load_visualprm_dataset(
            dataset_name=args.dataset_name,
            config_name=config_name,
            split=args.dataset_split,
            streaming=args.dataset_streaming,
            max_rows=args.max_source_rows,
        )
        input_source = f"{args.dataset_name}/{args.dataset_config or 'default'}:{args.dataset_split}"
    if not traces:
        raise ValueError(f"No valid VisualPRM traces loaded from {input_source}")
    tqdm.write(f"Loaded {len(traces)} valid traces with {len(load_errors)} load errors")
    selected_traces = balanced_sample_traces(traces, args.max_traces, args.seed)
    tqdm.write(f"Selected {len(selected_traces)} traces")
    requests, segment_errors = build_segment_requests(selected_traces)
    if not requests:
        raise ValueError(
            f"No eligible numeric/symbolic segments found in {len(selected_traces)} selected traces from {input_source}"
        )
    if args.mock_generation:
        cases, generation_errors, raw_generations = generate_cases_with_mock(requests)
    else:
        cases, generation_errors, raw_generations = generate_cases_with_vllm(
            requests=requests,
            model_name=args.generator_model,
            tensor_parallel_size=args.tensor_parallel_size,
            generation_batch_size=args.generation_batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        )

    write_jsonl_once(output_dir / "raw" / "numeric_cases.jsonl", cases)
    write_jsonl_once(output_dir / "errors" / "generation_errors.jsonl", [*load_errors, *segment_errors, *generation_errors])
    write_jsonl_once(output_dir / "generations" / "numeric_generations.jsonl", raw_generations)
    metadata = {
        "num_input_traces": len(traces),
        "num_selected_traces": len(selected_traces),
        "num_segment_requests": len(requests),
        "num_cases": len(cases),
        "num_generation_errors": len(load_errors) + len(segment_errors) + len(generation_errors),
        "input_source": input_source,
    }
    return cases, metadata


def score_cases(args: argparse.Namespace, output_dir: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("No VisualPRM cases available for scoring")
    pair_inputs, pair_rows = score_pair_rows_from_cases(cases)
    if not pair_inputs:
        raise ValueError("No VisualPRM positive/negative pairs available for scoring")
    tqdm.write(f"Built {len(pair_inputs)} scoring pairs from {len(cases)} cases")
    scores, scorer_errors, loaded_scorers = run_scorers(
        parse_csv_arg(args.scorers),
        pair_inputs,
        pair_rows,
        args.batch_size,
        args.device,
    )
    write_jsonl_once(output_dir / "raw" / "numeric_scores.jsonl", scores)
    write_jsonl_once(output_dir / "errors" / "scorer_errors.jsonl", scorer_errors)
    aggregate(output_dir)
    return {
        "num_pairs": len(pair_rows),
        "num_scores": len(scores),
        "requested_scorers": parse_csv_arg(args.scorers),
        "loaded_scorers": loaded_scorers,
        "num_scorer_errors": len(scorer_errors),
    }


def main() -> None:
    args = parse_args()
    print('validating args')
    validate_args(args)
    set_cache_defaults()
    random.seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    print('ensuring layout')
    ensure_layout(output_dir)
    mode = "score_only" if args.score_only else "generate_only" if args.generate_only else "full"
    print('write args.json')
    write_json(config_path(output_dir, "args.json", mode), vars(args))
    append_log(output_dir, f"mode={mode}")

    print('start summary')
    summary: dict[str, Any] = {"output_dir": str(output_dir), "mode": mode}
    if args.score_only:
        cases = read_jsonl(Path(args.cases_jsonl))
        summary["num_cases"] = len(cases)
        summary.update(score_cases(args, output_dir, cases))
    else:
        print('start generation')
        cases, generation_summary = generate_cases(args, output_dir)
        summary.update(generation_summary)
        if not args.generate_only:
            summary.update(score_cases(args, output_dir, cases))

    write_json(config_path(output_dir, "run_metadata.json", mode), summary)
    append_log(output_dir, json.dumps(summary, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
