from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from rlpt.src.metrics.reliable.aggregate_hn_neg import aggregate
from metrics.reliable.data_loaders import load_negbench, load_sugarcrepe, load_sugarcrepepp
from metrics.reliable.data_loaders.common import Case, write_jsonl
from metrics.reliable.scorers import SCORER_REGISTRY, ScorerError, make_device


LOADERS = {
    "sugarcrepe": load_sugarcrepe,
    "sugarcrepepp": load_sugarcrepepp,
    "negbench": load_negbench,
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


def load_cases(dataset_names: list[str], max_cases: int | None) -> tuple[list[Case], list[dict[str, Any]]]:
    cases: list[Case] = []
    errors: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        loader = LOADERS.get(dataset_name)
        if loader is None:
            errors.append(
                {
                    "dataset": dataset_name,
                    "subcategory": "__dataset__",
                    "error_type": "ValueError",
                    "error_message": f"Unknown dataset: {dataset_name}",
                }
            )
            continue
        loaded, load_errors = loader(max_cases)
        cases.extend(loaded)
        errors.extend(asdict(error) for error in load_errors)
    return cases, errors


def build_pair_rows(cases: list[Case]) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    pairs: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        if case.mode == "ranking":
            pair_specs = [("positive_pair", None, case.anchor, case.positive)]
            pair_specs.extend(("negative_pair", i, case.anchor, negative) for i, negative in enumerate(case.negatives))
        elif case.mode == "separation":
            pair_specs = [("negative_pair", i, case.positive, negative) for i, negative in enumerate(case.negatives)]
        else:
            continue
        for pair_type, negative_index, text_a, text_b in pair_specs:
            pairs.append((text_a, text_b))
            rows.append(
                {
                    "case_id": case.case_id,
                    "dataset": case.dataset,
                    "subcategory": case.subcategory,
                    "mode": case.mode,
                    "pair_type": pair_type,
                    "negative_index": negative_index,
                    "text_a": text_a,
                    "text_b": text_b,
                    "metadata": case.metadata,
                }
            )
    return pairs, rows


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
    for scorer_name in scorer_names:
        scorer_cls = SCORER_REGISTRY.get(scorer_name)
        if scorer_cls is None:
            errors.append(asdict(ScorerError(scorer_name, "ValueError", f"Unknown scorer: {scorer_name}")))
            continue
        try:
            scorer = scorer_cls(device)
            loaded.append(scorer_name)
            scorer_values = scorer.score_pairs(pair_inputs, batch_size)
            if len(scorer_values) != len(pair_rows):
                raise RuntimeError(f"{scorer_name} returned {len(scorer_values)} scores for {len(pair_rows)} pairs")
            for base, values in zip(pair_rows, scorer_values):
                row = dict(base)
                row["scorer"] = scorer_name
                row.update(values)
                all_scores.append(row)
        except Exception as exc:  # noqa: BLE001 - persisted for inspection.
            errors.append(asdict(ScorerError.from_exception(scorer_name, exc)))
    return all_scores, errors, loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="sugarcrepe,sugarcrepepp,negbench")
    parser.add_argument("--scorers", default="sbert,nli,bertscore")
    parser.add_argument("--output_dir", default="src/outputs/reliable")
    parser.add_argument("--max_cases_per_subcategory", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_cache_defaults()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv_arg(args.datasets)
    scorers = parse_csv_arg(args.scorers)
    cases, load_errors = load_cases(datasets, args.max_cases_per_subcategory)
    cases = sorted(cases, key=lambda case: (case.dataset, case.subcategory, case.case_id))
    pair_inputs, pair_rows = build_pair_rows(cases)
    scores, scorer_errors, loaded_scorers = run_scorers(scorers, pair_inputs, pair_rows, args.batch_size, args.device)

    write_jsonl(output_dir / "cases_top3.jsonl", [case.public_dict() for case in cases])
    write_jsonl(output_dir / "scores_top3.jsonl", scores)
    write_jsonl(output_dir / "load_errors.jsonl", load_errors)
    write_jsonl(output_dir / "scorer_errors.jsonl", scorer_errors)
    aggregate(output_dir)

    summary = {
        "output_dir": str(output_dir),
        "num_cases": len(cases),
        "num_pairs": len(pair_rows),
        "num_scores": len(scores),
        "datasets": datasets,
        "requested_scorers": scorers,
        "loaded_scorers": loaded_scorers,
        "load_errors": len(load_errors),
        "scorer_errors": len(scorer_errors),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
