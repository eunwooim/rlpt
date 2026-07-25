#!/usr/bin/env python
"""Lightweight tests for raw VisualProcessBench evaluation semantics."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from process_reward.visualprocessbench import (
    EvaluationExample,
    _class_indices,
    _prediction_rows,
    compute_metrics,
    load_evaluation_examples,
    parse_args,
)


def _example(label: int, index: int) -> EvaluationExample:
    return EvaluationExample(
        row_index=1,
        anchor="Question context",
        generated=f"step {index}",
        compatibility_label=label,
        data_source="source",
        policy_model="policy",
        pair_id=f"pair_{index}",
        source_sample_id="sample",
        step_index=index,
        input_mode="context_step",
        metadata={},
    )


class VisualProcessBenchMetricsTest(unittest.TestCase):
    def test_class_indices_are_confirmed_from_model_config(self) -> None:
        self.assertEqual(
            _class_indices({"num_labels": 2, "reward_class_index": 1}),
            (1, 0),
        )
        self.assertEqual(
            _class_indices({"num_labels": 2, "reward_class_index": 0}),
            (0, 1),
        )
        with self.assertRaisesRegex(ValueError, "reward_class_index"):
            _class_indices({"num_labels": 2})
        with self.assertRaisesRegex(ValueError, "exactly two labels"):
            _class_indices({"num_labels": 3, "reward_class_index": 1})

    def test_neutral_exclusion(self) -> None:
        examples = [_example(1, 0), _example(-1, 1), _example(0, 2)]
        scores = [0.9, 0.1, 0.99]
        metrics, predictions = compute_metrics(
            examples, scores, threshold=0.0, runtime=1.0
        )
        self.assertEqual(predictions, [1, -1, 1])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["overall"]["evaluated_steps"], 2)
        self.assertEqual(metrics["overall"]["neutral_steps"], 1)
        self.assertEqual(
            metrics["grouped_metrics"]["policy_model"]["policy"]["evaluated_steps"],
            2,
        )
        rows = _prediction_rows(examples, scores, predictions)
        self.assertTrue(rows[2]["is_neutral"])
        self.assertIsNone(rows[2]["correct"])

    def test_margin_threshold_is_strict(self) -> None:
        examples = [_example(1, 0), _example(1, 1)]
        metrics, predictions = compute_metrics(
            examples,
            [0.75, 0.7501],
            threshold=0.5,
            runtime=1.0,
        )
        self.assertEqual(predictions, [-1, 1])
        self.assertEqual(metrics["overall"]["threshold_used"], 0.5)

    def test_default_margin_threshold_is_zero(self) -> None:
        args = parse_args(
            ["--data_path", "data.jsonl", "--model_path", "checkpoint"]
        )
        self.assertEqual(args.threshold, 0.0)

    def test_test_set_threshold_tuning_is_not_available(self) -> None:
        argv = ["--data_path", "data.jsonl", "--model_path", "checkpoint"]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([*argv, "--auto"])
        metrics, _ = compute_metrics(
            [_example(1, 0), _example(-1, 1)],
            [0.9, 0.1],
            threshold=0.0,
            runtime=1.0,
        )
        self.assertNotIn("auto_search", metrics)
        self.assertEqual(metrics["overall"]["threshold_used"], 0.0)

    def test_primary_f1_is_pooled_macro_f1(self) -> None:
        examples = [
            _example(1, 0),
            _example(1, 1),
            _example(1, 2),
            _example(-1, 3),
        ]
        metrics, _ = compute_metrics(
            examples,
            [0.9, 0.9, 0.1, 0.1],
            threshold=0.0,
            runtime=1.0,
        )
        self.assertAlmostEqual(metrics["overall"]["f1_positive"], 0.8)
        self.assertAlmostEqual(metrics["overall"]["f1_negative"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["f1"], 11.0 / 15.0)
        self.assertEqual(metrics["f1"], metrics["overall"]["macro_f1_pooled"])
        self.assertIn("weighted_macro_f1_over_sources", metrics["overall"])
        self.assertNotIn("micro_over_sources", metrics["overall"])

    def test_empty_response_step_is_rejected(self) -> None:
        row = {
            "question": "Question?",
            "response": {
                "steps": ["valid", "   "],
                "process_correctness": [1, 0],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty response step"):
                load_evaluation_examples(path, input_mode="context_step")

    def test_context_projection_excludes_current_step(self) -> None:
        row = {
            "question": "What is shown?",
            "policy_model": "policy",
            "data_source": "source",
            "response": {
                "steps": ["first reasoning step", "current reasoning step"],
                "process_correctness": [1, -1],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            examples = load_evaluation_examples(path, input_mode="context_step")
        self.assertIn("What is shown?", examples[1].anchor)
        self.assertIn("first reasoning step", examples[1].anchor)
        self.assertNotIn("current reasoning step", examples[1].anchor)
        self.assertEqual(examples[1].generated, "current reasoning step")

    def test_raw_length_mismatch(self) -> None:
        row = {
            "question": "Question?",
            "policy_model": "policy",
            "data_source": "source",
            "response": {
                "steps": ["first", "second"],
                "process_correctness": [1],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "step/label length mismatch"):
                load_evaluation_examples(path, input_mode="context_step")


if __name__ == "__main__":
    unittest.main()
