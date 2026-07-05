# Reliability Experiment Details

## Ranking Mode

Ranking mode uses an anchor caption, a semantically matching positive caption, and one or more hard negatives. A scorer passes a case when the anchor-positive score is greater than the anchor-negative score.

## Separation Mode

Separation mode uses a positive/control caption and one or more hard negatives. The diagnostic criterion is whether `raw_score(positive, negative)` stays below a threshold `tau`.

## Raw Values

Every scorer-specific raw value is saved because calibration will happen later. The first pass should not discard directional NLI probabilities, BERTScore precision/recall/F1, or raw cosine scores.

## Deferred Tau Calibration

Tau calibration is not tuned in this experiment. Tables report a fixed tau grid only so scorer behavior can be inspected before stochastic hyperparameter search or threshold calibration is added.

## Tables

- `subcategory_counts.csv`: verifies how many cases and scored pairs were produced for each dataset/subcategory/mode.
- `cases_preview.csv`: shows representative normalized examples for quick loader inspection.
- `table_ranking_by_subcategory.csv`: reports ranking accuracy, positive/negative means, and margins for ranking cases.
- `table_separation_by_subcategory.csv`: reports false positive rates at fixed tau values for separation cases.
- `table_scorer_overall.csv`: summarizes scorer behavior by dataset and score field.
- `load_errors.jsonl`: records unavailable or malformed datasets/subcategories.
- `scorer_errors.jsonl`: records scorer load or execution failures.
