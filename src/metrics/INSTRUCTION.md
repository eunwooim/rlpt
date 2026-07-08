# Instructions: Dataset-Balanced NLI Calibration

## Goal

Implement calibration for NLI-based compatibility scoring using already-processed reliability score files.

This pass is calibration only.

Do not regenerate data.  
Do not rerun scorer inference.  
Do not implement RL training.  
Do not implement VisualPRM generation changes in this pass.

## Method

Use **dataset-balanced logistic calibration** over frozen NLI outputs.

The pretrained NLI model is frozen. We only learn a lightweight calibration layer over cached NLI probabilities.

For a reference/candidate pair, define:

```text
E_ab = P(entailment | premise=a, hypothesis=b)
E_ba = P(entailment | premise=b, hypothesis=a)
C_max = max(C_ab, C_ba)
N_max = max(N_ab, N_ba)
```

Feature vector:

```text
x = [E_ab, E_ba, C_max, N_max]
```

Learn:

```text
score = sigmoid(w1 * E_ab + w2 * E_ba + w3 * C_max + w4 * N_max + b)
```

where:

```text
w1, w2, w3, w4, b
```

are learned calibration parameters.

Expected signs:

```text
w1 > 0
w2 > 0
w3 < 0
w4 < 0
```

Do not hard-code these signs unless implementation is simple. Save learned weights and inspect them.

## Why This Is Not Training a New NLI Model

Do not fine-tune DeBERTa or any NLI model.

This is a small supervised calibration layer over frozen NLI features.

Use wording:

```text
dataset-balanced logistic calibration
```

or:

```text
lightweight calibration layer over frozen NLI outputs
```

## Constraints

- Do not install packages.
- Do not modify conda environments.
- Do not overwrite previous outputs.
- Use timestamped run directories.
- Do not save new results directly into a flat output directory.
- Use already-processed score JSONL files only.
- Do not rerun generation.
- Do not rerun scorer inference.
- Do not implement random/grid/coordinate search as the primary method.
- If running commands is necessary, use the existing tmux session named `rlpt`.
- Do not run long GPU jobs. Calibration should be CPU/tabular work.

## Output Layout

Use:

```text
src/outputs/reliable/calibration/run_YYYYMMDD_HHMMSS/
  raw/
  tables/
  logs/
  errors/
  configs/
```

Save:

```text
configs/args.json
configs/run_metadata.json
configs/nli_calibration_best.json
raw/nli_calibration_predictions.jsonl
logs/run.log
errors/calibration_errors.jsonl
```

## Input

Add:

```text
src/metrics/calibrate_nli.py
```

Support multiple existing score files:

```bash
--score_jsonl path1 path2 path3
```

Expected inputs include already-processed reliability outputs:

```text
src/outputs/reliable/caption_negation/run_*/raw/scores_top3.jsonl
src/outputs/reliable/visualprm_numeric/run_*/raw/numeric_scores.jsonl
```

The script must not assume one dataset.

## Required Rows

Use rows where:

```text
scorer == "nli"
```

Required fields:

```text
E_ab
E_ba
C_ab
C_ba
N_ab
N_ba
```

If fields are nested in `all_raw_values`, read from there.

If required fields are missing, skip the row and log to:

```text
errors/calibration_errors.jsonl
```

## Labels

### Ranking datasets

Examples:

```text
sugarcrepepp
visualprm
```

Use:

```text
pair_type == positive -> label 1
pair_type == negative -> label 0
```

Ranking metrics:

```text
ranking_acc = mean(score_positive > score_negative)
mean_margin = mean(score_positive - score_negative)
```

Compare positives and negatives within the same:

```text
dataset
case_id
negative_type if available
```

### Separation-only datasets

Examples:

```text
sugarcrepe
negbench
```

Use:

```text
negative pair -> label 0
```

Separation metric:

```text
fpr_at_tau = mean(score_negative >= tau)
```

## Dataset and Label Balancing

Do not let VisualPRM dominate.

Use sample weights for logistic regression.

Default weighting:

```text
dataset-label balanced weighting
```

For each block:

```text
(dataset, label)
```

assign each row weight:

```text
weight_i = 1 / count(dataset=d, label=y)
```

Then normalize weights so total weight is stable.

This makes each dataset-label block contribute equal total mass.

If a dataset has only negatives, it contributes through its negative block only.

Also support optional caps:

```bash
--max_cases_per_dataset 50000
--max_pairs_per_dataset 100000
```

Sampling must be deterministic with `--seed`.

## Splits

Split by `case_id`, not by row.

Default:

```text
--calib_frac 0.7
--seed 42
```

Support:

```bash
--use_all_for_calibration
```

If `--use_all_for_calibration` is set:

- fit on all available data
- select tau on all available data
- still write all tables
- set held-out metrics to null
- record this clearly in `configs/nli_calibration_best.json`

Default should remain split-based calibration.

## Logistic Regression

Use scikit-learn if available:

```python
sklearn.linear_model.LogisticRegression
```

Recommended defaults:

```text
solver = lbfgs
max_iter = 1000
C = 1.0
```

Fit with:

```text
sample_weight
```

If scikit-learn is unavailable, implement a small PyTorch or NumPy fallback.

Do not install scikit-learn.

## Tau Selection

Do not random-search tau jointly with weights.

After fitting logistic regression, compute predicted compatibility scores.

Select `tau` on the calibration split by sweeping candidate thresholds from predicted scores.

Choose tau that maximizes dataset-balanced objective:

For ranking datasets:

```text
component = ranking_acc
```

For separation-only datasets:

```text
component = 1 - fpr_at_tau
```

Overall:

```text
selection_score = mean(component over datasets)
```

Tie-breakers:

```text
1. higher average ranking mean_margin
2. lower average fpr_at_tau
```

## Held-Out Reporting

After selecting weights and tau on the calibration split, evaluate on the held-out split.

Report both:

```text
calib metrics
heldout metrics
```

If `--use_all_for_calibration` is set, held-out metrics should be null/unavailable.

## Outputs

### Best Config

Write:

```text
configs/nli_calibration_best.json
```

Schema:

```json
{
  "method": "dataset_label_balanced_logistic_calibration",
  "features": ["E_ab", "E_ba", "C_max", "N_max"],
  "weights": {
    "E_ab": 0.0,
    "E_ba": 0.0,
    "C_max": 0.0,
    "N_max": 0.0
  },
  "bias": 0.0,
  "tau": 0.0,
  "selection_metric": "mean_dataset_component",
  "calib_metrics": {},
  "heldout_metrics": {},
  "datasets": [],
  "score_jsonl": [],
  "use_all_for_calibration": false
}
```

### Predictions

Write:

```text
raw/nli_calibration_predictions.jsonl
```

Each row:

```json
{
  "case_id": "...",
  "dataset": "...",
  "subcategory": "...",
  "negative_type": "...",
  "pair_type": "positive",
  "label": 1,
  "split": "calib",
  "E_ab": 0.0,
  "E_ba": 0.0,
  "C_max": 0.0,
  "N_max": 0.0,
  "calibrated_score": 0.0,
  "tau": 0.0,
  "predicted_compatible": true
}
```

### Tables

Write:

```text
tables/table_nli_calibration_overall.csv
tables/table_nli_calibration_by_dataset.csv
tables/table_nli_calibration_by_subcategory.csv
tables/table_nli_calibration_by_negative_type.csv
```

At minimum include:

```text
split
dataset
mode
num_cases
num_pairs
weight_E_ab
weight_E_ba
weight_C_max
weight_N_max
bias
tau
ranking_acc
fpr_at_tau
mean_positive_score
mean_negative_score
mean_margin
component
```

Use empty values for non-applicable fields.

## CLI

Required arguments:

```bash
--score_jsonl
--output_dir
--calib_frac
--seed
```

Optional arguments:

```bash
--max_cases_per_dataset
--max_pairs_per_dataset
--use_all_for_calibration
```

Do not require `--num_trials`; this is not random search.

## Commands to Document

Held-out calibration:

```bash
python src/metrics/calibrate_nli.py \
  --score_jsonl \
    src/outputs/reliable/caption_negation/run_YYYYMMDD_HHMMSS/raw/scores_top3.jsonl \
    src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS/raw/numeric_scores.jsonl \
  --output_dir src/outputs/reliable/calibration/run_YYYYMMDD_HHMMSS \
  --calib_frac 0.7 \
  --max_pairs_per_dataset 100000 \
  --seed 42
```

Use all processed reliability data:

```bash
python src/metrics/calibrate_nli.py \
  --score_jsonl \
    src/outputs/reliable/caption_negation/run_YYYYMMDD_HHMMSS/raw/scores_top3.jsonl \
    src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS/raw/numeric_scores.jsonl \
  --output_dir src/outputs/reliable/calibration/run_YYYYMMDD_HHMMSS \
  --max_pairs_per_dataset 100000 \
  --use_all_for_calibration \
  --seed 42
```

## Validation

Run only lightweight checks:

```bash
python src/metrics/calibrate_nli.py --help
```

If a tiny local synthetic JSONL is needed to check parsing, create it under:

```text
src/outputs/reliable/calibration_debug/
```

Do not run long jobs.

## Report After Implementation

Report:

```text
files added/modified
validation commands run
whether scikit-learn was available
whether dataset-label balancing is implemented
exact full calibration command to run
assumptions about score JSONL schema
```

Do not paste large tables into markdown.
