# Reliability Metrics Handbook

This directory contains the cached-score reliability pipeline for text scorers used as compatibility rewards. The pipeline has two parts:

1. Run frozen scorers once and save JSONL scores.
2. Rebuild reliability tables from the saved JSONL files.

The table commands below do not rerun scorer inference, VisualPRM generation, or NLI calibration.

## Notation

Let `s(a,b)` be a normalized compatibility score for a reference/candidate pair.

Reranker scores are logits, so the BGE reranker is normalized before thresholding or reward scaling:

```text
s_bge(a,b) = sigmoid(raw_reranker_score)
```

The NLI heuristic baseline is:

```text
C_max = max(C_ab, C_ba)
s_nli(a,b) = 0.5 * E_ab + 0.5 * E_ba - C_max
```

Reward scaling uses `kappa`, never `tau`:

```text
tilde_s_kappa(a,b) = max(0, s(a,b) - kappa) / (1 - kappa)
```

The reward tables report averages of `tilde_s_kappa`. The default master-script value is `KAPPA=0.0`.

Threshold rejection and FPR analysis use `tau`, never `kappa`:

```text
fpr_at_tau = mean(s_negative >= tau)
```

## Current Tables

Primary reward tables report:

```text
mean_positive_reward = average tilde_s_kappa(a,b) on positive pairs
mean_negative_reward = average tilde_s_kappa(a,b) on negative pairs
reward_margin = mean_positive_reward - mean_negative_reward
```

For ranking datasets, the primary reward tables do not report ranking accuracy. For separation datasets, the primary reward tables do not report FPR. FPR appears only in threshold-sensitivity appendix tables where `tau` is the operating threshold.

The calibrated NLI logistic scorer is reported as a separate scorer row using its learned `calibrated_score`, then the same `kappa` scaling is applied for reward-table summaries.

## Cached Inputs

Expected completed score files:

```text
src/outputs/reliable/caption_negation/initial_run/scores_top3.jsonl
src/outputs/reliable/visualprm_numeric/initial_run/raw/numeric_scores.jsonl
src/outputs/reliable/calibration/<calibration_run>/raw/nli_calibration_predictions.jsonl
```

The current master script defaults to:

```text
DATA_RUN_NAME=initial_run
CALIBRATION_RUN=run_20260706_224211
CALIBRATION_WEIGHTS=src/ckpts/calibration/v1/nli_calibration_weights.json
KAPPA=0.0
```

If `nli_calibration_predictions.jsonl` is missing, `reward_tables.py` applies the saved logistic weights directly to cached NLI probabilities. This is still cached-output post-processing, not scorer inference.

## Rebuild Reward Tables

Run from the repository root:

```bash
bash src/metrics/run.bash
```

Override the reward scaling threshold with:

```bash
KAPPA=0.5 bash src/metrics/run.bash
```

Equivalent explicit command flow:

```bash
python src/metrics/aggregate_hn_neg.py \
  --output_dir src/outputs/reliable/caption_negation/initial_run \
  --kappa 0.0

python src/metrics/aggregate_visualprm_numeric.py \
  --output_dir src/outputs/reliable/visualprm_numeric/initial_run \
  --kappa 0.0

python src/metrics/reward_tables.py \
  --caption_scores src/outputs/reliable/caption_negation/initial_run/scores_top3.jsonl \
  --numeric_scores src/outputs/reliable/visualprm_numeric/initial_run/raw/numeric_scores.jsonl \
  --calibration_predictions src/outputs/reliable/calibration/run_20260706_224211/raw/nli_calibration_predictions.jsonl \
  --calibration_weights src/ckpts/calibration/v1/nli_calibration_weights.json \
  --kappa 0.0 \
  --output_dir src/outputs/reliable/reward_tables/initial_run
```

Primary outputs:

```text
src/outputs/reliable/caption_negation/initial_run/table_scorer_overall.csv
src/outputs/reliable/caption_negation/initial_run/table_scorer_by_subcategory.csv
src/outputs/reliable/caption_negation/initial_run/table_scorer_by_mode.csv
src/outputs/reliable/visualprm_numeric/initial_run/tables/table_numeric_overall.csv
src/outputs/reliable/visualprm_numeric/initial_run/tables/table_numeric_by_subcategory.csv
src/outputs/reliable/visualprm_numeric/initial_run/tables/table_numeric_by_perturbation.csv
src/outputs/reliable/reward_tables/initial_run/table_reward_by_dataset.csv
src/outputs/reliable/reward_tables/initial_run/table_reward_by_source_dataset.csv
src/outputs/reliable/reward_tables/initial_run/table_reward_by_subcategory.csv
src/outputs/reliable/reward_tables/initial_run/table_reward_by_dataset.md
```

In these tables, BGE reranker rows use `score_field=sigmoid_raw_score`.

## Tau Sensitivity

Tau sensitivity is an appendix for rejection/FPR behavior only:

```bash
python src/metrics/threshold_sensitivity.py \
  --caption_scores src/outputs/reliable/caption_negation/initial_run/scores_top3.jsonl \
  --numeric_scores src/outputs/reliable/visualprm_numeric/initial_run/raw/numeric_scores.jsonl \
  --calibration_predictions src/outputs/reliable/calibration/run_20260706_224211/raw/nli_calibration_predictions.jsonl \
  --calibration_best src/outputs/reliable/calibration/run_20260706_224211/configs/nli_calibration_best.json \
  --output_dir src/outputs/reliable/threshold_sensitivity/initial_run
```

Fixed `tau` values:

```text
0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
```

Outputs:

```text
src/outputs/reliable/threshold_sensitivity/initial_run/appendix_threshold_sensitivity.csv
src/outputs/reliable/threshold_sensitivity/initial_run/appendix_threshold_sensitivity.md
src/outputs/reliable/threshold_sensitivity/initial_run/appendix_calibrated_operating_threshold.csv
src/outputs/reliable/threshold_sensitivity/initial_run/appendix_calibrated_operating_threshold.md
```

## Scorer Inference Commands

Use these only when intentionally creating or replacing cached score files.

Caption and negation reliability:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/metrics/run_hn_neg.py \
  --datasets sugarcrepe,sugarcrepepp,negbench \
  --scorers sbert,nli,cross_nli,bge_reranker \
  --output_dir src/outputs/reliable/caption_negation/initial_run \
  --batch_size 32 \
  --device auto \
  --seed 42
```

VisualPRM numeric generation plus scoring:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/metrics/run_visualprm_numeric.py \
  --output_dir src/outputs/reliable/visualprm_numeric/initial_run \
  --dataset_name OpenGVLab/VisualPRM400K-v1.1-Raw \
  --dataset_config default \
  --dataset_split train \
  --dataset_streaming \
  --max_traces 10000 \
  --scorers sbert,nli,cross_nli,bge_reranker \
  --generator_model Qwen/Qwen3-32B \
  --tensor_parallel_size 4 \
  --generation_batch_size 64 \
  --batch_size 32 \
  --device auto \
  --seed 42
```

VisualPRM scoring only from existing numeric cases:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/metrics/run_visualprm_numeric.py \
  --cases_jsonl src/outputs/reliable/visualprm_numeric/initial_run/raw/numeric_cases.jsonl \
  --output_dir src/outputs/reliable/visualprm_numeric/initial_run \
  --score_only \
  --scorers sbert,nli,cross_nli,bge_reranker \
  --batch_size 32 \
  --device auto \
  --seed 42
```

## NLI Logistic Calibration

Calibration learns only a lightweight logistic layer over frozen NLI probabilities:

```text
x = [E_ab, E_ba, max(C_ab, C_ba), max(N_ab, N_ba)]
score = sigmoid(w dot x + b)
```

Calibration selects an operating threshold `tau` for rejection/FPR diagnostics. It does not set `kappa`.

Held-out calibration command:

```bash
DATA_RUN_NAME="initial_run"
CALIB_TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME="nli_calibration_${CALIB_TS}"

python src/metrics/calibrate_nli.py \
  --score_jsonl \
    src/outputs/reliable/caption_negation/${DATA_RUN_NAME}/scores_top3.jsonl \
    src/outputs/reliable/visualprm_numeric/${DATA_RUN_NAME}/raw/numeric_scores.jsonl \
  --output_dir "src/outputs/reliable/calibration/run_${CALIB_TS}" \
  --exp_name "${EXP_NAME}" \
  --calib_frac 0.85 \
  --max_pairs_per_dataset 100000 \
  --tau_grid 0.50:0.95:0.05 \
  --tau_selection_metric fpr_at_fixed_tpr \
  --target_tpr 0.96 \
  --seed 42
```

The learned weights are saved to:

```text
src/ckpts/calibration/${EXP_NAME}/nli_calibration_weights.json
```

## Lightweight Checks

These checks only exercise argument parsing and imports:

```bash
python src/metrics/aggregate_hn_neg.py --help
python src/metrics/aggregate_visualprm_numeric.py --help
python src/metrics/reward_tables.py --help
python src/metrics/threshold_sensitivity.py --help
python src/metrics/calibrate_nli.py --help
bash -n src/metrics/run.bash
```
