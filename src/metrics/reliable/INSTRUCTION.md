# Instructions: VisualPRM Segment Ranking and NLI Calibration

## Goal

Implement:

1. VisualPRM segment-level ranking.
2. NLI coefficient and tau calibration.

Full runs are expected unless the user explicitly says otherwise.

## Constraints

- Do not install packages.
- Do not modify conda environments.
- Do not overwrite previous outputs.
- Use timestamped run directories.
- Do not save new results directly into a flat output directory.
- If running commands is necessary, use the existing tmux session.

## Output Layout

```text
src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS/
  raw/
  tables/
  logs/
  errors/
  configs/
  generations/

src/outputs/reliable/calibration/run_YYYYMMDD_HHMMSS/
  raw/
  tables/
  logs/
  errors/
  configs/
```

Save run arguments and metadata:

```text
configs/args.json
configs/run_metadata.json
logs/run.log
```

## Part A: VisualPRM Segment Ranking

### Task

For each selected VisualPRM reasoning chain:

1. Apply punctuation-based sentence segmentation.
2. Keep eligible numerical/symbolic segments.
3. Generate a positive and adversarial negatives for every eligible segment.
4. Score each `(reference, positive)` and `(reference, negative)` pair.
5. Aggregate ranking metrics.

If a reasoning chain has `N` eligible segments, it should produce approximately:

```text
N positive pairs
N * num_negative_types negative pairs
```

### Files

Add:

```text
src/metrics/reliable/run_visualprm_numeric.py
src/metrics/reliable/visualprm_numeric.py
src/metrics/reliable/aggregate_visualprm_numeric.py
src/metrics/reliable/scorers/cross_nli.py
src/metrics/reliable/scorers/bge_reranker.py
src/metrics/reliable/calibrate_nli.py
```

Reuse existing scorer utilities.

### Scorers

Use existing:

```text
sbert
nli
bertscore
```

Add:

```text
cross_nli: cross-encoder/nli-deberta-v3-large
bge_reranker: BAAI/bge-reranker-large
```

For `cross_nli`, use entailment probability as `raw_score` if NLI logits are available.

For `bge_reranker`, save model output as `raw_score`.

### Segmentation

Use **punctuation-based sentence segmentation**.

Split on sentence boundaries and newlines. Do not split on commas.

Drop trivial marker-only segments such as `Given`, `Therefore`, `Thus`, `Hence`, `Now`, `Final answer`, and pure numbering.

Keep eligible segments containing numerical or symbolic content.

### Balanced Sampling

Balance by trace-level subcategory, not segment-level subcategory.

Subcategory:

1. use explicit `subcategory` if present
2. else use `source`
3. clean source filename when possible
4. else use `unknown`

Sampling:

1. group traces by subcategory
2. repeatedly select a trace from the least-filled subcategory
3. break ties randomly with `--seed`
4. segment the selected trace
5. generate cases for every eligible segment in that trace
6. stop at `--max_traces`

Default full run:

```text
--max_traces 10000
```

### Generation Prompt

Use one prompt per segment.

```text
You are generating controlled adversarial examples for evaluating semantic similarity metrics.

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
{
  "positive": "...",
  "negative_value_flip": "... or null",
  "negative_entity_flip": "... or null",
  "negative_relation_flip": "... or null",
  "notes": {
    "changed_value": "... or null",
    "changed_entity": "... or null",
    "changed_relation": "... or null"
  }
}
```

### Case Output

Write:

```text
raw/numeric_cases.jsonl
```

Schema:

```json
{
  "case_id": "visualprm_000001_seg_03",
  "trace_id": "visualprm_000001",
  "dataset": "visualprm",
  "subcategory": "mavis_function_poly",
  "source": "mavis_function_poly_prm.jsonl",
  "question": "...",
  "segment_index": 3,
  "reference": "f(2) = 3",
  "positive": "The function value at x = 2 is 3.",
  "negatives": [
    { "type": "value_flip", "text": "The function value at x = 3 is 3." },
    { "type": "entity_flip", "text": "The function value at y = 2 is 3." },
    { "type": "relation_flip", "text": "The function value at x = 2 is not 3." }
  ],
  "mode": "ranking"
}
```

Invalid generations:

```text
errors/generation_errors.jsonl
```

### Score Output

Write:

```text
raw/numeric_scores.jsonl
```

Schema:

```json
{
  "case_id": "...",
  "trace_id": "...",
  "dataset": "visualprm",
  "subcategory": "...",
  "negative_type": "value_flip",
  "pair_type": "negative",
  "scorer": "nli",
  "score_field": "nli_score_coverage",
  "score": 0.91,
  "all_raw_values": {}
}
```

For positive pairs:

```text
negative_type = null
pair_type = positive
```

### Aggregation

Produce:

```text
tables/table_numeric_overall.csv
tables/table_numeric_by_subcategory.csv
tables/table_numeric_by_perturbation.csv
tables/numeric_subcategory_counts.csv
tables/numeric_cases_preview.csv
```

Metrics:

```text
ranking_acc = mean(score(reference, positive) > score(reference, negative))
fpr_at_0_6 = mean(score(reference, negative) >= 0.6)
mean_margin = mean(score(reference, positive) - score(reference, negative))
```

Compare scores within the same `case_id`, scorer, score field, and negative type.

Summary fields:

```text
sbert/raw_score
bertscore/bertscore_f1
nli/nli_score_coverage
nli/nli_score_equiv
nli/E_ab
nli/E_ba
cross_nli/raw_score
bge_reranker/raw_score
```

Save `C_ab`, `C_ba`, `N_ab`, and `N_ba`, but do not treat contradiction fields as similarity scores in summary tables.

## Part B: NLI Coefficient and Tau Calibration

### Task

Fit:

```text
score = clip(
  alpha * E_ab
  + beta * E_ba
  - gamma * max(C_ab, C_ba)
  - delta * max(N_ab, N_ba),
  0,
  1
)
```

Tune:

```text
tau
```

### Input

Support multiple score files:

```bash
--score_jsonl path1 path2
```

### Search

Use random search with fixed seed.

Defaults:

```text
alpha ∈ [0, 1]
beta ∈ [0, 1]
gamma ∈ [0, 2]
delta ∈ [0, 2]
tau ∈ [0.1, 0.9]
num_trials = 5000
```

Constraint:

```text
alpha + beta > 0
```

Split by `case_id`, not pair row.

Default:

```text
--calib_frac 0.7
--seed 42
```

### Objective

For ranking data, optimize:

```text
ranking_acc
```

Tie-breakers:

```text
higher mean_margin
lower fpr_at_tau
```

For separation-only data, optimize:

```text
lower fpr_at_tau
```

### Calibration Outputs

```text
raw/nli_calibration_trials.jsonl
configs/nli_calibration_best.json
tables/table_nli_calibration_overall.csv
tables/table_nli_calibration_by_dataset.csv
tables/table_nli_calibration_by_subcategory.csv
tables/table_nli_calibration_by_negative_type.csv
```

## Commands to Document

VisualPRM full run:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/metrics/reliable/run_visualprm_numeric.py \
  --input_jsonl /path/to/visualprm_subset.jsonl \
  --output_dir src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS \
  --max_traces 10000 \
  --scorers sbert,nli,cross_nli,bge_reranker \
  --batch_size 32 \
  --device auto \
  --seed 42
```

Calibration full run:

```bash
python src/metrics/reliable/calibrate_nli.py \
  --score_jsonl src/outputs/reliable/visualprm_numeric/run_YYYYMMDD_HHMMSS/raw/numeric_scores.jsonl \
  --output_dir src/outputs/reliable/calibration/run_YYYYMMDD_HHMMSS \
  --num_trials 5000 \
  --calib_frac 0.7 \
  --seed 42
```
