# VisualPRM Process-Reward Pipeline

This package implements a fast, modular text-pair process-reward pipeline. Each stage is independently runnable and writes a timestamped, non-overwriting run directory.

```text
OpenGVLab/VisualPRM400K-v1.1
  -> load_visualprm.py -> canonical.jsonl
  -> generate.py       -> pairs.jsonl
                         |-> cache_nli.py -> nli_features.jsonl (optional analysis)
                         `-> train.py     -> grouped splits + LoRA adapter + MLP head
  -> inference.py      -> compatibility probability
OpenGVLab/VisualProcessBench-derived pairs
  -> vpb_process.py -> evaluator pair JSONL
  -> visualprocessbench.py -> metrics.json + predictions.jsonl
```

VisualProcessBench is not used for loading, prompt examples, generation, or training.

## Canonical schema

Only `load_visualprm.py` knows the source dataset's field names. It emits:

```json
{
  "question": "...",
  "images": ["..."],
  "steps": ["..."],
  "step_labels": [1, 0, -1],
  "answer": "...",
  "metadata": {
    "source_sample_id": "...",
    "dataset": "OpenGVLab/VisualPRM400K-v1.1",
    "split": "train"
  }
}
```

Downstream stages consume only this schema. Neutral steps (`step_labels == 0`) are skipped during generation. Positive steps produce a compatibility label of `1`; negative steps produce a hard-negative pair with label `0`.

## 1. Export canonical data

```bash
python src/process_reward/load_visualprm.py \
  --dataset_name OpenGVLab/VisualPRM400K-v1.1 \
  --split train \
  --streaming \
  --max_samples 10000 \
  --shuffle_buffer_size 10000 \
  --seed 42
```

Output:

```text
src/outputs/process_reward_canonical/run_YYYYMMDD_HHMMSS/
  configs/args.json
  data/canonical.jsonl
  errors/load_errors.jsonl
  logs/run.log
  metrics/summary.json
```

## 2. Generate pairs

vLLM:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/process_reward/generate.py \
  --canonical_jsonl src/outputs/process_reward_canonical/run_<TIMESTAMP>/data/canonical.jsonl \
  --backend vllm \
  --model_name Qwen/Qwen3-32B \
  --prompt_version v1 \
  --tensor_parallel_size 4 \
  --generation_batch_size 64 \
  --num_shots 0 \
  --seed 42
```

OpenAI:

```bash
OPENAI_API_KEY=<configured-in-environment> \
python src/process_reward/generate.py \
  --canonical_jsonl src/outputs/process_reward_canonical/run_<TIMESTAMP>/data/canonical.jsonl \
  --backend openai \
  --model_name gpt-4o-mini \
  --prompt_version v1 \
  --num_shots 0 \
  --seed 42
```

The optional in-context pool is a JSON array. Each record must contain a matching `generation_type`, `anchor`, and `generated`; `id` or `example_id` is optional:

```json
[
  {
    "id": "positive_001",
    "generation_type": "positive_paraphrase",
    "anchor": "The total is 12.",
    "generated": "The sum equals 12."
  },
  {
    "id": "negative_001",
    "generation_type": "hard_negative",
    "anchor": "The total is 12.",
    "generated": "The total is 21."
  }
]
```

Positive requests sample only `positive_paraphrase` examples, and negative requests sample only `hard_negative` examples. Sampling uses each request's deterministic `generation_seed`. If `--num_shots` exceeds either required label-specific pool, generation fails before loading a backend model. Raw generation records save the selected example IDs and original JSON-array indices.

Enable few-shot generation by adding:

```bash
--in_context_pool_json path/to/in_context_pool.json --num_shots 2
```

## 3. Cache frozen NLI features (optional)

```bash
CUDA_VISIBLE_DEVICES=4 python src/process_reward/cache_nli.py \
  --pairs_jsonl src/outputs/process_reward_generation/run_<TIMESTAMP>/data/pairs.jsonl \
  --nli_model microsoft/deberta-xlarge-mnli \
  --batch_size 32 \
  --device cuda
```

The cache contains forward and reverse entailment, contradiction, and neutral
probabilities for analysis. It is not an input requirement for judge training.
The reward model trains solely on `anchor`, `generated`, and
`compatibility_label`.

## 4. Train the reward model

```bash
CUDA_VISIBLE_DEVICES=4 python src/process_reward/train.py \
  --data_path src/outputs/process_reward_generation/run_<TIMESTAMP>/data \
  --base_model microsoft/deberta-v3-large \
  --lora_target_modules query_proj,value_proj \
  --train_ratio 0.90 \
  --validation_ratio 0.05 \
  --test_ratio 0.05 \
  --split_seed 42 \
  --max_length 256 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --mlp_hidden_dim 256 \
  --report_to wandb \
  --wandb_mode offline \
  --seed 42
```

`--data_path` accepts either `pairs.jsonl` or its containing `data/`
directory. Before model or tokenizer loading, `train.py` creates
`pairs_train.jsonl`, `pairs_val.jsonl`, and `pairs_test.jsonl` beside the
unsplit file. All pairs from one `source_sample_id` stay in the same split.
If all three split files already exist, they are validated for non-empty,
source-disjoint contents and then reused. A partial split set is rejected so
stale and newly generated splits cannot be mixed.

Training writes:

```text
src/outputs/process_reward_training/run_YYYYMMDD_HHMMSS/
  configs/
  logs/
  checkpoints/checkpoint-*/
  checkpoints/final/
  metrics/
```

W&B receives only train loss, validation loss, accuracy, F1, and learning rate. Predictions are not uploaded.

## 5. Score pairs

```bash
CUDA_VISIBLE_DEVICES=4 python src/process_reward/inference.py \
  --checkpoint src/outputs/process_reward_training/run_<TIMESTAMP>/checkpoints/final \
  --anchor "The total is 12." \
  --generated "The sum equals 12." \
  --device cuda
```

For a JSONL batch containing `anchor` and `generated` fields, use `--pairs_jsonl` and optionally `--output_jsonl`. Python callers can use `ProcessRewardScorer.from_pretrained(...)` followed by `score_pairs(...)`.

## 6. Evaluate VisualProcessBench

`visualprocessbench.py` has two explicit input modes.

### Raw context-step mode

Use `--input_mode context_step` for raw VisualProcessBench records containing
`question`, `policy_model`, `data_source`, and parallel
`response.steps`/`response.process_correctness` lists. Each step is projected
as:

```text
anchor    = question + preceding response steps
generated = current response step
```

Labels are `1=correct`, `-1=incorrect`, and `0=neutral`. Step and label lists
must have equal length. Neutral steps are scored and retained in
`predictions.jsonl`, but excluded from loss, AUC, F1, accuracy, precision, and
recall.

This is an explicit distribution-shift evaluation: the judge was trained on
`(original step, generated variant)` pairs, not
`(question + preceding response steps, current step)` pairs. Do not interpret
context-step results as in-distribution judge performance.

Validate the local raw file without loading the checkpoint:

```bash
python src/process_reward/visualprocessbench.py \
  --input_mode context_step \
  --validate_data_only \
  --data_path /mnt/data1/eunwooim/VisualProcessBench/test.jsonl \
  --model_path PLACEHOLDER
```

Run raw evaluation:

```bash
CUDA_VISIBLE_DEVICES=4 python src/process_reward/visualprocessbench.py \
  --input_mode context_step \
  --data_path /mnt/data1/eunwooim/VisualProcessBench/test.jsonl \
  --model_path src/outputs/process_reward_judge/v1_1_0/checkpoints/final \
  --output_dir src/outputs/process_reward_evaluation/run_<TIMESTAMP> \
  --batch_size 32 \
  --max_length 256 \
  --threshold 0.5 \
  --device cuda
```

For binary softmax probabilities, prediction uses the strict margin rule:

```text
prediction = 1 if (p_positive - p_negative) > threshold else -1
```

### Existing pair mode

Pair mode remains the default and evaluates the judge on its trained input
shape:

```json
{
  "pair_id": "pair_001",
  "source_sample_id": "sample_001",
  "step_index": 0,
  "data_source": "geometry",
  "policy_model": "optional_policy_model",
  "anchor": "The reference reasoning segment.",
  "generated": "The candidate reasoning segment.",
  "compatibility_label": 1
}
```

Use `vpb_process.py` only to normalize records that already define those two
texts:

```bash
python src/process_reward/vpb_process.py \
  --input_jsonl path/to/defined_pairs.jsonl \
  --output_jsonl path/to/evaluator_pairs.jsonl

CUDA_VISIBLE_DEVICES=4 python src/process_reward/visualprocessbench.py \
  --input_mode pair \
  --data_path path/to/evaluator_pairs.jsonl \
  --model_path src/outputs/process_reward_judge/v1_1_0/checkpoints/final \
  --batch_size 32 \
  --max_length 256 \
  --threshold 0.5 \
  --device cuda
```

Omit `--output_dir` for a timestamped run. `--max_samples N` limits flattened
steps in context-step mode or records in pair mode. `--auto` retains the
weighted per-data-source macro-F1 threshold search and should be treated as an
in-dataset diagnostic. Inference shows a step-level `tqdm` progress bar; use
`--no_progress` to disable it.

Each run writes `metrics.json`, `predictions.jsonl`, and `run_config.json`.
Metrics include overall results plus grouped results by `policy_model` and
`data_source`. Prediction rows contain both class probabilities, their margin,
the `-1`/`1` prediction, and all neutral rows.

## Lightweight validation

These checks do not access datasets or load models:

```bash
python -m py_compile src/process_reward/*.py
python -m unittest discover -s src/process_reward -p 'test_visualprocessbench.py' -v

for script in load_visualprm generate cache_nli dataset model train inference vpb_process visualprocessbench; do
  python "src/process_reward/${script}.py" --help
done

python src/process_reward/load_visualprm.py --config_only
python src/process_reward/generate.py --config_only --canonical_jsonl PLACEHOLDER.jsonl
python src/process_reward/cache_nli.py --config_only --pairs_jsonl PLACEHOLDER.jsonl
python src/process_reward/dataset.py --config_only --input_jsonl PLACEHOLDER.jsonl
python src/process_reward/model.py --config_only --lora_target_modules query_proj,value_proj
python src/process_reward/train.py --config_only \
  --data_path PLACEHOLDER.jsonl \
  --lora_target_modules query_proj,value_proj
python src/process_reward/inference.py --config_only --checkpoint PLACEHOLDER
python src/process_reward/vpb_process.py --config_only \
  --input_jsonl PLACEHOLDER.jsonl \
  --output_jsonl PLACEHOLDER_PAIRS.jsonl
python src/process_reward/visualprocessbench.py --config_only \
  --data_path PLACEHOLDER.jsonl \
  --model_path PLACEHOLDER

python -c "import sys; sys.path.insert(0, 'src'); import process_reward.load_visualprm, process_reward.generate, process_reward.cache_nli, process_reward.dataset, process_reward.model, process_reward.train, process_reward.inference, process_reward.vpb_process, process_reward.visualprocessbench"
```

Validate pair or raw data without loading a checkpoint:

```bash
python src/process_reward/visualprocessbench.py \
  --validate_data_only \
  --data_path path/to/visualprocessbench_pairs.jsonl \
  --model_path PLACEHOLDER
```

```bash
python src/process_reward/visualprocessbench.py \
  --input_mode context_step \
  --validate_data_only \
  --data_path /mnt/data1/eunwooim/VisualProcessBench/test.jsonl \
  --model_path PLACEHOLDER
```
