# VisualPRM Process-Reward Pipeline

This package implements a fast, modular text-pair process-reward pipeline. Each stage is independently runnable and writes a timestamped, non-overwriting run directory.

```text
OpenGVLab/VisualPRM400K-v1.1
  -> load_visualprm.py -> canonical.jsonl
  -> generate.py       -> pairs.jsonl
                         |-> cache_nli.py -> nli_features.jsonl (optional analysis)
                         `-> train.py     -> grouped splits + LoRA adapter + MLP head
  -> inference.py      -> compatibility probability
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

## Lightweight validation

These checks do not access datasets or load models:

```bash
python -m py_compile src/process_reward/*.py

for script in load_visualprm generate cache_nli dataset model train inference; do
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

python -c "import sys; sys.path.insert(0, 'src'); import process_reward.load_visualprm, process_reward.generate, process_reward.cache_nli, process_reward.dataset, process_reward.model, process_reward.train, process_reward.inference"
```
