# Task

Complete the process-reward evaluation pipeline.

Current milestone:

1. Generated positive and negative reasoning segments from VisualPRM400K.
2. Fine-tuned DeBERTaV3 with LoRA and an MLP classification head.
3. Verified that the compatibility judge converges on the current train/validation/test split.
4. Revise and complete:

```text
src/process_reward/visualprocessbench.py
```

The file is currently incomplete, but its existing evaluation logic should be preserved.

Do not implement RL training or the final reward yet.

---

# Dataset

Current sources:

- OpenGVLab/VisualPRM400K-v1.1
- OpenGVLab/VisualProcessBench

Generated pair schema:

```json
{
  "pair_id": "pair_...",
  "source_sample_id": "visualprm_...",
  "step_index": 0,
  "compatibility_label": 0,
  "anchor": "...",
  "generated": "..."
}
```

Train, validation, and test splits are grouped by `source_sample_id`.

---

# Current Goal

Complete:

```text
src/process_reward/visualprocessbench.py
```

Requirements:

- preserve the existing evaluation logic
- complete missing functions and placeholders
- add configurable data and model paths
- load the trained DeBERTaV3 LoRA compatibility judge
- run batched inference
- compute and save the existing evaluation metrics
- write compact prediction outputs
- avoid hardcoded paths
- do not require removed generation metadata

Expose at least:

```text
--data_path
--model_path
--output_dir
--batch_size
--max_length
--threshold
```

---

# Reward Model

Base encoder:

```text
microsoft/deberta-v3-large
```

Trained components:

- LoRA adapters
- small MLP classifier

Input:

```text
(anchor, generated)
```

Output:

```text
binary compatibility probability
```

---

# Evaluation Concern

The current IID test result confirms successful optimization, but not out-of-domain generalization.

Potential shortcut risks include:

- positives dominated by paraphrasing
- negatives dominated by small numerical edits
- lexical differences between positive and negative examples

Do not claim generalization from the current IID metrics alone.

Where practical, preserve metadata that allows later evaluation by error type or transformation category.

---

# Logging

Log only:

- evaluation loss
- accuracy
- precision
- recall
- F1
- runtime
- samples per second

Do not upload every prediction to W&B.

---

# Outputs

Evaluation:

```text
src/outputs/process_reward_evaluation/
```

Recommended files:

```text
metrics.json
predictions.jsonl
run_config.json
```

Use timestamped or explicitly named runs.

---

# Validation

Only run:

- `--help`
- import checks
- config parsing
- data-schema validation
- a small synthetic or limited evaluation test

Do not run full evaluation, generation, training, or RL by default.
