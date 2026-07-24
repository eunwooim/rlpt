# Task

Implement a fast process-reward data pipeline.

Current milestone:

1. Generate positive/negative reasoning segments from VisualPRM400K.
2. Fine-tune DeBERTaV3 with LoRA and an MLP classification head.
3. The trained classifier will later be used as the process reward model.

Do not implement RL training or our final reward yet.

---

# Dataset

Current source:

OpenGVLab/VisualPRM400K-v1.1

Each sample contains:

- question
- image
- reasoning steps
- process_correctness labels

---

# Data Generation

Supported backends (default-models):

- vllm (Qwen/Qwen3-32B-Instruct)
- openai (gpt-4o-mini. I will allow you install openai library just for one time.)

Backend-specific models should NOT be hardcoded.
Leave placeholders and expose `--model_name`.

Skip all neutral steps.

```
process_correctness == 1
```

Generate:

- one positive paraphrase

Requirements:

- preserve reasoning
- preserve mathematical meaning
- natural wording

```
process_correctness == -1
```

Generate:

- one locally plausible hard negative

Requirements:

- minimal edit
- still incorrect
- preserve context

---

# In-context Examples

Support few-shot prompting.

For now:

```
IN_CONTEXT_POOL = [
    ...
]
```

Leave placeholders only.

Randomly sample

```
--num_shots
```

examples using a fixed random seed.

We will populate the pool later.

---

# Reward Model

Base encoder:

```
microsoft/deberta-v3-large
```

Train:

- LoRA
- small MLP classifier

Input:

(anchor, generated)

Output:

binary correctness probability

---

# Logging

Use W&B.

Log only:

- train loss
- validation loss
- accuracy
- F1
- learning rate

Do not upload every prediction.

---

# Outputs

Generation:

```
src/outputs/process_reward_generation/
```

Training:

```
src/outputs/process_reward_training/
```

Use timestamped runs.

---

# Validation

Only run

- --help
- import checks
- config parsing

Do not run generation or training.
