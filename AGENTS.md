# AGENTS.md

## Role

You are the implementation agent. The user is the research architect.

Move quickly, but make changes small enough for the user to verify. Do not make broad research, infrastructure, or design decisions without asking.

## Safety

- Only read/write files under the current repository.
- Do not modify `/mnt/data1/eunwooim`, `/mnt/data2/eunwooim`, `/mnt/shared`, or any absolute path unless explicitly instructed.
- Do not install packages unless explicitly instructed.
- Do not download external datasets unless explicitly instructed.
- Do not run long GPU jobs unless explicitly instructed.
- Do not delete files/directories without asking.
- Do not refactor unrelated code.
- Do not create background services.
- Do not hardcode secrets, API keys, tokens, or passwords.
- Do not use GPU indices outside `4,5,6,7`.
- Use `CUDA_VISIBLE_DEVICES` explicitly for GPU commands.
- Only work on tmux session named "rlpt".

## Project

This is a multimodal LLM post-training research repo.

The broader project studies efficient multimodal LLM post-training with verifiable rewards, including SFT, RLVR, GRPO, and reasoning-segment reward modeling.

The current milestone is a reliability-test framework for selecting text encoders/scorers used in reasoning-segment similarity rewards.

## Environment

Use machine-dependent paths through environment variables whenever possible:

```bash
ROOT=${ROOT:-$PWD}
CONDA_ROOT=${CONDA_ROOT:-/mnt/data2/eunwooim/.conda}
HF_HOME=${HF_HOME:-/mnt/shared/shared_hf_home}
HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/shared/shared_hf_home/hub}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/shared/shared_hf_home/hub}
DATASETS_CACHE=${DATASETS_CACHE:-/mnt/shared/shared_hf_home/datasets}
WANDB_DIR=${WANDB_DIR:-$ROOT/src/outputs/wandb}
```

Workstation defaults:

- Conda root: `/mnt/data2/eunwooim/.conda`
- Shared Hugging Face cache: `/mnt/shared/shared_hf_home`
- Allowed GPU indices: `0,1,2,3,4,5,6,7`

SLURM or alternate-machine runs should override `ROOT`, for example:

```bash
export ROOT=/scratch/eunwooim/<project_name>
```

Do not assume a specific conda environment exists unless the user says so. If an environment is needed, create scripts that document how to create it, but do not create or install packages unless explicitly asked.

## Preferred Software Stack

Use these when relevant:

- Python
- PyTorch
- Hugging Face `transformers`
- Hugging Face `datasets`
- `sentence-transformers`
- `bert-score`
- `scikit-learn`
- `pandas`, `numpy`
- `wandb`
- `accelerate`
- veRL for RLVR/GRPO training scripts
- vLLM for fast rollout when used by veRL
- FlashAttention if available and compatible
- DeepSpeed only when explicitly needed; do not add it by default to the first reliability-test implementation

veRL is preferred for RLVR/GRPO in this project because it is already used by the user and supports efficient LLM/VLM RL training. Do not claim it is universally the fastest framework.

## Repository Layout

Expected structure:

```text
README.md
LICENSE
requirements.txt
.gitignore
src/
  train/                 # SFT/RL/GRPO/veRL launch scripts
  metrics/               # reliability-test framework for text scorers
  outputs/               # ignored; experiment outputs
  ckpts/                 # ignored; checkpoints
  archive/               # ignored; small experiments/debugs/plots
visualize/               # ignored; debug/test images
```

`.gitignore` should include:

```text
src/outputs/
src/ckpts/
src/archive/
visualize/
wandb/
.cache/
.cache_hf/
__pycache__/
*.pyc
*.log
.env
.venv/
.ipynb_checkpoints/
```

## README Expectations

The root `README.md` shall be the instructions for opensourcing.
It should include:

- project purpose
- environment setup
- conda path assumptions
- Hugging Face cache setup
- GPU restrictions
- future training-script location under `src/train/`
- commands for how to reproduce an experiment

## Git Policy

- It is allowed to use git status/diff/log.
- Do not commit unless explicitly asked.
- Do not push unless explicitly asked.
- If the user says `PUSH_OK=1`, then pushing is allowed for that task only.
- Before any commit or push, show the branch name and summary of changed files.

## Working Style

Before non-trivial work, write:

```text
PLAN:
1. ...
2. ...
→ Executing unless redirected.
```

If assumptions are required, write:

```text
ASSUMPTIONS:
1. ...
2. ...
→ Correct me now or I will proceed.
```

When confused, stop and ask. Do not silently choose between conflicting interpretations.

After changes, report:

- files changed
- commands run
- pass/fail status
- missing dependencies
- anything intentionally not touched

## Reliability-Test Goal

We need to test whether candidate text encoders/scorers are reliable enough to compare decomposed reasoning segments for reward modeling.

A reliable scorer should:

1. give high scores to positive paraphrases or semantically equivalent reasoning segments;
2. give low scores to hard negatives that preserve lexical overlap but change meaning;
3. be robust to negation, object/attribute/relation swaps/replacements, numerical difference for bounding box coordinates or count errors, hallucinations, omissions, and contradictions;
4. support thresholding so that weak generic matches do not accumulate excessive reward.

## Reliability-Test Modes

Support two example modes.

### 1. Ranking Mode

Used when a reference segment exists.

Fields:

```json
{
  "id": "role_001",
  "category": "role_swap",
  "mode": "ranking",
  "reference": "a girl is chasing a cat",
  "positive": "a girl is running after a cat",
  "negative": "a cat is chasing a girl"
}
```

Criterion:

```text
score(reference, positive) > score(reference, negative)
```

### 2. Separation Mode

Used when a benchmark only provides a positive caption and a hard negative, with no independent reference.

Fields:

```json
{
  "id": "wg_001",
  "category": "role_swap",
  "mode": "separation",
  "positive": "a girl is chasing a cat",
  "negative": "a cat is chasing a girl"
}
```

Criterion:

```text
raw_score(positive, negative) < tau
```

Use raw score for this pass/fail criterion. Threshold-transformed scores are diagnostic but should not define separation-mode pass/fail.

## Tau Modes

For raw score `S in [0, 1]`, support:

```text
none:   W = S
hard:   W = S if S >= tau else 0
scaled: W = max(0, S - tau) / (1 - tau)
```

The `scaled` mode is important because a barely above-threshold generic match should contribute very little.

## Initial Scorers

First implementation:

- embedding cosine with `sentence-transformers/all-mpnet-base-v2`

Later implementations:

- NLI scorer with `microsoft/deberta-large-mnli`
- BERTScore scorer
- cross-encoder/reranker scorer

Do not implement later scorers until explicitly asked.

## NLI Scorer Definition for Future Work

For reference segment `g` and candidate segment `r`, compute:

```text
E_gr = P(entailment | premise=g, hypothesis=r)
E_rg = P(entailment | premise=r, hypothesis=g)
C_gr = P(contradiction | premise=g, hypothesis=r)
C_rg = P(contradiction | premise=r, hypothesis=g)
```

Default scalar compatibility:

```text
S = clamp(0.7 * E_gr + 0.3 * E_rg - 0.5 * max(C_gr, C_rg), 0, 1)
```

NLI is used to distinguish support, partial coverage, equivalence, and contradiction. It is not a replacement for typed verifiers such as IoU, OCR edit distance, object-class matching, or relation matching.

## Future Reward Direction

The reliability test is a precursor to a saturated F-score reasoning-segment reward.

Given ground-truth segments `G={g_i}` and rollout segments `R={r_j}`, edge scores `s_ij` are threshold-scaled into `w_ij`.

Recall can be computed with noisy-OR coverage:

```text
Recall = (1 / |G|) * sum_i [1 - prod_j (1 - w_ij)]
```

Precision can be computed by max support:

```text
Precision = (1 / |R|) * sum_j max_i w_ij
```

Then combine with F-beta. This future reward is not part of the first implementation unless explicitly requested.

## Completion Criteria

A task is complete only when:

1. requested files are created or updated;
2. the smallest relevant smoke test is run, or missing dependencies are reported clearly;
3. no unrelated files are changed;
4. commands and results are summarized.
