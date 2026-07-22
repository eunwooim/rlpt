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

## Completion Criteria

A task is complete only when:

1. requested files are created or updated;
2. the smallest relevant smoke test is run, or missing dependencies are reported clearly;
3. no unrelated files are changed;
4. commands and results are summarized.
