# RLPT

Repository for multimodal LLM post-training research. This README is limited to environment reconstruction, repository layout, and experiment reproduction commands.

## Repository Layout

```text
README.md
LICENSE
requirements.txt
.gitignore
src/
  train/                 # SFT, RLVR, GRPO, and veRL launch scripts
  metrics/
    reliable/            # reliability tests for text encoders/scorers
  outputs/               # ignored experiment outputs
  ckpts/                 # ignored checkpoints
  archive/               # ignored debug artifacts and small experiments
visualize/               # ignored visualization/debug images
```

## Environment Setup

`requirements.txt` is expected to contain a fully pinned package snapshot from the working environment. Use the same Python version and pinned package versions across workstations and SLURM nodes.

Reference environment:

```text
Python: 3.10.20
pip: 26.1.2
setuptools: 81.0.0
wheel: 0.47.0
Conda root: /mnt/data2/eunwooim/.conda
Shared Hugging Face cache: /mnt/shared/shared_hf_home
Allowed GPU indices: 4,5,6,7
```

### Clone

```bash
git clone <repo-url> rlpt
cd rlpt
```

### Runtime Paths

Set machine-dependent paths through environment variables.

```bash
export ROOT=${ROOT:-$PWD}
export CONDA_ROOT=${CONDA_ROOT:-/mnt/data2/eunwooim/.conda}
export HF_HOME=${HF_HOME:-/mnt/shared/shared_hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
export DATASETS_CACHE=${DATASETS_CACHE:-$HF_HOME/datasets}
export WANDB_DIR=${WANDB_DIR:-$ROOT/src/outputs/wandb}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
```

For SLURM or another filesystem, override before setup:

```bash
export ROOT=/scratch/eunwooim/rlpt
export HF_HOME=/scratch/eunwooim/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
export DATASETS_CACHE=$HF_HOME/datasets
export WANDB_DIR=$ROOT/src/outputs/wandb
```

### Create Environment

Recommended clean reproduction:

```bash
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda create -y -n rlpt python=3.10.20 pip
conda activate rlpt
python -m pip install pip==26.1.2 setuptools==81.0.0 wheel==0.47.0
python -m pip install --no-deps -r requirements.txt
```

Use `--no-deps` when `requirements.txt` is a full transitive freeze. This avoids resolver changes from silently altering the package snapshot.

If `rlpt` already exists and should not be overwritten, create a machine-specific environment:

```bash
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda create -y -n rlpt_repro python=3.10.20 pip
conda activate rlpt_repro
python -m pip install pip==26.1.2 setuptools==81.0.0 wheel==0.47.0
python -m pip install --no-deps -r requirements.txt
```

### Verify Environment

```bash
python --version
python -m pip freeze > /tmp/rlpt.freeze
diff -u requirements.txt /tmp/rlpt.freeze
```

The diff should be empty for an identical pip-visible environment.

Check key packages:

```bash
python - <<'PY'
import importlib
import os

mods = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "wandb",
    "numpy",
    "pandas",
    "scipy",
]

for name in mods:
    m = importlib.import_module(name)
    print(name, getattr(m, "__version__", "unknown"))

import torch
print("cuda available", torch.cuda.is_available())
print("torch cuda", torch.version.cuda)
print("visible cuda devices", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("HF_HOME", os.environ.get("HF_HOME"))
print("DATASETS_CACHE", os.environ.get("DATASETS_CACHE"))
print("WANDB_DIR", os.environ.get("WANDB_DIR"))
PY
```

`python -m pip check` may be useful diagnostically, but exact reproduction is defined by matching `requirements.txt` unless the environment is intentionally refreshed.

## Experiment Reproduction Commands

Append finalized commands here as experiments are added.

### Reliability Test

```bash
# TODO
```

### SFT

```bash
# TODO
```

### RLVR / GRPO

```bash
# TODO
```

### Evaluation

```bash
# TODO
```
