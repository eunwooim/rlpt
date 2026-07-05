# RLPT

Repository for multimodal LLM post-training research. This README contains environment setup, repository structure, and links to experiment reproduction commands. Detailed experiment notes are elablrated at each README.md files.

## Repository structure

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

## Environment setup

`requirements.txt` includes fully pinned package snapshot from the working environment. Use the same Python version and pinned package versions across workstations and SLURM nodes.

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

### Runtime paths

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
git clone <repo-url> rlpt
cd $ROOT/rlpt
conda create -y -n rlpt python=3.10.20 pip
conda activate rlpt
python -m pip install pip==26.1.2 setuptools==81.0.0 wheel==0.47.0
python -m pip install --no-deps -r requirements.txt
```

Use `--no-deps` when `requirements.txt` is a full transitive freeze. This avoids resolver changes from silently altering the package snapshot.
