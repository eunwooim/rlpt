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
  metrics/               # reliability tests for text encoders/scorers
  data/                  # dataset generation pipelines
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

## Process-reward pipeline

The fast VisualPRM generation, frozen-NLI caching, DeBERTaV3 LoRA training, and inference flow is documented in [`src/data/process_reward/README.md`](src/data/process_reward/README.md). Each stage is independently runnable and exchanges dataset-agnostic JSONL records.

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

- NLI scorer with `microsoft/deberta-xlarge-mnli`
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
