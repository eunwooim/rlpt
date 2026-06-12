# RLPT — Common Environment Setup

RL post-training of Qwen2.5-VL-3B on ScienceQA with a composite reward
(format + SBERT-bipartite reasoning match + hard answer − hallucination penalty).
Results storyline: `docs/results_storyline.md` · raw tables: `docs/results/` ·
full project notes: `CLAUDE.md`.

Everyone on the project builds the **same environment from the same lockfiles in
this repo** — that is the common environment. Filesystem sharing isn't possible
(`/scratch/<user>/` is group-private and we're in different groups), so identical
builds, not a shared prefix, are the contract. Same Python, same 219 pins, same
paths relative to `$USER` — results are reproducible across accounts.

There are two envs, kept separate so the heavy RL stack never disturbs the
data/scoring stack:

| Env | Prefix (per user) | Lockfile | Purpose |
|---|---|---|---|
| `rlpt-data` | `/scratch/$USER/envs/rlpt-data` | `requirements.txt` (repo root) | dataset building, inspection, CPU scoring |
| `rlpt-train` | `/scratch/$USER/envs/rlpt-train` | `src/train/requirements-train.lock.txt` | GRPO training, vLLM generation, GPU evals |

---

## 1. Build (once per user)

Run on a login/OOD shell (needs internet; nothing compiles):

```bash
module load mamba/latest

# training env — the pins are load-bearing, do not upgrade ad hoc:
# "latest everything" hangs verl's Ray workers, transformers 5.x breaks
# Qwen2.5-VL rope for vLLM, vllm<0.9 lacks the async server verl 0.8 needs.
mamba create -y -p /scratch/$USER/envs/rlpt-train python=3.11
source activate /scratch/$USER/envs/rlpt-train
pip install -r src/train/requirements-train.lock.txt   # verl 0.8.0 · vllm 0.10.1.1 · torch 2.7.1+cu126 · transformers 4.56.2
python -m spacy download en_core_web_sm
source deactivate

# data env
mamba create -y -p /scratch/$USER/envs/rlpt-data python=3.11
source activate /scratch/$USER/envs/rlpt-data
pip install -r requirements.txt
```

**Do NOT `pip install flash-attn`.** Real flash-attn wheels need glibc ≥ 2.32 and
Sol is RHEL8 (glibc 2.28). The repo ships a pure-torch/SDPA shim
(`src/train/flash_attn_shim/`) that verl and vLLM import via `PYTHONPATH` instead
(see below); installing the real package into the env breaks the import order.

## 2. Per-user paths and runtime variables

The repo's sbatch scripts currently hardcode `sghos104` paths. If you're running
from your own clone, set these to **your own** locations (and adjust the prefixes
inside any sbatch you submit):

```bash
export HF_HOME=/scratch/$USER/rlpt_hf_cache        # models/datasets cache (~30 GB on first download)
export NLTK_DATA=/scratch/$USER/rlpt_nltk_data
unset ROCR_VISIBLE_DEVICES                         # Sol exports the AMD var; clashes with CUDA

# training / generation / eval only:
export PYTHONPATH=<repo-root>/src/train/flash_attn_shim:$PYTHONPATH   # mandatory (see above)
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn          # standalone vLLM scripts (outside Ray)
export HF_HUB_OFFLINE=1                            # on COMPUTE nodes only, after the cache is warm
```

First use: warm the cache from a login shell (internet, no `HF_HUB_OFFLINE`) —
`Qwen/Qwen2.5-VL-3B-Instruct`, `sentence-transformers/all-MiniLM-L6-v2`,
`derek-thomas/ScienceQA`, plus NLTK `punkt`/`punkt_tab` into `$NLTK_DATA`.
Compute nodes then run fully offline.

## 3. Verify

```bash
P=/scratch/$USER/envs/rlpt-train/bin/python
$P -c "import verl, vllm, torch, transformers as t; \
  print(verl.__version__, vllm.__version__, torch.__version__, t.__version__)"
# expect: 0.8.0 0.10.1.1 2.7.1+cu126 4.56.2

# reward stack import (CPU is fine; run from the repo root)
RLPT_REWARD_DEVICE=cpu $P -c "from tools.graph_match_reward import compute_score; print('reward OK')"
```

## 4. Cluster notes (Sol)

- Partition `public` (or `htc` for jobs < 4 h); **always `#SBATCH --no-requeue`**.
- Login/OOD shells are for editing and downloads only — anything that computes goes
  through `sbatch`/`srun` (the OOD VSCode shell has 1 CPU and no GPU).
- Evals fit on 1×A100; training used 2×A100 with `--mem=400G`.
- Hyperparameters for the runs behind `docs/results/`:
  `src/train/grpo_scienceqa_qwen2_5vl.sbatch` (main arm) and
  `src/train/grpo_scienceqa_ablation_nomatch.sbatch` (w_match=0 ablation);
  reward weights live in `tools/graph_match_reward.py`
  (`RLPT_W_{FORMAT,MATCH,ANSWER,PUN}`, defaults 1/2/5/1).

## 5. Keeping the lockfiles authoritative

If you must change a package version, change it **via the lockfile in a commit**
(refreeze with `pip freeze | grep -v flash` after testing a full smoke run:
`src/train/grpo_scienceqa_smoke.sbatch`), so both environments stay identical.
`src/train/requirements-train.txt` and `setup_train_env.sh` are the historical
pre-bring-up notes — the lockfile supersedes them.
