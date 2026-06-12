# RLPT — Environment Setup (shared)

RL post-training of Qwen2.5-VL-3B on ScienceQA with a composite reward
(format + SBERT-bipartite reasoning match + hard answer − hallucination penalty).
Results storyline: `docs/results_storyline.md` · raw tables: `docs/results/` ·
full project notes: `CLAUDE.md`.

This document is the **common environment reference** for everyone working on the
project (Sounak + supervisor). There are two environments, both prebuilt on Sol and
**group-readable by `grp_bshettah`** — if you are in the group you can use them
directly, no installation needed.

| Env | Prefix | Purpose |
|---|---|---|
| `rlpt-data` | `/scratch/sghos104/envs/rlpt-data` | dataset building, inspection, scoring-only scripts |
| `rlpt-train` | `/scratch/sghos104/envs/rlpt-train` | GRPO training, vLLM generation, all GPU evals |

They are kept separate on purpose: the RL stack's pins (torch 2.7.1, transformers
4.56.2) conflict with nothing in the data env, and rebuilding one never breaks the
other.

---

## 1. Use the shared environments (recommended)

No activation required — invoke the interpreter by absolute path:

```bash
# data / scoring work
/scratch/sghos104/envs/rlpt-data/bin/python src/data_factory/smoke_test.py

# training / generation / eval work
/scratch/sghos104/envs/rlpt-train/bin/python tools/gen_eval_outputs.py --help
```

Or activate interactively:

```bash
module load mamba/latest
source activate /scratch/sghos104/envs/rlpt-train   # or .../rlpt-data
```

All sbatch scripts in `src/train/` and `src/data_factory/` already point at these
prefixes — submitting them works for any group member as-is.

### Required runtime environment variables

The sbatch scripts set these; for ad-hoc shells you need them too:

```bash
export HF_HOME=/scratch/sghos104/rlpt/data/hf_cache   # all models/datasets cached here
export HF_HUB_OFFLINE=1                               # compute nodes: never hit the Hub
export NLTK_DATA=/scratch/sghos104/rlpt/data/nltk_data
unset ROCR_VISIBLE_DEVICES                            # Sol sets the AMD var; clashes with CUDA

# training/generation only:
export PYTHONPATH=/scratch/sghos104/rlpt/src/train/flash_attn_shim:$PYTHONPATH
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn             # standalone vLLM scripts (outside Ray)
```

The `flash_attn_shim` entry is **mandatory** for training/rollout: real flash-attn
wheels need glibc ≥ 2.32 and Sol is RHEL8 (glibc 2.28), so the repo ships a
pure-torch/SDPA shim (`src/train/flash_attn_shim/`) that verl and vLLM import
instead. Do not `pip install flash-attn` into the env.

---

## 2. Rebuild from scratch (only if you can't use the shared prefixes)

The exact working stack is frozen in **`src/train/requirements-train.lock.txt`**
(verl 0.8.0 · vllm 0.10.1.1 · torch 2.7.1+cu126 · transformers 4.56.2). These pins
are load-bearing — "latest everything" hangs verl's Ray workers, transformers 5.x
breaks Qwen2.5-VL rope for vLLM, and vllm < 0.9 lacks the async server verl 0.8
requires. Build on a node with internet (login/OOD shell is fine; nothing compiles):

```bash
module load mamba/latest
mamba create -y -p /scratch/$USER/envs/rlpt-train python=3.11
source activate /scratch/$USER/envs/rlpt-train
pip install -r src/train/requirements-train.lock.txt
python -m spacy download en_core_web_sm
```

(`src/train/setup_train_env.sh` + `requirements-train.txt` are the original
pre-bring-up build notes; prefer the lockfile, which is frozen from the live env.)

For the data env, `requirements.txt` at the repo root is a full freeze of
`rlpt-data`:

```bash
mamba create -y -p /scratch/$USER/envs/rlpt-data python=3.11
source activate /scratch/$USER/envs/rlpt-data
pip install -r requirements.txt
```

---

## 3. Verify

```bash
P=/scratch/sghos104/envs/rlpt-train/bin/python
$P -c "import verl, vllm, torch, transformers as t; \
  print(verl.__version__, vllm.__version__, torch.__version__, t.__version__)"
# expect: 0.8.0 0.10.1.1 2.7.1+cu126 4.56.2

# reward stack (CPU is fine)
RLPT_REWARD_DEVICE=cpu $P -c "from tools.graph_match_reward import compute_score; print('reward OK')"
```

---

## 4. Cluster notes (Sol)

- Partition `public` (or `htc` for jobs < 4 h); **always `#SBATCH --no-requeue`**.
- Login/OOD shells are for editing and downloads only — anything that computes goes
  through `sbatch`/`srun` (the OOD VSCode shell has 1 CPU and no GPU).
- Evals fit on 1×A100; training used 2×A100 with `--mem=400G`.
- Hyperparameters for the runs the results came from:
  `src/train/grpo_scienceqa_qwen2_5vl.sbatch` (main arm) and
  `src/train/grpo_scienceqa_ablation_nomatch.sbatch` (w_match=0 ablation);
  reward weights live in `tools/graph_match_reward.py`
  (`RLPT_W_{FORMAT,MATCH,ANSWER,PUN}`, defaults 1/2/5/1).
