# DeBERTa-v3-small chunk-boundary classifier ("chunker")

Token-level classifier that predicts where a raw reasoning trajectory should be
split into steps, reproducing the human step segmentation in VisualPRM400K-v1.1.
It segments model outputs before a downstream DeBERTa process-reward model scores
each segment. This is **token classification** (2 labels: split / no-split).

## Files
| file | role |
|---|---|
| `chunker_common.py` | **Single source of truth**: expanded candidate rule, label construction, windowing, question hash, language bucket. Imported by everything so train & inference can't diverge. |
| `prepare_data.py` | One pass over `canonical.jsonl` → question-grouped train/val/test, tokenize + window, cache arrow datasets, report class balance. |
| `train.py` | Finetune with HF Trainer; split-F1 (threshold-swept) over candidate tokens; early stopping; auto-resume. |
| `chunker.py` | Inference: `chunk(text, min_tokens=8, max_tokens=220)` with windowed prob-merging + DP constrained decoding. |
| `evaluate.py` | Test-set token split-F1 (overall + by language), over-split (1-step) & under-split (multi) analysis. |
| `train.sbatch` | Sol launcher (public / a100:1 / no-requeue). |
| `profile_step0.py`, `test_labels.py` | Step-0 profiling and the label unit test. |

## Environment
Dedicated env at `env/` (Python 3.11; torch 2.13+cu130, transformers 5.14).
```bash
export HF_HOME=/scratch/sghos104/.hf_cache
PY=/scratch/sghos104/rlpt/chunker/env/bin/python
```

## Pipeline
```bash
# 1. Prepare data (CPU, ~20-40 min; run on the login/interactive node in background
#    or a CPU job — no GPU needed). --smoke makes a small fast subset.
$PY prepare_data.py --out_dir data                 # full
$PY prepare_data.py --out_dir data_smoke --smoke   # smoke subset

# 2. Train (GPU). Submit and forget — auto-resumes on requeue/timeout.
sbatch train.sbatch
#   smoke: $PY train.py --data_dir data_smoke --output_dir runs/smoke --max_train_samples 60000 --epochs 1

# 3. Evaluate on the untouched test split.
$PY evaluate.py --model_dir runs/deberta_chunker/best

# 4. Inference demo (predicted vs gold boundaries on 5 held-out trajectories).
$PY chunker.py --model_dir runs/deberta_chunker/best
```

## Resuming
`train.sbatch` calls `train.py`, which detects the newest checkpoint in
`runs/deberta_chunker/` via `get_last_checkpoint` and passes it to
`trainer.train(resume_from_checkpoint=...)`. To resume after a timeout, just
`sbatch train.sbatch` again. Checkpoints every 2000 steps, `save_total_limit=3`.

## Using the chunker in code
```python
import chunker as ck
ck.load("runs/deberta_chunker/best")          # loads model + trained threshold
out = ck.chunk(raw_text)                       # min_tokens=8, max_tokens=220
out["chunks"]                                  # list[str] segments
out["split_char_offsets"]                      # char offset where each chunk ends
```

## Key design decisions (see docs / commit history for the numbers)
- **Split grouped by question** (99% of records share a question; naive splits leak).
  Question-groups with >1000 total rollouts are forced train-only so no single
  question dominates evaluation.
- **Expanded candidate rule** (ASCII + CJK punctuation + math closers `] $ ) } %`
  + trailing digits + list items): raises boundary recall 84% → 98.7% over the
  multilingual / heavy-LaTeX data. Defined once in `chunker_common.py`.
- **Labels anchored on the last non-whitespace char** of each non-final step, by
  character offset (robust to SentencePiece newline absorption and `[UNK]` chars).
- **Known limitation**: deberta-v3-small `[UNK]`s some CJK characters; `evaluate.py`
  reports split-F1 by language bucket to quantify it. If CJK F1 is badly degraded,
  mDeBERTa-v3-base is the planned follow-up.
