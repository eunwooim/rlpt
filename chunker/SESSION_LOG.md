# Chunker session log — 2026-07-22

DeBERTa-v3-small **chunk-boundary classifier** ("chunker"): a token-level
classifier (2 labels: split / no-split) that predicts where a raw reasoning
trajectory should be segmented into steps, reproducing the human step
segmentation in VisualPRM400K-v1.1. Downstream, it segments model outputs before
a DeBERTa **process-reward model** scores each segment. NOT generation, NOT
sequence classification.

Everything lives in `/scratch/sghos104/rlpt/chunker/`.

---

## Environment (important — this Claude node has NO GPU)

- **Claude session node `sc001` is CPU-only, 1 core, no GPU** (the unmetered
  interactive node). Decision (user-approved): **build code + run CPU profiling /
  prepare here; drive ALL GPU work via `sbatch` to `public` a100.**
- **Dedicated env**: `/scratch/sghos104/rlpt/chunker/env` (Python 3.11.15;
  torch **2.13.0+cu130**, transformers **5.14.1**, datasets 5.0.0, sentencepiece,
  scikit-learn, numpy). Built with `/packages/apps/mamba/2.6.2/bin/mamba` (the
  `module load mamba` PATH edit is lost in pipes — call the absolute binary).
  ~5.4 GB (torch bundles CUDA-13 libs). On shared `/scratch`, so the GPU job
  reuses it with no reinstall.
- `export HF_HOME=/scratch/sghos104/.hf_cache`. Tokenizer = `DebertaV2Tokenizer`,
  `is_fast=True`, offset mapping works.
- **Data**: `/scratch/sghos104/rlpt/canonical.jsonl`, **9,478,315 lines, 18.3 GB**,
  read-only. Schema: `question`, `images`, `steps` (list[str]), `step_labels`
  (list[int]), `answer`, `metadata{source_sample_id, dataset, split, source_index,
  source_metadata}`.

## Step-0 profiling (full file; `profile_report.json`, `profile_sample.jsonl`)

- **steps/record**: 1 → 8,982,559 (94.77%); multi **495,756 (5.23%)**
  = {2:34,422, 3:23,566, 4:65,307, 5:60,551, 6+:311,910}. **6+ is 63% of multi**.
- **Schema notes**: `step_labels` are **binary {+1:7.28M, −1:5.06M}, no 0**
  (brief said ternary — irrelevant to chunker). `answer` empty 99.9994% (unusable).
  0 empty steps; label/step length match 100%. Single-step records embed their
  own "Step N:" markers (5.0%) → real "don't split despite internal markers" negs.
- **Leakage (decisive)**: only **969,152 unique questions**; **99.0% of records
  share a question** (max 87,502 rollouts). `source_sample_id`: 5.68M unique, 59%
  shared. **8.3%** of multi records reuse a single-step's exact text as a step.
  → **split grouped by question**.
- **Token lengths** (DeBERTa tok): joined multi mean 307 / p50 233 / p90 633 /
  p99 1024 / max 2625, **17.3% > 512**. per-step mean 47 / p50 35 / p99 211.
  single-step mean 277, 13.2% > 512.
- **Boundary cues**: 70.2% end in ASCII `.!?;:`; the 29.8% "other" =
  56.8% LaTeX `\]`, 6% `$`, ~10% `)`, **~9% CJK `。：？！` (ASCII set missed)**,
  ~11% bare digit/letter.
- **Candidate-rule recall over gold**: ASCII-only **84.25%** → expanded
  (+CJK punct +math-closers `] $ ) } %` +trailing digit) **98.75%**.

## Design decisions (all user-approved)

1. **Split grouped by `question`**; question-groups with **>1000 total rollouts
   → TRAIN only**; val/test drawn from small/medium groups, preferring multi-rich
   / single-poor groups so no one question dominates eval; **verify val/test land
   near 15k multi / 4k single** (measured in `prepare_report.json`).
2. **Expanded candidate rule as ONE shared function** (`chunker_common.
   candidate_token_mask`) imported by prepare & inference so they can't diverge;
   **re-report 1-vs-0 class balance before training**.
3. **Gold labels anchored on the last non-whitespace char** of each non-final
   step (by char offset — robust to SentencePiece newline absorption & `[UNK]`).
4. **`evaluate.py` reports split-F1 by language bucket** (en/cjk/other). If CJK
   F1 is badly degraded → **mDeBERTa-v3-base** is the planned follow-up (no change
   to the current plan).

## Two bugs caught at the Gate-4 code review — BOTH FIXED

1. **CJK candidate gate bug**: the "next char must be whitespace/EOT" gate (added
   to protect decimals like `3.14`) killed nearly all CJK candidates — Chinese/
   Japanese put no space after `。！？`. Invisible in training (gold sits at our
   `\n` joins; gold-always-wins) but would cripple inference on raw CJK text.
   **Fix**: CJK sentence punctuation is always a candidate; a following CJK char
   also counts as a valid break. **Verified**: a space-free Chinese paragraph now
   yields candidates at all 3 `。`.
2. **`min_tokens` was soft** (MINPEN=2) — could leak sub-8-token fragments that
   break downstream bipartite matching. **Fix**: hard min-8 when the text is long
   enough to permit a feasible solution (`hard_min=True`), soft fallback only for
   degenerate-short texts. Added **`frac_under_min`** to `evaluate.py`.
- Token-level gold selectability after fixes: **96.5%** (rest covered by DP force-split).

## Files (all in `chunker/`)

| file | role |
|---|---|
| `chunker_common.py` | SHARED truth: `candidate_token_mask`, `build_token_labels`, `make_windows` (content 510 + CLS/SEP, **step 384**), `gold_boundary_char_positions`, `qhash`, `lang_bucket`. Labels {1 gold, 0 candidate-non-gold, −100 ignore}; metrics over label≠−100. |
| `profile_step0.py` | Step-0 profiling (done). |
| `test_labels.py` | Gate-3 unit test — PASS on 5 boundary flavors. |
| `prepare_data.py` | extraction + question-grouped split + tokenize/window; writes `data/{train,val,test}` (arrow), `data/{val,test}_traj.jsonl`, `data/prepare_report.json`. `--smoke` for a fast subset. |
| `train.py` | HF Trainer token-classification; threshold-swept (0.1–0.9) split-F1 over candidate tokens; early-stop on `best_f1` (patience 4); bf16; eff batch ~128; auto-resume via `get_last_checkpoint`; saves `best/` + `chunker_meta.json` (threshold). |
| `chunker.py` | `chunk(text, min_tokens=8, max_tokens=220)`: windowed prob-merge, shared candidate rule, **DP constrained decoding** (score = P−threshold; max hard via −PEN forced splits; min hard-then-soft). Returns chunks / split_token_indices / split_char_offsets / segment_token_lengths. `__main__` demo. |
| `evaluate.py` | test-set token split-F1 (overall + by language), over-split (1-step), under-split (multi boundary P/R/F1), `frac_under_min`. |
| `train.sbatch` | Sol launcher: `account=grp_bshettah`, `public/public`, `--no-requeue`, `a100:1`, 12h, chunker env python. Re-submit to resume. |
| `README.md` | pipeline / resume / inference docs. |

## Gate progress (brief's 7-step order)

1. Env + hardware report — ✓ (found no GPU → sbatch execution model chosen).
2. Step-0 profiling — ✓ shown & approved.
3. Label unit test — ✓ PASS, approved.
4. Code review — ✓ approved **with 2 fixes (both applied & verified)**.
5. **Smoke train — IN PROGRESS**. Prep DONE (`data/prepare_report.json`):
   - split trajectories: train 465,480 multi / 119,995 single; **val 15,210 /
     4,005; test 15,066 / 4,000** (hits 15k/4k targets exactly).
   - windows: train **702,450** / val 20,662 / test 20,093.
   - **class balance (1:0 over candidate tokens): train 1:6.85** (under 1:10, no
     rebalance), val 1:3.15, test 1:4.59; by kind multi 1:5.28, single all-neg.
   - **IMPORTANT lesson**: this Claude node is capped at **4 GB** cgroup
     (`SLURM_MEM_PER_NODE=4096`); the in-session prepare OOM'd. Fix: `process()`
     streams windows via `Dataset.from_generator` (bounded mem) AND prep now runs
     as a **CPU sbatch** (`prepare.sbatch`, `--mem=64G`, no GPU). Data prep must
     NOT be run in-session.
   - Smoke train submitted: **`sbatch smoke_train.sbatch`** (job 59556382,
     `runs/smoke`, 60k train windows, 1 epoch, eval every 200). Waiter watching.
6. Full run — **TWO ARMS RUNNING OVERNIGHT (2026-07-22 eve)**. Smoke (job
   59556382) exposed a **NaN blow-up**: bf16 numerically unstable for DeBERTa-v3
   (probe of the saved ckpt returned NaN P(split); step-200 eval_loss spiked to
   3.15). Fix: **fp32/TF32** (not bf16), plus a `NanGuard` callback that stops+
   flags non-finite loss. Also added a **discrimination-gap** metric (mean
   P(split) gold vs non-gold, logged every eval) and **class-weighted CE**
   (`--pos_weight`, `WeightedTrainer`) — both verified on synthetic data.
   - **Arm A plain CE**: job **59556473** (`chunker_plain`) → `runs/full_plain/`,
     log `logs/slurm-full_plain-59556473.out`.
   - **Arm B weighted CE (pos_weight 6.85)**: job **59556474**
     (`chunker_weighted`) → `runs/full_weighted/`, log
     `logs/slurm-full_weighted-59556474.out`.
   - Identical otherwise: fp32, seed 42, 3 epochs, early-stop val best_f1
     (patience 5), eval every 1000, eff batch 128, 12h, checkpoint+resume.
   - Each job auto-runs `summarize_val.py` after training →
     `runs/full_{plain,weighted}/summary_val.json` (best_f1, threshold, P/R,
     gap curve, best_f1 curve, **token split-F1 by language en/cjk/other**).
     **NO test-set eval** (waits until we pick the winner).
   - Overnight monitor task `b58oudoq1` watches for early crash / completion.
   - **Status @ launch+~1min**: Arm A RUNNING on `sg002`, training started clean,
     **no NaN** (fp32 stability fix holding). Arm B PENDING (waiting for 2nd A100).
     Watch-point: fp32 ~3.8 s/it early; 16,464 total opt-steps — if 3 epochs
     don't fit in 12h, resume via re-`sbatch` (train.py auto-detects last ckpt) or
     rely on early-stop. Summary is written from the best ckpt at train end;
     if a job is walltime-killed mid-train, re-submit to finish + summarize.
   - **MORNING TODO**: user will ask to compare the two `summary_val.json`,
     pick the winning arm, then run **test-set eval** (`evaluate.py`) +
     `chunker.py` demo on the winner. New launchers: `full_plain.sbatch`,
     `full_weighted.sbatch`, `summarize_val.py`.
7. Final TEST eval + `chunker.py` demo on the winning arm — PENDING (morning).

New launchers this stage: `prepare.sbatch` (CPU prep, `full`|`smoke` arg),
`smoke_train.sbatch` (GPU smoke). Arrow data at `data/{train,val,test}` +
`data/{val,test}_traj.jsonl` + `data/prepare_report.json`.

## Immediate next step for a fresh session

1. `tail chunker/prepare_full.log`; when `data/prepare_report.json` exists, read
   it and present **split_trajectories** (val/test vs 15k/4k) and
   **class_balance_over_candidate_tokens** (train ratio_1_to_0).
2. Smoke train (small GPU job), then submit `train.sbatch` for the full run.
3. After training: `evaluate.py` (watch CJK bucket F1) + `chunker.py` demo.

## Standing constraints in force
No commits mid-experiment; git identity `SounakGhosh-10 <sghos104@asu.edu>`;
`canonical.jsonl` read-only; sbatch on `public` with `--no-requeue`; the four
`run_scale_*.sbatch` training runs remain gated on separate user approval
(unrelated to this task — do NOT submit).

## Single-step negative policy retrain (2026-08-07/09)

Finding: 83% of single-step VisualPRM records contain explicit step markers,
numbered lists, or blank-line paragraph breaks (40 strided blocks over
canonical.jsonl; reservoir estimate 86.3%). len(steps)==1 means UNSEGMENTED,
not ATOMIC — segmentation is a per-question property (0.6% of questions have
both segmented and unsegmented rollouts) and only ~5% of questions got it.
Selector unknown: not image presence (text-only 100% single-step, but image
records only 8.6% multi-step), not length (p50 202 vs 194 words). Sub-source
field is absent from canonical.jsonl (source_metadata.id == -1 everywhere).
The 120k singles used as negative-only "do not split here" examples were
therefore suppressing real boundaries.

Four arms, val/test frozen by question hash (prepare_arms.py; prepare_data.py
could NOT be re-run per arm — one random.Random(seed) stream feeds the
single-step reservoir before the val/test group shuffles, so each arm would
have silently got a different eval set). Baseline reproduced v2: 696,618 train
windows vs 696,892, balance 6.62 vs 6.64.

Val, each arm at its own best eligible threshold:
  keep  thr 0.25  bndF1 0.8017  zero 2.09%  over/1step 2.87
  mask  thr 0.40  bndF1 0.8205  zero 1.35%  over/1step 3.86   <- winner
  drop  thr 0.45  bndF1 0.8202  zero 1.41%  over/1step 4.44
  clean thr 0.45  bndF1 0.8195  zero 1.32%  over/1step 3.91
Selection rule (fixed before results): primary val boundary F1 at own best
threshold; tiebreak within 0.01 = lower single-step false-split rate. mask
wins on both.

TEST (one run, mask @ 0.40): boundary F1 0.8163 (P 0.847 / R 0.788),
val->test gap -0.004 (splits exchangeable). vs v2 shipped 0.799 (P 0.844 /
R 0.758): +0.017, ALL of it recall, precision unchanged.

Threshold curve no longer trapped: every setting 0.20-0.90 stays under the
zero-cut cap for mask; keep was capped at 0.45.

Caveat: early stopping was silently disabled in all four arms
(metric_for_best_model looked for eval_best_f1, not found), so all ran the
full 3 epochs. keep peaked at epoch 2.57; the treated arms report 3.00 and
were still improving. Identical across arms so the comparison holds, but
mask's ceiling is likely above 0.8205.

Note: over/1step rose 2.43 -> 3.89, but that reference is the unreliable
label — 83% of the 4,000 held-out single-step trajectories carry structural
markers, so some "false" splits are boundaries the pipeline never marked.

Shipped: release/chunker-deberta-v3-small-v2-mask (thr 0.40, min/max 8/220).

## Single-step behavioural eval + gold structural baseline (2026-08-10)

Sample: 32,971 single-step IMAGE records, source-stratified across all 38
annotation subsets (1,000/source where available). Join to source via
question-text after stripping the rendered prompt template; 24.6% of singles
matched. SCOPE: quality-selected slice (questions present in the filtered
pipeline outputs), not the raw 8.86M corpus.

Marker cut-rates (rates, NOT accuracy -- no reference segmentation exists for
single-step records):
  stepN    1,993/2,231    = 0.893
  numlist 82,957/115,361  = 0.719
  blank  149,556/215,477  = 0.694
  bullet   4,226/142,101  = 0.030
  cuts landing at NO marker: 2.7%   zero-cut records: 5.72%

Hand-inspected 6 of the 4,250 cut_no_marker cases: all land after a completed
sentence or closed LaTeX display block, at genuine reasoning transitions.

Structural rates vs GOLD baseline (canonical.jsonl, all 495,756 multi-step
records, 2,860,657 boundaries, job 60885363):
  flag                        GOLD%   MODEL%   delta
  mid_sentence_pair            0.12     0.01   -0.11
  prev_ends_on_operator        0.84     0.14   -0.70
  prev_odd_ending              2.78     0.66   -2.12
  next_starts_continuation     9.22     7.18   -2.03
  prev_no_terminal_punct       1.74     2.42   +0.68
  unbalanced_[]                0.35     0.61   +0.27
  unbalanced_()                0.45     0.64   +0.19
  unbalanced_latex             0.38     0.48   +0.10

CONCLUSION: the chunker cuts mid-sentence and mid-equation LESS often than the
gold annotation itself. Only the unbalanced-delimiter flags run above gold,
concentrated in mavis_function_{cos,sin,tan} (3-4% vs 0.16% in poly) -- long
LaTeX display blocks hitting the max_tokens=220 cap and forcing an in-block
cut. Decoder-config issue, not classifier; argues for raising the cap once the
PRM's per-step token budget is known.

Cross-corpus (xdataset V2 strip-and-resegment, macro-F1):
                 v1 base   keep@.35   mask@.35   mask@.40
  prm800k          0.940      0.940      0.941     0.941
  processbench     0.646      0.595      0.716     0.722
  visualprm        0.796      0.782      0.823     0.822
Gains land where the model was weakest; PRM800K has no headroom (cov 0.929).
ProcessBench density 1.62-1.70 = over-segmenting; F1 gain comes from coverage
rising faster than precision falls. keep@.35 scoring below the v1 baseline
bounds run-to-run variance.
