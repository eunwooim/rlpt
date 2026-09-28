# Filtered (tau=0.85) vs unfiltered VisualPRM: 2x2 SFT/RLVR comparison — smoke scale

**Date:** 2026-07-18 · **Scale caveat:** these runs used the smoke-scale budget
(TOKEN_BUDGET=1e6 nominal → 245 SFT optimizer steps at effective batch 1; 4 GRPO
steps). They establish the pipeline and directional findings; the DATA_RATIO=1
scale runs (effective batch 128, TOKEN_BUDGET 1e8) are being configured as of this
writing and supersede these numbers when done.

## 1. What was run

| Run | Job | Output dir | Wall |
|---|---|---|---|
| SFT unfiltered (baseline) | 58944776 | `src/outputs/train/visualprm_sft_first_baseline/run_20260712_045241` | 13:59 |
| RLVR unfiltered (baseline) | 58945708 | `src/outputs/train/visualprm_rlvr_first_baseline/run_20260712_060302` | 21:44 |
| SFT filtered tau=0.85 | 59198050 | `src/outputs/train/visualprm_sft_filtered_tau085/run_20260718_001034` | 15:02 |
| RLVR filtered tau=0.85 | 59198051 | `src/outputs/train/visualprm_rlvr_filtered_tau085/run_20260718_001542` | 28:12 |
| Eval 5 models x 2 benchmarks | 59199137 FAILED (vLLM fork-CUDA; fixed with `VLLM_WORKER_MULTIPROC_METHOD=spawn`) → 59199631 COMPLETED | `src/outputs/eval_visualprm_2x2/` | 1:02:38 |

**Configs actually used.** Filtered runs are byte-identical to the unfiltered
baselines (seed 42, lr 1e-5, max_seq_len 4096, TOKEN_BUDGET 1e6, SFT batch 1x1,
RLVR train_batch 32 / mini 8 / micro 1 / rollout_n 4 / prompt+response 2048+2048,
rewards accuracy+format via `baseline_rewards.py`, no bipartite term, `steps`
field unused) **except** three deliberate control changes:
`RLPT_FILTERED_JSONL=<filtered jsonl>` (activates `FilteredVisualPRMJsonlSource`,
Edit 6 — re-reads full raw records by (source_file, line_index), so prompts and
responses match the unfiltered pipeline byte-for-byte), `DATA_RATIO=1`, and
`MAX_TRAIN_SAMPLES=5651`. The latter two reproduce the unfiltered runs' exact
sample geometry (5,651 = floor(565,149 x 0.01), seed-42 draw; RLVR val 113 /
train 5,538 exactly). Verified: 8/8 sampled adapter records `==` raw; RLVR
parquet prompt/ground_truth equality.

## 2. Evaluation setup

- Benchmarks: ScienceQA test (1,836) and MMK12 (1,024), parquets on disk with the
  `<think>/<answer>` prompt template; greedy vLLM generation, max 1,024 new tokens,
  identical prompts/settings for all five models (`src/eval/generate_eval.py`).
- Scoring (`src/eval/score_eval.py`): extraction = last `<answer>` block, else
  "Final answer:" line, else last non-empty line; `answer_correct` replicates the
  old validation track's semantics (exact/letter/index/unambiguous-substring).
- **Note:** the unfiltered checkpoints had never been evaluated before this
  session; the ScienceQA/MMK12 numbers in `docs/results_scienceqa_mmk12.md` are
  from the earlier match-reward track and are not comparable.

## 3. Results

### 3.1 Primary scorer (pre-registered extraction chain)

acc / format_strict / format_any:

| Model | ScienceQA | MMK12 |
|---|---|---|
| base (Qwen2.5-VL-3B-Instruct) | 0.7369 / 0.480 / 0.767 | 0.3486 / 0.169 / 0.823 |
| sft_unfiltered | 0.6520 / 0.605 / 0.861 | 0.3252 / 0.255 / 0.852 |
| sft_filtered_tau085 | **0.6770** / 0.676 / 0.918 | **0.3682** / 0.406 / 0.881 |
| rlvr_unfiltered | 0.7484 / 0.929 / 0.966 | 0.3340 / 0.357 / 0.798 |
| rlvr_filtered_tau085 | 0.7473 / 0.009 / 0.398 | 0.1895 / 0.008 / 0.174 |

### 3.2 The filtered-RLVR "MMK12 collapse" is a format/extraction artifact

The 0.1895 looked like a training collapse; it is not. Diagnosis on the
generations themselves:

- Only 8.0% of filtered-RLVR MMK12 outputs hit the 1,024-token cap (unfiltered:
  3.2%); mean length 556 vs 445 tokens. Non-termination is NOT the story.
- 82.5% of filtered-RLVR MMK12 outputs contain neither `<answer>` nor a
  "Final answer:" line — but they end **cleanly**, with the answer in a LaTeX
  `\boxed{...}` (e.g. "The correct choice is \(\boxed{10.5}\)."). The model
  drifted off the `<think>/<answer>` template (`<think>` present in 1.1% of
  outputs vs 90.2% for unfiltered) toward the boxed style; the primary scorer's
  last-line fallback then fails on trailing LaTeX.
- Adding one uniform fallback — extract the last `\boxed{...}` before the
  last-line fallback, applied identically to all five models — gives:

| Model | ScienceQA (aug) | MMK12 (aug) |
|---|---|---|
| base | 0.7369 | 0.3584 |
| sft_unfiltered | 0.6520 | 0.3389 |
| sft_filtered_tau085 | 0.6770 | 0.3750 |
| rlvr_unfiltered | 0.7484 | 0.3633 |
| rlvr_filtered_tau085 | 0.7500 | 0.3545 |

### 3.3 Significance (McNemar exact, paired by question_id)

| Comparison (filtered vs unfiltered) | Primary scorer | Boxed-aware scorer |
|---|---|---|
| SFT, ScienceQA | 197 vs 151, **p=0.016** | 197 vs 151, **p=0.016** |
| SFT, MMK12 | 134 vs 90, **p=0.004** | 134 vs 97, **p=0.018** |
| RLVR, ScienceQA | 186 vs 188, p=0.96 | 188 vs 185, p=0.92 |
| RLVR, MMK12 | 73 vs 221, p=1.8e-18 (artifact, see 3.2) | 156 vs 165, p=0.66 |

**Headline findings (at smoke scale):**
1. **Unfiltered SFT hurts the base model** on both benchmarks (SQA 0.737→0.652,
   MMK12 0.349→0.325) — direct confirmation of the noisy-imitation concern.
2. **Filtering significantly improves SFT** on both benchmarks (p=0.016 /
   p=0.004-0.018), and filtered SFT is the only SFT cell above base on MMK12.
3. **RLVR arms are statistically tied** on both benchmarks once scoring is
   format-robust; both sit ~+1.1 pt over base on ScienceQA after only 4 GRPO steps.
4. **Filtered RLVR drifts off the answer template** (a real behavioral
   difference — entropy 0.72→1.44 over 4 steps vs 0.80→0.67 unfiltered — even
   though its task accuracy is unchanged). Worth watching at scale: a reward with
   a real format term should suppress this.

## 4. RLVR training trajectories

| Step | unfiltered reward mean | filtered reward mean |
|---|---|---|
| 1 | 0.523 | 0.547 |
| 2 | 0.500 | 0.555 |
| 3 | 0.414 | 0.484 |
| 4 | 0.547 | 0.500 |

- Val reward: unfiltered 0.5376 → 0.5553; filtered 0.5022 → 0.5022 (identical to
  16 digits at steps 0 and 4).
- Entropy: unfiltered 0.800 → 0.670; filtered 0.721 → 1.442.
- Every step in both runs: reward min/max = {0.25, 0.75} — the **degenerate
  accuracy reward** signature (parquet ground_truth is the full response text; no
  "Final answer:" pattern in `_extract_answer` → first-number fallback; format
  term ≡ 0.5). Both arms trained on this same degenerate signal by design (user
  instruction: keep rewards identical; bipartite term pending supervisor
  discussion). All RLVR conclusions carry this caveat.
- **Within-group reward variance:** verl does not log per-group variance and
  rollouts are not persisted, so it cannot be computed post-hoc — stated as a
  limitation rather than fabricated. Evidence of mixed groups:
  `critic/advantages/max = 1.4999` at every step in both runs (a GRPO group with
  both 0.25 and 0.75 rewards); with rewards Bernoulli-valued in {0.25, 0.75}, a
  group's variance is determined by its mean, so the reward means above are a
  proxy. For the scale runs, per-group variance logging should be added if wanted.

SFT loss (35-step bucket means): unfiltered 8.24→2.23, filtered ~8.7→2.2 —
indistinguishable at this scale (different pools, same schedule).

## 5. Anomalies and honest-reporting notes

- Eval job 59199137 failed pre-generation (vLLM V1 EngineCore forked after CUDA
  init); fix documented in `run_eval_2x2.sbatch`; resubmitted as 59199631. RLVR
  checkpoint merges (`verl.model_merger`, FSDP→HF) completed in the failed job
  and were reused.
- RLVR filtered job log ends with a harmless post-completion DataLoader teardown
  traceback — same signature as the unfiltered baseline (see run ledger).
- Both RLVR runs are only 4 GRPO steps; treat all RLVR deltas as noise-prone.
- format_strict for base/SFT models is low mostly because they emit multiple
  `<think>` blocks or none; format_any is the meaningful adherence measure there.

## 6. Files

- Generations + TSV: `src/outputs/eval_visualprm_2x2/` (`{tag}__{bench}.jsonl`,
  `results_2x2.tsv`, merged RLVR models `merged_rlvr_*`).
- Eval code: `src/eval/generate_eval.py`, `src/eval/score_eval.py`,
  `run_eval_2x2.sbatch`.
- Training sbatch: `run_filtered_sft.sbatch`, `run_filtered_rlvr.sbatch`.
- Adapter: Edit 6 in `docs/local_edits_to_training_code.md`
  (`FilteredVisualPRMJsonlSource` in `src/train/train_common.py`).
