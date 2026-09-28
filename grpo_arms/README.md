# Six-arm GRPO campaign — match reward vs VisualPRM, chunker vs marker, 3B vs 7B

| # | model | reward | segmentation |
|---|-------|--------|--------------|
| 1 | Qwen2.5-VL-3B-Instruct | match (NLI bipartite) | chunker `release/chunker-deberta-v3-small-v2-mask`, thr 0.40 |
| 2 | 3B | match | marker regex (blank line, `Step N:`, `\d+[.)]`, `[-*•]`) |
| 3 | 3B | VisualPRM-8B min step score | VisualPRM's own `\n\n` split |
| 4 | 7B | match | chunker |
| 5 | 7B | match | marker |
| 6 | 7B | VisualPRM | `\n\n` |

Reward (arms 1/2/4/5): `R = 1*format + 2*match + 5*answer`;
match = Hungarian assignment over s(i,j) = 0.5·E(i→j)+0.5·E(j→i) − max(C(i→j),C(j→i)) clip[0,1],
threshold τ (gate-1 sweep), precision-dominant F_β (β=0.5) on matched mass.
NLI = microsoft/deberta-xlarge-mnli, fp32+TF32, label indices from id2label.
Arms 3/6 replace match with min over VisualPRM-8B `generate_steps_with_soft_score`
(revision `7b7c9c4fecbc`), served cross-process from `envs/vprm-judge`.

## Data
- Train: `data/train_subset.jsonl` — 6,000 frozen prompts, sha256 in
  `data/train_subset.sha256`, funnel in `data/subset_manifest.json`.
  Filter: last-step score > 0.5 (answer-correct), ≥5 body steps, no nlvr2,
  dedup (image,question)→max min-body-MC, cap 300/source, seed 42.
- Validate: VPB questions+answers only (`data/visualprocessbench/test.jsonl`),
  single-image rows (2,856/2,866), same set for all six arms.

## Files
- `build_subset.py` — frozen subset builder
- `segmentation.py` — marker (granularity struct rule, verbatim) + `\n\n` splits
- `nli_match.py` — bidirectional NLI pair score + Hungarian F_β
- `score_server.py` / `vprm_server.py` — Unix-socket scoring servers (models
  cannot load inside verl's reward worker)
- `arm_reward.py` — verl custom_reward_function (env-configured; returns
  per-component dict → reward_extra_info → `trainer.rollout_data_dir` dumps)
- `train_arm.py` — verl launcher on the frozen subset (val = fixed 120-row
  seed-0 holdout, identical across arms; resume: fixed OUTDIR + resume_mode=auto)
- `tau_sweep.py` — gate 1 (identity vs hard_negative on granularity/nli_pairs.jsonl)
- `preflight.py` — gate-0 checks
- `run_arm.sbatch` — parameterized arm launcher (server lifecycle + Ray + verl)

## Gates
1. τ sweep report → wait for go-ahead
2. Arm 1 to completion (curves + component independence + VPB accuracy) → wait
3. Remaining five arms
