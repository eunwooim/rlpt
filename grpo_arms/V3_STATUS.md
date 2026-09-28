# V3 STATUS — arm-1 v3 overnight (answer-only control, v3a no-std, v3b no-std + n8 + filter), started 2026-09-11 23:31

Live file. The monitor (grpo_arms/monitor_v3.py, login node, every 30 min) rewrites the "Live monitor" section; training jobs append
start/end blocks at the bottom (run_arm.sbatch, STATUS_FILE=grpo_arms/V3_STATUS.md). Every number is copied from a log or file.

## Job table
| run | training job | resubmit 1 | resubmit 2 | eval jobs (vpb_test, 2048 tok) | notes |
|---|---|---|---|---|---|
| preflight: probe bit-identity | 63051579 (was 63050273: landed on a 20 GB MIG slice, no progress in 46 min → cancelled) | – | – | – | reward_redesign/redteam-63051579.log; resubmitted with --exclude of the 7 MIG-slice nodes |
| preflight: VPB split | 63050275 (htc, cpu) | – | – | – | COMPLETED rc=0, 1m30 |
| smoke v3b (1 step, htc) | 1-GPU software smoke 63059277 (earlier attempts: 63050373 pinned/cancelled, 63051617 2-GPU cancelled as too late, 63051689 wrong NGPU, 63052231 reward-migration bug, 63057803 sleeping-replica crash) | – | – | – | NGPU=1, MAXRESP=1024, n=8, filter on; validates port + dual val logging + acc plumbing; OUTDIR runs/v3_smoke1g_v3b |
| v3b | 63051619 FAILED 02:22 (pre-fix); resubmit 63051620 ran 07:30 → STOPPED BY USER 15:30 at step 46 | 63051620 | 63051621 (cancelled) | 63101265 (step 40 on vpb_test; 63051622/63051623 cancelled) | runs/arm1_3b_matchv3b_nostd_n8_filter |
| v3a | 63051624 (public, 2×A100, 14:00:00) — COMPLETED 10:01 rc=0, 90/90 | 63051625 | 63051626 | 63051627 (step 90), 63051628 (step 60) | runs/arm1_3b_matchv3a_nostd |
| ctrl | 63051629 (public, 2×A100, 14:00:00) | 63051630 | 63051631 | 63051632 (step 90), 63051633 (step 60) | runs/arm1_3b_ctrl_answeronly |
| refs on vpb_test (base3b, cs25, arm1v2 step183) | – | – | – | 63050431 (cs25_2ktest written 00:44; TIMEOUT at 60 min during base3b — harmless, 63050432 wrote base3b_2ktest 00:58), 63050432 (step183, on a 20 GB MIG slice, at risk of timeout), backup 63053943 (step183, non-MIG) COMPLETED 02:20 (443 s gen) → all three reference gens on vpb_test exist | 1-h gpu:1 jobs, 2048 tok, EVAL_SET=vpb_test |
| final report job | 63101266 (htc CPU, afterany:63101265; 63051641 cancelled) | – | – | – | writes grpo_arms/V3_REPORT.md |
| step 7 re-aggregation | 63051634 (htc, gpu:1, no MIG) | – | – | – | writes grpo_arms/REWARD_REAGG.md (steps 40/80/120/160 of the v2 run) |

## Step 0 — preflight
1. Queue at 23:31: only the OOD vscode session (grp_vgupt140). Nothing on grp_bshettah.
2. verl 0.8.0 flags (grepped in /scratch/sghos104/envs/rlpt-train/lib/python3.11/site-packages/verl):
   - `algorithm.norm_adv_by_std_in_grpo`: present — trainer/ppo/ray_trainer.py:1621 reads `self.config.algorithm.get("norm_adv_by_std_in_grpo", True)`
     and passes it to compute_advantage; default True in trainer/config/ppo_trainer.yaml:74. Override `algorithm.norm_adv_by_std_in_grpo=False` works as-is.
   - `algorithm.filter_groups`: ONLY the config dataclass exists (trainer/config/algorithm.py:43 `FilterGroupsConfig(enable, metric, max_num_gen_batches)`,
     `AlgoConfig.filter_groups: Optional[FilterGroupsConfig] = None` at :660). There is NO implementation anywhere in the installed package:
     no reference in trainer/ppo/ray_trainer.py and no `recipe/` directory at all (the DAPO recipe is not shipped in this wheel).
     DECISION: ported the minimal DAPO dynamic-sampling loop into our own trainer — grpo_arms/v3_trainer.py (FilterGroupsTrainer subclasses
     RayPPOTrainer, fit() copied from the installed 0.8.0 with the filter loop inserted after reward computation; V3TaskRunner subclasses
     main_ppo.TaskRunner) + entrypoint grpo_arms/v3_main.py (hydra pkg://verl.trainer.config). train_arm.py got `--entrypoint`, `--extra_val_files`,
     `--total_epochs`. Config keys are passed as `+algorithm.filter_groups.enable=True +algorithm.filter_groups.metric=acc
     +algorithm.filter_groups.max_num_gen_batches=3` (the `+` because ppo_trainer.yaml lacks the key). Since the raise never existed here,
     the "partial-batch patch" IS part of the port: when max_num_gen_batches is exhausted it warns and trains on the mixed groups collected
     (truncated to a multiple of ppo_mini_batch_size=8 prompts). Diff kept at grpo_arms/patches/filter_groups_partial_batch.patch.
     v3b also gets `trainer.total_epochs=2` so the dataloader can wrap: up to 3 gen batches x 32 prompts per step x 90 steps could exceed the
     5,880-prompt epoch; `total_training_steps=90` still stops it at 90.
   - `metric=acc`: arm_reward.compute_score now emits `acc` (= answer correctness after the gates; gated rollouts count 0) in the reward dict,
     which the naive reward manager forwards as reward_extra_info (verified: workers/reward_manager/naive.py:92-96 collects dict keys).
3. arm1v2 optimizer config (runs/arm1_3b_matchv2_final/logs/verl_command.txt, configs/arm_config.json) — UNCHANGED tonight:
   lr = 1e-05 (actor_rollout_ref.actor.optim.lr), KL: use_kl_loss=True with the verl default kl_loss_coef = 0.001 (kl_coef 0.001 in the
   step-101 metrics of v1 = `actor/kl_coef:0.001`), entropy_coeff = 0 (no entropy bonus), ppo_micro_batch_size_per_gpu = 1 with
   use_dynamic_bsz=True and ppo_max_token_len_per_gpu = 8192, gpu_memory_utilization = 0.40, rollout.n = 5, seed 42, temperature = verl default 1.0.
   Note for the morning: entropy_coeff=0 + KL 0.001 means nothing in the objective resists entropy collapse except the tiny KL term.

## Step 1 — VPB dev/test split
Job 63050275 (CPU, 1m30): vpb_dev 400 / vpb_test 2,456, stratified by source, seed 0 (recipe + per-source counts + shas in grpo_arms/DATA_SPLITS.md).
vpb_dev.parquet sha256 41d302ba…, vpb_test.jsonl sha256 1515742c…, vpb_dev.jsonl sha256 8cba4db8….
vpb_dev is passed as the second data.val_files entry (`--extra_val_files`) with data_source=vpb_dev; its rows have no gold steps, so
arm_reward scores them answer-only without calling the NLI server (score = 5·answer + 1·format, truncation-gated) — `val-aux/vpb_dev/answer/mean@1`
is the dev accuracy. Confirmed in the smoke: (pending).

## Step 2 — reward plumbing (no scoring change)
- reward_v2.score_new: new keyword `mode` (default env REWARD_MODE, "match"); `answer_only` returns 5·answer + 1·format after the same three
  gates and the same dedupe/format computation, with match = pun = 0 and no NLI call. The match path is byte-for-byte the previous code
  (diff shows only the mode lines); score_old untouched. Backups: reward_v2.py.bak_v3, arm_reward.py.bak_v3, score_server.py.bak_v3.
- score_server op reward_v2 passes `mode` through and adds `acc`; arm_reward passes REWARD_MODE and emits `acc`.
- Bit-identity check: job 63050273 reruns the probe in match mode and compares every row/field with probe_scored_final.jsonl (job 62990551,
  the final-spec run: gold 7.97 / clean 5.47 / giant 5.40 / repeat6 4.67 / degenerate 0.00). NOTE: the numbers quoted in the brief
  (gold 8.00 > clean 5.00 > giant 4.38 > repeat6 3.29) are from the PRE-final reward (job 62701227, τ-gated mass version); the frozen spec
  is the final one, so identity is checked against 62990551. Result: job 63051579 (H100 node sg023, 28 min): 905/905 rows, `mismatching (row,field) count=0 max_abs_diff=0.000e+00 -> BIT-IDENTICAL` (reward_redesign/redteam-63051579.log, rc=0). PASS.

## Step 3 — training runs
Per-run sbatch files: grpo_arms/v3/{ctrl,v3a,v3b}.sbatch = run_arm.sbatch with an export block on top; diffs in grpo_arms/v3/*.diff (below).
Common to all three (from the export block): MODEL = cold-start checkpoint-25, REWARD=match_v2 (server op reward_v2), SEG=native, TAU=0.45 (unused
by score_new), MAXRESP=2048, NGPU=2, GPUMEM=0.40, token caps 8192 (ppo / ref log_prob / rollout log_prob), --max_steps 90 --save_freq 10
--test_freq 10, max_actor_ckpt_to_keep=4, second val file grpo_arms/data/vpb_dev.parquet, seed 42 (train_arm default, same as arm1v2),
WATCHDOG=1, STATUS_FILE=grpo_arms/V3_STATUS.md. Memory for v3b (n=8): rollout memory is bounded by gpu_memory_utilization=0.40 (vLLM
batches 256 instead of 160 sequences inside the same KV budget) and the actor/ref/rollout log-prob passes are token-capped at 8192 tokens
per micro-batch with use_dynamic_bsz=True, so peak GPU memory does not scale with n; micro-batch sizes were therefore NOT reduced
(arm1v2 peaked at 40.9 GB of 80 GB at n=5). Verified by the 1-step smoke (below).

`diff grpo_arms/run_arm.sbatch grpo_arms/v3/ctrl.sbatch`:
```diff
2c2
< #SBATCH --job-name=grpo_arm
---
> #SBATCH --job-name=v3_ctrl
16a17,24
> # ===== v3 run "ctrl" (2026-09-11): every run-specific value is set here so `diff grpo_arms/v3/ctrl.sbatch grpo_arms/run_arm.sbatch`
> # shows exactly what differs from the arm1v2 training sbatch. Everything below this block is run_arm.sbatch verbatim.
> export ARM=1 MODEL=/scratch/sghos104/rlpt/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25 REWARD=match_v2 SEG=native TAU=0.45 MAXRESP=2048 NGPU=2 GPUMEM=0.40 WATCHDOG=1
> export REWARD_MODE=answer_only
> export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_answeronly
> export STATUS_FILE=grpo_arms/V3_STATUS.md
> export EXTRA="--max_steps 90 --save_freq 10 --test_freq 10 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override trainer.max_actor_ckpt_to_keep=4 --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5"
> export RESUB="${RESUB:-0}"
```
`diff grpo_arms/run_arm.sbatch grpo_arms/v3/v3a.sbatch`:
```diff
2c2
< #SBATCH --job-name=grpo_arm
---
> #SBATCH --job-name=v3_v3a
16a17,24
> # ===== v3 run "v3a" (2026-09-11): every run-specific value is set here so `diff grpo_arms/v3/v3a.sbatch grpo_arms/run_arm.sbatch`
> # shows exactly what differs from the arm1v2 training sbatch. Everything below this block is run_arm.sbatch verbatim.
> export ARM=1 MODEL=/scratch/sghos104/rlpt/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25 REWARD=match_v2 SEG=native TAU=0.45 MAXRESP=2048 NGPU=2 GPUMEM=0.40 WATCHDOG=1
> export REWARD_MODE=match
> export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv3a_nostd
> export STATUS_FILE=grpo_arms/V3_STATUS.md
> export EXTRA="--max_steps 90 --save_freq 10 --test_freq 10 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override trainer.max_actor_ckpt_to_keep=4 --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override algorithm.norm_adv_by_std_in_grpo=False"
> export RESUB="${RESUB:-0}"
```
`diff grpo_arms/run_arm.sbatch grpo_arms/v3/v3b.sbatch`:
```diff
2c2
< #SBATCH --job-name=grpo_arm
---
> #SBATCH --job-name=v3_v3b
16a17,24
> # ===== v3 run "v3b" (2026-09-11): every run-specific value is set here so `diff grpo_arms/v3/v3b.sbatch grpo_arms/run_arm.sbatch`
> # shows exactly what differs from the arm1v2 training sbatch. Everything below this block is run_arm.sbatch verbatim.
> export ARM=1 MODEL=/scratch/sghos104/rlpt/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25 REWARD=match_v2 SEG=native TAU=0.45 MAXRESP=2048 NGPU=2 GPUMEM=0.40 WATCHDOG=1
> export REWARD_MODE=match
> export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv3b_nostd_n8_filter
> export STATUS_FILE=grpo_arms/V3_STATUS.md
> export EXTRA="--max_steps 90 --save_freq 10 --test_freq 10 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override trainer.max_actor_ckpt_to_keep=4 --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 8 --total_epochs 2 --entrypoint v3_main --verl_extra_override algorithm.norm_adv_by_std_in_grpo=False --verl_extra_override +algorithm.filter_groups.enable=True --verl_extra_override +algorithm.filter_groups.metric=acc --verl_extra_override +algorithm.filter_groups.max_num_gen_batches=3"
> export RESUB="${RESUB:-0}"
```

## Step 7 — offline reward re-aggregation (job 63051634, H100, rc=0, 20 min) → grpo_arms/REWARD_REAGG.md
Rollout dumps of the v2 run exist for every step (183 files); per-pair NLI scores are not stored, so the job recomputed s(i,j) with
reward_v2.Scorers for steps 40/80/120/160 (160 rollouts each). `original` reproduces the recorded training-time match exactly
(0.385/0.502/0.638/0.743 vs dump 0.385/0.502/0.638/0.742). Answer share of within-group |advantage| (original → tau045 → zero_allok → prec_only):
step 40: 0.44 → 0.43 → **0.74** → 0.44 · step 80: 0.17 → 0.17 → **0.36** → 0.17 · step 120: 0.03 → 0.03 → **0.09** → 0.03 · step 160: 0.21 → 0.21 → **0.42** → 0.21.
Reading: the τ gate and precision-only aggregation do NOT change who owns the advantage (the group-normalisation erases the 5:2 weighting
regardless of how match is aggregated); zeroing match on all-correct groups doubles the answer share but only by removing the gradient from
those groups entirely ("groups w/ signal" 1.00 → 0.59/0.56/0.34/0.50). Nothing in the reward aggregation fixes the saturation — consistent
with tonight's premise that the optimizer (std-normalisation, group mixing) is the lever.

## Live monitor (monitor_v3.py, 2026-09-13 07:04)

### ctrl — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_answeronly
progress bar: 90/90 · last metric step: 90 · checkpoints with actor/: [60, 70, 80, 90] · DONE=True · STOP_WATCHDOG=False
  [arm1] host=sg036 job=63051629 start=Sat Sep 12 03:31:21 MST 2026
  [arm1] done=Sat Sep 12 07:29:43 MST 2026 rc=0
step-0 val: train-val acc 0.623, vpb_dev acc 0.302
| step | s/step | trainval acc | vpb_dev acc | resp_len | clip | match | answer | entropy | mixed grp frac | n_seg |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 176 | - | - | 441 | 0.013 | 0.000 | 0.431 | 1.671 | 0.812 | 8.4 |
| 5 | 140 | - | - | 266 | 0.000 | 0.000 | 0.556 | 1.453 | 0.710 | 5.5 |
| 10 | 175 | 0.604 | 0.287 | 273 | 0.000 | 0.000 | 0.681 | 0.786 | 0.594 | 5.0 |
| 20 | 164 | 0.428 | 0.217 | 255 | 0.000 | 0.000 | 0.519 | 0.515 | 0.323 | 6.2 |
| 30 | 172 | 0.338 | 0.142 | 378 | 0.006 | 0.000 | 0.494 | 0.285 | 0.281 | 7.2 |
| 40 | 171 | 0.440 | 0.175 | 301 | 0.000 | 0.000 | 0.419 | 0.361 | 0.156 | 6.4 |
| 50 | 154 | 0.423 | 0.163 | 285 | 0.000 | 0.000 | 0.406 | 0.364 | 0.281 | 5.4 |
| 60 | 167 | 0.452 | 0.193 | 313 | 0.000 | 0.000 | 0.338 | 0.275 | 0.250 | 4.8 |
| 70 | 179 | 0.320 | 0.142 | 442 | 0.000 | 0.000 | 0.388 | 0.332 | 0.062 | 5.4 |
| 80 | 180 | 0.423 | 0.193 | 553 | 0.000 | 0.000 | 0.569 | 0.398 | 0.125 | 5.4 |
| 90 | 169 | 0.359 | 0.140 | 580 | 0.000 | 0.000 | 0.444 | 0.279 | 0.156 | 5.0 |
**FLAGS:** step 69: truncation clip_ratio=0.006 > 0 · step 73: truncation clip_ratio=0.006 > 0 · step 74: truncation clip_ratio=0.006 > 0 · step 76: truncation clip_ratio=0.013 > 0 · step 77: truncation clip_ratio=0.050 > 0 · step 78: truncation clip_ratio=0.019 > 0 · step 79: truncation clip_ratio=0.006 > 0 · step 88: truncation clip_ratio=0.025 > 0

### v3a — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv3a_nostd
progress bar: 90/90 · last metric step: 90 · checkpoints with actor/: [60, 70, 80, 90] · DONE=True · STOP_WATCHDOG=False
  [arm1] host=sg016 job=63051624 start=Sat Sep 12 03:20:14 MST 2026
  [arm1] done=Sat Sep 12 10:01:04 MST 2026 rc=0
step-0 val: train-val acc 0.598, vpb_dev acc 0.320
| step | s/step | trainval acc | vpb_dev acc | resp_len | clip | match | answer | entropy | mixed grp frac | n_seg |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 328 | - | - | 447 | 0.019 | 0.215 | 0.425 | 1.920 | 0.906 | 7.1 |
| 5 | 246 | - | - | 257 | 0.006 | 0.245 | 0.569 | 1.014 | 0.839 | 5.2 |
| 10 | 281 | 0.645 | 0.280 | 300 | 0.006 | 0.296 | 0.688 | 0.781 | 0.500 | 5.4 |
| 20 | 251 | 0.678 | 0.235 | 206 | 0.000 | 0.267 | 0.644 | 0.483 | 0.323 | 4.0 |
| 30 | 300 | 0.542 | 0.215 | 276 | 0.000 | 0.280 | 0.650 | 0.336 | 0.281 | 5.2 |
| 40 | 252 | 0.523 | 0.195 | 244 | 0.000 | 0.253 | 0.550 | 0.352 | 0.312 | 3.7 |
| 50 | 272 | 0.501 | 0.228 | 294 | 0.006 | 0.244 | 0.463 | 0.377 | 0.344 | 4.1 |
| 60 | 243 | 0.454 | 0.210 | 191 | 0.000 | 0.261 | 0.644 | 0.316 | 0.312 | 4.5 |
| 70 | 259 | 0.421 | 0.230 | 211 | 0.000 | 0.237 | 0.650 | 0.302 | 0.281 | 4.1 |
| 80 | 266 | 0.472 | 0.147 | 244 | 0.006 | 0.246 | 0.550 | 0.288 | 0.219 | 3.9 |
| 90 | 273 | 0.457 | 0.210 | 271 | 0.000 | 0.253 | 0.519 | 0.242 | 0.312 | 4.2 |
**FLAGS:** step 45: truncation clip_ratio=0.006 > 0 · step 49: truncation clip_ratio=0.006 > 0 · step 50: truncation clip_ratio=0.006 > 0 · step 73: truncation clip_ratio=0.006 > 0 · step 75: truncation clip_ratio=0.013 > 0 · step 76: truncation clip_ratio=0.075 > 0 · step 77: truncation clip_ratio=0.013 > 0 · step 80: truncation clip_ratio=0.006 > 0

### v3b — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv3b_nostd_n8_filter
progress bar: 46/90 · last metric step: 46 · checkpoints with actor/: [10, 20, 30, 40] · DONE=False · STOP_WATCHDOG=False
  [arm1] host=sg036 job=63051619 start=Sat Sep 12 02:06:16 MST 2026
  [arm1] done=Sat Sep 12 02:22:42 MST 2026 rc=1
  [arm1] host=sg036 job=63051620 start=Sat Sep 12 07:29:58 MST 2026
  [arm1] TERM received Sat Sep 12 15:29:30 MST 2026 (wall limit in ~15 min) — stopping training for a clean resume
step-0 val: train-val acc 0.639, vpb_dev acc 0.307
| step | s/step | trainval acc | vpb_dev acc | resp_len | clip | match | answer | entropy | mixed grp frac | n_seg | gen batches | groups trained | partial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 802 | - | - | 428 | 0.012 | 0.208 | 0.484 | 1.821 | 0.938 | 8.0 | 2 | 32 | 0 |
| 5 | 545 | - | - | 296 | 0.020 | 0.244 | 0.527 | 1.482 | 1.000 | 5.6 | 2 | 32 | 0 |
| 10 | 572 | 0.463 | 0.242 | 368 | 0.008 | 0.202 | 0.465 | 1.057 | 1.000 | 6.5 | 2 | 32 | 0 |
| 20 | 496 | 0.303 | 0.158 | 344 | 0.027 | 0.166 | 0.457 | 0.676 | 1.000 | 8.0 | 2 | 32 | 0 |
| 30 | 653 | 0.309 | 0.102 | 323 | 0.000 | 0.177 | 0.411 | 0.454 | 0.875 | 7.5 | 3 | 24 | 1 |
| 40 | 788 | 0.213 | 0.052 | 353 | 0.027 | 0.151 | 0.543 | 0.358 | 0.969 | 7.3 | 3 | 32 | 0 |
| 46 | 657 | - | - | 414 | 0.086 | 0.142 | 0.457 | 0.254 | 1.000 | 11.3 | 3 | 32 | 0 |
filter summary: 47 steps; gen batches/step mean 2.36 (max 3); groups trained/step mean 31.7 (min 24); partial-batch fallback fired 2x
**FLAGS:** step 39: truncation clip_ratio=0.023 > 0 · step 40: truncation clip_ratio=0.027 > 0 · step 41: truncation clip_ratio=0.051 > 0 · step 42: truncation clip_ratio=0.055 > 0 · step 43: truncation clip_ratio=0.021 > 0 · step 44: truncation clip_ratio=0.066 > 0 · step 45: truncation clip_ratio=0.059 > 0 · step 46: truncation clip_ratio=0.086 > 0

### ctrl_base — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_answeronly_base
progress bar: 90/90 · last metric step: 90 · checkpoints with actor/: [60, 70, 80, 90] · DONE=True · STOP_WATCHDOG=False
  [arm1] host=sg037 job=63123541 start=Sat Sep 12 23:18:49 MST 2026
  [arm1] done=Sun Sep 13 03:28:55 MST 2026 rc=0
step-0 val: train-val acc 0.461, vpb_dev acc 0.290
| step | s/step | trainval acc | vpb_dev acc | resp_len | clip | match | answer | entropy | mixed grp frac | n_seg |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 168 | - | - | 338 | 0.000 | 0.000 | 0.356 | 0.739 | 0.531 | 6.5 |
| 5 | 138 | - | - | 266 | 0.000 | 0.000 | 0.544 | 0.465 | 0.677 | 6.5 |
| 10 | 188 | 0.639 | 0.278 | 294 | 0.000 | 0.000 | 0.694 | 0.365 | 0.438 | 6.2 |
| 20 | 187 | 0.585 | 0.250 | 219 | 0.000 | 0.000 | 0.700 | 0.292 | 0.355 | 4.7 |
| 30 | 168 | 0.527 | 0.212 | 282 | 0.000 | 0.000 | 0.588 | 0.261 | 0.375 | 5.9 |
| 40 | 210 | 0.487 | 0.205 | 314 | 0.006 | 0.000 | 0.600 | 0.310 | 0.406 | 5.1 |
| 50 | 149 | 0.332 | 0.172 | 359 | 0.000 | 0.000 | 0.544 | 0.286 | 0.312 | 7.0 |
| 60 | 173 | 0.424 | 0.155 | 496 | 0.013 | 0.000 | 0.556 | 0.210 | 0.344 | 6.0 |
| 70 | 182 | 0.430 | 0.165 | 510 | 0.006 | 0.000 | 0.606 | 0.120 | 0.250 | 5.8 |
| 80 | 180 | 0.425 | 0.133 | 554 | 0.006 | 0.000 | 0.625 | 0.136 | 0.062 | 5.5 |
| 90 | 176 | 0.434 | 0.120 | 738 | 0.031 | 0.000 | 0.500 | 0.091 | 0.188 | 7.4 |
**FLAGS:** step 73: truncation clip_ratio=0.006 > 0 · step 76: truncation clip_ratio=0.006 > 0 · step 77: truncation clip_ratio=0.050 > 0 · step 78: truncation clip_ratio=0.013 > 0 · step 80: truncation clip_ratio=0.006 > 0 · step 86: truncation clip_ratio=0.006 > 0 · step 87: mixed-group fraction < 0.1 for 3 consecutive logged steps · step 90: truncation clip_ratio=0.031 > 0


## Decisions taken without you
- 00:30 The 2×A100 htc smoke was estimated to start 10:42 tomorrow (pinned behind drained node scg008); resubmitted as 2×any-GPU excluding MIG
  slices (est. ~04:42) AND launched the three real chains now (public 2×A100, est. start 02:12–05:16) rather than serialising on the smoke.
  Rationale: the runs would not start earlier anyway; if the smoke exposes a port bug I cancel the chains before/at start (I keep polling);
  a startup failure costs ~20 min per attempt and the afternotok resubmits are capped at 2.
- 00:30 Wall times from arm1v2 measurements: ctrl/v3a 14 h (90 steps × 5.1 min incl. val/ckpt × 1.5 + 45 min + 9 extra vpb_dev vals ≈ 13.5 h);
  v3b 36 h (n=8 and up to 3 gen batches/step: gen ≤ 4.8× → ≤ 14 min/step × 90 × 1.5 + 45 min ≈ 32 h).
- 00:25 Probe bit-identity job 63050273 sat on a 20 GB A100 MIG slice with no progress for 46 min → cancelled, resubmitted excluding MIG nodes.
- 23:35 filter_groups has no implementation in the installed verl → ported it (see Step 0.2) instead of switching entrypoints to a recipe that does not exist.
- 23:40 bit-identity reference = the final-spec probe run (62990551), not the pre-final numbers quoted in the brief.
- 00:28 added a 1-GPU software smoke 63051689 (→ resubmitted, see below) (NGPU=1, MAXRESP=1024, GPUMEM=0.30, n=8, filter on, any non-MIG GPU) because the 2-GPU smoke 63051617 is estimated for 08:56; it validates the port / dual val logging / acc plumbing early, the 2-GPU smoke still validates memory at n=8 + 2048.
- 00:36 bug: DO_BASE=1 leaked via --export=ALL into the arm1v2 step-183 ref job (63050432), so it also generates base3b_2ktest (duplicate of 63050431's work, same greedy output). Fixed launch_vpb_eval.sh (per-step jobs export DO_BASE=0); the already-submitted v3 eval chains never had DO_BASE set.
### training job 63051689 started Sat Sep 12 00:42:29 MST 2026 on sg014 (RESUB=0, wall=1:30:00, existing ckpts: 0)
- 00:46 smoke1g 63051689 died at Ray start (my sed missed: the export line still said NGPU=2 on a 1-GPU allocation); fixed and resubmitted as 63052231.

### training job 63051689 ended Sat Sep 12 00:47:03 MST 2026 rc=124 (RESUB=0, timed_out=1, watchdog_stop=no, ckpts: )
```
[monitor] 0 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/v3_smoke1g_v3b/monitor.csv
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
gate-2: fewer than 3 steps with rollout dumps — no correlation yet
RED-FLAG none
```
- 01:45 QUEUE SITUATION (nothing I can fix): Sol is saturated tonight. Slurm start estimates at 01:44 — backup step-183 ref eval 63053943: 05:07;
  1-GPU smoke 63052231: 06:35; v3b 63051619: 08:58; v3a 63051624: 09:46; ctrl 63051629: 11:13 (all "Priority"; fairshare after ~80 A100-h
  this week). Everything is chained and self-driving (resubmits, evals afterok, report afterany), the monitor loop updates this file every
  30 min. Cancelled the 2-GPU smoke 63051617 (est. 08:56 = same time as v3b, so it could not gate anything); the 1-GPU smoke still validates
  the port / dual-val logging / acc plumbing before v3b starts. Expected timeline: training 09:00–~23:00 (v3b longest), evals + report after.
### training job 63052231 started Sat Sep 12 01:59:30 MST 2026 on sg045 (RESUB=0, wall=1:30:00, existing ckpts: 0)
### training job 63051619 started Sat Sep 12 02:06:16 MST 2026 on sg036 (RESUB=0, wall=1-12:00:00, existing ckpts: 0)

### training job 63052231 ended Sat Sep 12 02:15:40 MST 2026 rc=1 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: )
```
[monitor] 0 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/v3_smoke1g_v3b/monitor.csv
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
gate-2: fewer than 3 steps with rollout dumps — no correlation yet
RED-FLAG none
```
- 02:20 smoke1g 63052231 (sg045, 1 GPU) got through Ray start, V3TaskRunner and model load, then died in step-0 validation with
  `NotImplementedError: Reward function is not implemented for data_source='CLEVR_math_en_...'` — verl's reward loop used its default scorer
  because my entrypoint skipped `migrate_legacy_reward_impl(config)` (main_ppo.main() calls it before run_ppo; it moves custom_reward_function
  into config.reward.*). Fixed grpo_arms/v3_main.py (auto_set_device + migrate_legacy_reward_impl, exactly as main_ppo.main). Python files are
  read at job start, so the queued v3b job (63051619) picks the fix up automatically; ctrl/v3a use the stock entrypoint and were never affected.
  Smoke resubmitted (see job table).

### training job 63051619 ended Sat Sep 12 02:22:42 MST 2026 rc=1 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: )
```
[monitor] 0 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv3b_nostd_n8_filter/monitor.csv
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
gate-2: fewer than 3 steps with rollout dumps — no correlation yet
RED-FLAG none
```
### training job 63057803 started Sat Sep 12 02:23:51 MST 2026 on sg036 (RESUB=0, wall=1:30:00, existing ckpts: 0)

### training job 63057803 ended Sat Sep 12 02:37:10 MST 2026 rc=1 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: )
```
[monitor] 0 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/v3_smoke1g_v3b/monitor.csv
step-0 val (source-avg): answer=0.697  format=0.888  gated=0.013  inv_frac=0.231  match=0.243  n_dup=0.000  n_segments=2.733  pun=0.000  score=4.858
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
gate-2: fewer than 3 steps with rollout dumps — no correlation yet
RED-FLAG none
```
- 02:37 smoke1g 63057803: step-0 val PASSED on both files (train-val sources report acc; vpb_dev acc 0.3175 over 400, gated 0.0625 —
  cs25 on full VPB was 0.3183, so the dev split is representative) and the filter loop ran (gen batch 1: 23/32 groups mixed → pulled a 2nd
  batch), but the 2nd generation crashed vLLM ("CUDA error: an illegal memory access"): my port slept the rollout replicas right after each
  generation (as the stock fit does) and then generated again on sleeping replicas. Fixed v3_trainer.py: sleep_replicas() now runs once the
  step proceeds to training. Patch file regenerated. Smoke resubmitted as 63059277.
### training job 63059277 started Sat Sep 12 02:38:19 MST 2026 on sg036 (RESUB=0, wall=1:30:00, existing ckpts: 0)
- 03:04 smoke1g 63059277: the filter loop now works end-to-end (gen batch 1: 22/32 mixed, gen batch 2: 27/32 → 49/64 mixed, trained 32,
  partial=0) and old_log_prob / ref / advantage ran on the merged batch, but the actor worker died inside update_actor
  ("A worker died or was killed ... unexpected system error", no CUDA OOM text). The 1-GPU smoke ran with --mem=160G (the real runs use 320G
  and 2 GPUs; arm1v2's FSDP offload needed >200G host RSS) → host cgroup OOM CONFIRMED: sstat MaxRSS = 167,771,208 KB = the 160 G limit. Resubmitted the same 1-GPU smoke with
  --mem=320G as 63061395 (OUTDIR runs/v3_smoke1g_v3b_m320). If it passes, the port is fully validated before v3b starts (~09:00); if it dies the
  same way on 1 GPU, update_actor is the known 1-GPU limitation (62710890/62715132) and the 2-GPU real run is the test.

### training job 63059277 ended Sat Sep 12 03:04:59 MST 2026 rc=1 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: )
```
[monitor] 0 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/v3_smoke1g_v3b/monitor.csv
step-0 val (source-avg): answer=0.797  format=0.886  gated=0.014  inv_frac=0.245  match=0.222  n_dup=0.000  n_segments=2.800  pun=0.007  score=5.307
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
gate-2: fewer than 3 steps with rollout dumps — no correlation yet
RED-FLAG none
```
### training job 63061395 started Sat Sep 12 03:05:30 MST 2026 on sg036 (RESUB=0, wall=1:30:00, existing ckpts: 0)
### training job 63051624 started Sat Sep 12 03:20:14 MST 2026 on sg016 (RESUB=0, wall=14:00:00, existing ckpts: 0)

### training job 63061395 ended Sat Sep 12 03:30:59 MST 2026 rc=1 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: )
```
[monitor] 0 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/v3_smoke1g_v3b_m320/monitor.csv
step-0 val (source-avg): answer=0.729  format=0.886  gated=0.014  inv_frac=0.221  match=0.223  n_dup=0.000  n_segments=2.867  pun=0.007  score=4.965
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
gate-2: fewer than 3 steps with rollout dumps — no correlation yet
RED-FLAG none
```
- 03:35 smoke1g_m320 63061395 (320 G host, 1 GPU): MaxRSS 174.6 GB (so the earlier 160 G kill was host OOM), filter loop 24/32 + 25/32 → 49/64
  mixed, trained 32, old_log_prob/ref/advantage fine, then `torch.OutOfMemoryError` in update_actor: sleeping vLLM process 22.79 GiB + actor
  process 54.31 GiB on one 80 GB GPU — the SAME footprint as the stock trainer's 1-GPU failure on 2026-09-06 (62715132: 22.35 + 54.62 GiB).
  Conclusion: this is the known 1-GPU limitation, not the port; on 2 GPUs the actor is FSDP-sharded (arm1v2 peaked at 40.9 GB) and the
  update_actor pass is token-capped (8192/micro-batch), so n=8 does not raise it. SMOKE VERDICT: port validated up to and including advantage
  computation on the merged batch; per-source val accuracy is logged for both val files; `acc` reaches filter_groups. No further smokes —
  the 2-GPU v3b job (63051619, est. ~09:00) is the remaining test, guarded by the afternotok resubmits and the watchdog.
### training job 63051629 started Sat Sep 12 03:31:21 MST 2026 on sg036 (RESUB=0, wall=14:00:00, existing ckpts: 0)
- 03:40 v3b 63051619 actually started at 02:06 (Slurm's 08:58 estimate was wrong by 7 h) — 14 minutes BEFORE the v3_main reward-migration
  fix — and failed the same way as smoke 63052231 (NotImplementedError in step-0 val). Its afternotok resubmit 63051620 is now the live v3b
  attempt (reads the fixed v3_main/v3_trainer at start; OUTDIR data reused). One resubmit (63051621) remains in reserve. v3a 63051624 started
  03:20 (stock entrypoint, unaffected). ctrl 63051629 estimated 04:35.
- 03:42 ctrl 63051629 started 03:31 on sg036 (so all three lanes are live: v3a running, ctrl running, v3b = resub 63051620 pending).
  Reference numbers on vpb_test (2,456 q, 2048 tok, greedy; scored on the login node with score_vpb's scorer — the report job re-scores):
  base3b 0.3025 (length 3.2 %), cs25 0.3192 (3.9 %), arm1v2 step183 0.2406 (0.1 %). v3a step-0 val: vpb_dev acc 0.3175 (400 prompts,
  gated 0.075), train-val acc 0.578 (source-avg over 35 sources; `acc` counts gated rollouts as wrong, unlike the older `answer` mean 0.650).
- 03:50 monitor_v3 verified on the live runs. ctrl step 1: match 0.000 / answer 0.431 / score 2.63 (answer-only mode confirmed), 176 s/step
  (no NLI matrix); v3a steps 1-2: 328/306 s, peak GPU 41.2 GB, match .215→.236, answer .425→.450, entropy 1.92→2.02. NOISE FLOOR for dev
  decisions: the identical step-0 model scored vpb_dev 0.3175 (v3a) vs 0.3025 (ctrl) — vLLM batch-order nondeterminism ≈ ±0.015 on 400
  prompts; treat dev differences under ~0.02 as ties. The "truncation > 0" flag fires at clip_ratio 0.013-0.019 (a few of 160 rollouts hit
  the 2048 cap at the cold-start init) — reported as specified, not a concern at that level.
- 04:31 monitor loop had died with my session; restarted detached (setsid, pid 1387675). v3a 11/90, ctrl 19/90, v3b resub1 still pending (Priority).
- 04:55 YARDSTICK CHECK (ctrl): vpb_dev acc fell 0.3025 → 0.2875 (step 10) → 0.2175 (step 20) while vpb_dev gated rose 0.06 → 0.055 → 0.2175
  and format 0.94 → 0.78; among ungated dev outputs accuracy is ~0.28 at step 20 vs ~0.32 at step 0. On the TRAINING prompts the marker rate
  is 1.000 at step 20 (styles: boxed 115 / "Final answer" 45 of 160; drifting toward \boxed), response length flat ~255 tok, entropy 1.67 →
  0.51. So the dev drop is mostly answers that the vpb_dev gate rejects (no marker or truncated), not visible in dumps because verl does not
  save validation generations. Scorer cross-check on the existing vpb_test generations: arm_reward's marker regex and reward_v2's FINAL_RE
  agree on 2453/2456 (cs25) and 2420/2456 (base3b) outputs, with 0 "missed-but-correct" answers — no scorer blind spot at init. To see what
  the trained policy emits on VPB prompts, job 63067174 generates ctrl's step-20 checkpoint on vpb_dev (tag ctrl_step20_dev); analysed below.
- 05:10 ctrl step-20 checkpoint generated on vpb_dev (job 63067174, tag ctrl_step20_dev, 2048 tok greedy): acc 0.2300 vs cs25 0.3100 on the
  SAME 400 questions (from the earlier full-VPB 2k run). Cause: 93/400 (23 %) hit the 2048-token cap (cs25: 23/400), mean 699 tokens vs 455;
  the truncated outputs are degenerate loops (repeated table rows, restarted enumerations). Among completed outputs with a marker (305)
  accuracy is 0.3016 ≈ cs25's 0.31 — the policy is not worse when it finishes; it fails to finish OOD prompts. Meanwhile on the training
  prompts responses are SHORT (154–207 words, clip 0), so the truncation gate never fires in training and cannot teach against it.
  In-distribution train-val acc (120 VisualPRM prompts) also collapsed: 0.623 → 0.604 → 0.421 → 0.324 (steps 0/10/20/30); entropy
  1.67 → 0.79 → 0.51 → 0.29; mixed-correctness groups 0.81 → 0.59 → 0.32 → 0.28. vpb_dev acc 0.3025 → 0.2875 → 0.2175 → 0.1375.
  EMERGING DECISION (branch 3, "ctrl < cs25"): the answer-only control degrades the policy on its own, so the training setup, not the match
  reward, is the primary problem. First suspects (step 0.3 values): lr 1e-05 (10× the usual 1e-06 for GRPO on a 3B VLM), kl_loss_coef
  0.001, entropy_coeff 0 → rapid entropy collapse. Flags per the brief: none of the auto-stop conditions is met (train resp_len < 800,
  mixed frac > 0.1, truncation only sporadic on train), so ctrl runs to 90 as planned — the vpb_test rows are needed for the table.
- 05:12 ctrl's in-distribution collapse is also mostly gating: train-val (greedy, 120 prompts) gated 0.095 → 0.096 → 0.248 → 0.317 (steps 0/10/20/30)
  while acc 0.623 → 0.604 → 0.428 → 0.338 — i.e. under greedy decoding the low-entropy policy loops and hits the cap / drops the marker, on
  VisualPRM prompts too. Sampled training rollouts (temperature 1) stay short and marked, which is why the reward never penalises it.
  v3a @20: vpb_dev 0.23 (gated 0.085), train-val gated 0.042 — v3a's drop is less gating, more wrong answers. v3b resub still pending.

### training job 63051629 ended Sat Sep 12 07:29:43 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 10 20 30 40 50 60 70 80 90 )
```
[monitor] 90 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_answeronly/monitor.csv
step-0 val (source-avg): answer=0.614  format=0.738  gated=0.094  inv_frac=0.000  match=0.000  n_dup=0.138  n_segments=6.098  pun=0.000  score=3.684
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.631    441.2    0.013    0.169    0.206    8.400    0.000    0.000    0.431    0.662    0.000      -      0.000    0.000      -        -      176.4
      20    3.519    255.1    0.000    0.000    0.150    6.219    0.000    0.000    0.519    0.925    0.000      -      0.000    0.000    0.422    0.000    164.5
      40    3.069    301.5    0.000    0.013    0.606    6.400    0.000    0.000    0.419    0.975    0.000      -      0.000    0.000    0.433    0.000    170.7
      60    2.644    312.9    0.000    0.013    0.412    4.787    0.000    0.000    0.338    0.988    0.000      -      0.000    0.000    0.445    0.000    167.3
      80    3.825    553.2    0.000    0.006    0.781    5.425    0.000    0.000    0.569    0.981    0.000      -      0.000    0.000    0.417    0.000    179.7
      90    3.213    579.8    0.000    0.000    1.038    4.963    0.000    0.000    0.444    0.994    0.000      -      0.000    0.000    0.353    0.000    168.7
gate-2: over 90 steps, Pearson r(match/mean, answer/mean) = +nan
  first quarter: match 0.000 answer 0.555 r=+nan
  last quarter: match 0.000 answer 0.438 r=+nan
  within-step strata (all steps): mean match|answer=1 = 0.000, match|answer=0 = 0.000
  JUDGMENT (auto, rule-based): answer moved (-0.117) while match stayed flat (+0.000) — the match term is not what training optimised. Within steps, correct-answer rollouts score +0.000 higher on match than wrong-answer ones (match is essentially answer-blind at the sample level).
RED-FLAG WARN n_dup/mean rising: 0.444 (steps 51-70) -> 0.827 (steps 71-90)
RED-FLAG WARN response_length/mean up 331->476 while score up 3.13->3.17 (steps 51-70 vs 71-90)
```
### training job 63051620 started Sat Sep 12 07:29:58 MST 2026 on sg036 (RESUB=1, wall=1-12:00:00, existing ckpts: 0)
- 07:30 ctrl 63051629 COMPLETED 90/90, rc=0 (03:31 → 07:29, 3h58m ≈ 2.6 min/step), resubmits 63051630/63051631 cancelled by the job, checkpoints
  10..90 all on disk (verl keep=4 not applied, as in v2). vpb_dev acc by step: 0.3025 (0) .2875 (10) .2175 (20) .1375 (30) .175 (40) .1625 (50)
  .1925 (60) .1425 (70) .1925 (80) **.14 (90)**. Post-run monitor: n_dup/mean rising 0.44 → 0.83 (steps 51-70 → 71-90) and response length
  331 → 476 tok with flat score — the answer-only control drifts into repetition late. Eval jobs 63051632 (step 90) / 63051633 (step 60)
  on vpb_test are now released.
- 07:30 v3b resub1 63051620 STARTED on sg036 (the node ctrl released); ctrl evals 63051632 (step 90, sg048 = 20 GB MIG slice, watch for timeout) and 63051633 (step 60, scg022) running.
- 07:56 v3b step 1 on 2×A100: filter loop 27/32 + 25/32 mixed → 52/64, trained 32 (partial 0), peak actor memory 41.07 GB (n=8 fits, as
  argued), entropy 1.82, resp_len 428, score 3.02, **802 s/step** (2 gen batches of 8 rollouts + update) → 90 steps ≈ 20 h → finish ~03:30
  on 09-13 if every step needs 2 gen batches (36 h wall). step-0 vpb_dev 0.3075.
- 07:56 FIRST vpb_test NUMBER: ctrl step 60 = 0.1922 (length rate 4.8 %, mean 459 tok) vs cs25 0.3192 (3.9 %, 409 tok) → −0.127. With
  truncation this low on the test set, the answer-only control has lost real accuracy, not just answer markers. Branch 3 of the decision
  tree ("ctrl < cs25") is now all but certain; the final table waits for step 90 (63051632 running on a MIG slice) and the v3a/v3b rows.
- 08:04 backup ctrl step-90 test eval 63077476 submitted off-MIG (63051632 is on the 20 GB slice, 35 min in; idempotent).
- 08:23 ctrl step-90 vpb_test generation written by backup 63077476; cancelled the MIG-slice job 63051632 (52 min in) to avoid a partial rewrite of the file.
- 08:25 ctrl on vpb_test (2,456 q, 2048 tok): step 60 = 0.1922 (marker .951, length 4.8 %, 459 tok), **step 90 = 0.1539** (marker .821,
  length 17.9 %, 976 tok) vs cs25 0.3192 (length 3.9 %, 409 tok). The answer-only control ends −0.165 below its init, with a late repetition
  blow-up (mean length doubles between steps 60 and 90, matching the post-run n_dup warning). Both ctrl rows for the report table exist.

### training job 63051624 ended Sat Sep 12 10:01:04 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 10 20 30 40 50 60 70 80 90 )
```
[monitor] 90 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv3a_nostd/monitor.csv
step-0 val (source-avg): answer=0.590  format=0.709  gated=0.115  inv_frac=0.246  match=0.270  n_dup=0.150  n_segments=6.651  pun=0.031  score=4.070
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    3.083    447.4    0.019    0.144    0.094    7.138    0.096    0.215    0.425    0.750    0.257    0.323    0.266    0.177      -        -      327.6
      20    4.662    205.6    0.000    0.006    0.044    4.013    0.010    0.267    0.644    0.919    0.257    0.456    0.313    0.185    0.666    0.311    251.5
      40    4.179    244.3    0.000    0.013    0.075    3.744    0.003    0.253    0.550    0.925    0.283    0.166    0.274    0.228    0.514    0.278    252.4
      60    4.705    191.3    0.000    0.006    0.163    4.513    0.017    0.261    0.644    0.981    0.237    0.228    0.281    0.225    0.447    0.243    243.2
      80    4.153    244.0    0.006    0.019    0.169    3.869    0.006    0.246    0.550    0.981    0.271    0.067    0.252    0.238    0.463    0.200    265.8
      90    4.089    270.5    0.000    0.000    0.163    4.206    0.004    0.253    0.519    0.994    0.264    0.253    0.274    0.230    0.450    0.239    273.1
gate-2: over 90 steps, Pearson r(match/mean, answer/mean) = +0.461
  first quarter: match 0.282 answer 0.614 r=+0.762
  last quarter: match 0.246 answer 0.578 r=+0.329
  within-step strata (all steps): mean match|answer=1 = 0.287, match|answer=0 = 0.230
  JUDGMENT (auto, rule-based): match (-0.035) and answer (-0.037) moved together; the across-step Pearson r = +0.461. They share direction but are not locked (r <= 0.7) — match still varies independently of answer. Within steps, correct-answer rollouts score +0.056 higher on match than wrong-answer ones (match is answer-sensitive at the sample level).
RED-FLAG none
```
- 10:01 v3a 63051624 COMPLETED 90/90, rc=0 (03:20 → 10:01, 6h41m ≈ 4.5 min/step), resubmits 63051625/63051626 cancelled by the job.
  vpb_dev acc by step 0..90: 0.3175 0.28 0.23 0.2125 0.195 0.2275 0.21 0.23 0.1475 0.21 (final 0.21 vs 0.32 at init). Evals 63051627 (step 90) / 63051628 (step 60) on vpb_test released.
- 10:17 v3a on vpb_test (2,456 q, 2048 tok), scored on the login node:
    v3a_step60_2ktest      n=2456 acc=0.2423 marker=0.969 length_rate=0.033 mean_tokens=267
    v3a_step90_2ktest      n=2456 acc=0.2341 marker=0.970 length_rate=0.030 mean_tokens=297
    vs cs25 0.3192 / base3b 0.3025 / arm1v2 step183 0.2406. Remaining rows: v3b step 60/90 (training at ~13 min/step, ETA ~03:30 09-13).
- 11:05 v3b step 20: vpb_dev 0.15 (0.3075 → 0.2425 @10 → 0.15 @20), entropy 0.68, resp_len 344, score 2.44, mixed-group fraction 0.61 (filter
  keeps every step at 2 gen batches, 32 trained, no partial fallback so far). Sharpest early drop of the three arms. None of the brief's
  stop flags is met (truncation 0, length < 800, mixed frac > 0.1), so v3b continues to 90 (ETA ~03:30 09-13, ~13 min/step).
- 15:25 monitor loop restarted (pid 1866895); v3b 46/90, vpb_dev 0.3075→.2425→.15→.0925→.0425 (steps 0-40), entropy 0.25, train clip 8.6 %, filter mean 2.35 gen batches/step, 2 partial fallbacks.
- 15:30 USER DECISION: stop v3b. Cancelled leaf-first: report 63051641, evals 63051622/63051623, resub2 63051621, then training
  63051620 (stopped at step 46; vpb_dev had fallen to 0.0425 at step 40). Checkpoints 10/20/30/40 on disk. Submitted v3b step-40 eval on
  vpb_test (63101265, 1-h gpu:1, non-MIG) and the report job (63101266, afterany:63101265). report_v3.py now treats v3b step 40 as its final row.

### V3_REPORT.md written by job 63101266 at Sat Sep 12 15:45:52 MST 2026 (rc=0)

## DONE (15:50) — v3b stopped at step 46 on your instruction; its step-40 checkpoint evaluated (63101265, 0.0774 on vpb_test); V3_REPORT.md
written by job 63101266 (rc=0) with an interpretation addendum (§5). Decision: branch 3 — the training setup degrades regardless of reward.
Monitor loop stopped. All jobs finished; nothing queued.

## Follow-up (queued 21:07) — answer-only control from the BASE model (ctrl_base)
Tests the cold-start-init confound: identical to ctrl (REWARD_MODE=answer_only, n=5, std-norm, lr 1e-05, KL 0.001, 90 steps, MAXRESP 2048,
vpb_dev as 2nd val file) except MODEL=Qwen/Qwen2.5-VL-3B-Instruct. sbatch grpo_arms/v3/ctrl_base.sbatch (diff vs ctrl: grpo_arms/v3/ctrl_base.diff —
only job name, MODEL, OUTDIR). OUTDIR runs/arm1_3b_ctrl_answeronly_base.
| run | training job | resubmit 1 | resubmit 2 | eval jobs (vpb_test, 2048 tok) | report |
|---|---|---|---|---|---|
| ctrl_base | 63123541 (public, 2×A100, 14:00:00) | 63123542 | 63123543 | 63123544 (step 90), 63123545 (step 60) | 63123586 (afterany on both evals; regenerates V3_REPORT.md with a §3b verdict) |
Reading rule for §3b: ctrl_base ≥ base3b (0.3025 on vpb_test) → the cold-start init is the confound; ctrl_base < base3b → optimizer config / prompt mix.
Monitor loop restarted with ctrl_base added.
### training job 63123541 started Sat Sep 12 23:18:49 MST 2026 on sg037 (RESUB=0, wall=14:00:00, existing ckpts: 0)
- 23:30 ctrl_base 63123541 started 23:18 on sg037; monitor loop restarted (pid 2202865).
- 23:47 ctrl_base step 0: vpb_dev 0.29 (gated 0.065), train-val acc 0.304 (cs25 had 0.62 — the cold-start SFT mostly taught the VisualPRM prompt formats); step 1: score 1.77, resp_len 338, entropy 0.74 (lower than cs25's 1.67 at step 1), 168 s/step.
- 00:10 ctrl_base step 10: train-val acc 0.304 → 0.639 (formats learned), vpb_dev 0.29 → 0.2775 (noise), entropy 0.74 → 0.37 already, score 4.31, resp_len 294, clip 0.
- 00:33 ctrl_base step 20: vpb_dev 0.25 (0.29 → .2775 → .25), train-val 0.585 (peak 0.639 @10), entropy 0.29, resp_len 219, clip 0 — same shape as the cold-start control (dev .3025 → .2875 → .2175); the base init does not prevent the collapse.
- 00:57 ctrl_base step 30: vpb_dev 0.2125 (0.29 → .2775 → .25 → .2125), train-val 0.527 (peak .639 @10), entropy 0.26, resp_len 282, clip 0. Verdict forming: the base init degrades on the same schedule as the cold-start control → the optimizer config (lr 1e-05 / KL 0.001 / entropy 0) or the VisualPRM prompt mix is the cause, not the SFT init.
- 01:20 ctrl_base step 40: vpb_dev 0.205, train-val 0.487, entropy 0.31, resp_len 314, clip 0.006.
- 01:44 ctrl_base step 50: vpb_dev 0.17, train-val 0.332 (from .639 @10), entropy 0.29, resp_len 359 — collapse independent of init.
- 02:10 ctrl_base step 60: vpb_dev 0.155, train-val 0.424, entropy 0.21, resp_len 496 (rising), clip 0.013.
- 02:35 ctrl_base step 70: vpb_dev 0.165, train-val 0.425, entropy 0.12, resp_len 510.
- 03:00 ctrl_base step 80: vpb_dev 0.1325, train-val 0.425, entropy 0.14, resp_len 554.

### training job 63123541 ended Sun Sep 13 03:28:55 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 10 20 30 40 50 60 70 80 90 )
```
[monitor] 90 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_answeronly_base/monitor.csv
step-0 val (source-avg): answer=0.457  format=0.642  gated=0.311  inv_frac=0.000  match=0.000  n_dup=0.129  n_segments=5.242  pun=0.000  score=2.161
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    1.775    338.1    0.000    0.319    0.169    6.475    0.000    0.000    0.356    0.556    0.000      -      0.000    0.000      -        -      168.4
      20    4.456    219.4    0.000    0.006    0.144    4.725    0.000    0.000    0.700    0.956    0.000      -      0.000    0.000    0.575    0.000    187.3
      40    3.987    314.4    0.006    0.006    0.519    5.094    0.000    0.000    0.600    0.988    0.000      -      0.000    0.000    0.479    0.000    209.6
      60    3.750    495.9    0.013    0.019    0.819    5.963    0.000    0.000    0.556    0.969    0.000      -      0.000    0.000    0.417    0.000    172.8
      80    4.113    553.7    0.006    0.006    0.750    5.463    0.000    0.000    0.625    0.988    0.000      -      0.000    0.000    0.417    0.000    180.2
      90    3.463    737.8    0.031    0.031    2.450    7.394    0.000    0.000    0.500    0.963    0.000      -      0.000    0.000    0.426    0.000    176.1
gate-2: over 90 steps, Pearson r(match/mean, answer/mean) = +nan
  first quarter: match 0.000 answer 0.590 r=+nan
  last quarter: match 0.000 answer 0.620 r=+nan
  within-step strata (all steps): mean match|answer=1 = 0.000, match|answer=0 = 0.000
  JUDGMENT (auto, rule-based): answer moved (+0.030) while match stayed flat (+0.000) — the match term is not what training optimised. Within steps, correct-answer rollouts score +0.000 higher on match than wrong-answer ones (match is essentially answer-blind at the sample level).
RED-FLAG WARN n_segments collapsing: 5.936 (steps 51-70) -> 5.338 (steps 71-90)
```

### V3_REPORT.md written by job 63123586 at Sun Sep 13 03:49:13 MST 2026 (rc=0)
- 03:50 ctrl_base 63123541 COMPLETED 90/90 rc=0 (23:18 → 03:28, 4h10m); resubmits cancelled; evals 63123544/63123545 done; V3_REPORT.md
  regenerated by 63123586 (rc=0) with §3b. vpb_test: ctrl_base step 60 = 0.1832 (length 22.5 %), step 90 = 0.1706 (length 33.5 %) vs base3b
  0.3025 → **ctrl_base < base3b**: the answer-only RLVR degrades from the base model too, on the same schedule as from the cold-start
  (vpb_dev 0.29 → .278 → .25 → .2125 → .205 → .17 → .155 → .165 → .1325 → .1175; entropy 0.74 → 0.09; resp_len 294 → 738 with 33 % truncation
  at the end). The cold-start init is NOT the confound; the optimizer config (lr 1e-05, KL 0.001, entropy_coeff 0) and/or the VisualPRM
  prompt mix is. Note: the report's step-0 "trainval acc" for ctrl_base (0.461) is the ungated `answer` mean; the gated `acc` mean was 0.304.
  Nothing queued. Monitor loop stopped.
