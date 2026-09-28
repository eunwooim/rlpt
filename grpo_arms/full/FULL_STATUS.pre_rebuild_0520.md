# FULL STATUS — run M (bipartite match, resumed from v4 B step 60 → 183) and run R (RLVR + adaptive entropy, 183 steps), account grp_vgupt140
Started 2026-09-18 02:28. Live file: the monitor job rewrites "## Live monitor (monitor_full.py, 2026-09-18 05:16, job 63574075)

### M — M: match + soft gates, resumed from v4 B@60 — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate
progress bar: 77/183 · last metric step: 77 · FSDP checkpoints: [60] (latest=60) · merged: [30, 60] · DONE=False · STOP_WATCHDOG=False
  [arm1] host=sg030 job=63148593 start=Sun Sep 13 05:48:02 MST 2026
  [arm1] done=Sun Sep 13 11:01:39 MST 2026 rc=0
  [arm1] host=sg235 job=63567988 start=Fri Sep 18 03:47:41 MST 2026
  [36m(TaskRunner pid=2291420)[0m Setting global step to 60
  [36m(TaskRunner pid=2291420)[0m Resuming from /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate/checkpoints/global_step_60
step-0 val: train-val acc 0.673, vpb_dev acc 0.345
| step | s/step | trainval acc | vpb_dev acc | match | answer | format | entropy | ent coeff | KL | resp_len | n_seg | mixed grp frac | soft-gate fired | trunc (clip) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 410 | - | - | 0.240 | 0.412 | 0.775 | 2.343 | off | 0.0005 | 471 | 8.2 | 0.781 | 0.081 | 0.031 |
| 5 | 278 | - | - | 0.243 | 0.406 | 0.706 | 1.826 | off | 0.0012 | 344 | 6.7 | 0.645 | 0.169 | 0.013 |
| 10 | 283 | 0.699 | 0.305 | 0.253 | 0.506 | 0.794 | 1.499 | off | 0.0022 | 322 | 6.0 | 0.719 | 0.056 | 0.006 |
| 20 | 252 | 0.690 | 0.338 | 0.268 | 0.506 | 0.812 | 1.197 | off | 0.0067 | 292 | 5.4 | 0.645 | 0.044 | 0.000 |
| 30 | 328 | 0.683 | 0.338 | 0.293 | 0.531 | 0.850 | 0.607 | off | 0.0125 | 347 | 6.4 | 0.656 | 0.050 | 0.000 |
| 40 | 255 | 0.659 | 0.325 | 0.313 | 0.675 | 0.906 | 0.467 | off | 0.0226 | 316 | 4.8 | 0.594 | 0.019 | 0.000 |
| 50 | 256 | 0.735 | 0.320 | 0.339 | 0.594 | 0.963 | 0.419 | off | 0.0212 | 321 | 5.1 | 0.656 | 0.013 | 0.000 |
| 60 | 284 | 0.722 | 0.318 | 0.313 | 0.681 | 0.950 | 0.453 | off | 0.0276 | 295 | 4.4 | 0.656 | 0.019 | 0.000 |
| 61 | 325 | - | - | 0.340 | 0.606 | 0.956 | 0.487 | off | 0.0247 | 345 | 4.7 | 0.688 | 0.025 | 0.000 |
| 70 | 242 | 0.733 | 0.312 | 0.331 | 0.756 | 0.981 | 0.579 | off | 0.0303 | 270 | 4.5 | 0.656 | 0.006 | 0.000 |
| 77 | 229 | - | - | 0.310 | 0.606 | 0.969 | 0.622 | off | 0.0285 | 297 | 5.0 | 0.625 | 0.006 | 0.000 |
flags: none

### R — R: RLVR answer-only + adaptive entropy (target 0.6) — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy
progress bar: 0/183 · last metric step: 11 · FSDP checkpoints: [] (latest=0) · merged: [] · DONE=False · STOP_WATCHDOG=False
  [arm1] host=sg016 job=63570807 start=Fri Sep 18 04:43:34 MST 2026
| step | s/step | trainval acc | vpb_dev acc | match | answer | format | entropy | ent coeff | KL | resp_len | n_seg | mixed grp frac | soft-gate fired | trunc (clip) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | - | - | - | n/a | 0.338 | 0.731 | - | - | - | - | 7.8 | 0.781 | n/a | - |
| 5 | - | - | - | n/a | 0.494 | 0.719 | - | - | - | - | 6.5 | 0.839 | n/a | - |
| 10 | - | - | - | n/a | 0.506 | 0.781 | - | - | - | - | 6.8 | 0.625 | n/a | - |
| 11 | - | - | - | n/a | 0.594 | 0.787 | - | - | - | - | 6.2 | 0.625 | n/a | - |
flags: none

events (last 25; full log grpo_arms/full/monitor_events.log):

- 2026-09-18 02:48 pruned FSDP checkpoint M global_step_30 (44G; merge verified: 2 shards + bundle ok; newest=60, keep=[60, 120, 183])
- 2026-09-18 04:19 resume verification M: first_step=60 overlap=2 -> FAIL
- 2026-09-18 04:46 resume verification M: first_step=61 overlap=2 -> PASS

## Job table

| run | job | role | state | notes |
|---|---|---|---|---|
| M | 63567988 | training | RUNNING | wall 14:00:00, launched 2026-09-18 02:42 |
| M | 63567989 | resubmit 1 (afternotok) | PENDING | wall 14:00:00, launched 2026-09-18 02:42 |
| M | 63567990 | resubmit 2 (afternotok) | PENDING | wall 14:00:00, launched 2026-09-18 02:42 |
| M | – | eval step 90 | waiting for checkpoint | vpb_gen_B_matchv4_softgate_step90_2ktest.jsonl |
| M | – | eval step 120 | waiting for checkpoint | vpb_gen_B_matchv4_softgate_step120_2ktest.jsonl |
| M | – | eval step 150 | waiting for checkpoint | vpb_gen_B_matchv4_softgate_step150_2ktest.jsonl |
| M | – | eval step 183 | waiting for checkpoint | vpb_gen_B_matchv4_softgate_step183_2ktest.jsonl |
| R | 63570807 | training | CANCELLED | wall 10:00:00, launched 2026-09-18 03:42 |
| R | 63570808 | resubmit 1 (afternotok) | CANCELLED | wall 10:00:00, launched 2026-09-18 03:42 |
| R | 63570809 | resubmit 2 (afternotok) | CANCELLED | wall 10:00:00, launched 2026-09-18 03:42 |
| R | – | eval step 30 | waiting for checkpoint | vpb_gen_R_rlvr_entropy_step30_2ktest.jsonl |
| R | – | eval step 60 | waiting for checkpoint | vpb_gen_R_rlvr_entropy_step60_2ktest.jsonl |
| R | – | eval step 90 | waiting for checkpoint | vpb_gen_R_rlvr_entropy_step90_2ktest.jsonl |
| R | – | eval step 120 | waiting for checkpoint | vpb_gen_R_rlvr_entropy_step120_2ktest.jsonl |
| R | – | eval step 150 | waiting for checkpoint | vpb_gen_R_rlvr_entropy_step150_2ktest.jsonl |
| R | – | eval step 183 | waiting for checkpoint | vpb_gen_R_rlvr_entropy_step183_2ktest.jsonl |
| smoke | 63567991 | adaptive-entropy smoke (htc, 1 A100, 3 steps) | COMPLETED | grpo_arms/full/smoke_entropy.sbatch |
| monitor | 63574075 | this monitor (public CPU) | running | every 30 min |

## Preflight (02:28)
1. myfairshare: grp_vgupt140 / sghos104 RawUsage_CHE 3832.4, RawFairShare 0.164363, RealFairShare **0.164363** (≥ 0.15 → proceed; queue times
   will be longer than on bshettah). grp_bshettah 1654.3 / 0.451266 (untouched from here on). Queue: OOD vscode (vgupt140) + one unrelated
   pending job (storymem-encode, bshettah, not mine to touch). Nothing of mine running.
2. runs/arm1_3b_matchv4_softgate/checkpoints/global_step_60: actor/ {model,optim,extra_state}×2 ranks + fsdp_config + huggingface/, and
   data.pt (dataloader state); latest_checkpointed_iteration.txt = 60; 44 GB. Also global_step_30 (44 GB). Run M can resume in place.
3. grpo_arms/patches/adaptive_entropy.diff does NOT exist and there is no v5 directory or smoke record → implement + smoke-test (below).
4. Disk: runs/ total ≈ 1.5 TB (v1 arm1_3b_match_chunker 224 GB, v3 runs 187/187/180 GB, ctrl_v4 101 GB, matchv4_softgate 101 GB, others);
   scratch: 456 TB free → far above the 400 GB bar; nothing pruned, v1 run untouched.
5. cs25 processor bundle: 8/8 files. reward_v2.score_new has gate_mode (env GATE_MODE, default hard, soft = truncation-hard / parse+last-line-soft);
   score_server emits answer_valid + acc = correct ∧ valid; arm_reward emits acc/soft_gated. Templates grpo_arms/v4/A_ctrl_v4.sbatch and
   B_matchv4_softgate.sbatch intact (account grp_bshettah in their headers — the copies here switch to grp_vgupt140).

## Run M — match + soft gates, v4 B resumed in place 60 → 183 (submitted 02:42)
- sbatch grpo_arms/full/M_match_full.sbatch = copy of grpo_arms/v4/B_matchv4_softgate.sbatch; launcher grpo_arms/full/launch_full.sh (train + 2 afternotok resubs, no eval chain);
  chain.json in the OUTDIR records train 63567988, resubs 63567989/63567990 and prev_train_logs = v4 B's Slurm logs (63148593/4/5) so the monitor's
  M table starts at step 0. v4 B's `DONE` was renamed `DONE.v4_step60` and its chain.json `chain_v4.json` (the sbatch is a no-op while DONE exists).
- Resume path: verl `trainer.resume_mode=auto` (default) finds `checkpoints/latest_checkpointed_iteration.txt` = 60 → loads actor/ (FSDP model +
  optimizer + extra_state) and `data.pt` (dataloader state; 60 mod 2303 steps/epoch ≠ 0 so verl restores it) → global_steps = 60, first trained step = 61,
  same seed-42 permutation. The monitor verifies this once rollouts/61.jsonl exists (section "Resume verification (M)").
- Only differences vs the v4 B sbatch (`diff grpo_arms/v4/B_matchv4_softgate.sbatch grpo_arms/full/M_match_full.sbatch`, saved as grpo_arms/full/M_vs_v4B.diff):
  account, job name, header comment, `ENTROPY_MODE=off` (explicit no-op), STATUS_FILE, `--max_steps 183`, and `trainer.max_actor_ckpt_to_keep=-1`
  (v4 kept 2 FSDP checkpoints; -1 = verl keeps all, the monitor prunes to {60, 120, 183} + newest after a verified merge — with keep=2 verl
  would have deleted step 60 at step 120). Hydra applies the later duplicate override (checked on the login node).
```
2,3c2,3
< #SBATCH --job-name=v4_B_match
< #SBATCH --account=grp_bshettah
---
> #SBATCH --job-name=full_M_match
> #SBATCH --account=grp_vgupt140
17c17
< # ===== v4 run "B_matchv4_softgate" (2026-09-13, account grp_bshettah, conservative optimizer): every run-specific value is set here so `diff grpo_arms/v3/ctrl.sbatch grpo_arms/run_arm.sbatch`
---
> # ===== FULL run "M" (2026-09-18): v4 B resumed in place from global_step_60 to 183, account grp_vgupt140, v4 optimizer unchanged: every run-specific value is set here so `diff grpo_arms/v3/ctrl.sbatch grpo_arms/run_arm.sbatch`
21a22
> export ENTROPY_MODE=off
23,24c24,25
< export STATUS_FILE=grpo_arms/V4_STATUS.md
< export EXTRA="--max_steps 60 --save_freq 30 --test_freq 10 --learning_rate 1e-6 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override actor_rollout_ref.actor.kl_loss_coef=0.01 --verl_extra_override actor_rollout_ref.actor.clip_ratio_low=0.2 --verl_extra_override actor_rollout_ref.actor.clip_ratio_high=0.28"
---
> export STATUS_FILE=grpo_arms/FULL_STATUS.md
> export EXTRA="--max_steps 183 --save_freq 30 --test_freq 10 --learning_rate 1e-6 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override actor_rollout_ref.actor.kl_loss_coef=0.01 --verl_extra_override actor_rollout_ref.actor.clip_ratio_low=0.2 --verl_extra_override actor_rollout_ref.actor.clip_ratio_high=0.28 --verl_extra_override trainer.max_actor_ckpt_to_keep=-1"
```

## Run R — RLVR answer-only, hard gates, cs25 init, adaptive entropy, 183 steps (launched by the post-smoke job on PASS)
- sbatch grpo_arms/full/R_rlvr_entropy.sbatch = copy of grpo_arms/v4/A_ctrl_v4.sbatch; only differences (grpo_arms/full/R_vs_v4A.diff): account, job name,
  header comment, `ENTROPY_MODE=adaptive`, OUTDIR runs/arm1_3b_rlvr_entropy, STATUS_FILE, `--max_steps 183`, entropy plumbing
  (`--entrypoint entropy_main`, `actor_rollout_ref.actor.calculate_entropy=True` so verl produces the entropy tensor inside update_actor),
  `trainer.max_actor_ckpt_to_keep=-1` (as M), and the TERM-handler pkill pattern also matching `entropy_main`.
```
2,3c2,3
< #SBATCH --job-name=v4_A_ctrl
< #SBATCH --account=grp_bshettah
---
> #SBATCH --job-name=full_R_rlvr
> #SBATCH --account=grp_vgupt140
17c17
< # ===== v4 run "A_ctrl_v4" (2026-09-13, account grp_bshettah, conservative optimizer): every run-specific value is set here so `diff grpo_arms/v3/ctrl.sbatch grpo_arms/run_arm.sbatch`
---
> # ===== FULL run "R" (2026-09-18): RLVR (answer-only, hard gates) from cs25 with ADAPTIVE ENTROPY, 183 steps, account grp_vgupt140, v4 optimizer: every run-specific value is set here so `diff grpo_arms/v3/ctrl.sbatch grpo_arms/run_arm.sbatch`
22,24c22,25
< export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_v4
< export STATUS_FILE=grpo_arms/V4_STATUS.md
< export EXTRA="--max_steps 60 --save_freq 30 --test_freq 10 --learning_rate 1e-6 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override actor_rollout_ref.actor.kl_loss_coef=0.01 --verl_extra_override actor_rollout_ref.actor.clip_ratio_low=0.2 --verl_extra_override actor_rollout_ref.actor.clip_ratio_high=0.28"
---
> export ENTROPY_MODE=adaptive
> export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy
> export STATUS_FILE=grpo_arms/FULL_STATUS.md
> export EXTRA="--max_steps 183 --save_freq 30 --test_freq 10 --learning_rate 1e-6 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override actor_rollout_ref.actor.kl_loss_coef=0.01 --verl_extra_override actor_rollout_ref.actor.clip_ratio_low=0.2 --verl_extra_override actor_rollout_ref.actor.clip_ratio_high=0.28 --entrypoint entropy_main --verl_extra_override actor_rollout_ref.actor.calculate_entropy=True --verl_extra_override trainer.max_actor_ckpt_to_keep=-1"
162c163
<   pkill -u "$USER" -f "verl.trainer.main_ppo" 2>/dev/null
---
>   pkill -u "$USER" -f "verl.trainer.main_ppo|entropy_main" 2>/dev/null
```

## Adaptive entropy (grpo_arms/adaptive_entropy.py + grpo_arms/entropy_main.py; patch grpo_arms/patches/adaptive_entropy.diff)
- No pre-existing patch → implemented 02:50–03:10. Nothing in the rlpt-train env is edited. ENTROPY_MODE=off (default) = byte-identical behaviour.
- Rule (Skywork-OR1 style): target 0.6; coeff starts 0; after each optimizer step `coeff ← clip(coeff + 0.002·(0.6 − entropy_measured), 0, 0.01)` with
  entropy_measured = verl's `actor/entropy` of that step (token-mean entropy from the old-log-prob pass over the batch, before the update).
  The coefficient multiplies verl's existing entropy bonus (`policy_loss −= coeff · entropy_loss` in workers/utils/losses.ppo_loss).
- Mechanics: driver — `AdaptiveEntropyTrainer(RayPPOTrainer)` overrides `_compute_old_log_prob` (measure) and `_update_actor` (inject
  `batch.meta_info["entropy_coeff"]`, update after the step, log `actor/entropy_coeff`, `actor/entropy_coeff_next`, `actor/entropy_target`,
  append to checkpoints/entropy_coeff_log.jsonl; restored on resume from the row of the resumed step). Workers — `engine_workers.ppo_loss` is rebound to
  `adaptive_ppo_loss`, which calls verl's ppo_loss and adds `−coeff·entropy_loss` with the coefficient read from the batch (NonTensorData survives
  the mini/micro-batch splits — checked on the login node); it logs `actor/entropy_coeff_applied` so driver and worker values are cross-checked.
  Worker subclass `AdaptiveEntropyWorker` (no overrides) makes every actor process import the module; `AdaptiveEntropyTaskRunner` registers it.
- Note on the production target: at cs25 the batch entropy starts ≈2.3 (v4 logs) and fell to ≈0.42–0.45 by steps 50–60 in v4, so the coefficient stays
  clipped at 0 until entropy drops below 0.6 and then grows by ≈0.0003–0.0004 per step — it acts as a floor, not a constant bonus.
- Smoke: grpo_arms/full/smoke_entropy.sbatch (htc, 1×A100, 3 steps, MAXRESP 512, ENTROPY_TARGET=3.0 so the coefficient must rise from 0 within
  3 steps; production target 0.6 would keep it at 0 for a 3-step smoke). Differences vs the R sbatch (grpo_arms/full/smoke_vs_R.diff):
```
2c2
< #SBATCH --job-name=full_R_rlvr
---
> #SBATCH --job-name=full_R_smoke
4c4
< #SBATCH --partition=public
---
> #SBATCH --partition=htc
9,12c9,13
< #SBATCH --cpus-per-task=16
< #SBATCH --mem=320G
< #SBATCH --gres=gpu:a100:2
< #SBATCH --time=48:00:00
---
> #SBATCH --cpus-per-task=8
> #SBATCH --mem=180G
> #SBATCH --gres=gpu:a100:1
> #SBATCH --exclude=scg001,scg002,scg003,scg004,sg048,sg049,sg050
> #SBATCH --time=02:30:00
19c20
< export ARM=1 MODEL=/scratch/sghos104/rlpt/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25 REWARD=match_v2 SEG=native TAU=0.45 MAXRESP=2048 NGPU=2 GPUMEM=0.40 WATCHDOG=1
---
> export ARM=1 MODEL=/scratch/sghos104/rlpt/src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25 REWARD=match_v2 SEG=native TAU=0.45 MAXRESP=512 NGPU=1 GPUMEM=0.30 WATCHDOG=0
23,25c24,27
< export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy
< export STATUS_FILE=grpo_arms/FULL_STATUS.md
< export EXTRA="--max_steps 183 --save_freq 30 --test_freq 10 --learning_rate 1e-6 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override actor_rollout_ref.actor.kl_loss_coef=0.01 --verl_extra_override actor_rollout_ref.actor.clip_ratio_low=0.2 --verl_extra_override actor_rollout_ref.actor.clip_ratio_high=0.28 --entrypoint entropy_main --verl_extra_override actor_rollout_ref.actor.calculate_entropy=True --verl_extra_override trainer.max_actor_ckpt_to_keep=-1"
---
> export ENTROPY_TARGET=3.0   # SMOKE ONLY: production target is 0.6 (default in adaptive_entropy.py)
> export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/full_smoke_entropy
> export STATUS_FILE=grpo_arms/full/smoke_status.md
> export EXTRA="--max_steps 3 --save_freq 1000 --test_freq 1000 --max_train_samples 96 --verl_extra_override trainer.val_before_train=False --learning_rate 1e-6 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override actor_rollout_ref.actor.kl_loss_coef=0.01 --verl_extra_override actor_rollout_ref.actor.clip_ratio_low=0.2 --verl_extra_override actor_rollout_ref.actor.clip_ratio_high=0.28 --entrypoint entropy_main --verl_extra_override actor_rollout_ref.actor.calculate_entropy=True --verl_extra_override trainer.max_actor_ckpt_to_keep=-1"
```
  Checker grpo_arms/full/check_smoke.py (asserts: metrics present each step; worker-applied == driver coefficient; coeff_next follows the rule;
  coefficient rises; rc=0, no Traceback). Post-smoke CPU job 63568198 (afterany) runs it and on PASS launches R (10 h + 2 resubs); on FAIL it
  launches nothing and records the failure here.

## Monitoring / evals / pruning / report
- grpo_arms/monitor_full.py as public CPU job 63568199 (2 cores, 24 h, self-chaining until grpo_arms/full/ALL_DONE): every 30 min rewrites
  "Job table" + "Live monitor"; submits a 1-h htc gpu:1 vpb_test eval (2048 tok, greedy; MIG nodes excluded) per new checkpoint
  (M 90/120/150/183, R 30/60/90/120/150/183; ≤3 attempts); prunes FSDP dirs after a verified merge (keep {60, 120, 183} + newest);
  verifies M's resume; submits grpo_arms/report_full.sbatch (afterany on every training + eval job) once all evals are queued and also runs
  report_full.py itself when everything has settled → grpo_arms/FULL_REPORT.md (tables, curves, McNemar, verdict, fairshare).

## Decisions
- 02:30 disk far above threshold → no pruning decision needed; the v1 run is untouched.
- 02:42 M submitted before any new code (brief: 'Submit M first'); smoke submitted in parallel; R waits for the smoke PASS (post-smoke job).
- 02:55 keep-all FSDP checkpoints (`trainer.max_actor_ckpt_to_keep=-1`) on both runs so the monitor's {60,120,183} policy is possible; a later duplicate Hydra override wins (tested).
- 03:42 smoke PASSED (coefficient 0 → 0.002756 → 0.006107 → 0.009094 exactly per rule; workers applied the driver value each step; ~4–6 min/step on 1 GPU at 512 tok). R launched from the login node because the post-smoke CPU job was still queued; it no-ops on seeing R's chain.json. The benign 'DataLoader worker killed' traceback at Ray teardown (also in v4 B's successful log) is ignored by the checker.
- 03:05 smoke uses ENTROPY_TARGET=3.0 (not 0.6): with cs25's starting entropy ≈2.3 the production rule keeps the coefficient at 0 for the first ~50 steps, which would make a 3-step smoke prove nothing about the worker-side plumbing.

## Smoke test (adaptive entropy) — job 63567991, checked 2026-09-18 03:42: **PASS**

htc, 1×A100, cs25, answer-only hard gates, 3 steps × 32 prompts × n=5, MAXRESP 512, ENTROPY_TARGET=3.0 (production 0.6) so the coefficient must rise from 0.

| step | entropy (actor/entropy) | entropy_loss | coeff used (driver) | coeff applied (worker) | coeff next | s/step |
|---|---|---|---|---|---|---|
| 1 | 1.622247 | 1.560556 | 0.000000 | 0.000000 | 0.002756 | 369.438252 |
| 2 | 1.324191 | 1.297632 | 0.002756 | 0.002756 | 0.006107 | 219.551432 |
| 3 | 1.506399 | 1.384980 | 0.006107 | 0.006107 | 0.009094 | 264.822892 |

checks:

- PASS: installed in driver + workers — 4 install lines
- PASS: ENABLED with target 3.0
- PASS: no Traceback other than the benign DataLoader-teardown one
- PASS: job rc=0
- PASS: step 1: metrics present — {'entropy': 1.6222467422485352, 'entropy_loss': 1.560556374490261, 'ent_coeff': 0.0, 'ent_coeff_applied': 0.0}
- PASS: step 1: worker applied == driver coeff — 0.0 vs 0.0
- PASS: step 1: coeff_next follows the rule — next=0.00275550651550293 expected=0.002756
- PASS: step 2: metrics present — {'entropy': 1.3241909742355347, 'entropy_loss': 1.2976315543055534, 'ent_coeff': 0.00275550651550293, 'ent_coeff_applied': 0.00275550651550293}
- PASS: step 2: worker applied == driver coeff — 0.00275550651550293 vs 0.00275550651550293
- PASS: step 2: coeff_next follows the rule — next=0.006107124567031861 expected=0.006107
- PASS: step 2: coefficient moved in the expected direction — 0.000000 -> 0.002756 (entropy 1.324 vs target 3.0)
- PASS: step 3: metrics present — {'entropy': 1.506399154663086, 'entropy_loss': 1.3849801793694496, 'ent_coeff': 0.006107124567031861, 'ent_coeff_applied': 0.006107124567031861}
- PASS: step 3: worker applied == driver coeff — 0.006107124567031861 vs 0.006107124567031861
- PASS: step 3: coeff_next follows the rule — next=0.00909432625770569 expected=0.009094
- PASS: step 3: coefficient moved in the expected direction — 0.002756 -> 0.006107 (entropy 1.506 vs target 3.0)

- 2026-09-18 03:42 smoke 63567991 PASSED (all checks; rc=0) -> run R launched from the login node (post-smoke job 63568198 was still queued): {"run": "R_rlvr_entropy", "train": "63570807", "resub": ["63570808", "63570809"], "wall": "10:00:00", "outdir": "/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy", "account": "grp_vgupt140", "prev_train_logs": [], "launched": "2026-09-18 03:42"}
### training job 63567988 started Fri Sep 18 03:47:41 MST 2026 on sg235 (RESUB=0, wall=14:00:00, existing ckpts: 1)

## Resume verification (M, 2026-09-18 04:46)

- job 63567988: [36m(TaskRunner pid=2291420)[0m Setting global step to 60
- job 63567988: [36m(TaskRunner pid=2291420)[0m Resuming from /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate/checkpoints/global_step_60
- first TRAINING `step:N` metric line in the resumed job: **61** (expected 61; verl also logs the resume-time validation as step:60)
- rollouts/61.jsonl prompt groups: 32; identical to step 1's batch: 2/32 (a restart would give 32/32)
- text overlap of step 61 with the union of steps 1-60 (1802 distinct prompts): **2** — within-run baseline (v4 B step 60 vs 1-59): 4; the train subset has 171 question texts that occur on >1 row (different images), e.g. 'Find x.' ×75
- **PASS** — the dataloader continued from batch 61
