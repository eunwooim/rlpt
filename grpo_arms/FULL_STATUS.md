# FULL STATUS — run M (bipartite match, resumed from v4 B step 60 → 183) and run R (RLVR + adaptive entropy, 183 steps), account grp_vgupt140



Started 2026-09-18 02:28. Live file: the monitor job (grpo_arms/monitor_full.py, public CPU) rewrites the "Job table" and "Live monitor" sections every 30 min;
training jobs append their start/end blocks to the last section. Everything else is written once and dated.

## Job table

| run | job | role | state | notes |
|---|---|---|---|---|
| M | 63567988 | training | COMPLETED | wall 14:00:00, launched 2026-09-18 02:42 |
| M | 63567989 | resubmit 1 (afternotok) | CANCELLED | wall 14:00:00, launched 2026-09-18 02:42 |
| M | 63567990 | resubmit 2 (afternotok) | CANCELLED | wall 14:00:00, launched 2026-09-18 02:42 |
| M | 63580419 | eval step 90 | generation present | vpb_gen_B_matchv4_softgate_step90_2ktest.jsonl |
| M | 63587597 | eval step 120 | generation present | vpb_gen_B_matchv4_softgate_step120_2ktest.jsonl |
| M | 63597712 | eval step 150 | generation present | vpb_gen_B_matchv4_softgate_step150_2ktest.jsonl |
| M | 63605758 | eval step 183 | generation present | vpb_gen_B_matchv4_softgate_step183_2ktest.jsonl |
| R | 63570804 | training | COMPLETED | wall 10:00:00, launched 2026-09-18 03:42 |
| R | 63570805 | resubmit 1 (afternotok) | CANCELLED | wall 10:00:00, launched 2026-09-18 03:42 |
| R | 63570806 | resubmit 2 (afternotok) | CANCELLED | wall 10:00:00, launched 2026-09-18 03:42 |
| R | 63578736 | eval step 30 | generation present | vpb_gen_R_rlvr_entropy_step30_2ktest.jsonl |
| R | 63583940 | eval step 60 | generation present | vpb_gen_R_rlvr_entropy_step60_2ktest.jsonl |
| R | 63587598 | eval step 90 | generation present | vpb_gen_R_rlvr_entropy_step90_2ktest.jsonl |
| R | 63592042 | eval step 120 | generation present | vpb_gen_R_rlvr_entropy_step120_2ktest.jsonl |
| R | 63597713 | eval step 150 | generation present | vpb_gen_R_rlvr_entropy_step150_2ktest.jsonl |
| R | 63600730 | eval step 183 | generation present | vpb_gen_R_rlvr_entropy_step183_2ktest.jsonl |
| smoke | 63567991 | adaptive-entropy smoke (htc, 1 A100, 3 steps) | COMPLETED | grpo_arms/full/smoke_entropy.sbatch |
| report | 63605759 | FULL_REPORT.md post-job (afterany) | COMPLETED | grpo_arms/report_full.sbatch |
| monitor | 63577023 | this monitor (public CPU) | running | every 30 min |

## Live monitor (monitor_full.py, 2026-09-18 14:11, job 63577023)

### M — M: match + soft gates, resumed from v4 B@60 — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate
progress bar: 183/183 · last metric step: 183 · FSDP checkpoints: [60, 120, 180, 183] (latest=183) · merged: [30, 60, 90, 120, 150, 183] · DONE=True · STOP_WATCHDOG=False
  [arm1] done=Sun Sep 13 11:01:39 MST 2026 rc=0
  [arm1] host=sg235 job=63567988 start=Fri Sep 18 03:47:41 MST 2026
  [36m(TaskRunner pid=2291420)[0m Setting global step to 60
  [36m(TaskRunner pid=2291420)[0m Resuming from /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate/checkpoints/global_step_60
  [arm1] done=Fri Sep 18 13:39:49 MST 2026 rc=0
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
| 80 | 246 | 0.703 | 0.340 | 0.359 | 0.781 | 0.950 | 0.521 | off | 0.0290 | 321 | 4.8 | 0.469 | 0.019 | 0.000 |
| 90 | 318 | 0.773 | 0.325 | 0.337 | 0.731 | 0.938 | 0.557 | off | 0.0253 | 355 | 4.9 | 0.500 | 0.019 | 0.000 |
| 100 | 270 | 0.766 | 0.357 | 0.330 | 0.756 | 0.950 | 0.661 | off | 0.0293 | 293 | 4.6 | 0.406 | 0.000 | 0.000 |
| 110 | 242 | 0.747 | 0.340 | 0.332 | 0.625 | 0.950 | 0.881 | off | 0.0354 | 274 | 4.4 | 0.548 | 0.019 | 0.000 |
| 120 | 338 | 0.750 | 0.355 | 0.341 | 0.694 | 0.950 | 0.563 | off | 0.0379 | 375 | 4.8 | 0.594 | 0.031 | 0.000 |
| 130 | 264 | 0.790 | 0.343 | 0.390 | 0.738 | 0.981 | 0.506 | off | 0.0373 | 353 | 4.4 | 0.531 | 0.000 | 0.000 |
| 140 | 246 | 0.699 | 0.343 | 0.382 | 0.688 | 1.000 | 0.489 | off | 0.0433 | 318 | 4.4 | 0.500 | 0.000 | 0.000 |
| 150 | 299 | 0.768 | 0.347 | 0.353 | 0.681 | 0.994 | 0.545 | off | 0.0485 | 303 | 4.5 | 0.355 | 0.006 | 0.000 |
| 160 | 261 | 0.798 | 0.362 | 0.377 | 0.775 | 0.981 | 0.531 | off | 0.0400 | 329 | 4.8 | 0.594 | 0.000 | 0.000 |
| 170 | 285 | 0.757 | 0.362 | 0.404 | 0.794 | 0.981 | 0.493 | off | 0.0501 | 355 | 4.8 | 0.344 | 0.006 | 0.000 |
| 180 | 279 | 0.734 | 0.340 | 0.415 | 0.725 | 0.981 | 0.615 | off | 0.0533 | 358 | 4.6 | 0.531 | 0.019 | 0.000 |
| 183 | 296 | 0.759 | 0.380 | 0.434 | 0.787 | 1.000 | 0.553 | off | 0.0565 | 345 | 4.6 | 0.406 | 0.000 | 0.000 |
**FLAGS:** step 146: entropy=0.390 < 0.4

### R — R: RLVR answer-only + adaptive entropy (target 0.6) — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy
progress bar: 183/183 · last metric step: 183 · FSDP checkpoints: [60, 120, 180, 183] (latest=183) · merged: [30, 60, 90, 120, 150, 183] · DONE=True · STOP_WATCHDOG=False
  [arm1] host=sg031 job=63570804 start=Fri Sep 18 04:26:33 MST 2026
  [arm1] done=Fri Sep 18 12:08:26 MST 2026 rc=0
step-0 val: train-val acc 0.656, vpb_dev acc 0.315
| step | s/step | trainval acc | vpb_dev acc | match | answer | format | entropy | ent coeff | KL | resp_len | n_seg | mixed grp frac | soft-gate fired | trunc (clip) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 178 | - | - | n/a | 0.338 | 0.731 | 2.076 | 0.00000 | 0.0005 | 441 | 7.8 | 0.781 | n/a | 0.019 |
| 5 | 141 | - | - | n/a | 0.494 | 0.719 | 1.161 | 0.00000 | 0.0018 | 304 | 6.5 | 0.839 | n/a | 0.000 |
| 10 | 153 | 0.691 | 0.318 | n/a | 0.506 | 0.781 | 1.804 | 0.00000 | 0.0026 | 337 | 6.8 | 0.625 | n/a | 0.013 |
| 20 | 142 | 0.711 | 0.302 | n/a | 0.525 | 0.775 | 1.114 | 0.00000 | 0.0063 | 309 | 6.1 | 0.677 | n/a | 0.000 |
| 30 | 168 | 0.674 | 0.323 | n/a | 0.575 | 0.856 | 0.926 | 0.00000 | 0.0103 | 367 | 7.5 | 0.594 | n/a | 0.000 |
| 40 | 145 | 0.667 | 0.325 | n/a | 0.669 | 0.875 | 0.958 | 0.00000 | 0.0159 | 337 | 6.0 | 0.562 | n/a | 0.000 |
| 50 | 118 | 0.692 | 0.328 | n/a | 0.575 | 0.944 | 0.601 | 0.00000 | 0.0148 | 338 | 6.2 | 0.531 | n/a | 0.000 |
| 60 | 175 | 0.770 | 0.333 | n/a | 0.656 | 0.963 | 0.641 | 0.00011 | 0.0181 | 330 | 5.7 | 0.688 | n/a | 0.000 |
| 61 | 132 | - | - | n/a | 0.606 | 0.956 | 0.572 | 0.00003 | 0.0153 | 402 | 6.2 | 0.594 | n/a | 0.006 |
| 70 | 139 | 0.801 | 0.328 | n/a | 0.706 | 0.938 | 0.802 | 0.00000 | 0.0221 | 289 | 5.6 | 0.719 | n/a | 0.000 |
| 80 | 137 | 0.791 | 0.328 | n/a | 0.725 | 0.963 | 0.543 | 0.00014 | 0.0239 | 316 | 5.8 | 0.469 | n/a | 0.000 |
| 90 | 148 | 0.767 | 0.333 | n/a | 0.688 | 0.950 | 0.482 | 0.00010 | 0.0244 | 356 | 6.1 | 0.500 | n/a | 0.000 |
| 100 | 131 | 0.786 | 0.362 | n/a | 0.719 | 0.975 | 0.610 | 0.00119 | 0.0287 | 297 | 5.5 | 0.406 | n/a | 0.000 |
| 110 | 135 | 0.779 | 0.365 | n/a | 0.613 | 0.963 | 0.981 | 0.00008 | 0.0296 | 285 | 6.0 | 0.613 | n/a | 0.000 |
| 120 | 182 | 0.815 | 0.362 | n/a | 0.731 | 0.938 | 0.643 | 0.00004 | 0.0257 | 400 | 6.3 | 0.469 | n/a | 0.000 |
| 130 | 122 | 0.754 | 0.360 | n/a | 0.706 | 0.969 | 0.712 | 0.00000 | 0.0274 | 340 | 5.6 | 0.375 | n/a | 0.000 |
| 140 | 136 | 0.792 | 0.365 | n/a | 0.669 | 0.956 | 0.779 | 0.00000 | 0.0303 | 340 | 6.1 | 0.469 | n/a | 0.006 |
| 150 | 154 | 0.781 | 0.345 | n/a | 0.713 | 0.956 | 0.921 | 0.00000 | 0.0320 | 309 | 6.3 | 0.452 | n/a | 0.000 |
| 160 | 145 | 0.780 | 0.378 | n/a | 0.769 | 0.975 | 0.750 | 0.00000 | 0.0293 | 339 | 5.9 | 0.469 | n/a | 0.000 |
| 170 | 124 | 0.774 | 0.367 | n/a | 0.825 | 1.000 | 0.647 | 0.00000 | 0.0321 | 305 | 5.4 | 0.188 | n/a | 0.000 |
| 180 | 141 | 0.795 | 0.350 | n/a | 0.744 | 0.988 | 0.759 | 0.00000 | 0.0357 | 296 | 5.4 | 0.500 | n/a | 0.000 |
| 183 | 174 | 0.806 | 0.380 | n/a | 0.869 | 0.975 | 0.695 | 0.00000 | 0.0343 | 286 | 5.2 | 0.250 | n/a | 0.000 |
**FLAGS:** step 93: entropy=0.395 < 0.4

events (last 25; full log grpo_arms/full/monitor_events.log):

- 2026-09-18 06:40 eval M:90: submitted job 63580419 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 06:40 eval R:30: generation present (vpb_gen_R_rlvr_entropy_step30_2ktest.jsonl)
- 2026-09-18 07:10 eval M:90: generation present (vpb_gen_B_matchv4_softgate_step90_2ktest.jsonl)
- 2026-09-18 07:40 eval R:60: submitted job 63583940 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 07:40 pruned FSDP checkpoint R global_step_30 (44G; merge verified: 2 shards + bundle ok; newest=60, keep=[60, 120, 183])
- 2026-09-18 08:10 eval R:60: generation present (vpb_gen_R_rlvr_entropy_step60_2ktest.jsonl)
- 2026-09-18 08:40 eval M:120: submitted job 63587597 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 08:40 eval R:90: submitted job 63587598 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 08:40 pruned FSDP checkpoint M global_step_90 (44G; merge verified: 2 shards + bundle ok; newest=120, keep=[60, 120, 183])
- 2026-09-18 09:10 eval M:120: generation present (vpb_gen_B_matchv4_softgate_step120_2ktest.jsonl)
- 2026-09-18 09:40 eval R:90: generation present (vpb_gen_R_rlvr_entropy_step90_2ktest.jsonl)
- 2026-09-18 09:40 eval R:120: submitted job 63592042 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 09:40 pruned FSDP checkpoint R global_step_90 (44G; merge verified: 2 shards + bundle ok; newest=120, keep=[60, 120, 183])
- 2026-09-18 10:10 eval R:120: generation present (vpb_gen_R_rlvr_entropy_step120_2ktest.jsonl)
- 2026-09-18 11:11 eval M:150: submitted job 63597712 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 11:11 eval R:150: submitted job 63597713 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 11:41 eval M:150: generation present (vpb_gen_B_matchv4_softgate_step150_2ktest.jsonl)
- 2026-09-18 11:41 eval R:150: generation present (vpb_gen_R_rlvr_entropy_step150_2ktest.jsonl)
- 2026-09-18 12:11 eval R:183: submitted job 63600730 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 12:11 pruned FSDP checkpoint R global_step_150 (44G; merge verified: 2 shards + bundle ok; newest=183, keep=[60, 120, 183])
- 2026-09-18 12:41 eval R:183: generation present (vpb_gen_R_rlvr_entropy_step183_2ktest.jsonl)
- 2026-09-18 13:41 eval M:183: submitted job 63605758 (htc, gpu:1, 1 h, vpb_test 2048 tok)
- 2026-09-18 13:41 pruned FSDP checkpoint M global_step_150 (44G; merge verified: 2 shards + bundle ok; newest=183, keep=[60, 120, 183])
- 2026-09-18 13:41 report post-job 63605759 submitted (afterany on 16 training + eval jobs)
- 2026-09-18 14:11 eval M:183: generation present (vpb_gen_B_matchv4_softgate_step183_2ktest.jsonl)

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
- 2026-09-18 03:42 smoke 63567991 PASSED (all checks; rc=0) -> run R launched from the login node (post-smoke job 63568198 was still queued): {"run": "R_rlvr_entropy", "train": "63570807", "resub": ["63570808", "63570809"], "wall": "10:00:00", "outdir": "/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy", "account": "grp_vgupt140", "prev_train_logs": [], "launched": "2026-09-18 03:42"}
- 2026-09-18 04:47 INCIDENT, resolved: the post-smoke CPU job 63568198 started at 03:42 in the same minute I launched R by hand, so two R chains were
  submitted for the same OUTDIR (63570804/05/06 by the post-smoke job, and 63570807/08/09 by me — my launcher wrote chain.json last, so the
  file pointed at the wrong chain until I corrected it at 05:36; the monitor's R table lacked training metrics until then). Both training jobs got nodes at 04:25 / 04:43. I cancelled my chain (63570807/08/09) at 04:45; job 63570807 had only
  printed its header (no score server, no Ray, no verl, nothing written under the OUTDIR) before the cancel, so run R = 63570804 (sg031, started
  04:25, resubs 63570805/63570806) is unaffected. Root cause: the launcher's "already launched" guard checked chain.json, which neither job had
  written yet when the other started. Lesson: with a Slurm post-job in flight, do not also launch by hand.
- 2026-09-18 04:47 M resume verified by hand from the log: "Resuming from .../global_step_60", "Setting global step to 60", train dataloader
  183 batches/epoch (the 5,880-prompt train subset → 183 steps = 1 epoch; 60 mod 183 ≠ 0 so verl restored data.pt), first TRAINING metric line
  step:61 (verl also logs the resume-time validation as step:60, which the monitor's first version mis-read as the first step). Step-61 prompt
  overlap with steps 1-60 = 2/32 by question text; v4 B's own step 60 overlapped 4/32 with steps 1-59 the same way because 171 question texts
  recur on several rows with different images ("Find x." ×75). Not a restart (2/32 of step 61 equal step 1's batch, a restart would give 32/32).
  Monitor check fixed accordingly and the monitor job restarted (63568199 → 63574075); the section below now reads PASS.
- 04:47 M pace: ~4 min/step at steps 61-69 (ETA ≈ 7.5 h of the 14 h wall); entropy 0.44-0.57; resp_len ≈ 280-345.
- 05:36 R health from its real log (63570804): entropy 2.08 → 0.98 over steps 1-21, coefficient 0 throughout (correct: entropy > 0.6), KL 0.005,
  resp_len ≈ 310-340, ~2.4 min/step (ETA ≈ 6.7 h of the 10 h wall), vpb_dev 0.315 (step 0) → 0.318 (10) → 0.303 (20), no Traceback.
- 18:35 DONE. M (63567988) finished 183/183 at 13:39 (9 h 52 min, rc=0, no resubmit needed); R (63570804) finished at 12:08 (7 h 42 min, rc=0);
  both afternotok resubmit chains were auto-cancelled on success. All 10 evals generated on the first attempt; FSDP pruned to {60, 120, 183}
  (+ global_step_180 in each run: saved by save_freq 30 three steps before the final save, never an eval step so never merged, hence not pruned
  under the "after verified merge" rule — 44 GB each, delete on your word). Final report written by the post-job 63605759 (14:04) and refreshed
  18:31: grpo_arms/FULL_REPORT.md. Fairshare grp_vgupt140: RawUsage_CHE 3832.4 → 6346.1 (+2513.7), RealFairShare 0.164 → 0.050. grp_bshettah untouched.
- Headline (vpb_test, 2,456 q, greedy, 2048 tok): M 90/120/150/183 = 0.3335 / 0.3465 / 0.3599 / **0.3632** (vs v4 B@60 0.3436 +0.0195, McNemar p=0.036;
  vs cs25 +0.0440, p<0.001). R 30/60/90/120/150/183 = 0.3265 / 0.3359 / 0.3498 / 0.3493 / 0.3481 / **0.3563** (vs v4 A@60 0.3282 +0.0281, p=0.003;
  vs cs25 +0.0371, p<0.001; no post-30 decline). M vs R at 183: +0.0069, p=0.471 (not separable). M entropy dipped below 0.4 exactly once
  (0.390 at step 146; last-30 mean 0.544); R's entropy floor engaged only briefly (steps 59-63 and ~80-100, max coeff 1.4e-3) and R settled
  at 0.65-0.92 without it. Both runs' only Traceback is the benign DataLoader-teardown one after rc=0.
- 09:45 interim: M@120 = 0.3465 (vs cs25 +0.0273, McNemar p=0.003; vs v4 B@60 0.3436 +0.0029, p=0.787; length rate 0.012).
  R@90 = 0.3498 (vs cs25 +0.0305, p=0.001; vs v4 A@60 0.3282 +0.0216, p=0.020; vs M@90 0.3335 +0.0163, p=0.069) — R rises monotonically
  30 → 60 → 90 (0.3265 / 0.3359 / 0.3498), no v4-A-style decline. R vpb_dev 0.3625 (100) / 0.365 (110) / 0.3625 (120); entropy oscillates
  0.61 (100) / 0.98 (110) / 0.64 (120) / 0.76 (122) with the coefficient pulsing 1.2e-3 (100) → 8e-5 (110) → 4e-5 (120) → 0 (122).
  M vpb_dev 0.355 (120) / 0.3425 (130); entropy 0.56 (120) → 0.51 (130) → 0.43 (134) — approaching the 0.4 flag line; resp_len 300–375.
  Evals: R:120 running (63592042); R global_step_90 pruned after verified merge. No errors. Progress M 134/183, R 122/183.
- 08:45 interim: R@60 = 0.3359 on vpb_test (R@30 0.3265 → no post-30 decline so far; vs cs25 +0.0167, McNemar p=0.070; vs v4 A@60 0.3282,
  p=0.422; vs v4 B@60 p=0.416). R entropy 0.64 (60) → 0.80 (70) → 0.54 (80) → 0.48 (90) → 0.51 (97); coefficient now engaging: 1.4e-4 (80),
  1.0e-4 (90), 1.2e-3 (97). M: step 120 saved (vpb_dev 0.355; 0.3575 at 100, 0.34 at 110), entropy 0.56–0.88 over 100–120, resp_len 275–375.
  Evals queued: M:120 (63587597), R:90 (63587598); M global_step_90 pruned after verified merge (60 and 120 kept). No errors in either log.
- 07:45 interim vpb_test (report_full.py, partial FULL_REPORT.md): M@90 = 0.3335 (v4 B@60 0.3436: −0.0101, McNemar b=217 c=242 p=0.263;
  vs cs25 +0.0143, p=0.129; length rate 0.027); R@30 = 0.3265 (vs cs25 +0.0073, p=0.439; vs v4 A@60 p=0.892; v4 A@30 was 0.3384).
  Curves: M vpb_dev 0.3125 (70) / 0.34 (80) / 0.325 (90) / 0.3575 (100), entropy 0.52–0.73 at 80–107 (min 0.44 at step 62), KL ≈ 0.03;
  R vpb_dev 0.3225 (30) / 0.325 (40) / 0.3275 (50) / 0.3325 (60) / 0.3275 (70), entropy 0.93 (30) → 0.60 (50) → 0.64 (60) → 0.80 (70):
  the coefficient first became positive at step 59 (1.1e-4), peaked 2.0e-4 at step 62 and returned to 0 by step 63–70 as entropy
  rebounded above 0.6 — the floor engages and releases as specified. Pace: M ≈ 4.0 min/step (ETA step 183 ≈ 13:00), R ≈ 2.4 min/step (ETA ≈ 11:15).
- 05:25 status-file repair: the monitor's section writer matched the words "## Live monitor" inside this file's intro sentence, so the live table
  was written at the top without a proper header, and bullets I had appended after the last section were wiped when that section was rewritten.
  Fixed (headers matched at line start; run_arm.sbatch's appended lines now go to a dedicated last section), file rebuilt from
  grpo_arms/full/FULL_STATUS.pre_rebuild_0520.md, monitor restarted again (63574075 → see job table). Monitor pass time also cut (log/rollout caches).

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

## Resume verification (M, 2026-09-18 04:46)

- job 63567988: [36m(TaskRunner pid=2291420)[0m Setting global step to 60
- job 63567988: [36m(TaskRunner pid=2291420)[0m Resuming from /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate/checkpoints/global_step_60
- first TRAINING `step:N` metric line in the resumed job: **61** (expected 61; verl also logs the resume-time validation as step:60)
- rollouts/61.jsonl prompt groups: 32; identical to step 1's batch: 2/32 (a restart would give 32/32)
- text overlap of step 61 with the union of steps 1-60 (1802 distinct prompts): **2** — within-run baseline (v4 B step 60 vs 1-59): 4; the train subset has 171 question texts that occur on >1 row (different images), e.g. 'Find x.' ×75
- **PASS** — the dataloader continued from batch 61

## Training job log (appended by run_arm.sbatch; keep this the LAST section)
### training job 63567988 started Fri Sep 18 03:47:41 MST 2026 on sg235 (RESUB=0, wall=14:00:00, existing ckpts: 1)

### training job 63570804 ended Fri Sep 18 12:08:26 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 60 120 150 180 183 )
```
[monitor] 183 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_rlvr_entropy/monitor.csv
step-0 val (source-avg): answer=0.646  format=0.747  gated=0.082  inv_frac=0.000  match=0.000  n_dup=0.161  n_segments=6.412  pun=0.000  score=3.834
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.231    440.9    0.019    0.150    0.119    7.838    0.000    0.000    0.338    0.731    0.000      -      0.000    0.000      -        -      178.4
      20    3.338    309.5    0.000    0.037    0.094    6.088    0.000    0.000    0.525    0.775    0.000      -      0.000    0.000    0.700    0.000    142.2
      40    4.219    336.6    0.000    0.006    0.069    6.025    0.000    0.000    0.669    0.875    0.000      -      0.000    0.000    0.658    0.000    145.4
      60    4.213    330.1    0.000    0.013    0.119    5.737    0.000    0.000    0.656    0.963    0.000      -      0.000    0.000    0.758    0.000    174.9
      80    4.556    315.6    0.000    0.006    0.094    5.844    0.000    0.000    0.725    0.963    0.000      -      0.000    0.000    0.779    0.000    137.3
     100    4.569    296.6    0.000    0.006    0.044    5.500    0.000    0.000    0.719    0.975    0.000      -      0.000    0.000    0.774    0.000    130.6
     120    4.562    400.4    0.000    0.031    0.087    6.294    0.000    0.000    0.731    0.938    0.000      -      0.000    0.000    0.802    0.000    181.7
     140    4.300    340.1    0.006    0.006    0.125    6.056    0.000    0.000    0.669    0.956    0.000      -      0.000    0.000    0.781    0.000    136.0
     160    4.819    339.3    0.000    0.000    0.175    5.944    0.000    0.000    0.769    0.975    0.000      -      0.000    0.000    0.768    0.000    144.8
     180    4.706    296.2    0.000    0.013    0.050    5.381    0.000    0.000    0.744    0.988    0.000      -      0.000    0.000    0.782    0.000    141.1
     183    5.319    285.8    0.000    0.000    0.037    5.237    0.000    0.000    0.869    0.975    0.000      -      0.000    0.000    0.794    0.000    173.9
gate-2: over 183 steps, Pearson r(match/mean, answer/mean) = +nan
  first quarter: match 0.000 answer 0.557 r=+nan
  last quarter: match 0.000 answer 0.746 r=+nan
  within-step strata (all steps): mean match|answer=1 = 0.000, match|answer=0 = 0.000
  JUDGMENT (auto, rule-based): answer moved (+0.189) while match stayed flat (+0.000) — the match term is not what training optimised. Within steps, correct-answer rollouts score +0.000 higher on match than wrong-answer ones (match is essentially answer-blind at the sample level).
RED-FLAG none
```

### training job 63567988 ended Fri Sep 18 13:39:49 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 60 120 150 180 183 )
```
[monitor] 183 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate/monitor.csv
step-0 val (source-avg): answer=0.664  format=0.773  gated=0.038  inv_frac=0.263  match=0.299  n_dup=0.160  n_segments=5.613  pun=0.054  score=4.540
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.957    471.4    0.031    0.025    0.131    8.250    0.236    0.240    0.412    0.775    0.325    0.342    0.291    0.204      -        -      410.4
      20    3.646    291.9    0.000    0.000    0.069    5.394    0.046    0.268    0.506    0.812    0.283    0.429    0.323    0.212    0.681    0.317    251.6
      40    4.882    315.8    0.000    0.006    0.075    4.756    0.024    0.313    0.675    0.906    0.245    0.366    0.352    0.232    0.649    0.331    254.5
      60    4.978    295.2    0.000    0.000    0.087    4.406    0.004    0.313    0.681    0.950    0.247    0.329    0.345    0.245    0.711    0.337    284.4
      80    5.464    321.5    0.000    0.000    0.100    4.775    0.016    0.359    0.781    0.950    0.231    0.385    0.385    0.267    0.693    0.350    246.2
     100    5.384    292.7    0.000    0.000    0.075    4.581    0.007    0.330    0.756    0.950    0.246    0.387    0.363    0.226    0.755    0.389    269.5
     120    5.056    375.0    0.000    0.000    0.056    4.812    0.014    0.341    0.694    0.950    0.244    0.288    0.370    0.277    0.739    0.391    337.5
     140    5.198    318.4    0.000    0.000    0.100    4.381    0.003    0.382    0.688    1.000    0.219    0.418    0.426    0.285    0.689    0.390    245.9
     160    5.605    329.2    0.000    0.000    0.062    4.781    0.004    0.377    0.775    0.981    0.226    0.368    0.405    0.278    0.786    0.393    261.0
     180    5.434    358.1    0.000    0.000    0.087    4.638    0.003    0.415    0.725    0.981    0.205    0.456    0.453    0.315    0.723    0.419    279.2
     183    5.801    344.8    0.000    0.000    0.056    4.600    0.004    0.434    0.787    1.000    0.200    0.464    0.470    0.299    0.748    0.416    296.1
gate-2: over 183 steps, Pearson r(match/mean, answer/mean) = +0.834
  first quarter: match 0.282 answer 0.560 r=+0.850
  last quarter: match 0.375 answer 0.737 r=+0.367
  within-step strata (all steps): mean match|answer=1 = 0.374, match|answer=0 = 0.264
  JUDGMENT (auto, rule-based): match (+0.093) and answer (+0.177) moved together; the across-step Pearson r = +0.834. They track each other closely — match adds little beyond answer over the run. Within steps, correct-answer rollouts score +0.110 higher on match than wrong-answer ones (match is answer-sensitive at the sample level).
RED-FLAG WARN response_length/mean up 320->370 while score up 5.33->5.48 (steps 144-163 vs 164-183)
```

- 2026-09-18 14:04 FULL_REPORT.md written by job 63605759 (rc=0)
