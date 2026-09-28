# V4 STATUS — answer-only control (A) + match with soft gates (B) at a conservative optimizer, account grp_bshettah
Started 2026-09-13 04:30. Live file: the monitor job rewrites "## Live monitor (monitor_v4.py, 2026-09-13 11:18)

### A_ctrl_v4 — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_v4
progress bar: 60/60 · last metric step: 60 · checkpoints with actor/: [30, 60] · DONE=True · STOP_WATCHDOG=False
  [arm1] host=sg043 job=63147498 start=Sun Sep 13 04:34:26 MST 2026
  [arm1] done=Sun Sep 13 04:55:08 MST 2026 rc=1
  [arm1] host=sg043 job=63147499 start=Sun Sep 13 04:55:15 MST 2026
  [arm1] done=Sun Sep 13 07:25:41 MST 2026 rc=0
step-0 val: train-val acc 0.679, vpb_dev acc 0.330
| step | s/step | trainval acc | vpb_dev acc | resp_len | clip | match | answer | entropy | KL | mixed grp frac | n_seg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 172 | - | - | 419 | 0.013 | 0.000 | 0.381 | 1.738 | 0.0005 | 0.750 | 7.7 |
| 5 | 143 | - | - | 333 | 0.013 | 0.000 | 0.463 | 1.831 | 0.0015 | 0.677 | 6.5 |
| 10 | 149 | 0.735 | 0.340 | 346 | 0.006 | 0.000 | 0.537 | 1.492 | 0.0059 | 0.656 | 6.3 |
| 20 | 136 | 0.766 | 0.335 | 315 | 0.000 | 0.000 | 0.525 | 1.172 | 0.0066 | 0.742 | 6.3 |
| 30 | 163 | 0.771 | 0.315 | 368 | 0.000 | 0.000 | 0.613 | 0.843 | 0.0096 | 0.562 | 7.1 |
| 40 | 137 | 0.659 | 0.345 | 325 | 0.000 | 0.000 | 0.631 | 0.746 | 0.0155 | 0.531 | 6.0 |
| 50 | 114 | 0.748 | 0.330 | 326 | 0.000 | 0.000 | 0.556 | 0.607 | 0.0131 | 0.750 | 6.4 |
| 60 | 153 | 0.718 | 0.320 | 300 | 0.000 | 0.000 | 0.662 | 0.649 | 0.0174 | 0.656 | 5.5 |
flags: none

### B_matchv4_softgate — /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate
progress bar: 60/60 · last metric step: 60 · checkpoints with actor/: [30, 60] · DONE=True · STOP_WATCHDOG=False
  [arm1] host=sg030 job=63148593 start=Sun Sep 13 05:48:02 MST 2026
  [arm1] done=Sun Sep 13 11:01:39 MST 2026 rc=0
step-0 val: train-val acc 0.673, vpb_dev acc 0.345
| step | s/step | trainval acc | vpb_dev acc | resp_len | clip | match | answer | entropy | KL | mixed grp frac | n_seg | soft-gate fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 410 | - | - | 471 | 0.031 | 0.240 | 0.412 | 2.343 | 0.0005 | 0.781 | 8.2 | 0.081 |
| 5 | 278 | - | - | 344 | 0.013 | 0.243 | 0.406 | 1.826 | 0.0012 | 0.645 | 6.7 | 0.169 |
| 10 | 283 | 0.699 | 0.305 | 322 | 0.006 | 0.253 | 0.506 | 1.499 | 0.0022 | 0.719 | 6.0 | 0.056 |
| 20 | 252 | 0.690 | 0.338 | 292 | 0.000 | 0.268 | 0.506 | 1.197 | 0.0067 | 0.645 | 5.4 | 0.044 |
| 30 | 328 | 0.683 | 0.338 | 347 | 0.000 | 0.293 | 0.531 | 0.607 | 0.0125 | 0.656 | 6.4 | 0.050 |
| 40 | 255 | 0.659 | 0.325 | 316 | 0.000 | 0.313 | 0.675 | 0.467 | 0.0226 | 0.594 | 4.8 | 0.019 |
| 50 | 256 | 0.735 | 0.320 | 321 | 0.000 | 0.339 | 0.594 | 0.419 | 0.0212 | 0.656 | 5.1 | 0.013 |
| 60 | 284 | 0.751 | 0.330 | 295 | 0.000 | 0.313 | 0.681 | 0.453 | 0.0276 | 0.656 | 4.4 | 0.019 |
flags: none


## Job table
| run | training job | resubmit 1 | resubmit 2 | eval jobs (vpb_test, 2048 tok, steps 60/30) | notes |
|---|---|---|---|---|---|
| probe (hard + soft) | 63147505 DONE rc=0 (soft: go; hard: bit-identical) | – | – | – | reward_redesign/redteam_v4gate.sbatch → probe_scored_v4_{soft,hard}.jsonl |
| A ctrl_v4 | 63147498 FAILED (soft_gated KeyError) → resubmit 63147499 COMPLETED 07:25 rc=0, 60/60 | 63147499 (live) | 63147500 | 63147501 (step 60), 63147502 (step 30) | grpo_arms/v4/A_ctrl_v4.sbatch; OUTDIR runs/arm1_3b_ctrl_v4 |
| B matchv4_softgate | 63148593 COMPLETED 11:01 rc=0, 60/60 (05:48 → 11:01 on sg030) | 63148594 | 63148595 | 63148596 (step 60), 63148597 (step 30) | grpo_arms/v4/B_matchv4_softgate.sbatch; OUTDIR runs/arm1_3b_matchv4_softgate |
| monitor | 63147506 (public CPU, 04:35 → cancelled 11:25 after both runs finished) | – | – | – | grpo_arms/monitor_v4.sbatch → monitor_v4.py --loop --interval 1800 |
| report | 63148598 COMPLETED 11:18 rc=0 | – | – | – | afterany on all 4 eval jobs → grpo_arms/V4_REPORT.md (report_v4.py) |

## Step 0 — preflight (04:31)
- Fairshare BEFORE (myfairshare): grp_bshettah / sghos104 RawUsage_CHE 1262.7, RawFairShare 0.492369, TargetFairShare 0.916194, RealFairShare 0.492369.
  sshare: account grp_bshettah RawUsage 39,591,820 (EffectvUsage 0.923589); user sghos104 under it RawUsage 4,545,880, FairShare 0.492369.
  (reference, untouched: grp_vgupt140 RawUsage_CHE 6227.8, RealFairShare 0.051862.)
- Queue: only the OOD vscode session (grp_vgupt140). Nothing of mine running.
- verl 0.8.0 config keys (trainer/config/actor/actor.yaml → actor_rollout_ref.actor.*): `clip_ratio` 0.2, `clip_ratio_low` 0.2, `clip_ratio_high` 0.2
  (decoupled clip: core_algos uses clip_ratio_low/high; `clip_ratio_c` 3.0 is the dual-clip constant), `kl_loss_coef` 0.001, `kl_loss_type`
  low_var_kl, `entropy_coeff` 0, `optim.lr` (arm1v2: 1e-05 via --learning_rate), rollout `temperature` 1.0 (validation uses 0).
  Verified against the arm1v2 job's resolved config dump (logs/arm-62998194.log): clip_ratio_low 0.2, clip_ratio_high 0.2, kl_loss_coef 0.001,
  kl_loss_type low_var_kl, entropy_coeff 0, temperature 1.0 — all as expected.
- cs25 processor bundle: all 8 files present in checkpoint-25. Template: grpo_arms/v3/ctrl.sbatch bills --account=grp_vgupt140 in its header;
  the v4 copies set --account=grp_bshettah in the header AND every sbatch call passes --account=grp_bshettah explicitly (evals/report/monitor too).
  Partition public, qos public, 2×A100, 320G, mail END/FAIL added on the command line as in v3.

## Step 1 — run A (answer-only, conservative optimizer)
Submitted 04:34 via `bash grpo_arms/launch_v4.sh A_ctrl_v4 06:00:00` (every sbatch call carries --account=grp_bshettah; wall from the v3 ctrl
measurement 2.6 min/step × 60 × 1.5 + 45 min + 7 vpb_dev vals ≈ 5.5 h). Changes vs grpo_arms/v3/ctrl.sbatch — exactly: account, job name,
GATE_MODE=hard (explicit, = previous behaviour), OUTDIR, STATUS_FILE, and EXTRA: `--max_steps 60 --save_freq 30 --test_freq 10 --learning_rate 1e-6`,
dropped `trainer.max_actor_ckpt_to_keep=4` (default retention), added `actor_rollout_ref.actor.kl_loss_coef=0.01` (kl_loss_type stays low_var_kl),
`clip_ratio_low=0.2`, `clip_ratio_high=0.28`. n=5, MAXRESP 2048, token caps 8192, seed 42, cs25 init, vpb_dev as 2nd val file, REWARD_MODE=answer_only —
all unchanged. No entropy bonus, no filter_groups, std-normalisation at the verl default (on).
```diff
2,3c2,3
< #SBATCH --job-name=v3_ctrl
< #SBATCH --account=grp_vgupt140
---
> #SBATCH --job-name=v4_A_ctrl
> #SBATCH --account=grp_bshettah
17c17
< # ===== v3 run "ctrl" (2026-09-11): every run-specific value is set here so `diff grpo_arms/v3/ctrl.sbatch grpo_arms/run_arm.sbatch`
---
> # ===== v4 run "A_ctrl_v4" (2026-09-13, account grp_bshettah, conservative optimizer): every run-specific value is set here so `diff grpo_arms/v3/ctrl.sbatch grpo_arms/run_arm.sbatch`
21,23c21,24
< export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_answeronly
< export STATUS_FILE=grpo_arms/V3_STATUS.md
< export EXTRA="--max_steps 90 --save_freq 10 --test_freq 10 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override trainer.max_actor_ckpt_to_keep=4 --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5"
---
> export GATE_MODE=hard
> export OUTDIR=/scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_v4
> export STATUS_FILE=grpo_arms/V4_STATUS.md
> export EXTRA="--max_steps 60 --save_freq 30 --test_freq 10 --learning_rate 1e-6 --extra_val_files /scratch/sghos104/rlpt/grpo_arms/data/vpb_dev.parquet --verl_extra_override actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 --verl_extra_override actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 --rollout_n 5 --verl_extra_override actor_rollout_ref.actor.kl_loss_coef=0.01 --verl_extra_override actor_rollout_ref.actor.clip_ratio_low=0.2 --verl_extra_override actor_rollout_ref.actor.clip_ratio_high=0.28"
```


## Step 2 — GATE_MODE (soft gates) + probe
- reward_v2.score_new gained `gate_mode` (env GATE_MODE, default "hard" = previous behaviour). soft: truncation → R=0 (unchanged); no parsable
  answer or answer not on the last line → answer term 0, match/format/pun computed as normal. score_server returns `answer_valid`
  (parsable ∧ last line ∧ not truncated) and `acc` = correct ∧ valid; arm_reward swaps in its own answer checker only when valid, emits
  `soft_gated` (soft gate fired) and keeps `acc` = "final answer correct AND on last line" in both modes. Matching/weights/segmentation/dedupe/
  order/pun untouched; score_old byte-identical. Diff: grpo_arms/v4/reward_v2_GATE_MODE.diff. Mock check (hard vs soft): perfect 8/8, no-answer
  0/2.0, answer-first 0/3.0, truncated 0/0, wrong-answer 3/3.
- Probe job 63147505 (htc gpu:1, chunker env), SOFT mode (reward_redesign/probe_scored_v4_soft.jsonl):
```
population           n   R_med  R_mean  match   prec    rec    pun   off  gated   ans%  segs  dup
attack_giant       175    5.40    3.43   0.30   0.72   0.11   0.00  0.29      0  57.1%     1    0
attack_repeat6     175    4.67    2.83   0.10   0.12   0.11   0.33  0.40      0  57.1%     6    5
clean              185    5.47    4.03   0.33   0.39   0.25   0.00  0.20      6  53.0%     5    0
degenerate         185    0.00    0.00   0.00   0.00   0.00   0.00  0.00    185   0.0%    92    0
gold               185    7.97    7.72   0.99   0.99   0.99   0.00  0.00      0 100.0%     6    0
```
  vs HARD (frozen spec, job 62990551): identical medians (gold 7.97 / clean 5.47 / giant 5.40 / repeat6 4.67 / hacked 0.00); the only change is
  clean gated 12 → 6 (six clean rows with a missing/misplaced answer are now scored on match+format instead of 0; the other six are truncated).
  Hacked stays exactly 0 because all 185 hacked rows are truncated at the cap — the truncation gate is still hard by design. (The numbers
  quoted in the brief — 8.00/5.00/4.38/3.29 — are the pre-final-spec probe; the frozen spec's hard numbers are the ones above.)
  GO CONDITION: hacked (0.00) is 5.47 below clean, ordering gold > clean > {giant, repeat6} > hacked holds → run B submitted (05:10).
- Hard-mode rerun (same job, second pass) is checked against probe_scored_final.jsonl for bit-identity; result appended below when it finishes.

## Step 3 — run B
(pending)

## Live monitor
(written by the monitor job)

## Decisions taken without you
- 04:35 the brief asks for the monitor as a 12-h htc CPU job; htc MaxTime is 4 h, so the monitor runs on public (2 cores, 4 GB, no gres, 12 h).
### training job 63147498 started Sun Sep 13 04:34:26 MST 2026 on sg043 (RESUB=0, wall=6:00:00, existing ckpts: 0)
- 04:53 run A resolved config (logs/arm-63147498.log): actor lr 1e-06, kl_loss_coef 0.01, kl_loss_type low_var_kl, clip_ratio_low 0.2, clip_ratio_high 0.28, entropy_coeff 0, rollout temperature 1.0 — as intended. Probe 63147505 running (soft mode first, then hard).

### training job 63147498 ended Sun Sep 13 04:55:08 MST 2026 rc=1 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: )
```
[monitor] 0 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_v4/monitor.csv
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
gate-2: fewer than 3 steps with rollout dumps — no correlation yet
RED-FLAG none
```
### training job 63147499 started Sun Sep 13 04:55:15 MST 2026 on sg043 (RESUB=1, wall=6:00:00, existing ckpts: 0)
- 04:56 run A 63147498 FAILED in step-0 validation: `KeyError: 'soft_gated'` — verl's reward loop needs every reward dict to carry the same
  keys, and arm_reward's reference-free vpb_dev path did not emit the new `soft_gated` key. Fixed (soft_gated=0 on that path; diff refreshed
  in grpo_arms/v4/reward_v2_GATE_MODE.diff). The afternotok resubmit 63147499 becomes the live run A attempt (reads the fixed file at start;
  OUTDIR data reused). One resubmit (63147500) left in reserve.
- 05:10 run A resub 63147499 healthy: step-0 vpb_dev 0.33; step 1/2: entropy 1.74 / 2.04, KL 0.0005 / 0.0008, clipfrac 2e-4, score 2.27 / 2.14, resp_len 419 / 346, 172 / 139 s/step.
- 05:24 probe job 63147505 finished: HARD-mode rerun vs probe_scored_final.jsonl (job 62990551) = 0 mismatching fields → BIT-IDENTICAL; the GATE_MODE switch changes nothing in hard mode. Soft go-gap 5.47.
- 05:28 run A step 10: vpb_dev 0.34 (0.33 @0), train-val acc 0.729 (0.62 at init), entropy 1.49, KL 0.0059, score 3.38, resp_len 346, clip 0.006 — first run in the campaign where entropy stays > 1 and dev rises. (v3 ctrl @10: entropy 0.79, dev 0.2875.) B 63148593 pending (Resources).
### training job 63148593 started Sun Sep 13 05:48:02 MST 2026 on sg030 (RESUB=0, wall=8:00:00, existing ckpts: 0)
- 05:48 run B 63148593 started on sg030.
- 05:52 run A step 20: vpb_dev 0.335, train-val 0.766, entropy 1.17, KL 0.0066, resp_len 315, clip 0. Run B: server ping ok, Ray isolated, starting.
- 06:14 run B step 0 vpb_dev 0.345; step 1: soft-gate fired 8.1 % of rollouts, truncated 2.5 %, acc 0.388, match 0.24, format 0.775, entropy 2.34, KL 0.0005, resp_len 471, 410 s/step. Wall risk: 60 × 410 s + 7 vals ≈ 7.6 h vs 8 h wall — if it times out, the TERM trap exits 124 and resub 63148594 resumes from the step-30 checkpoint (adds queue time, no lost work).
- 06:15 run A step 30: vpb_dev 0.315 (0.33/0.34/0.335/0.315 at 0/10/20/30), train-val 0.771, entropy 0.84 (1.74 → 1.49 → 1.17 → 0.84 — trending toward the 0.4 floor), KL 0.0096, resp_len 368, clip 0, checkpoint global_step_30 saved (34 s).
- 06:38 run A step 40: vpb_dev 0.345 (best so far), train-val 0.659 (dip from 0.771 @30), entropy 0.75, KL 0.0155, resp_len 325.
- 07:01 run A step 50: vpb_dev 0.33, train-val 0.748, entropy 0.61, KL 0.013, resp_len 326. Run B step 10 rollouts: soft-gate fired 5.6 % (8.1 % @1), truncated 1.3 %, acc 0.50, match 0.253, format 0.794.
- 07:01 run B step 10: vpb_dev 0.305 (0.345 @0), entropy 1.50, KL 0.0022, resp_len 322.

### training job 63147499 ended Sun Sep 13 07:25:41 MST 2026 rc=0 (RESUB=1, timed_out=0, watchdog_stop=no, ckpts: 30 60 )
```
[monitor] 60 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_ctrl_v4/monitor.csv
step-0 val (source-avg): answer=0.669  format=0.750  gated=0.078  inv_frac=0.000  match=0.000  n_dup=0.187  n_segments=5.853  pun=0.000  score=3.926
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.269    418.8    0.013    0.188    0.113    7.688    0.000    0.000    0.381    0.644    0.000      -      0.000    0.000      -        -      172.1
      20    3.338    314.6    0.000    0.069    0.119    6.312    0.000    0.000    0.525    0.775    0.000      -      0.000    0.000    0.754    0.000    136.1
      40    4.081    325.0    0.000    0.006    0.181    6.025    0.000    0.000    0.631    0.925    0.000      -      0.000    0.000    0.650    0.000    136.9
      60    4.238    300.1    0.000    0.013    0.163    5.506    0.000    0.000    0.662    0.956    0.000      -      0.000    0.000    0.707    0.000    153.2
gate-2: over 60 steps, Pearson r(match/mean, answer/mean) = +nan
  first quarter: match 0.000 answer 0.491 r=+nan
  last quarter: match 0.000 answer 0.645 r=+nan
  within-step strata (all steps): mean match|answer=1 = 0.000, match|answer=0 = 0.000
  JUDGMENT (auto, rule-based): answer moved (+0.154) while match stayed flat (+0.000) — the match term is not what training optimised. Within steps, correct-answer rollouts score +0.000 higher on match than wrong-answer ones (match is essentially answer-blind at the sample level).
RED-FLAG none
```
- 07:26 run A (resub 63147499) COMPLETED 60/60 rc=0 (04:57 → 07:25, 2h28m ≈ 2.3 min/step incl. 7 vals), checkpoints 30 and 60, resub 63147500 cancelled.
  vpb_dev by step 0..60: 0.33 / 0.34 / 0.335 / 0.315 / 0.345 / 0.33 / **0.32**; train-val acc 0.62(init) → 0.729 / 0.766 / 0.771 / 0.659 / 0.748 / **0.718**;
  entropy at logged steps 1.49 / 1.17 / 0.84 / 0.75 / 0.61 / 0.65 (min 0.61 ≥ 0.4); KL 0.006 → 0.017; resp_len 300–370, clip 0 at every logged step.
  A HOLDS on the dev criteria (vpb_dev@60 0.32 ≥ 0.29, entropy ≥ 0.4 throughout) and is not stagnant (train-val +0.10). Evals 63147501 (step 60) /
  63147502 (step 30) released.

## 07:35 state re-established (new session; the previous one died mid-poll at ~07:26 — nothing in Slurm depended on it)
- Queue: A's evals 63147501 (step 60) / 63147502 (step 30) RUNNING since 07:25; B 63148593 RUNNING (1h44m, step 15/60, not near the 8 h wall;
  resubs 63148594/63148595 pending on afternotok); B evals 63148596/63148597 + report 63148598 pending on dependency; monitor 63147506
  RUNNING on grp_bshettah (12 h, started 04:35) — no resubmission needed. V4_REPORT.md not written yet (report job waits for all 4 evals).
- B live: step 10 vpb_dev 0.305 (0.345 @0), entropy 1.50 → 1.13 @15, KL 0.0022 → 0.0028, resp_len 322 → 334, clip 0.006; rollouts:
  soft-gate fired 8.1 % @1 / 16.9 % @5 / 5.6 % @10 / 6.9 % @15, truncated 2.5 % → 0.6 %, acc .39/.39/.50/.41, match .24/.24/.25/.25, format .78/.71/.79/.84.
- 3a. step-0 vpb_dev: A 0.33 (gated 0.06) vs B 0.345 (gated 0.0575). Same init (cs25), same greedy val (rollout val temperature 0 in both
  resolved configs), same gate; the 0.015 (6 of 400 questions) is the vLLM batch-order nondeterminism already measured in v3 (cs25 scored
  0.3175 vs 0.3025 in two jobs). Treat A-vs-B dev differences under ~0.02 as ties.
- 3b. Both runs compute vpb_dev with the same gate (arm_reward's reference-free path: acc = correct AND marker present AND not truncated;
  score-server path: acc = correct AND answer_valid). arm_reward.py on disk: sha256 c45ab153…, mtime 04:55:17; A's resubmit started 04:55:13
  but its reward worker imports the module minutes later (after Ray/model start) — proof it loaded the fixed file: A's step-0 val passed
  (the old file raised KeyError there) and A's rollouts/1.jsonl carries the `soft_gated` key. B started 05:48 > mtime. No edits since.
  Same file content for both.
- Probe gap (item 4): added `attack_answer_first` (answer line first, then the rollout's own steps as reference-shaped padding, finish=stop)
  to redteam_reward.py (backup .bak_v4); job 63160283 (htc gpu:1 non-MIG, grp_bshettah) runs soft then hard → probe_scored_v4atk_{soft,hard}.jsonl.

## Run A on vpb_test (2,456 q, greedy, 2048 tok; scored 07:40 on the login node with score_vpb's scorer; report job re-scores)
```
cs25_2ktest              acc=0.3192 Δcs25=+0.0000 Δbase=+0.0167 marker=0.947 length_rate=0.039 tokens=409 | DynaMath=0.457 MMMU_DEV_=0.398 MathVerse=0.236 MathVisio=0.208 WeMath=0.548
base3b_2ktest            acc=0.3025 Δcs25=-0.0167 Δbase=+0.0000 marker=0.927 length_rate=0.032 tokens=434 | DynaMath=0.449 MMMU_DEV_=0.353 MathVerse=0.227 MathVisio=0.206 WeMath=0.476
A_ctrl_v4_step30_2ktest  acc=0.3384 Δcs25=+0.0192 Δbase=+0.0359 marker=0.976 length_rate=0.023 tokens=411 | DynaMath=0.461 MMMU_DEV_=0.421 MathVerse=0.279 MathVisio=0.216 WeMath=0.536
A_ctrl_v4_step60_2ktest  acc=0.3282 Δcs25=+0.0090 Δbase=+0.0257 marker=0.980 length_rate=0.020 tokens=387 | DynaMath=0.461 MMMU_DEV_=0.434 MathVerse=0.247 MathVisio=0.240 WeMath=0.476
```
A holds on vpb_test as well: step 30 = 0.3384 (+0.019 vs cs25 0.3192, +0.036 vs base 0.3025), step 60 = 0.3282 (+0.009 / +0.026), marker rate
0.976/0.980, truncation 2.3 % / 2.0 % (cs25 3.9 %), mean length 411 / 387 tokens. Gains are on MMMU (+0.02–0.04), MathVerse (+0.04 @30), MathVision
(+0.03 @60); WeMath flat-to-down at 60 (0.476 vs 0.548). First run in the campaign that ends above its init.
- 07:54 run B step 20: vpb_dev 0.3375 (0.345 / 0.305 / 0.3375 at 0/10/20), train-val 0.690, entropy 1.20, KL 0.0067, resp_len 292, clip 0. A's evals 63147501/63147502 COMPLETED.
- 08:12 attack probe (job 63160283), SOFT mode with the new `attack_answer_first` row (answer line first, then the rollout's own steps, finish=stop, n=175):
  R_med 1.51 / R_mean 1.36, match 0.33 (= clean's 0.33 — the steps are the same text), format kept, answer 0 (not on the last line), gated 0/175.
  Ordering: gold 7.97 > clean 5.47 > giant 5.40 > repeat6 4.67 > answer_first 1.51 > hacked 0.00. Paired median R(clean) − R(answer_first) = 5.00
  (exactly the answer term); "attack ≥ clean" on 35.4 % of pairs = the clean rows whose own answer is wrong (they tie: same steps, no answer credit).
  Reading for arms 2–6: under soft gates a front-loaded answer + reference-shaped padding is worth the reasoning credit only (≈ 1.5 of 8) —
  it cannot beat an honest correct rollout, but it is not zero as under hard gates. Hard-mode pass running for the side-by-side row.
- 08:30 attack probe 63160283 DONE (rc=0). Hard vs soft, R medians (n: gold/clean 185, attackers 175):
```
population            hard R_med  soft R_med   match(soft)  format(soft)  gated hard/soft
gold                   7.97        7.97         0.99         0.92          0 / 0
clean                  5.47        5.47         0.33         0.81          12 / 6
attack_giant           5.40        5.40         0.30         0.00          0 / 0
attack_repeat6         4.67        4.67         0.10         0.07          0 / 0
attack_answer_first    0.00        1.51         0.33         0.74          175 / 0
degenerate (hacked)    0.00        0.00         0.00         0.00          185 / 185
```
  paired R(clean) − R(answer_first): hard 5.64 (attack ≥ clean 1.7 %), soft 5.00 (35.4 %, = the clean rows that are themselves wrong).
  Files: reward_redesign/probe_scored_v4atk_{hard,soft}.jsonl. Informational for arms 2–6; does not affect B.
- 08:43 run B step 30: vpb_dev 0.3375 (flat vs @20), train-val 0.683, entropy 0.61 (1.50 @10 → 1.20 @20 → 0.61 @30; A was 0.84 @30), KL 0.0125, resp_len 347, clip 0; checkpoint 30 saved. Rollouts @30: soft-gate 0.050, truncated 0.000, acc 0.525, match 0.293, format 0.850.
- 09:26 run B step 40: vpb_dev 0.325, train-val 0.659, entropy 0.47 (approaching the 0.4 flag), KL 0.0226, resp_len 316, clip 0.
- 10:11 run B step 50: vpb_dev 0.32, train-val 0.718, entropy 0.42 (0.02 above the 0.4 flag), KL 0.0212, resp_len 321, clip 0.

### training job 63148593 ended Sun Sep 13 11:01:39 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 30 60 )
```
[monitor] 60 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv4_softgate/monitor.csv
step-0 val (source-avg): answer=0.664  format=0.773  gated=0.038  inv_frac=0.263  match=0.299  n_dup=0.160  n_segments=5.613  pun=0.054  score=4.540
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.957    471.4    0.031    0.025    0.131    8.250    0.236    0.240    0.412    0.775    0.325    0.342    0.291    0.204      -        -      410.4
      20    3.646    291.9    0.000    0.000    0.069    5.394    0.046    0.268    0.506    0.812    0.283    0.429    0.323    0.212    0.681    0.317    251.6
      40    4.882    315.8    0.000    0.006    0.075    4.756    0.024    0.313    0.675    0.906    0.245    0.366    0.352    0.232    0.649    0.331    254.5
      60    4.978    295.2    0.000    0.000    0.087    4.406    0.004    0.313    0.681    0.950    0.247    0.329    0.345    0.245    0.739    0.347    284.4
gate-2: over 60 steps, Pearson r(match/mean, answer/mean) = +0.823
  first quarter: match 0.257 answer 0.485 r=+0.604
  last quarter: match 0.329 answer 0.659 r=+0.355
  within-step strata (all steps): mean match|answer=1 = 0.336, match|answer=0 = 0.232
  JUDGMENT (auto, rule-based): match (+0.072) and answer (+0.174) moved together; the across-step Pearson r = +0.823. They track each other closely — match adds little beyond answer over the run. Within steps, correct-answer rollouts score +0.104 higher on match than wrong-answer ones (match is answer-sensitive at the sample level).
RED-FLAG WARN n_segments collapsing: 5.346 (steps 21-40) -> 4.784 (steps 41-60)
```
- 11:02 run B 63148593 COMPLETED 60/60 rc=0 (05:48 → 11:01, 5h13m ≈ 5.2 min/step incl. 7 vals; never near the 8 h wall), checkpoints 30 and 60,
  resubs 63148594/63148595 cancelled. vpb_dev by step 0..60: 0.345 / 0.305 / 0.3375 / 0.3375 / 0.325 / 0.32 / **0.33**; train-val 0.62(init) →
  0.690 (20) / 0.683 / 0.659 / 0.718 / **0.751**; entropy 1.50 / 1.20 / 0.61 / 0.47 / 0.42 / 0.45 (min 0.42 ≥ 0.4, closer to the floor than A's 0.61);
  KL 0.0022 → 0.0276; resp_len 290–350, clip 0 at every logged step. Rollout stats @10: soft-gate 0.056 trunc 0.013 acc 0.500 match 0.253 fmt 0.794 | @20: soft-gate 0.044 trunc 0.000 acc 0.469 match 0.268 fmt 0.812 | @30: soft-gate 0.050 trunc 0.000 acc 0.525 match 0.293 fmt 0.850 | @40: soft-gate 0.019 trunc 0.006 acc 0.675 match 0.313 fmt 0.906 | @50: soft-gate 0.013 trunc 0.000 acc 0.588 match 0.339 fmt 0.963 | @60: soft-gate 0.019 trunc 0.000 acc 0.681 match 0.313 fmt 0.950.
  The soft gate fired on 16.9 % of rollouts at step 5, 4–6 % at steps 10–30 and 1.3–1.9 % at steps 40–60 — the policy learns the answer format without the hard gate; format term 0.79 → 0.95.
  Evals 63148596 (step 60) / 63148597 (step 30) RUNNING on sg037 / sg029; report 63148598 follows.

### V4_REPORT.md written by job 63148598 at Sun Sep 13 11:18:36 MST 2026 (rc=0)

## DONE (11:30)
- A: 0.3384 @30 / 0.3282 @60; B: 0.3176 @30 / 0.3436 @60 (vpb_test, 2048 tok; cs25 0.3192, base 0.3025). Verdict branch 1: optimizer fixed and
  match + soft gates ≥ control (B@60 vs A@60 McNemar p = 0.089 — treat as "at least as good"; B@60 vs cs25 p = 0.011). Full write-up: V4_REPORT.md
  (§7 has the caveats: the §4 "stagnant" line is an ungated-vs-gated metric artefact; B's entropy is near the 0.4 floor at step 50–60).
- Fairshare AFTER (myfairshare): grp_bshettah / sghos104 RawUsage_CHE 2580.2, RealFairShare 0.293071 (before: 1262.7 / 0.492369; Δ +1317.5 CHE).
- Monitor 63147506 cancelled at 11:25 (nothing left to monitor). Nothing queued. grp_vgupt140 untouched throughout.
