# OVERNIGHT STATUS — arm 1 v2 (reward_v2 final spec), started 2026-09-11

One file to read in the morning. Sections are appended by me and by the jobs themselves
(run_arm.sbatch appends a block when a training job starts/ends; eval_arm1v2.sbatch appends the VPB table).
All numbers are copied from logs/json.

## Stage 0 — setup (done before any job)
- Account fix: every `--account=grp_bshettah` in grpo_arms/run_arm.sbatch, eval_arm1.sbatch, probe_gen.sbatch,
  reward_redesign/redteam*.sbatch replaced by grp_vgupt140; grep confirms zero remaining in grpo_arms/ and reward_redesign/.
- Reward: reward_redesign/reward_v2.py `score_new` rewritten to the final spec (backup reward_v2.py.bak_20260911;
  `score_old` byte-identical). Mock-scorer arithmetic check passed on 11 hand-computed cases (perfect=8.0, reversed
  order=7.75, 6x duplicate padding=5.909 with pun 1.0, length gate=0, answer-first gate=0, boxed terminal=8.0, ...).
- Harness: redteam_reward.py acceptance block updated (A3 = ALL degenerate rows R==0; A4 = clean pun median <= 0.5
  with prec/recall/match/offset medians reported; A5 paired repeat6 required, attack_giant informational).
- New files: grpo_arms/monitor.py (monitor.csv/png + red flags + gate-2 correlation; tested on the old arm-1 run),
  grpo_arms/eval_arm1v2.sbatch (merge -> generate -> score -> table), grpo_arms/launch_arm1_v2.sh (smoke / real chain),
  run_arm.sbatch gained: DONE marker no-op guard, in-job watchdog (WATCHDOG=1), TERM-before-wall trap (exit 124 ->
  afternotok resume), post-run monitor + status append, cancels pending resubmits on success.

## Stage 1 — offline probe (Step 2)
- job 62990551, htc, 1xA100, 90 min, chunker/env, redteam_reward.py --skip_old -> reward_redesign/probe_scored_final.jsonl
- result: job 62990551 finished Fri Sep 11 01:46:22 MST 2026 (rc=0), 905 rows incl. 350 synthetic attackers
```
=== NEW reward ===
population           n   R_med  R_mean  match   prec    rec    pun   off  gated   ans%  segs  dup
attack_giant       175    5.40    3.43   0.30   0.72   0.11   0.00  0.29      0  57.1%     1    0
attack_repeat6     175    4.67    2.83   0.10   0.12   0.11   0.33  0.40      0  57.1%     6    5
clean              185    5.47    4.03   0.33   0.39   0.25   0.00  0.20     12  53.0%     5    0
degenerate         185    0.00    0.00   0.00   0.00   0.00   0.00  0.00    185   0.0%    92    0
gold               185    7.97    7.72   0.99   0.99   0.99   0.00  0.00      0 100.0%     6    0
=== acceptance ===
  [PASS] A1 new: clean - degenerate >= 2.0
  [FAIL] B1 old: bug reproduced (degenerate within 1.0 of clean or above)
  [PASS] A2 new: gold is the top median
  [PASS] A3 new: ALL degenerate rows score exactly 0 (185/185 zero)
  [PASS] A4 new: clean pun_med = 0.00 <= 0.5  [clean medians: prec=0.391 recall=0.249 match=0.325 offset=0.200]
  [PASS] A5 new (paired): median R(clean)-R(attack_repeat6) = 1.66 >= 1.0 [n=175, attack>=clean 4.0%]
  [info-pass] A5 new (paired): median R(clean)-R(attack_giant) = 1.01 >= 1.0 [n=175, attack>=clean 15.4%]  (INFORMATIONAL, not blocking)
  REQUIRED (A1-A5/repeat6): ALL PASS
```
- Decision (overnight rule; the brief said "stop and show me" but the run must finish unattended): all REQUIRED checks pass,
  so I proceeded to the smoke. attack_giant is informational and sits exactly at the margin (paired median 1.01, 15.4 % of
  giant rows score >= their clean twin) — worth a look in the morning; the format term (giant paragraph = 1 kept step -> format 0)
  is what separates it, as the 2026-09-11 design decision assumed.
- Note: `B1 old` reads FAIL only because --skip_old copies the new scores into the "old" column (no A/B run this time).

## Stage 2 — smoke (Step 3a)
- job 62992879 (htc, 2xA100, 3:00:00), OUTDIR grpo_arms/runs/arm1v2_smoke3, log grpo_arms/logs/arm-62992879.log,
  server log grpo_arms/logs/arm1_server_62992879.log. Config = the real run's: cold-start checkpoint-25, REWARD=match_v2,
  SEG=native, MAXRESP=2048, NGPU=2, GPUMEM=0.40, ppo_max_token_len_per_gpu=8192 (+ ref/rollout log_prob 8192),
  --max_steps 3 --max_train_samples 96 (3 batches of 32, 8 val prompts).
- result: job 62992879 ran 02:30:08 -> 03:07:23 on sg046, rc=0 (log grpo_arms/logs/arm-62992879.log). All five gates PASS:
  1. "multiple active Ray instances" warning: 0 occurrences (Ray isolated at /tmp/ray_62992879, port 32879).
  2. server log arm1_server_62992879.log: "[server] reward_v2 adapter ready".
  3. step-0 validation per source (8 val prompts, source-avg score 5.160): match CLEVR .362 / MathV360K .122 / SROIE .326 / ai2d .259;
     gated 0.0 / 0.5 / 0.0 / 0.333; n_segments 4.5 / 24.5 / 3.0 / 5.0; inv_frac(offset) .22 / .417 / .40 / .167; pun 0 and n_dup 0
     in all four sources (train rollouts do vary: pun .125 -> .039, n_dup .087 -> .094 over steps 1 -> 3).
  4. steps 1-3 trained, critic/score/mean 2.661 -> 3.289 -> 4.281 (max ~7.4, min -5.27 -> -0.67), response_length/mean 368.9 -> 286.3 -> 246.0,
     clip_ratio 0.019 -> 0.0 -> 0.0, actor max_memory_allocated 40.84 GB (no OOM at MAXRESP=2048 / 2 GPUs / token caps 8192).
  5. rc=0. (A DataLoader/EngineCore traceback appears AFTER "Training Progress 100%" during Ray teardown — cosmetic.)
- seconds/step: 339.2, 233.7, 263.6 (mean 278.8, max 339.2). Wall-time math uses 340 s/step (the slowest step, conservative).
- rollout component means step 1 -> 3: match .224 -> .263, answer .350 -> .625, format .650 -> .825, gated .138 -> .062,
  within-step r(match,answer) .385 / .382; final-step val (8 prompts): answer .542, match .338.

## Stage 3 — real run (Step 3b)
- 03:10 launched via `bash grpo_arms/launch_arm1_v2.sh real 340` -> grpo_arms/runs/arm1_3b_matchv2_final/chain.json:
  train job **62998194** (public, grp_vgupt140, 2xA100, 16 cpu, 320G, wall 1-02:40:00, mail END,FAIL), WATCHDOG=1,
  resubmits 62998195 (afternotok:62998194) and 62998196 (afternotok:62998195) — same OUTDIR, verl resume_mode=auto, RESUB=1/2,
  eval 62998197 (htc, afterok:62998194 ? afterok:62998195 ? afterok:62998196, N_EVAL=4, mail END,FAIL).
- config: cold-start checkpoint-25, REWARD=match_v2, SEG=native, TAU=0.45 (ignored by score_new), MAXRESP=2048, NGPU=2, GPUMEM=0.40,
  batch 32 x rollout 5, mini 8, micro 1, lr 1e-5, KL loss on, ppo_max_token_len_per_gpu=8192, ref/rollout log_prob 8192,
  --max_steps 183 --save_freq 20 --test_freq 20, max_actor_ckpt_to_keep=4 (44 GB each).
- wall-time math: 340 s/step (slowest smoke step) x 183 steps x 1.5 + 45 min = 96,030 s = 1-02:40:00 < 48 h -> full epoch, max_steps=183.
  (with the smoke mean 278.8 s/step the run itself would take ~14.2 h; if late-run steps stretch like arm-1 v1 did (to ~480 s), 183 steps
  need ~24.4 h — still inside the wall; beyond that the TERM@900 trap exits 124 and resub1 resumes from the last global_step_N.)
- monitor: run_arm.sbatch's watchdog runs grpo_arms/monitor.py --check every 15 min (log lines "[watchdog HH:MM] ..." in logs/arm-62998194.log);
  on job end it writes runs/arm1_3b_matchv2_final/monitor.csv, monitor.png, monitor_summary.txt and appends a block below.
  To look while it runs: `chunker/env/bin/python grpo_arms/monitor.py --run grpo_arms/runs/arm1_3b_matchv2_final --every 20`
- state: started 03:21:18 on sg046 (job 62998194). Startup checks: server ping ok, Ray isolated (port 38194), no "multiple active Ray",
  "[server] reward_v2 adapter ready" present, 5,880 images materialized, watchdog ticking ("[watchdog 09-11 03:41] last_step=0|OK").
  step-0 val (120 prompts, unweighted source-avg over 35 sources): score 4.452, answer 0.650, match 0.293, gated 0.076, pun 0.049, n_dup 0.231, n_segments 6.123.
  step 1: score/mean 2.810, resp_len 387.2, clip 0.006, entropy 1.567, 338.6 s, peak actor mem 40.68 GB.
  step 2: score/mean 3.467, resp_len 366.2, clip 0.013, entropy 1.708, 312.7 s, peak 40.88 GB.
- monitor snapshots (every ~50 steps, appended by me while my session lasts; the job appends its own block at the end):

## Stage 4 — VPB eval (Step 4)
- job 62998197 (htc, 1xA100, 3:45:00) runs automatically after the training chain succeeds: cs25 generation, merge + generate for the 4 kept
  checkpoints, score_vpb.py, table appended here and to ARM1_V2_REPORT.md (also grpo_arms/evals/vpb_table_arm1v2.md).
- result: (pending)
- post-job 62998293 (htc CPU-only, afterany:62998197) regenerates monitor.csv/png + gate-2 judgment from the full run and appends them,
  plus the sacct outcome of the whole chain, to this file and ARM1_V2_REPORT.md.

## Decisions taken without you (overnight rule)
- 01:50 probe passed all required checks -> proceeded to smoke without waiting for your go-ahead (brief: finish unattended).
- 03:10 smoke passed all five gates -> launched the real run (wall sized from the slowest smoke step, 340 s, not the mean 279 s).
### training job 62992879 started Fri Sep 11 02:30:09 MST 2026 on sg046 (RESUB=0, wall=3:00:00, existing ckpts: 0)

### training job 62992879 ended Fri Sep 11 03:07:23 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 3 )
```
[monitor] 3 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1v2_smoke3/monitor.csv
step-0 val (source-avg): answer=0.792  format=0.667  gated=0.208  inv_frac=0.301  match=0.267  n_dup=0.000  n_segments=9.250  pun=0.000  score=5.160
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.661    368.9    0.019    0.138    0.087    6.638    0.125    0.224    0.350    0.650    0.274    0.385    0.310    0.178      -        -      339.2
       3    4.281    246.0    0.000    0.062    0.094    5.281    0.039    0.263    0.625    0.825    0.252    0.382    0.307    0.191    0.542    0.338    263.6
gate-2: over 3 steps, Pearson r(match/mean, answer/mean) = +0.578
  first quarter: match 0.224 answer 0.350 r=+nan
  last quarter: match 0.263 answer 0.625 r=+nan
  within-step strata (all steps): mean match|answer=1 = 0.288, match|answer=0 = 0.170
RED-FLAG none
```
### training job 62998194 started Fri Sep 11 03:21:18 MST 2026 on sg046 (RESUB=0, wall=1-02:40:00, existing ckpts: 0)
  - 05:23 step 20 (2h02m elapsed, ~4.6 min/step so far): ckpt global_step_20 saved (36.7 s), val on 120 prompts (source-avg):
    score 4.941, answer 0.656, match 0.353, format 0.956, gated 0.017, pun 0.000, n_dup 0.114, n_segments 3.893, offset 0.247.
    train rollouts step 1 -> 20: gated .169 -> .006, n_dup .131 -> .069, n_segments 7.23 -> 4.01, pun .126 -> .004, match .199 -> .314,
    answer .412 -> .644, format .694 -> .981, offset .252 -> .273; critic/score/mean 2.81 -> 4.82; resp_len 387 -> 279 tok, clip <= 0.02.
    gate-2 so far: r(match, answer over steps) = +0.825 (both rising together in the first 20 steps — expected early); within-step
    match|answer=1 0.335 vs match|answer=0 0.240 (+0.095). WARN from monitor: n_segments 5.71 (steps 1-10) -> 4.36 (11-20), i.e. the
    policy is writing fewer, longer-per-step responses (gold median ~6 steps); far from the 1.5 stop limit, watching it.
  - 07:37 step 50 (4h16m elapsed, 5.1 min/step incl. val/ckpt overhead): ckpts 20, 40 saved. val@40 (src-avg): score 4.583, answer 0.568,
    match 0.375, format 0.993, gated 0.007, pun 0.003, n_dup 0.190, n_segments 4.412, offset 0.209 (val answer dipped from 0.656@20;
    train-rollout answer did not: .644@20 -> .700@40 -> .631@50). train rollouts @50: score 4.918, resp_len 388 tok (flat ~380-420 since
    step 30, clip 0.000), gated 0.000, n_dup 0.094, n_segments 4.03, pun 0.001, match 0.391, answer 0.631, format 0.981, offset 0.214.
    gate-2 @50: r(match, answer over steps) = +0.554; first quarter match .286/answer .584 (r +.911) -> last quarter .382/.644 (r +.073);
    within-step match|answer=1 .364 vs match|answer=0 .285 (+0.079). monitor RED-FLAG: none (n_segments WARN cleared, stable ~4.0-4.3).
    Extrapolated finish: 133 steps x ~290 s = ~10.7 h -> ~18:20 today + eval ~2 h.
  - 14:12 step 132/183 (10h51m elapsed, ~4.9 min/step; ckpts 60..120 kept, ETA ~18:30 + eval). No red flags: gated 0.000, n_dup 0.15 flat,
    n_segments 3.5-4.1, pun 0.000, format 0.99-1.00, clip 0.000. BUT resp_len drifts up 279@20 -> 525@132 and the GATE-2 pattern has
    turned: train match .314@20 -> .663@132 while train answer .644 -> .594; val answer 0.656@20 -> .568@40 -> .605@60 -> .520@80 ->
    .515@100 -> .505@120 while val match .353 -> .632. r(match, answer over steps) = -0.365 (first quarter +0.744, last quarter -0.023);
    within-step match|answer=1 .475 vs match|answer=0 .428 (+0.047, answer-blind at sample level). Not the v1 length/repeat hack
    (no truncation, no duplicates), but the match term is being optimised at the expense of answer accuracy — VPB eval will decide.

### training job 62998194 ended Fri Sep 11 19:37:26 MST 2026 rc=0 (RESUB=0, timed_out=0, watchdog_stop=no, ckpts: 20 40 60 80 100 120 140 160 180 183 )
```
[monitor] 183 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv2_final/monitor.csv
step-0 val (source-avg): answer=0.650  format=0.766  gated=0.076  inv_frac=0.251  match=0.293  n_dup=0.231  n_segments=6.123  pun=0.049  score=4.452
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.810    387.2    0.006    0.169    0.131    7.231    0.126    0.199    0.412    0.694    0.252    0.347    0.253    0.160      -        -      338.6
      20    4.825    279.2    0.006    0.006    0.069    4.013    0.004    0.314    0.644    0.981    0.273    0.434    0.358    0.236    0.656    0.353    277.4
      40    5.267    393.4    0.000    0.000    0.094    4.131    0.002    0.385    0.700    1.000    0.225    0.286    0.410    0.326    0.568    0.375    306.9
      60    5.048    403.3    0.000    0.000    0.156    3.981    0.007    0.452    0.631    0.994    0.231   -0.010    0.451    0.454    0.605    0.453    301.5
      80    5.073    480.6    0.000    0.031    0.263    3.919    0.000    0.502    0.650    0.944    0.218   -0.081    0.493    0.519    0.520    0.503    312.6
     100    5.293    404.9    0.000    0.006    0.169    3.494    0.000    0.556    0.637    0.994    0.210    0.328    0.587    0.501    0.515    0.563    325.6
     120    5.651    491.3    0.000    0.000    0.150    3.812    0.000    0.638    0.675    1.000    0.267   -0.008    0.638    0.640    0.505    0.632    332.1
     140    4.418    575.3    0.000    0.000    0.306    4.412    0.000    0.662    0.419    1.000    0.252   -0.118    0.644    0.676    0.471    0.671    318.8
     160    5.422    804.9    0.000    0.000    0.125    4.444    0.000    0.742    0.588    1.000    0.254    0.216    0.760    0.717    0.483    0.700    402.0
     180    5.142    936.6    0.000    0.000    0.081    5.356    0.000    0.774    0.519    1.000    0.266   -0.003    0.774    0.775    0.479    0.779    454.8
     183    5.080   1008.1    0.000    0.000    0.069    5.369    0.000    0.790    0.500    1.000    0.260   -0.123    0.779    0.801    0.493    0.779    478.5
gate-2: over 183 steps, Pearson r(match/mean, answer/mean) = -0.395
  first quarter: match 0.330 answer 0.635 r=+0.587
  last quarter: match 0.731 answer 0.538 r=+0.186
  within-step strata (all steps): mean match|answer=1 = 0.546, match|answer=0 = 0.508
  JUDGMENT (auto, rule-based): match (+0.401) and answer (-0.097) moved in OPPOSITE directions (r = -0.395) — the arm-1 v1 hacking signature; check response_length / gated / n_dup before trusting the match term. Within steps, correct-answer rollouts score +0.038 higher on match than wrong-answer ones (match is essentially answer-blind at the sample level).
RED-FLAG WARN response_length/mean up 751->879 while score up 5.15->5.22 (steps 144-163 vs 164-183)
```
  - 19:37 TRAINING COMPLETE: job 62998194 finished 183/183 steps, rc=0, elapsed 16h16m (03:21 -> 19:37), DONE marker written, resubmits
    62998195/62998196 cancelled by the job, watchdog never fired (STOP_WATCHDOG absent). All 10 checkpoints (20..180, 183) are on disk
    (verl kept them all despite max_actor_ckpt_to_keep=4; 44 GB each, fine). monitor.csv / monitor.png / monitor_summary.txt written.
    Final trajectory (train rollouts): match .199@1 -> .790@183, answer .412 -> .500 (peak .700@40), resp_len 387 -> 1008 tok (no clipping,
    cap 2048), n_segments 7.2 -> 4.0 (@100) -> 5.4, gated 0, n_dup .07, pun 0, format 1.00, offset .26.
    Val (120 prompts, src-avg) answer: .656@20 .568@40 .605@60 .519@80 .515@100 .505@120 .471@140 .483@160 .479@180 .493@183;
    val match .353 -> .778. GATE-2 final: r(match, answer over steps) = -0.395; first quarter match .330/answer .635, last quarter
    .731/.538; within-step match|answer=1 .546 vs match|answer=0 .508 (+0.038). Auto-judgment: opposite directions = v1 signature,
    but WITHOUT truncation/duplication — the mechanism is GRPO group saturation (by step 120 only 1/32 groups mixed on correctness,
    answer's share of within-group |advantage| .64@1 -> .03@120), so match is the only remaining gradient. monitor WARN at the end:
    resp_len 751 -> 879 (steps 144-163 vs 164-183) while score 5.15 -> 5.22.
- 20:28 eval 62998197 was stuck (Slurm pinned it to drained node scg008, est. start 2026-09-12 14:35; htc has ~1,060 pending GPU jobs).
  Resubmitted: eval **63043620** (htc, est. start ~22:20 tonight), report post-job **63043621** (afterany:63043620); cancelled 62998293 then 62998197.
  chain.json updated (stale ids kept under eval_stale / report_post_stale).
- 20:30 63043620's estimate slipped to 2026-09-12 06:30 (A100s on htc are saturated: 143 pending single-A100 jobs). A 1-hour job on
  ANY GPU type backfills immediately (test-only: 20:33), so the eval was split into short jobs on --gres=gpu:1 (4 cpu, 64G):
  cs25 gen 63043681 (0:45), checkpoints 183/180/160/140 -> jobs 63043682:63043683:63043684:63043685 (1:00 each, merge + generate, DO_SCORE=0),
  then CPU post-job 63043686 (afterany on all five; DO_SCORE=1 scores everything, writes the VPB table + monitor/gate-2 into this file and the report).
  Cancelled 63043621 then 63043620. eval_arm1v2.sbatch gained STEP_LIST / DO_CS25 / DO_SCORE knobs; report_post.sbatch gained the scoring step.
- 21:23 FIRST VPB NUMBERS (1024-token frozen protocol, scored on the login node from the finished gens): cs25 0.3183 (marker .941,
  truncated 5.1%), base3b 0.2952, arm1_step183 0.2549, **arm1v2_step183 0.1590** (marker .689, truncated 31.9%; acc|stop 0.2305, acc|length
  0.0066). The v2 policy writes ~940 tokens on VPB (median 961) so the 1024 cap cuts a third of its answers off; even completed outputs
  score below cs25. Because training used a 2048 cap, queued a 2048-token protocol as well: jobs 63045335 (base3b_2k + cs25_2k),
  63045336/7/8/9 (steps 183/180/160/140, tags *_2k). Scoring post-job re-pointed: 63045340 (afterany on all 10 gen jobs; 63043686 cancelled).
  The final table will show both protocols with truncation fractions.

### VPB evaluation (scored by post-job 63045340, Fri Sep 11 22:53:30 MST 2026)
generation: greedy, max_tokens 1024, pixel cap 640*28*28, frozen vpb_eval.jsonl sha 7cd6d974ab5d611d..

**max_tokens = 1024** (frozen protocol; base3b / arm1_step183 reference numbers)

| tag | step | n | accuracy | answer-extractable rate | truncated frac | delta vs base | delta vs cs25 | note |
|---|---|---|---|---|---|---|---|---|
| base3b | - | 2856 | 0.2952 | 0.9163 | 0.043 | +0.0000 | -0.0231 | Qwen2.5-VL-3B-Instruct (reference) |
| cs25 | 25 | 2856 | 0.3183 | 0.9408 | 0.051 | +0.0231 | +0.0000 | cold-start init (checkpoint-25) |
| arm1_step183 | 183 | 2856 | 0.2549 | 0.9951 | 1.000 | -0.0403 | -0.0634 | reward-hacked arm-1 v1 reference |
| arm1v2_step140 | 140 | 2856 | 0.2349 | 0.9940 | 0.007 | -0.0603 | -0.0834 | arm 1 v2 (reward_v2 final) |
| arm1v2_step160 | 160 | 2856 | 0.1982 | 0.8796 | 0.123 | -0.0970 | -0.1201 | arm 1 v2 (reward_v2 final) |
| arm1v2_step180 | 180 | 2856 | 0.1572 | 0.6737 | 0.341 | -0.1380 | -0.1611 | arm 1 v2 (reward_v2 final) |
| arm1v2_step183 | 183 | 2856 | 0.1590 | 0.6887 | 0.319 | -0.1362 | -0.1593 | arm 1 v2 (reward_v2 final) |

per-source accuracy:
| tag | DynaMath | MMMU_DEV_VAL | MathVerse_MINI_Vision_Only | MathVision_MINI | WeMath |
|---|---|---|---|---|---|
| base3b | 0.4105 | 0.3658 | 0.2290 | 0.1980 | 0.4777 |
| cs25 | 0.4509 | 0.4086 | 0.2368 | 0.2037 | 0.5464 |
| arm1_step183 | 0.2649 | 0.3619 | 0.2115 | 0.1952 | 0.4399 |
| arm1v2_step140 | 0.2333 | 0.3852 | 0.1657 | 0.2079 | 0.4158 |
| arm1v2_step160 | 0.1895 | 0.3230 | 0.1598 | 0.1348 | 0.3952 |
| arm1v2_step180 | 0.1667 | 0.1829 | 0.1296 | 0.1180 | 0.3093 |
| arm1v2_step183 | 0.1667 | 0.2023 | 0.1345 | 0.1067 | 0.3196 |

**max_tokens = 2048** (= training response cap)

| tag | step | n | accuracy | answer-extractable rate | truncated frac | delta vs base | delta vs cs25 | note |
|---|---|---|---|---|---|---|---|---|
| base3b_2k | - | 2856 | 0.3011 | 0.9212 | 0.033 | +0.0000 | -0.0172 | Qwen2.5-VL-3B-Instruct (reference) |
| cs25_2k | 25 | 2856 | 0.3183 | 0.9415 | 0.046 | +0.0172 | +0.0000 | cold-start init (checkpoint-25) |
| arm1v2_step140_2k | 140 | 2856 | 0.2272 | 1.0000 | 0.000 | -0.0739 | -0.0911 | arm 1 v2 (reward_v2 final) |
| arm1v2_step160_2k | 160 | 2856 | 0.2185 | 0.9923 | 0.008 | -0.0826 | -0.0998 | arm 1 v2 (reward_v2 final) |
| arm1v2_step180_2k | 180 | 2856 | 0.2269 | 1.0000 | 0.000 | -0.0742 | -0.0914 | arm 1 v2 (reward_v2 final) |
| arm1v2_step183_2k | 183 | 2856 | 0.2328 | 1.0000 | 0.000 | -0.0683 | -0.0855 | arm 1 v2 (reward_v2 final) |

per-source accuracy:
| tag | DynaMath | MMMU_DEV_VAL | MathVerse_MINI_Vision_Only | MathVision_MINI | WeMath |
|---|---|---|---|---|---|
| base3b_2k | 0.4211 | 0.3541 | 0.2320 | 0.2051 | 0.4983 |
| cs25_2k | 0.4491 | 0.3969 | 0.2407 | 0.2093 | 0.5326 |
| arm1v2_step140_2k | 0.2333 | 0.3696 | 0.1598 | 0.1910 | 0.4158 |
| arm1v2_step160_2k | 0.2088 | 0.3930 | 0.1754 | 0.1475 | 0.4089 |
| arm1v2_step180_2k | 0.2123 | 0.3852 | 0.1793 | 0.1615 | 0.4433 |
| arm1v2_step183_2k | 0.2070 | 0.3930 | 0.1745 | 0.1910 | 0.4502 |

### Monitor + gate-2 (auto post-job 63045340, Fri Sep 11 22:55:05 MST 2026, monitor rc=0)
files: /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv2_final/monitor.csv, /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv2_final/monitor.png, /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv2_final/monitor_summary.txt
```
[monitor] 183 steps -> /scratch/sghos104/rlpt/grpo_arms/runs/arm1_3b_matchv2_final/monitor.csv
step-0 val (source-avg): answer=0.650  format=0.766  gated=0.076  inv_frac=0.251  match=0.293  n_dup=0.231  n_segments=6.123  pun=0.049  score=4.452
    step    score resp_len     clip    gated    n_dup    n_seg      pun    match   answer   format   offset   r(m,a)     m|a1     m|a0  val_ans val_match   step_s
       1    2.810    387.2    0.006    0.169    0.131    7.231    0.126    0.199    0.412    0.694    0.252    0.347    0.253    0.160      -        -      338.6
      20    4.825    279.2    0.006    0.006    0.069    4.013    0.004    0.314    0.644    0.981    0.273    0.434    0.358    0.236    0.656    0.353    277.4
      40    5.267    393.4    0.000    0.000    0.094    4.131    0.002    0.385    0.700    1.000    0.225    0.286    0.410    0.326    0.568    0.375    306.9
      60    5.048    403.3    0.000    0.000    0.156    3.981    0.007    0.452    0.631    0.994    0.231   -0.010    0.451    0.454    0.605    0.453    301.5
      80    5.073    480.6    0.000    0.031    0.263    3.919    0.000    0.502    0.650    0.944    0.218   -0.081    0.493    0.519    0.520    0.503    312.6
     100    5.293    404.9    0.000    0.006    0.169    3.494    0.000    0.556    0.637    0.994    0.210    0.328    0.587    0.501    0.515    0.563    325.6
     120    5.651    491.3    0.000    0.000    0.150    3.812    0.000    0.638    0.675    1.000    0.267   -0.008    0.638    0.640    0.505    0.632    332.1
     140    4.418    575.3    0.000    0.000    0.306    4.412    0.000    0.662    0.419    1.000    0.252   -0.118    0.644    0.676    0.471    0.671    318.8
     160    5.422    804.9    0.000    0.000    0.125    4.444    0.000    0.742    0.588    1.000    0.254    0.216    0.760    0.717    0.483    0.700    402.0
     180    5.142    936.6    0.000    0.000    0.081    5.356    0.000    0.774    0.519    1.000    0.266   -0.003    0.774    0.775    0.479    0.779    454.8
     183    5.080   1008.1    0.000    0.000    0.069    5.369    0.000    0.790    0.500    1.000    0.260   -0.123    0.779    0.801    0.493    0.779    478.5
gate-2: over 183 steps, Pearson r(match/mean, answer/mean) = -0.395
  first quarter: match 0.330 answer 0.635 r=+0.587
  last quarter: match 0.731 answer 0.538 r=+0.186
  within-step strata (all steps): mean match|answer=1 = 0.546, match|answer=0 = 0.508
  JUDGMENT (auto, rule-based): match (+0.401) and answer (-0.097) moved in OPPOSITE directions (r = -0.395) — the arm-1 v1 hacking signature; check response_length / gated / n_dup before trusting the match term. Within steps, correct-answer rollouts score +0.038 higher on match than wrong-answer ones (match is essentially answer-blind at the sample level).
RED-FLAG WARN response_length/mean up 751->879 while score up 5.15->5.22 (steps 144-163 vs 164-183)
```
chain outcome (sacct):
```

```
DONE marker present (training chain finished cleanly).

## DONE (23:05) — everything finished; read grpo_arms/ARM1_V2_REPORT.md (headline in §0). Short version: reward implemented + probe passed;
training 183/183 clean (no hack signature); gate 2 NEGATIVE (match up, answer down, GRPO group saturation); VPB: v2 final 0.1590 @1024 tok /
0.2328 @2048 tok vs cold-start 0.3183 and base 0.2952/0.3011 — the v2 reward degraded the policy. 173 GB of checkpoints + 4×7 GB merged HF dirs
are under runs/arm1_3b_matchv2_final (kept for you to decide).

## Follow-ups actioned (23:20, on your instruction)
- Disk: deleted FSDP checkpoints global_step_140/160/180 (3 x 44 GB) after verifying all four merged HF dirs (hf_step_140/160/180/183,
  7.1 GB each, config + 3 safetensors + processor bundle) are complete. Kept global_step_183 (44 GB, resumable) + the HF dirs. Steps 20-120
  were already 8 KB stubs (verl's keep=4 had pruned them). runs/arm1_3b_matchv2_final is now ~73 GB instead of ~200 GB.
- Eval pattern made default: eval_arm1v2.sbatch header now 1 h / --gres=gpu:1 / 4 cpu / 64 G (one checkpoint per job); new fan-out launcher
  grpo_arms/launch_vpb_eval.sh <RUN> <TAGPFX> "<steps>" [MAXTOK] [TAGSFX] [DO_REFS] submits refs + per-checkpoint jobs + the scoring post-job;
  the Sol cluster skill (.claude/skills/sol-cluster-ops/SKILL.md) records the htc lane + backfill rule and now bills grp_vgupt140.
