# ARM 1 v2 report — reward_v2 FINAL spec, cold-start init, VisualPRM train subset

## 0. Headline (2026-09-11, all stages complete)
- **Reward**: `score_new` implemented to the final spec; offline probe passed every required check (A1–A4, A5-repeat6; A5-giant informational 1.01).
- **Training**: job 62998194 ran 183/183 steps in 16h16m on 2×A100, rc=0. No reward hacking of the v1 kind: 0 % truncation, gated 0.000,
  duplicates 0.07/rollout, format 1.00, step count 4–5. Response length grew 387 → 1008 tokens (cap 2048).
- **Gate 2 — NEGATIVE**: match rose 0.20 → 0.79 while train answer went 0.41 → 0.50 (peak 0.70 @ step 40) and val answer fell 0.656 → 0.493;
  r(match, answer) over steps = −0.395; within-step match|answer=1 − match|answer=0 = +0.038 (answer-blind). Mechanism: GRPO group
  saturation — by step 120 only 1 of 32 prompt groups still disagrees on correctness, so the 5-point answer term contributes ~3 % of the
  within-group advantage and match is the only gradient left (table in §4).
- **VPB (2,856 q)**: cold-start init 0.3183 > base 0.2952 > v1-hacked 0.2549 > **v2 final 0.1590** at the frozen 1024-token protocol
  (32 % of v2 answers truncated). At the training cap (2048 tokens, no truncation) v2 = 0.2328 vs cold-start 0.3183 (−0.086) and base
  0.3011; accuracy declines monotonically with training (0.2349 @140 → 0.1590 @183 at 1024 tok). **The v2 reward did not improve the policy;
  it degraded it by 7–9 points on VPB relative to its own init.**

Started 2026-09-11 (unattended overnight run). Every number below is copied from a log or json file named next to it.
Live progress: `grpo_arms/OVERNIGHT_STATUS.md`.

## 1. Final reward spec as implemented (`reward_redesign/reward_v2.py::score_new`)

| term | implementation |
|---|---|
| gates | no parsable answer (`extract_final`: last "Final answer:"/"Answer:" line, else last `\boxed{}`) OR `finish_reason == "length"` OR the answer is not on the last non-empty line (`answer_at_end`) → R = 0, `gated = True`; the rollout stays in the GRPO group at 0 |
| steps | `<think>`/`</think>` stripped; split on blank lines; answer lines removed from matching — FINAL_RE ("Final answer:", "Answer:") anywhere, plus a terminal `\boxed{}` line (`is_answer_line`) |
| dedupe | SBERT all-MiniLM-L6-v2 cosine ≥ 0.90 to any EARLIER kept step → duplicate; duplicates are never matched but still count in `n_steps` |
| NLI | `s(i,j) = clip01(0.5·E(i→j) + 0.5·E(j→i) − max(C))`, microsoft/deberta-xlarge-mnli, kept steps × GT steps; steps over 512 NLI tokens get s = 0 |
| matching | Hungarian one-to-one (`linear_sum_assignment(-s)`); NO tau gate — every assigned pair counts with credit `s(i,j)`; an assignment with `s == 0` exactly carries no credit and is dropped from the pair set (otherwise arbitrary zero-cost ties would reduce `pun` and randomise `offset`) |
| credit / P / R | `credit = Σ s(i,j)`; `n_steps` = ALL rollout steps incl. duplicates; `n_gold` = GT steps with answer lines stripped; `prec = credit/n_steps`, `recall = credit/n_gold`, `F = F_β(prec, recall)`, β = 0.5 |
| order | `offset = mean over pairs of |i − j| / n_gold` (i = rollout step index, original order, 0-based, counting duplicates; j = GT index), clipped to [0, 1] so a full scramble costs at most 25 %; `match = F · (1 − 0.25·offset)` |
| pun | `n_unmatched = n_steps − |pairs|`; `pun = max(0, n_unmatched − 0.5·n_gold) / n_gold` (PUN_TOL = 0.5) |
| format | 1 iff 2 ≤ kept steps ≤ 12 AND exactly one answer line |
| answer | 1 iff `norm(extracted) == norm(gold)` (score_server side; `arm_reward.py` then swaps in its own numeric/MCQ-tolerant checker for the 5-point answer term, unchanged from arm 1) |
| R | `5·answer + 2·match + 1·format − 1·pun` |
| removed | length scale `min(1, 2·gold_mass/roll_mass)`, mass-based credit cap, TAU_MATCH gate (constant kept for `score_old` only) |
| unchanged | `Breakdown` dataclass fields (`cov` = recall, `inv_frac` = offset, `n_segs` = n_steps, `roll_mass`/`gold_mass` informational), so `grpo_arms/score_server.py` (op `reward_v2`) and `grpo_arms/arm_reward.py` (mode `match_v2`) run unmodified; `score_old` byte-identical to before |

### Diff summary of `score_new` (backup: `reward_v2.py.bak_20260911`)
- `n_steps, n_gold` computed once; gated Breakdown reports `n_steps`.
- Pair loop: `s >= TAU_MATCH` filter → `s > 0` filter with `credit += s`.
- Mass-based `m_roll`/`m_gold`/`cov`/`prec` → `prec = credit/n_steps`, `recall = credit/n_gold`.
- `match = fb * min(1, 2*gold_mass/roll_mass) * (1 - W_ORDER*inversion_fraction)` → `match = fb * (1 - W_ORDER*order_offset)`.
- `pun` on unmatched word mass → on unmatched step count.
- New helpers `is_answer_line` (terminal boxed line = answer line) and `order_offset`; module docstring rewritten to the spec.

### Arithmetic check (mock scorer, CPU, 2026-09-11)
perfect 4/4 in order → R = 8.000 · reversed order → offset 0.5, match 0.875, R = 7.750 · gold + 6 duplicates of step 1 → n_steps 10, prec 0.4, F 0.455, pun 1.0, R = 5.909 · finish=length → 0 gated · answer-first-then-padding → 0 gated · wrong answer + two 0.6 partial matches → match 0.375, R = 1.750 · terminal `\boxed{7}` → 8.000 · giant single paragraph → format 0, R = 5.750 · 13 junk steps → offset clipped 1.0, pun 2.5, format 0, R = 2.634.

## 2. Offline probe acceptance (Step 2)
Job 62990551 (htc, 1×A100, chunker/env, `redteam_reward.py --in probe.jsonl --out probe_scored_final.jsonl --skip_old`),
log `reward_redesign/redteam-62990551.log`, 905 rows incl. 350 synthetic attackers (555 probe rows: 185 degenerate / 185 clean / 185 gold; attackers built from clean rows that carry an answer line). PUN_TOL stayed at 0.5 (A4 passed first time, no rerun needed).
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
Columns: R_med/R_mean = reward; match/prec/rec/pun/off = population medians of match, precision, recall, pun, offset; gated = rows with R forced to 0; ans% = answer-correct rate; segs/dup = median step count / duplicate count.
`B1 old` reads FAIL only because `--skip_old` copies the new scores into the old column. attack_giant is informational by the 2026-09-11 design decision and passes at the margin (1.01).


## 3. Smoke gates (Step 3a)
Job 62992879 (htc, 2×A100, sg046, 02:30:08–03:07:23, rc=0), OUTDIR `grpo_arms/runs/arm1v2_smoke3`, identical overrides to the real run
(cold-start checkpoint-25, match_v2, native segmentation, MAXRESP=2048, GPUMEM=0.40, ppo/ref/rollout token caps 8192), 3 steps × 32 prompts.

| gate | evidence (from `logs/arm-62992879.log`, `logs/arm1_server_62992879.log`) | result |
|---|---|---|
| no "multiple active Ray instances" warning | 0 occurrences; Ray isolated at /tmp/ray_62992879 port 32879 | PASS |
| server log shows "[server] reward_v2 adapter ready" | present (line 5 of the server log) | PASS |
| step-0 val has match/pun/gated/n_dup per source with sane varying values | match .362/.122/.326/.259, gated 0/.5/0/.333, n_segments 4.5/24.5/3/5, offset .22/.417/.40/.167 (CLEVR/MathV360K/SROIE/ai2d); pun and n_dup 0 in val but .125→.039 and .087→.094 in train rollouts | PASS |
| steps 1–3 complete, critic/score/mean non-zero and varying | 2.661 → 3.289 → 4.281; response_length/mean 368.9 → 286.3 → 246.0; clip_ratio .019 → 0 → 0; peak actor memory 40.84 GB | PASS |
| rc=0 | `[arm1] done=Fri Sep 11 03:07:23 MST 2026 rc=0` | PASS |

seconds/step: 339.2, 233.7, 263.6 (mean 278.8). No OOM, so `ppo_max_token_len_per_gpu` stayed at 8192.


## 4. Real run (Step 3b)
**Job 62998194** (public, grp_vgupt140, 2×A100, 16 cpu, 320 GB, wall 1-02:40:00, mail END,FAIL), OUTDIR `grpo_arms/runs/arm1_3b_matchv2_final`,
resubmits 62998195 → 62998196 (afternotok chain, same OUTDIR, verl `resume_mode=auto`, max 2), eval 62998197 (afterok-OR on the three).

Config header (from `train_arm.py` [ARM CONFIG] / `configs/arm_config.json`): base_model = cold-start checkpoint-25, reward = match_v2, segmentation = native,
tau = 0.45 (ignored by score_new), subset sha bd15b1f1…, 5,880 train / 120 val, batch 32 × rollout_n 5, ppo_mini 8, micro 1, lr 1e-5, KL loss on,
max_prompt 2048, max_response 2048, image cap 640·28·28, reward_weights {answer 5, match 2, format 1, pun −1}, ppo_max_token_len_per_gpu 8192,
ref/rollout log_prob_max_token_len_per_gpu 8192, GPUMEM 0.40, save/test every 20, max_actor_ckpt_to_keep 4, max_steps 183.

Wall-time math: 340 s/step (slowest smoke step) × 183 × 1.5 + 45 min = 96,030 s = 1-02:40:00 < 48 h → full epoch, `--max_steps 183`.

Actual run: 03:21:18 → 19:37:26 (16h16m, mean 4.9 min/step incl. val + checkpoint), well inside the 1-02:40 wall; no resume needed
(resubmits cancelled by the job), watchdog never fired. Files: `runs/arm1_3b_matchv2_final/monitor.csv`, `monitor.png`, `monitor_summary.txt`.

### Monitor snapshots (train-rollout means per step; val = 120 prompts, unweighted source average)
| step | score | resp_len | clip | gated | n_dup | n_seg | pun | match | answer | format | offset | val_answer | val_match | s/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (val) | 4.452 | – | – | 0.076 | 0.231 | 6.12 | 0.049 | 0.293 | 0.650 | 0.766 | 0.251 | 0.650 | 0.293 | – |
| 1 | 2.810 | 387 | 0.006 | 0.169 | 0.131 | 7.23 | 0.126 | 0.199 | 0.412 | 0.694 | 0.252 | – | – | 339 |
| 20 | 4.825 | 279 | 0.006 | 0.006 | 0.069 | 4.01 | 0.004 | 0.314 | 0.644 | 0.981 | 0.273 | 0.656 | 0.353 | 277 |
| 40 | 5.267 | 393 | 0.000 | 0.000 | 0.094 | 4.13 | 0.002 | 0.385 | 0.700 | 1.000 | 0.225 | 0.568 | 0.375 | 307 |
| 60 | 5.048 | 403 | 0.000 | 0.000 | 0.156 | 3.98 | 0.007 | 0.452 | 0.631 | 0.994 | 0.231 | 0.605 | 0.453 | 302 |
| 80 | 5.073 | 481 | 0.000 | 0.031 | 0.263 | 3.92 | 0.000 | 0.502 | 0.650 | 0.944 | 0.218 | 0.520 | 0.503 | 313 |
| 100 | 5.293 | 405 | 0.000 | 0.006 | 0.169 | 3.49 | 0.000 | 0.556 | 0.637 | 0.994 | 0.210 | 0.515 | 0.563 | 326 |
| 120 | 5.651 | 491 | 0.000 | 0.000 | 0.150 | 3.81 | 0.000 | 0.638 | 0.675 | 1.000 | 0.267 | 0.505 | 0.632 | 332 |
| 140 | 4.418 | 575 | 0.000 | 0.000 | 0.306 | 4.41 | 0.000 | 0.662 | 0.419 | 1.000 | 0.252 | 0.471 | 0.671 | 319 |
| 160 | 5.422 | 805 | 0.000 | 0.000 | 0.125 | 4.44 | 0.000 | 0.742 | 0.588 | 1.000 | 0.254 | 0.483 | 0.700 | 402 |
| 180 | 5.142 | 937 | 0.000 | 0.000 | 0.081 | 5.36 | 0.000 | 0.774 | 0.519 | 1.000 | 0.266 | 0.479 | 0.779 | 455 |
| 183 | 5.080 | 1008 | 0.000 | 0.000 | 0.069 | 5.37 | 0.000 | 0.790 | 0.500 | 1.000 | 0.260 | 0.493 | 0.779 | 478 |

Red flags (brief's four): gated fraction never rose (0.000 from step 40 on); response_length/mean did trend up (279 @20 → 1008 @183) while
score rose 4.8 → 5.1–5.6 — the monitor's WARN fired at the end (751 → 879 over steps 144–163 vs 164–183); n_dup did not rise (0.31 peak @140,
0.07 @183); n_segments did not collapse (3.5–5.4). No STOP criterion was met, so the run was allowed to finish per the overnight rules.

### Gate 2 — does match move independently of answer?
Over 183 steps Pearson r(match/mean, answer/mean) = **−0.395** (first quarter: match 0.330 / answer 0.635, r = +0.587; last quarter: match
0.731 / answer 0.538, r = +0.186). Within steps, correct-answer rollouts score only +0.038 higher on match than wrong-answer ones
(0.546 vs 0.508 averaged over all steps). Val answer: 0.656 @20 → 0.568 → 0.605 → 0.519 → 0.515 → 0.505 → 0.471 → 0.483 → 0.479 → 0.493 @183;
val match 0.353 → 0.778.

**Judgment.** Match and answer are not locked together — they move in opposite directions, so match is not a proxy for answer; but it is not
carrying useful independent information either. The policy learned to write steps that entail the gold trace (match ×4) without those steps
leading to the right answer more often. This is not the v1 hack (no truncation, no duplicates, format 1.0, offset flat ≈ 0.26). The
mechanism is visible in the per-group reward decomposition below: GRPO normalises advantages within each prompt's 5 rollouts, so the
answer term only produces gradient in groups where correctness is mixed. Those groups disappear as easy prompts saturate and hard ones stay
all-wrong, leaving match (continuous, always varying) as essentially the whole gradient from step ~100 on.

| step | groups | all-wrong | all-right | mixed | mean R | R \| right | R \| wrong | answer share of within-group \|advantage\| |
|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 7 | 3 | 22 | 2.81 | 5.63 | 0.83 | 0.64 |
| 20 | 31 | 4 | 15 | 12 | 4.82 | 6.70 | 1.43 | 0.36 |
| 40 | 32 | 4 | 13 | 15 | 5.27 | 6.82 | 1.65 | 0.44 |
| 60 | 32 | 7 | 16 | 9 | 5.05 | 6.88 | 1.91 | 0.27 |
| 80 | 32 | 9 | 17 | 6 | 5.07 | 6.72 | 2.02 | 0.17 |
| 100 | 32 | 9 | 17 | 6 | 5.29 | 7.17 | 1.99 | 0.18 |
| 120 | 32 | 10 | 21 | 1 | 5.65 | 7.28 | 2.28 | 0.03 |
| 140 | 32 | 18 | 13 | 1 | 4.42 | 7.29 | 2.35 | 0.03 |
| 160 | 32 | 9 | 16 | 7 | 5.42 | 7.52 | 2.43 | 0.21 |
| 183 | 32 | 14 | 14 | 4 | 5.08 | 7.56 | 2.60 | 0.12 |

(`R | right` vs `R | wrong` shows the 5-point answer term is intact per sample; the last column is Σ|5·(answer − group mean)| / (that + Σ|2·(match − group mean)|).)

Fix candidates for the next run (not executed): a difficulty filter / curriculum that keeps groups mixed (drop prompts whose cold-start
pass@5 is 0 or 5), larger rollout_n so hard prompts occasionally produce a correct sample, or zeroing the match term on all-correct groups.


## 5. VPB evaluation (Step 4)
Protocol: frozen `grpo_arms/data/vpb_eval.jsonl` (2,856 single-image questions, sha 7cd6d974…), greedy decoding, pixel cap 640·28·28,
scorer = `score_vpb.py` (arm_reward's extraction cascade + normalized / MCQ-letter / numeric-tolerance match). Two generation caps are
reported: 1024 tokens (the frozen protocol the base3b / arm1_step183 references were run with) and 2048 tokens (the v2 training cap, added
because v2 writes ~940 tokens on VPB and 32 % of its step-183 answers were cut off at 1024). Checkpoints evaluated: 140, 160, 180, 183
(merged FSDP → HF under `runs/arm1_3b_matchv2_final/hf_step_<N>`); cold-start checkpoint-25 generated fresh (tag cs25).

Eval logistics: the original single 3h45 A100 job (62998197) was estimated to start 18 h later (htc: >1,000 pending GPU jobs), so it was
split into 1-hour `--gres=gpu:1` jobs (cs25 63043681; steps 183/180/160/140 63043682–5; 2048-token variants 63045335–9), which backfilled
onto H100/A100 nodes within minutes; scoring + tables by CPU post-job 63045340. Raw scores: `grpo_arms/evals/vpb_scores.json`; table:
`grpo_arms/evals/vpb_table_arm1v2.md`. (The post-job log shows one cosmetic `sacct: Bad job/step specified: split` from chain.json's
placeholder eval id.)

Truncation at 1024 tokens (finish_reason = length): base3b 4.3 %, cs25 5.1 %, arm1_step183 100 %, arm1v2_step183 31.9 %
(acc | stop = 0.2305, acc | length = 0.0066). At 2048 tokens: v2 checkpoints 0.0–0.8 %, base3b 3.3 %, cs25 4.6 %.

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

## 6. Housekeeping (2026-09-11 23:20)
- Checkpoints: FSDP dirs for steps 140/160/180 deleted (132 GB) after verifying the merged HF copies; `global_step_183` (44 GB) and
  `hf_step_{140,160,180,183}` (7.1 GB each) kept. Steps 20–120 were already pruned to stubs by verl.
- Eval queue pattern institutionalised: `eval_arm1v2.sbatch` defaults to one checkpoint per 1-hour `--gres=gpu:1` job;
  `grpo_arms/launch_vpb_eval.sh` fans out refs + checkpoints + the CPU scoring post-job; rule recorded in the sol-cluster-ops skill.
