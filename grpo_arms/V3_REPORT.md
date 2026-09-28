# V3 REPORT — arm-1 v3 runs (generated 2026-09-13 03:49 by grpo_arms/report_v3.py)

Reward spec frozen (reward_v2.score_new match path unchanged); tonight varied the optimizer only: answer-only control (ctrl),
GRPO without std-normalisation (v3a), and no-std + rollout n=8 + mixed-correctness group filter (v3b). Init = cold-start checkpoint-25,
90 steps, batch 32, MAXRESP 2048. Decisions on vpb_dev (400), numbers here on vpb_test (2,456, sha 1515742c78309548…).

## 1. VPB test (2,456 questions, greedy, max_tokens 2048)

| model | step | n | accuracy | Δ vs cs25 | finish=length rate |
|---|---|---|---|---|---|
| base3b | - | 2456 | 0.3025 | -0.0167 | 0.032 |
| cs25 (init) | 25 | 2456 | 0.3192 | +0.0000 | 0.039 |
| arm1v2 (v2 reward, std-norm) | 183 | 2456 | 0.2406 | -0.0786 | 0.001 |
| ctrl answer-only | 60 | 2456 | 0.1922 | -0.1270 | 0.048 |
| ctrl answer-only | 90 | 2456 | 0.1539 | -0.1653 | 0.179 |
| v3a match, no std | 60 | 2456 | 0.2423 | -0.0770 | 0.033 |
| v3a match, no std | 90 | 2456 | 0.2341 | -0.0851 | 0.030 |
| v3b match, no std, n8, filter (stopped at step 46 by user; last checkpoint) | 40 | 2456 | 0.0774 | -0.2419 | 0.065 |
| ctrl_base answer-only from BASE model (2026-09-12 follow-up) | 60 | 2456 | 0.1832 | -0.1360 | 0.225 |
| ctrl_base answer-only from BASE model (2026-09-12 follow-up) | 90 | 2456 | 0.1706 | -0.1486 | 0.335 |

per-source accuracy:

| model | step | DynaMath | MMMU_DEV_VAL | MathVerse_MINI_Vision_Only | MathVision_MINI | WeMath |
|---|---|---|---|---|---|---|
| base3b | - | 0.4490 | 0.3529 | 0.2265 | 0.2059 | 0.4760 |
| cs25 (init) | 25 | 0.4571 | 0.3982 | 0.2356 | 0.2075 | 0.5480 |
| arm1v2 (v2 reward, std-norm) | 183 | 0.2122 | 0.4072 | 0.1835 | 0.2026 | 0.4440 |
| ctrl answer-only | 60 | 0.1714 | 0.3846 | 0.1302 | 0.1520 | 0.3800 |
| ctrl answer-only | 90 | 0.1388 | 0.3077 | 0.1065 | 0.1454 | 0.2360 |
| v3a match, no std | 60 | 0.2755 | 0.3484 | 0.2016 | 0.2075 | 0.3120 |
| v3a match, no std | 90 | 0.2306 | 0.3348 | 0.1914 | 0.1879 | 0.4160 |
| v3b match, no std, n8, filter (stopped at step 46 by user; last checkpoint) | 40 | 0.1816 | 0.0588 | 0.0340 | 0.0866 | 0.0200 |
| ctrl_base answer-only from BASE model (2026-09-12 follow-up) | 60 | 0.2531 | 0.3167 | 0.0600 | 0.2173 | 0.2800 |
| ctrl_base answer-only from BASE model (2026-09-12 follow-up) | 90 | 0.2061 | 0.4163 | 0.0815 | 0.1536 | 0.2400 |

## 2. Training curves (train-val = 120 VisualPRM prompts source-avg; vpb_dev = 400; match/answer/mixed from rollout dumps)

### ctrl (progress 90/90)

| step | trainval acc | vpb_dev acc | match | answer | entropy | resp_len | clip | mixed grp frac |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.623 | 0.302 | - | - | - | - | - | - |
| 10 | 0.604 | 0.287 | 0.000 | 0.681 | 0.786 | 273 | 0.000 | 0.594 |
| 20 | 0.428 | 0.217 | 0.000 | 0.519 | 0.515 | 255 | 0.000 | 0.323 |
| 30 | 0.338 | 0.142 | 0.000 | 0.494 | 0.285 | 378 | 0.006 | 0.281 |
| 40 | 0.440 | 0.175 | 0.000 | 0.419 | 0.361 | 301 | 0.000 | 0.156 |
| 50 | 0.423 | 0.163 | 0.000 | 0.406 | 0.364 | 285 | 0.000 | 0.281 |
| 60 | 0.452 | 0.193 | 0.000 | 0.338 | 0.275 | 313 | 0.000 | 0.250 |
| 70 | 0.320 | 0.142 | 0.000 | 0.388 | 0.332 | 442 | 0.000 | 0.062 |
| 80 | 0.423 | 0.193 | 0.000 | 0.569 | 0.398 | 553 | 0.000 | 0.125 |
| 90 | 0.359 | 0.140 | 0.000 | 0.444 | 0.279 | 580 | 0.000 | 0.156 |

### v3a (progress 90/90)

| step | trainval acc | vpb_dev acc | match | answer | entropy | resp_len | clip | mixed grp frac |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.598 | 0.320 | - | - | - | - | - | - |
| 10 | 0.645 | 0.280 | 0.296 | 0.688 | 0.781 | 300 | 0.006 | 0.500 |
| 20 | 0.678 | 0.235 | 0.267 | 0.644 | 0.483 | 206 | 0.000 | 0.323 |
| 30 | 0.542 | 0.215 | 0.280 | 0.650 | 0.336 | 276 | 0.000 | 0.281 |
| 40 | 0.523 | 0.195 | 0.253 | 0.550 | 0.352 | 244 | 0.000 | 0.312 |
| 50 | 0.501 | 0.228 | 0.244 | 0.463 | 0.377 | 294 | 0.006 | 0.344 |
| 60 | 0.454 | 0.210 | 0.261 | 0.644 | 0.316 | 191 | 0.000 | 0.312 |
| 70 | 0.421 | 0.230 | 0.237 | 0.650 | 0.302 | 211 | 0.000 | 0.281 |
| 80 | 0.472 | 0.147 | 0.246 | 0.550 | 0.288 | 244 | 0.006 | 0.219 |
| 90 | 0.457 | 0.210 | 0.253 | 0.519 | 0.242 | 271 | 0.000 | 0.312 |

### v3b (progress 46/90)

| step | trainval acc | vpb_dev acc | match | answer | entropy | resp_len | clip | mixed grp frac | gen batches | trained | partial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.639 | 0.307 | - | - | - | - | - | - | - | - | - |
| 10 | 0.463 | 0.242 | 0.202 | 0.465 | 1.057 | 368 | 0.008 | 1.000 | 2 | 32 | 0 |
| 20 | 0.303 | 0.158 | 0.166 | 0.457 | 0.676 | 344 | 0.027 | 1.000 | 2 | 32 | 0 |
| 30 | 0.309 | 0.102 | 0.177 | 0.411 | 0.454 | 323 | 0.000 | 0.875 | 3 | 24 | 1 |
| 40 | 0.213 | 0.052 | 0.151 | 0.543 | 0.358 | 353 | 0.027 | 0.969 | 3 | 32 | 0 |

### ctrl_base (progress 90/90)

| step | trainval acc | vpb_dev acc | match | answer | entropy | resp_len | clip | mixed grp frac |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.461 | 0.290 | - | - | - | - | - | - |
| 10 | 0.639 | 0.278 | 0.000 | 0.694 | 0.365 | 294 | 0.000 | 0.438 |
| 20 | 0.585 | 0.250 | 0.000 | 0.700 | 0.292 | 219 | 0.000 | 0.355 |
| 30 | 0.527 | 0.212 | 0.000 | 0.588 | 0.261 | 282 | 0.000 | 0.375 |
| 40 | 0.487 | 0.205 | 0.000 | 0.600 | 0.310 | 314 | 0.006 | 0.406 |
| 50 | 0.332 | 0.172 | 0.000 | 0.544 | 0.286 | 359 | 0.000 | 0.312 |
| 60 | 0.424 | 0.155 | 0.000 | 0.556 | 0.210 | 496 | 0.013 | 0.344 |
| 70 | 0.430 | 0.165 | 0.000 | 0.606 | 0.120 | 510 | 0.006 | 0.250 |
| 80 | 0.425 | 0.133 | 0.000 | 0.625 | 0.136 | 554 | 0.006 | 0.062 |
| 90 | 0.434 | 0.120 | 0.000 | 0.500 | 0.091 | 738 | 0.031 | 0.188 |

## 3. Decision (tree from the brief; vpb_test accuracy at 2048 tokens, final step 90; step 60 in brackets)

cs25 = 0.3192; ctrl90 = 0.1539 [60: 0.1922]; v3a90 = 0.2341 [60: 0.2423]; v3b final = step 40 = 0.0774 (run stopped at step 46 on the user's instruction after vpb_dev fell to 0.0425)
**Decision: the training setup degrades regardless of reward** — ctrl < cs25. First suspects (from V3_STATUS.md step 0.3): lr 1e-05, KL loss coef 0.001, entropy_coeff 0, temperature 1.0, rollout n 5, ppo_mini_batch 8 / micro 1 (dynamic bsz, 8192 tok cap).

### 3b. Follow-up: answer-only control from the base model
base3b = 0.3025; ctrl_base 60 = 0.1832; ctrl_base 90 = 0.1706
**ctrl_base < base3b**: even from the base model the answer-only RLVR degrades, so the optimizer config (lr 1e-05, KL 0.001, entropy 0) or the VisualPRM prompt mix is the problem independent of the init.

## 4. Failures and resubmits

- ctrl: DONE=True
```
JobID|JobName|State|Elapsed|ExitCode
63051629|v3_ctrl|COMPLETED|03:58:25|0:0
63051630|v3_ctrl_resub1|CANCELLED by 2477195|00:00:00|0:0
63051631|v3_ctrl_resub2|CANCELLED by 2477195|00:00:00|0:0
63051632|vpb_s90_2ktest|CANCELLED by 2477195|00:53:07|0:0
63051633|vpb_s60_2ktest|COMPLETED|00:15:16|0:0
```
- v3a: DONE=True
```
JobID|JobName|State|Elapsed|ExitCode
63051624|v3_v3a|COMPLETED|06:40:56|0:0
63051625|v3_v3a_resub1|CANCELLED by 2477195|00:00:00|0:0
63051626|v3_v3a_resub2|CANCELLED by 2477195|00:00:00|0:0
63051627|vpb_s90_2ktest|COMPLETED|00:15:20|0:0
63051628|vpb_s60_2ktest|COMPLETED|00:15:20|0:0
```
- v3b: DONE=False
```
JobID|JobName|State|Elapsed|ExitCode
63051619|v3_v3b|FAILED|00:16:30|1:0
63051620|v3_v3b_resub1|CANCELLED by 2477195|07:59:35|0:0
63051621|v3_v3b_resub2|CANCELLED by 2477195|00:00:00|0:0
63101265|v3b_s40_test|COMPLETED|00:15:03|0:0
```
- ctrl_base: DONE=True
```
JobID|JobName|State|Elapsed|ExitCode
63123541|v3_ctrl_base|COMPLETED|04:10:13|0:0
63123542|v3_ctrl_base_resub1|CANCELLED by 2477195|00:00:00|0:0
63123543|v3_ctrl_base_resub2|CANCELLED by 2477195|00:00:00|0:0
63123544|vpb_s90_2ktest|COMPLETED|00:18:40|0:0
63123545|vpb_s60_2ktest|COMPLETED|00:17:46|0:0
```
