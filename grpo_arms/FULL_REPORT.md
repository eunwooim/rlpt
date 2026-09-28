# FULL REPORT — M (match + soft gates, v4 B resumed 60→183) vs R (RLVR answer-only + adaptive entropy, 0→183) on grp_vgupt140 (generated 2026-09-18 18:30 by grpo_arms/report_full.py)

Common config: lr 1e-6, kl_loss_coef 0.01 (low_var_kl), clip 0.2/0.28, n=5, batch 32, MAXRESP 2048, 8192-token caps, temperature 1.0, seed 42,
std-normalised GRPO, no filter_groups, 2×A100 public, 183 steps, save every 30. M = v4 B's process continued (same dataloader from batch 61,
REWARD_MODE=match GATE_MODE=soft, no entropy bonus). R = fresh from cs25, REWARD_MODE=answer_only GATE_MODE=hard, ENTROPY_MODE=adaptive
(target 0.6, coeff <- clip(coeff + 0.002·(0.6 − entropy), 0, 0.01) after every step; grpo_arms/adaptive_entropy.py).
Numbers on vpb_test (2,456, sha 1515742c78309548…), greedy, 2048 tokens; McNemar = exact binomial test on the discordant pairs (b = row right/reference wrong, c = reverse).

## 1. vpb_test

| model | step | n | accuracy | Δ vs cs25 | Δ vs base3b | finish=length | McNemar vs cs25 | McNemar vs v4 @60 (B for M, A for R) | McNemar M vs R (same step) |
|---|---|---|---|---|---|---|---|---|---|
| base3b (Qwen2.5-VL-3B-Instruct) | - | 2456 | 0.3025 | -0.0167 | +0.0000 | 0.032 | b=228 c=269 p=0.073 | - | - |
| cs25 (cold-start init) | 25 | 2456 | 0.3192 | +0.0000 | +0.0167 | 0.039 | b=0 c=0 p=1.000 | - | - |
| v4 A answer-only | 30 | 2456 | 0.3384 | +0.0191 | +0.0358 | 0.023 | b=276 c=229 p=0.041 | - | - |
| v4 A answer-only | 60 | 2456 | 0.3282 | +0.0090 | +0.0257 | 0.020 | b=260 c=238 p=0.347 | - | - |
| v4 B match + soft gates | 30 | 2456 | 0.3176 | -0.0016 | +0.0151 | 0.021 | b=220 c=224 p=0.887 | - | - |
| v4 B match + soft gates | 60 | 2456 | 0.3436 | +0.0244 | +0.0411 | 0.013 | b=299 c=239 p=0.011 | - | - |
| **M** match + soft gates (B resumed) | 90 | 2456 | 0.3335 | +0.0143 | +0.0309 | 0.027 | b=268 c=233 p=0.129 | b=217 c=242 p=0.263 | b=211 c=251 p=0.069 |
| **M** match + soft gates (B resumed) | 120 | 2456 | 0.3465 | +0.0273 | +0.0440 | 0.012 | b=278 c=211 p=0.003 | b=250 c=243 p=0.787 | b=220 c=227 p=0.777 |
| **M** match + soft gates (B resumed) | 150 | 2456 | 0.3599 | +0.0407 | +0.0574 | 0.015 | b=302 c=202 p=0.000 | b=265 c=225 p=0.078 | b=251 c=222 p=0.198 |
| **M** match + soft gates (B resumed) | 183 | 2456 | 0.3632 | +0.0440 | +0.0607 | 0.031 | b=320 c=212 p=0.000 | b=275 c=227 p=0.036 | b=255 c=238 p=0.471 |
| **R** RLVR + adaptive entropy | 30 | 2456 | 0.3265 | +0.0073 | +0.0240 | 0.030 | b=250 c=232 p=0.439 | b=243 c=247 p=0.892 | b=225 c=203 p=0.310 |
| **R** RLVR + adaptive entropy | 60 | 2456 | 0.3359 | +0.0167 | +0.0334 | 0.026 | b=265 c=224 p=0.070 | b=261 c=242 p=0.422 | b=235 c=254 p=0.416 |
| **R** RLVR + adaptive entropy | 90 | 2456 | 0.3498 | +0.0305 | +0.0472 | 0.019 | b=291 c=216 p=0.001 | b=277 c=224 p=0.020 | b=251 c=211 p=0.069 |
| **R** RLVR + adaptive entropy | 120 | 2456 | 0.3493 | +0.0301 | +0.0468 | 0.022 | b=295 c=221 p=0.001 | b=268 c=216 p=0.020 | b=227 c=220 p=0.777 |
| **R** RLVR + adaptive entropy | 150 | 2456 | 0.3481 | +0.0289 | +0.0456 | 0.021 | b=302 c=231 p=0.002 | b=272 c=223 p=0.031 | b=222 c=251 p=0.198 |
| **R** RLVR + adaptive entropy | 183 | 2456 | 0.3563 | +0.0371 | +0.0537 | 0.030 | b=324 c=233 p=0.000 | b=304 c=235 p=0.003 | b=238 c=255 p=0.471 |

per-source accuracy:

| model | step | DynaMath | MMMU_DEV_VAL | MathVerse_MINI_Vision_Only | MathVision_MINI | WeMath |
|---|---|---|---|---|---|---|
| base3b (Qwen2.5-VL-3B-Instruct) | - | 0.4490 | 0.3529 | 0.2265 | 0.2059 | 0.4760 |
| cs25 (cold-start init) | 25 | 0.4571 | 0.3982 | 0.2356 | 0.2075 | 0.5480 |
| v4 A answer-only | 30 | 0.4612 | 0.4208 | 0.2786 | 0.2157 | 0.5360 |
| v4 A answer-only | 60 | 0.4612 | 0.4344 | 0.2469 | 0.2402 | 0.4760 |
| v4 B match + soft gates | 30 | 0.4551 | 0.3846 | 0.2514 | 0.2026 | 0.5040 |
| v4 B match + soft gates | 60 | 0.4755 | 0.4344 | 0.2616 | 0.2435 | 0.5400 |
| **M** match + soft gates (B resumed) | 90 | 0.4714 | 0.4118 | 0.2469 | 0.2206 | 0.5760 |
| **M** match + soft gates (B resumed) | 120 | 0.4714 | 0.3801 | 0.2797 | 0.2402 | 0.5680 |
| **M** match + soft gates (B resumed) | 150 | 0.4898 | 0.3529 | 0.2797 | 0.2696 | 0.6160 |
| **M** match + soft gates (B resumed) | 183 | 0.5163 | 0.4027 | 0.2786 | 0.2500 | 0.6040 |
| **R** RLVR + adaptive entropy | 30 | 0.4653 | 0.4389 | 0.2390 | 0.2157 | 0.5360 |
| **R** RLVR + adaptive entropy | 60 | 0.4796 | 0.4299 | 0.2639 | 0.2255 | 0.4960 |
| **R** RLVR + adaptive entropy | 90 | 0.4776 | 0.4389 | 0.2707 | 0.2320 | 0.5880 |
| **R** RLVR + adaptive entropy | 120 | 0.4776 | 0.4118 | 0.2820 | 0.2288 | 0.5760 |
| **R** RLVR + adaptive entropy | 150 | 0.4592 | 0.4344 | 0.2582 | 0.2533 | 0.6040 |
| **R** RLVR + adaptive entropy | 183 | 0.4592 | 0.4163 | 0.2820 | 0.2761 | 0.5600 |

## 2. Training curves 0..183 (train-val = 120 VisualPRM prompts source-avg gated acc; vpb_dev = 400; match/answer/format/mixed/n_seg from rollout dumps; M's 0-60 = v4 B)

### M — M: match + soft gates, resumed from v4 B@60 (progress 183/183)

| step | trainval acc | vpb_dev acc | match | answer | format | entropy | ent coeff | KL | resp_len | trunc | mixed grp frac | n_seg | soft-gate fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.673 | 0.345 | - | - | - | - | - | - | - | - | - | - | - |
| 10 | 0.699 | 0.305 | 0.253 | 0.506 | 0.794 | 1.499 | off | 0.0022 | 322 | 0.006 | 0.719 | 6.0 | 0.056 |
| 20 | 0.690 | 0.338 | 0.268 | 0.506 | 0.812 | 1.197 | off | 0.0067 | 292 | 0.000 | 0.645 | 5.4 | 0.044 |
| 30 | 0.683 | 0.338 | 0.293 | 0.531 | 0.850 | 0.607 | off | 0.0125 | 347 | 0.000 | 0.656 | 6.4 | 0.050 |
| 40 | 0.659 | 0.325 | 0.313 | 0.675 | 0.906 | 0.467 | off | 0.0226 | 316 | 0.000 | 0.594 | 4.8 | 0.019 |
| 50 | 0.735 | 0.320 | 0.339 | 0.594 | 0.963 | 0.419 | off | 0.0212 | 321 | 0.000 | 0.656 | 5.1 | 0.013 |
| 60 | 0.722 | 0.318 | 0.313 | 0.681 | 0.950 | 0.453 | off | 0.0276 | 295 | 0.000 | 0.656 | 4.4 | 0.019 |
| 70 | 0.733 | 0.312 | 0.331 | 0.756 | 0.981 | 0.579 | off | 0.0303 | 270 | 0.000 | 0.656 | 4.5 | 0.006 |
| 80 | 0.703 | 0.340 | 0.359 | 0.781 | 0.950 | 0.521 | off | 0.0290 | 321 | 0.000 | 0.469 | 4.8 | 0.019 |
| 90 | 0.773 | 0.325 | 0.337 | 0.731 | 0.938 | 0.557 | off | 0.0253 | 355 | 0.000 | 0.500 | 4.9 | 0.019 |
| 100 | 0.766 | 0.357 | 0.330 | 0.756 | 0.950 | 0.661 | off | 0.0293 | 293 | 0.000 | 0.406 | 4.6 | 0.000 |
| 110 | 0.747 | 0.340 | 0.332 | 0.625 | 0.950 | 0.881 | off | 0.0354 | 274 | 0.000 | 0.548 | 4.4 | 0.019 |
| 120 | 0.750 | 0.355 | 0.341 | 0.694 | 0.950 | 0.563 | off | 0.0379 | 375 | 0.000 | 0.594 | 4.8 | 0.031 |
| 130 | 0.790 | 0.343 | 0.390 | 0.738 | 0.981 | 0.506 | off | 0.0373 | 353 | 0.000 | 0.531 | 4.4 | 0.000 |
| 140 | 0.699 | 0.343 | 0.382 | 0.688 | 1.000 | 0.489 | off | 0.0433 | 318 | 0.000 | 0.500 | 4.4 | 0.000 |
| 150 | 0.768 | 0.347 | 0.353 | 0.681 | 0.994 | 0.545 | off | 0.0485 | 303 | 0.000 | 0.355 | 4.5 | 0.006 |
| 160 | 0.798 | 0.362 | 0.377 | 0.775 | 0.981 | 0.531 | off | 0.0400 | 329 | 0.000 | 0.594 | 4.8 | 0.000 |
| 170 | 0.757 | 0.362 | 0.404 | 0.794 | 0.981 | 0.493 | off | 0.0501 | 355 | 0.000 | 0.344 | 4.8 | 0.006 |
| 180 | 0.734 | 0.340 | 0.415 | 0.725 | 0.981 | 0.615 | off | 0.0533 | 358 | 0.000 | 0.531 | 4.6 | 0.019 |
| 183 | 0.759 | 0.380 | 0.434 | 0.787 | 1.000 | 0.553 | off | 0.0565 | 345 | 0.000 | 0.406 | 4.6 | 0.000 |

### R — R: RLVR answer-only + adaptive entropy (target 0.6) (progress 183/183)

| step | trainval acc | vpb_dev acc | match | answer | format | entropy | ent coeff | KL | resp_len | trunc | mixed grp frac | n_seg | soft-gate fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.656 | 0.315 | - | - | - | - | - | - | - | - | - | - | - |
| 10 | 0.691 | 0.318 | n/a | 0.506 | 0.781 | 1.804 | 0.00000 | 0.0026 | 337 | 0.013 | 0.625 | 6.8 | n/a |
| 20 | 0.711 | 0.302 | n/a | 0.525 | 0.775 | 1.114 | 0.00000 | 0.0063 | 309 | 0.000 | 0.677 | 6.1 | n/a |
| 30 | 0.674 | 0.323 | n/a | 0.575 | 0.856 | 0.926 | 0.00000 | 0.0103 | 367 | 0.000 | 0.594 | 7.5 | n/a |
| 40 | 0.667 | 0.325 | n/a | 0.669 | 0.875 | 0.958 | 0.00000 | 0.0159 | 337 | 0.000 | 0.562 | 6.0 | n/a |
| 50 | 0.692 | 0.328 | n/a | 0.575 | 0.944 | 0.601 | 0.00000 | 0.0148 | 338 | 0.000 | 0.531 | 6.2 | n/a |
| 60 | 0.770 | 0.333 | n/a | 0.656 | 0.963 | 0.641 | 0.00011 | 0.0181 | 330 | 0.000 | 0.688 | 5.7 | n/a |
| 70 | 0.801 | 0.328 | n/a | 0.706 | 0.938 | 0.802 | 0.00000 | 0.0221 | 289 | 0.000 | 0.719 | 5.6 | n/a |
| 80 | 0.791 | 0.328 | n/a | 0.725 | 0.963 | 0.543 | 0.00014 | 0.0239 | 316 | 0.000 | 0.469 | 5.8 | n/a |
| 90 | 0.767 | 0.333 | n/a | 0.688 | 0.950 | 0.482 | 0.00010 | 0.0244 | 356 | 0.000 | 0.500 | 6.1 | n/a |
| 100 | 0.786 | 0.362 | n/a | 0.719 | 0.975 | 0.610 | 0.00119 | 0.0287 | 297 | 0.000 | 0.406 | 5.5 | n/a |
| 110 | 0.779 | 0.365 | n/a | 0.613 | 0.963 | 0.981 | 0.00008 | 0.0296 | 285 | 0.000 | 0.613 | 6.0 | n/a |
| 120 | 0.815 | 0.362 | n/a | 0.731 | 0.938 | 0.643 | 0.00004 | 0.0257 | 400 | 0.000 | 0.469 | 6.3 | n/a |
| 130 | 0.754 | 0.360 | n/a | 0.706 | 0.969 | 0.712 | 0.00000 | 0.0274 | 340 | 0.000 | 0.375 | 5.6 | n/a |
| 140 | 0.792 | 0.365 | n/a | 0.669 | 0.956 | 0.779 | 0.00000 | 0.0303 | 340 | 0.006 | 0.469 | 6.1 | n/a |
| 150 | 0.781 | 0.345 | n/a | 0.713 | 0.956 | 0.921 | 0.00000 | 0.0320 | 309 | 0.000 | 0.452 | 6.3 | n/a |
| 160 | 0.780 | 0.378 | n/a | 0.769 | 0.975 | 0.750 | 0.00000 | 0.0293 | 339 | 0.000 | 0.469 | 5.9 | n/a |
| 170 | 0.774 | 0.367 | n/a | 0.825 | 1.000 | 0.647 | 0.00000 | 0.0321 | 305 | 0.000 | 0.188 | 5.4 | n/a |
| 180 | 0.795 | 0.350 | n/a | 0.744 | 0.988 | 0.759 | 0.00000 | 0.0357 | 296 | 0.000 | 0.500 | 5.4 | n/a |
| 183 | 0.806 | 0.380 | n/a | 0.869 | 0.975 | 0.695 | 0.00000 | 0.0343 | 286 | 0.000 | 0.250 | 5.2 | n/a |

## 3. Verdict

**M** (reference v4 B@60 = 0.3436): vpb_test 90/120/150/183 = 0.3335 / 0.3465 / 0.3599 / 0.3632; entropy over steps 61-183: min 0.390 at step 146, mean of the last 30 logged steps 0.544 (123 steps logged)
- M verdict: **RISES**: 183 > B@60 by +0.0195 (McNemar p=0.036); entropy ≥ 0.4 throughout 61-183: **NO**.
**R** (reference v4 A: 30 = 0.3384, 60 = 0.3282, i.e. the post-30 decline of -0.0102): vpb_test 30/60/90/120/150/183 = 0.3265 / 0.3359 / 0.3498 / 0.3493 / 0.3481 / 0.3563; entropy: min 0.395 at step 93, mean of the last 30 steps 0.755; coefficient first > 0 at step 60, max 0.00140, mean over the last 30 steps 0.00000
- R verdict: **avoids the decline**: no later checkpoint falls more than one SE (0.0095) below step 30 (worst: step 60 +0.0094); 60 vs 30 McNemar p=0.303.
- best checkpoints: M = step 183 (0.3632); R = step 183 (0.3563); incl. v4 B@60 0.3436 the best match checkpoint is M@183
- M vs R at step 90: 0.3335 vs 0.3498 (-0.0163), McNemar b=211 c=251 p=0.069
- M vs R at step 120: 0.3465 vs 0.3493 (-0.0029), McNemar b=220 c=227 p=0.777
- M vs R at step 150: 0.3599 vs 0.3481 (+0.0118), McNemar b=251 c=222 p=0.198
- M vs R at step 183: 0.3632 vs 0.3563 (+0.0069), McNemar b=255 c=238 p=0.471 ← best M ← best R

## 4. Fairshare (grp_vgupt140)

before: grp_vgupt140 / sghos104 RawUsage_CHE 3832.4, RealFairShare 0.164363 (2026-09-18 02:28, FULL_STATUS.md preflight)
after (myfairshare):
```
Account             User      RawUsage_CHE  RawFairShare  TargetFairShare  RealFairShare
grp_bshettah        sghos104  1578.1        0.460263      0.8963822        0.4602630
grp_vgupt140        sghos104  6346.1        0.049549      0.6441127        0.0495490
class_cse476sprin+  sghos104  0.0           1.000000      1.0000000        1.0000000
```

## 5. Jobs, failures and resubmits

- M: DONE=True; pruned FSDP: ['M:30', 'M:90', 'M:150']
```
JobID|JobName|State|Elapsed|ExitCode
63567988|full_M_match|COMPLETED|09:52:11|0:0
63567989|M_match_full_resub1|CANCELLED by 2477195|00:00:00|0:0
63567990|M_match_full_resub2|CANCELLED by 2477195|00:00:00|0:0
63580419|full_M_s90|COMPLETED|00:13:02|0:0
63587597|full_M_s120|COMPLETED|00:12:19|0:0
63597712|full_M_s150|COMPLETED|00:13:13|0:0
63605758|full_M_s183|COMPLETED|00:16:04|0:0
```
- R: DONE=True; pruned FSDP: ['R:30', 'R:90', 'R:150']
```
JobID|JobName|State|Elapsed|ExitCode
63570804|full_R_rlvr|COMPLETED|07:41:56|0:0
63570805|R_rlvr_entropy_resub1|CANCELLED by 2477195|00:00:00|0:0
63570806|R_rlvr_entropy_resub2|CANCELLED by 2477195|00:00:00|0:0
63578736|full_R_s30|COMPLETED|00:14:37|0:0
63583940|full_R_s60|COMPLETED|00:17:13|0:0
63587598|full_R_s90|COMPLETED|00:12:46|0:0
63592042|full_R_s120|COMPLETED|00:13:12|0:0
63597713|full_R_s150|COMPLETED|00:11:53|0:0
63600730|full_R_s183|COMPLETED|00:13:16|0:0
```
