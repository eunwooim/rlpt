# V4 REPORT — answer-only control (A) vs match + soft gates (B) at lr 1e-6 / KL 0.01 / clip 0.2–0.28 (generated 2026-09-13 11:18 by grpo_arms/report_v4.py)

Account grp_bshettah. Init = cold-start checkpoint-25, 60 steps, batch 32, n=5, MAXRESP 2048, std-normalisation on, no filter_groups, no entropy bonus.
Reward matching unchanged; the only reward change is GATE_MODE=soft on run B (truncation still R=0; missing/misplaced answer costs only the answer term).
Decisions on vpb_dev (400), numbers here on vpb_test (2,456, sha 1515742c78309548…).

## 1. VPB test (2,456 questions, greedy, max_tokens 2048)

| model | step | n | accuracy | Δ vs cs25 | Δ vs base3b | finish=length rate |
|---|---|---|---|---|---|---|
| base3b | - | 2456 | 0.3025 | -0.0167 | +0.0000 | 0.032 |
| cs25 (init) | 25 | 2456 | 0.3192 | +0.0000 | +0.0167 | 0.039 |
| arm1v2 (v2 reward, std-norm) | 183 | 2456 | 0.2406 | -0.0786 | -0.0619 | 0.001 |
| ctrl answer-only | 60 | 2456 | 0.1922 | -0.1270 | -0.1103 | 0.048 |
| ctrl answer-only | 90 | 2456 | 0.1539 | -0.1653 | -0.1486 | 0.179 |
| v3a match, no std | 60 | 2456 | 0.2423 | -0.0770 | -0.0603 | 0.033 |
| v3a match, no std | 90 | 2456 | 0.2341 | -0.0851 | -0.0684 | 0.030 |
| v3b match, no std, n8, filter (stopped at 46) | 40 | 2456 | 0.0774 | -0.2419 | -0.2252 | 0.065 |
| ctrl_base answer-only, base init | 60 | 2456 | 0.1832 | -0.1360 | -0.1193 | 0.225 |
| ctrl_base answer-only, base init | 90 | 2456 | 0.1706 | -0.1486 | -0.1319 | 0.335 |
| **A** answer-only, lr 1e-6 / KL 0.01 / clip 0.2-0.28 | 30 | 2456 | 0.3384 | +0.0191 | +0.0358 | 0.023 |
| **A** answer-only, lr 1e-6 / KL 0.01 / clip 0.2-0.28 | 60 | 2456 | 0.3282 | +0.0090 | +0.0257 | 0.020 |
| **B** match + soft gates, same optimizer | 30 | 2456 | 0.3176 | -0.0016 | +0.0151 | 0.021 |
| **B** match + soft gates, same optimizer | 60 | 2456 | 0.3436 | +0.0244 | +0.0411 | 0.013 |

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
| v3b match, no std, n8, filter (stopped at 46) | 40 | 0.1816 | 0.0588 | 0.0340 | 0.0866 | 0.0200 |
| ctrl_base answer-only, base init | 60 | 0.2531 | 0.3167 | 0.0600 | 0.2173 | 0.2800 |
| ctrl_base answer-only, base init | 90 | 0.2061 | 0.4163 | 0.0815 | 0.1536 | 0.2400 |
| **A** answer-only, lr 1e-6 / KL 0.01 / clip 0.2-0.28 | 30 | 0.4612 | 0.4208 | 0.2786 | 0.2157 | 0.5360 |
| **A** answer-only, lr 1e-6 / KL 0.01 / clip 0.2-0.28 | 60 | 0.4612 | 0.4344 | 0.2469 | 0.2402 | 0.4760 |
| **B** match + soft gates, same optimizer | 30 | 0.4551 | 0.3846 | 0.2514 | 0.2026 | 0.5040 |
| **B** match + soft gates, same optimizer | 60 | 0.4755 | 0.4344 | 0.2616 | 0.2435 | 0.5400 |

## 2. Training curves (train-val = 120 VisualPRM prompts source-avg; vpb_dev = 400; match/answer/mixed from rollout dumps)

### A_ctrl_v4 (progress 60/60)

| step | trainval acc | vpb_dev acc | match | answer | entropy | KL | resp_len | clip | mixed grp frac |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.679 | 0.330 | - | - | - | - | - | - | - |
| 10 | 0.735 | 0.340 | 0.000 | 0.537 | 1.492 | 0.0059 | 346 | 0.006 | 0.656 |
| 20 | 0.766 | 0.335 | 0.000 | 0.525 | 1.172 | 0.0066 | 315 | 0.000 | 0.742 |
| 30 | 0.771 | 0.315 | 0.000 | 0.613 | 0.843 | 0.0096 | 368 | 0.000 | 0.562 |
| 40 | 0.659 | 0.345 | 0.000 | 0.631 | 0.746 | 0.0155 | 325 | 0.000 | 0.531 |
| 50 | 0.748 | 0.330 | 0.000 | 0.556 | 0.607 | 0.0131 | 326 | 0.000 | 0.750 |
| 60 | 0.718 | 0.320 | 0.000 | 0.662 | 0.649 | 0.0174 | 300 | 0.000 | 0.656 |

### B_matchv4_softgate (progress 60/60)

| step | trainval acc | vpb_dev acc | match | answer | entropy | KL | resp_len | clip | mixed grp frac | soft-gate fired |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.673 | 0.345 | - | - | - | - | - | - | - | - |
| 10 | 0.699 | 0.305 | 0.253 | 0.506 | 1.499 | 0.0022 | 322 | 0.006 | 0.719 | 0.056 |
| 20 | 0.690 | 0.338 | 0.268 | 0.506 | 1.197 | 0.0067 | 292 | 0.000 | 0.645 | 0.044 |
| 30 | 0.683 | 0.338 | 0.293 | 0.531 | 0.607 | 0.0125 | 347 | 0.000 | 0.656 | 0.050 |
| 40 | 0.659 | 0.325 | 0.313 | 0.675 | 0.467 | 0.0226 | 316 | 0.000 | 0.594 | 0.019 |
| 50 | 0.735 | 0.320 | 0.339 | 0.594 | 0.419 | 0.0212 | 321 | 0.000 | 0.656 | 0.013 |
| 60 | 0.751 | 0.330 | 0.313 | 0.681 | 0.453 | 0.0276 | 295 | 0.000 | 0.656 | 0.019 |


## 3. Probe (reward_redesign/probe.jsonl + attackers, 1,080 rows incl. the v4 answer-first attacker): hard vs soft gates

| population | hard R median | soft R median | soft gated |
|---|---|---|---|
| gold | 7.97 | 7.97 | 0/185 |
| clean | 5.47 | 5.47 | 6/185 |
| attack_giant | 5.40 | 5.40 | 0/175 |
| attack_repeat6 | 4.67 | 4.67 | 0/175 |
| attack_answer_first | 0.00 | 1.51 | 0/175 |
| degenerate | 0.00 | 0.00 | 185/185 |

## 4. Verdict (tree from the brief)

A: vpb_test 30/60 = 0.3384/0.3282; vpb_dev@60 = 0.3200; min entropy at logged steps = 0.6067; train-val 0→60 = 0.6790 → 0.7176
B: vpb_test 30/60 = 0.3176/0.3436
**Verdict: optimizer fixed and match + soft gates ≥ control** — this is the configuration for arms 2–6.
**Also:** A is stable but stagnant (train-val moved < +0.05 by step 60) — recommend 120 steps or lr 2e-6.

## 5. Fairshare cost (grp_bshettah)

after (myfairshare):
```
Account             User      RawUsage_CHE  RawFairShare  TargetFairShare  RealFairShare
grp_bshettah        sghos104  2580.1        0.293071      0.8362427        0.2930710
grp_vgupt140        sghos104  6060.7        0.055779      0.6569862        0.0557790
class_cse476sprin+  sghos104  0.0           1.000000      1.0000000        1.0000000
```
before: see V4_STATUS.md step 0 (RawUsage_CHE 1262.7, RealFairShare 0.492369).

## 6. Failures and resubmits

- A_ctrl_v4: DONE=True
```
JobID|JobName|State|Elapsed|ExitCode
63147498|v4_A_ctrl|FAILED|00:20:51|1:0
63147499|v4_A_ctrl_v4_resub1|COMPLETED|02:30:32|0:0
63147500|v4_A_ctrl_v4_resub2|CANCELLED by 2477195|00:00:00|0:0
63147501|vpb_s60_2ktest|COMPLETED|00:12:20|0:0
63147502|vpb_s30_2ktest|COMPLETED|00:12:30|0:0
```
- B_matchv4_softgate: DONE=True
```
JobID|JobName|State|Elapsed|ExitCode
63148593|v4_B_match|COMPLETED|05:13:40|0:0
63148594|v4_B_matchv4_softgate_resub1|CANCELLED by 2477195|00:00:00|0:0
63148595|v4_B_matchv4_softgate_resub2|CANCELLED by 2477195|00:00:00|0:0
63148596|vpb_s60_2ktest|COMPLETED|00:16:11|0:0
63148597|vpb_s30_2ktest|COMPLETED|00:16:30|0:0
```

## 7. Interpretation and caveats (written 11:30 after the auto-generated sections)
- **The conservative optimizer is the fix.** With lr 1e-6 / KL 0.01 / clip 0.2–0.28 (everything else as arm1v2) the answer-only control no longer
  collapses: entropy 1.49 → 0.61 over 60 steps instead of 1.67 → 0.29 in 30, no truncation at any logged step, vpb_dev 0.33 → 0.32, and vpb_test
  0.3384 @30 / 0.3282 @60 vs cs25 0.3192. Every v3 arm (lr 1e-5) ended 8–24 points below its init; both v4 arms end above it.
- **Match + soft gates ≥ control, but the A-vs-B gap is inside the noise.** Paired McNemar on the 2,456 vpb_test questions:
  B@60 vs A@60 0.3436 vs 0.3282, discordant 256/218, p = 0.089; B@60 vs cs25 p = 0.011 (significant); A@60 vs cs25 p = 0.347 (not);
  A@30 vs cs25 p = 0.041; B@30 vs cs25 p = 0.887. So B's step-60 gain over the init is the only individually significant improvement, and the
  verdict "B ≥ A" should be read as "B is at least as good as A, and the only arm significantly above cs25", not as a proven advantage of the
  match term. Per source at step 60: B ≥ A on DynaMath (+0.014), MathVerse (+0.015), WeMath (+0.064); tied on MMMU and MathVision.
- **Soft gates behave as intended.** The soft gate fired on 16.9 % of rollouts at step 5 and 1.3–1.9 % at steps 40–60 while the format term rose
  0.79 → 0.95: the policy learns the answer position from the 5-point answer term alone, without the R=0 cliff. The added probe attacker
  (answer first, then reference-shaped padding) earns 1.51 under soft gates vs 0 under hard — the reasoning credit only, 5 points below clean.
- **The "stagnant" flag in §4 is a metric artefact.** report_v4 reads verl's step-0 `answer` mean (0.679, ungated) but `acc` (gated) at later steps;
  on the gated `acc` used everywhere else A moved 0.62 → 0.718 (+0.10) and B 0.62 → 0.751 (+0.13). Neither run is stagnant; ignore that line.
- **Watch entropy in B.** B's entropy reached 0.42 at step 50 and 0.45 at step 60 (A: 0.61 / 0.65); KL 0.028 vs 0.017. A 120-step run of B at this
  optimizer would likely cross the 0.4 floor — for arms 2–6 at 90+ steps, add the adaptive entropy bonus or lower lr to 5e-7 after step 60.
- **Consistency checks (V4_STATUS.md):** step-0 vpb_dev A 0.33 vs B 0.345 = vLLM batch-order nondeterminism (6/400 questions, same init, greedy);
  both runs loaded the same arm_reward.py (sha256 c45ab153…); `acc` = correct ∧ on last line in both.
- **Cost:** grp_bshettah / sghos104 RawUsage_CHE 1262.7 → 2580.2 (+1317.5 CHE), RealFairShare 0.4924 → 0.2931, for: 2 training runs (2×A100,
  2h28m + 5h13m), 1 failed start (16 min), 2 probe jobs, 4 evals, monitor (6.7 h CPU), report. grp_vgupt140 untouched.
