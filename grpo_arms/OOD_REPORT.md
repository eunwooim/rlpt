# OOD REPORT — base3b / cs25 / M@183 / R@183 on seven external benchmarks (generated 2026-09-19 05:21 by grpo_arms/ood/report_ood.py)

Greedy, 2048 tokens, VPB prompt template + pixel cap (grpo_arms/ood/build_ood_sets.py), one 1-h htc gpu:1 job per (model, benchmark) on grp_bshettah.
**Extraction and scoring are rule-based** (no VLMEvalKit / lmms-eval installed on Sol): arm_reward.extract_answer + the per-benchmark normalisation in this
script's docstring. MMMU: image_1 only (43/900 rows have extra images). McNemar = exact binomial on discordant pairs (b = M right/R wrong, c = the reverse).

M183 = runs/arm1_3b_matchv4_softgate/hf_step_183 (match + soft gates); R183 = runs/arm1_3b_rlvr_entropy/hf_step_183 (RLVR + adaptive entropy); cs25 = cold-start init; base3b = Qwen2.5-VL-3B-Instruct.

## 1. Accuracy per benchmark

| benchmark | n | base3b | cs25 | M183 | R183 | Δ(M−R) | Δ(M−cs25) | Δ(R−cs25) | McNemar M vs R | length rate base/cs25/M/R | overlap with vpb_test sources |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MathVista testmini | 1000 | 0.5620 | 0.5470 | 0.6150 | 0.6300 | -0.0150 | +0.0680 | +0.0830 | b=95 c=110 p=0.328 | 0.023/0.038/0.021/0.016 | disjoint |
| MMK12 (1,024 balanced) | 1024 | 0.3975 | 0.3955 | 0.4561 | 0.4893 | -0.0332 | +0.0605 | +0.0938 | b=130 c=164 p=0.054 | 0.024/0.031/0.026/0.013 | disjoint |
| MathVerse testmini VO+VD | 1576 | 0.2532 | 0.2963 | 0.3230 | 0.3274 | -0.0044 | +0.0266 | +0.0311 | b=143 c=150 p=0.726 | 0.041/0.053/0.025/0.027 | overlaps vpb_test (MathVerse_MINI_Vision_Only) |
|  ↳ MathVerse_VisionOnly | 788 | 0.2398 | 0.2868 | 0.3274 | 0.3160 | +0.0114 | +0.0406 | +0.0292 | b=78 c=69 p=0.510 | | |
|  ↳ MathVerse_VisionDominant | 788 | 0.2665 | 0.3058 | 0.3185 | 0.3388 | -0.0203 | +0.0127 | +0.0330 | b=65 c=81 p=0.214 | | |
| MathVision test (full) | 3040 | 0.1970 | 0.2003 | 0.2299 | 0.2391 | -0.0092 | +0.0296 | +0.0388 | b=328 c=356 p=0.302 | 0.063/0.071/0.049/0.049 | overlaps vpb_test (MathVision_MINI ⊂ test) |
| WeMath testmini | 1740 | 0.5080 | 0.5500 | 0.6362 | 0.6333 | +0.0029 | +0.0862 | +0.0833 | b=223 c=218 p=0.849 | 0.012/0.028/0.028/0.014 | overlaps vpb_test (WeMath) |
| DynaMath 10×501 | 5010 | 0.4192 | 0.4641 | 0.5050 | 0.5040 | +0.0010 | +0.0409 | +0.0399 | b=449 c=444 p=0.894 | 0.019/0.025/0.017/0.017 | overlaps vpb_test (DynaMath) |
|  ↳ DynaMath per-seed worst-case (all 10 variants right) | 501 | 0.1118 | 0.1537 | 0.2016 | 0.1896 | +0.0120 | +0.0479 | +0.0359 | b=26 c=20 p=0.461 | | |
| MMMU validation | 900 | 0.4411 | 0.4567 | 0.4911 | 0.4911 | +0.0000 | +0.0344 | +0.0344 | b=105 c=105 p=1.000 | 0.027/0.037/0.013/0.020 | overlaps vpb_test (MMMU_DEV_VAL) |

## 2. Pooled M vs R (paired on every item, all seven benchmarks)

- all items (DynaMath per variant): n = 14290; M 0.4457 vs R 0.4509 (Δ -0.0052); **McNemar b=1473 c=1547 p=0.184**
- DynaMath collapsed to per-seed worst-case: n = 9781; M 0.4028 vs R 0.4103; McNemar b=1050 c=1123 p=0.122
- M vs cs25: n = 14290; b=2006 c=1358 p=0.000; R vs cs25: b=2002 c=1280 p=0.000
- M vs base3b: n = 14290; b=2292 c=1271 p=0.000; R vs base3b: b=2383 c=1288 p=0.000

## 3. Extraction diagnostics (fraction of outputs without an explicit answer marker; those fall back to the last line)

| benchmark | base3b | cs25 | M183 | R183 |
|---|---|---|---|---|
| MathVista testmini | 0.037 | 0.039 | 0.021 | 0.016 |
| MMK12 (1,024 balanced) | 0.030 | 0.031 | 0.026 | 0.013 |
| MathVerse testmini VO+VD | 0.079 | 0.074 | 0.025 | 0.027 |
| MathVision test (full) | 0.075 | 0.072 | 0.049 | 0.049 |
| WeMath testmini | 0.015 | 0.029 | 0.028 | 0.014 |
| DynaMath 10×501 | 0.071 | 0.028 | 0.017 | 0.017 |
| MMMU validation | 0.031 | 0.038 | 0.013 | 0.020 |

## 4. Per-source accuracy (subjects / variants)

### MMK12 (1,024 balanced) — by subject (top 4)

| subject | n | base3b | cs25 | M183 | R183 |
|---|---|---|---|---|---|
| math | 256 | 0.453 | 0.414 | 0.531 | 0.551 |
| physics | 256 | 0.344 | 0.391 | 0.367 | 0.449 |
| chemistry | 256 | 0.418 | 0.426 | 0.457 | 0.453 |
| biology | 256 | 0.375 | 0.352 | 0.469 | 0.504 |

### MathVerse testmini VO+VD

| source | base3b | cs25 | M183 | R183 |
|---|---|---|---|---|
| MathVerse_VisionDominant | 0.2665 | 0.3058 | 0.3185 | 0.3388 |
| MathVerse_VisionOnly | 0.2398 | 0.2868 | 0.3274 | 0.3160 |

### MathVision test (full) — by subject (top 12)

| subject | n | base3b | cs25 | M183 | R183 |
|---|---|---|---|---|---|
| metric geometry - area | 500 | 0.220 | 0.238 | 0.274 | 0.286 |
| metric geometry - length | 449 | 0.214 | 0.192 | 0.249 | 0.261 |
| algebra | 345 | 0.148 | 0.217 | 0.212 | 0.220 |
| combinatorial geometry | 308 | 0.175 | 0.156 | 0.218 | 0.195 |
| solid geometry | 244 | 0.143 | 0.205 | 0.189 | 0.217 |
| metric geometry - angle | 173 | 0.214 | 0.283 | 0.318 | 0.335 |
| transformation geometry | 168 | 0.244 | 0.173 | 0.173 | 0.232 |
| combinatorics | 168 | 0.137 | 0.095 | 0.131 | 0.155 |
| arithmetic | 140 | 0.314 | 0.279 | 0.271 | 0.321 |
| logic | 119 | 0.185 | 0.176 | 0.202 | 0.143 |
| descriptive geometry | 104 | 0.125 | 0.183 | 0.192 | 0.173 |
| graph theory | 90 | 0.144 | 0.111 | 0.122 | 0.078 |

### WeMath testmini — by subject (top 12)

| subject | n | base3b | cs25 | M183 | R183 |
|---|---|---|---|---|---|
| Properties and Understanding of Triangles | 41 | 0.659 | 0.585 | 0.732 | 0.634 |
| Area of Squares | 32 | 0.531 | 0.656 | 0.875 | 0.688 |
| Volume and Capacity of Rectangular Cuboids | 30 | 0.567 | 0.767 | 0.833 | 0.767 |
| Sum of Interior Angles of Triangles | 28 | 0.714 | 0.750 | 0.714 | 0.750 |
| Area of Circles | 27 | 0.630 | 0.593 | 0.778 | 0.667 |
| Circumference of Circles | 25 | 0.640 | 0.680 | 0.760 | 0.760 |
| Volume and Capacity of Cylinders | 25 | 0.480 | 0.640 | 0.760 | 0.640 |
| Area of Triangles | 25 | 0.720 | 0.800 | 0.800 | 0.760 |
| Area of Rectangles | 24 | 0.750 | 0.792 | 0.792 | 0.792 |
| Translation | 23 | 0.304 | 0.391 | 0.478 | 0.304 |
| Calculation and Comparison of Angles | 23 | 0.348 | 0.435 | 0.435 | 0.696 |
| Volume and Capacity of Cones | 22 | 0.636 | 0.727 | 0.682 | 0.773 |

### DynaMath 10×501

| source | base3b | cs25 | M183 | R183 |
|---|---|---|---|---|
| DynaMath_v1 | 0.4152 | 0.4431 | 0.5030 | 0.4830 |
| DynaMath_v10 | 0.4192 | 0.4770 | 0.5030 | 0.5070 |
| DynaMath_v2 | 0.4092 | 0.4651 | 0.5030 | 0.5269 |
| DynaMath_v3 | 0.4172 | 0.4770 | 0.5030 | 0.5110 |
| DynaMath_v4 | 0.3972 | 0.4331 | 0.4810 | 0.4850 |
| DynaMath_v5 | 0.4331 | 0.4790 | 0.5150 | 0.5170 |
| DynaMath_v6 | 0.4471 | 0.4770 | 0.5329 | 0.5250 |
| DynaMath_v7 | 0.4172 | 0.4890 | 0.5090 | 0.5150 |
| DynaMath_v8 | 0.4331 | 0.4810 | 0.5130 | 0.5030 |
| DynaMath_v9 | 0.4032 | 0.4192 | 0.4870 | 0.4671 |

### MMMU validation — by subject (top 12)

| subject | n | base3b | cs25 | M183 | R183 |
|---|---|---|---|---|---|
| Art | 60 | 0.500 | 0.583 | 0.583 | 0.583 |
| Accounting | 30 | 0.433 | 0.500 | 0.333 | 0.467 |
| Agriculture | 30 | 0.400 | 0.467 | 0.533 | 0.500 |
| Architecture | 30 | 0.333 | 0.367 | 0.200 | 0.300 |
| Basic | 30 | 0.467 | 0.433 | 0.567 | 0.533 |
| Biology | 30 | 0.233 | 0.367 | 0.400 | 0.333 |
| Chemistry | 30 | 0.333 | 0.333 | 0.233 | 0.200 |
| Clinical | 30 | 0.567 | 0.567 | 0.500 | 0.600 |
| Computer | 30 | 0.333 | 0.467 | 0.533 | 0.567 |
| Design | 30 | 0.700 | 0.667 | 0.700 | 0.667 |
| Diagnostics | 30 | 0.267 | 0.300 | 0.367 | 0.333 |
| Economics | 30 | 0.567 | 0.500 | 0.667 | 0.567 |

## 5. Jobs

```
63644906|ood_base3b_mathvista|COMPLETED|00:10:28|0:0
63644907|ood_cs25_mathvista|COMPLETED|00:10:03|0:0
63644908|ood_M183_mathvista|COMPLETED|00:03:24|0:0
63644909|ood_R183_mathvista|COMPLETED|00:03:24|0:0
63644910|ood_base3b_mmk12|COMPLETED|00:04:02|0:0
63644911|ood_cs25_mmk12|COMPLETED|00:02:53|0:0
63644912|ood_M183_mmk12|COMPLETED|00:02:56|0:0
63644913|ood_R183_mmk12|COMPLETED|00:02:33|0:0
63644914|ood_base3b_mathverse|COMPLETED|00:04:34|0:0
63644915|ood_cs25_mathverse|COMPLETED|00:07:04|0:0
63644916|ood_M183_mathverse|COMPLETED|00:04:05|0:0
63644917|ood_R183_mathverse|COMPLETED|00:04:05|0:0
63644918|ood_base3b_mathvision|COMPLETED|00:11:16|0:0
63644919|ood_cs25_mathvision|COMPLETED|00:06:30|0:0
63644920|ood_M183_mathvision|COMPLETED|00:10:39|0:0
63644921|ood_R183_mathvision|COMPLETED|00:10:29|0:0
63644922|ood_base3b_wemath|COMPLETED|00:03:23|0:0
63644923|ood_cs25_wemath|COMPLETED|00:04:42|0:0
63644924|ood_M183_wemath|COMPLETED|00:02:55|0:0
63644925|ood_R183_wemath|COMPLETED|00:02:53|0:0
63644926|ood_base3b_dynamath|COMPLETED|00:14:22|0:0
63644927|ood_cs25_dynamath|COMPLETED|00:13:36|0:0
63644928|ood_M183_dynamath|COMPLETED|00:07:54|0:0
63644929|ood_R183_dynamath|COMPLETED|00:07:52|0:0
63644930|ood_base3b_mmmu|COMPLETED|00:04:38|0:0
63644931|ood_cs25_mmmu|COMPLETED|00:04:16|0:0
63644932|ood_M183_mmmu|COMPLETED|00:04:24|0:0
63644933|ood_R183_mmmu|COMPLETED|00:04:20|0:0
```
