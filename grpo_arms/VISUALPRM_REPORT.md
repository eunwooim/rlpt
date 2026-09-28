# VisualPRM GRPO campaign — consolidated report

Qwen2.5-VL-3B-Instruct, cold-start SFT (cs25), then GRPO with the bipartite reasoning-match reward ("Bipartite", runs B/M/M2) versus answer-only RLVR ("RLVR", runs A/R/R2), epochs 1 and 2, in-distribution (VisualProcessBench test split) and out-of-distribution (seven external benchmarks), followed by the NLI-judge benchmark. Written 2026-09-27 from the per-run reports listed in §9; every number below is copied from those files.

---

## 1. Setup

### 1.1 Models
| tag | what it is |
|---|---|
| base3b | `Qwen/Qwen2.5-VL-3B-Instruct`, zero-shot |
| cs25 | LoRA SFT (r=16, α=32, q/k/v/o/gate/up/down; lr 1e-5; batch 4 × grad-accum 32 = 128 traces/step) of base3b on the **unfiltered** VisualPRM400K-v1.1-Raw traces, stopped at optimizer step 25 (3,200 traces, 0.57 % of the 565,149-trace epoch; train loss 3.20 → 0.69). It teaches the step-by-step + `Final answer:` format that the reward parses. `src/outputs/train/coldstart_vprm_unf_3b/run_v1/checkpoints/checkpoint-25`. |
| Bipartite (B → M → M2) | GRPO from cs25 with the composite match reward (§1.3), soft gates. B = steps 1–60 (v4 run), M = the same process resumed 61–183, M2 = epoch 2, 184–330 (cancelled at 330 for fairshare; 366 never reached). |
| RLVR (A / R → R2) | GRPO from cs25 with the answer-only reward, hard gates. A = v4 control, 60 steps, no entropy bonus. R = fresh 183-step run with the adaptive entropy controller; R2 = epoch 2, 184–366. |

### 1.2 Training data
VisualPRM400K-v1.1-Raw → frozen 6,000-prompt subset (`grpo_arms/data/train_subset.jsonl`, sha bd15b1f1…): answer-correct traces (last step MC score > 0.5), ≥ 5 body steps, nlvr2 excluded (two images), dedup by (image, question) keeping the trace with the highest minimum body-step MC score, cap 300 per source file, seed 42; 37 source files; body steps 5–23 (mean 7.06). Funnel: 538,710 rows → 215,281 eligible traces → 114,541 unique prompts → 10,847 after the cap → 6,000 sampled. In verl the last 120 rows are the training-val set, so each epoch is **5,880 prompts = 183 batches of 32**. Images bounded to 640·28·28 pixels. Epoch 2 reshuffles the same 5,880 prompts with dataloader seed 43.

### 1.3 Rewards (`reward_redesign/reward_v2.py`, final spec 2026-09-11)
Steps = the rollout with `<think>` tags stripped, split on **blank lines**, answer lines removed; duplicates (SBERT all-MiniLM-L6-v2 cosine ≥ 0.90 to an earlier step) are never matched. Judge = `microsoft/deberta-xlarge-mnli` (fp32 + TF32, 512-token truncation), bidirectional:

    s(i,j) = clip[0,1]( 0.5·E(i→j) + 0.5·E(j→i) − max C )

over kept rollout steps × gold steps; Hungarian one-to-one assignment, no threshold; credit = Σ s; precision = credit / n_rollout_steps (duplicates included); recall = credit / n_gold_steps; F = F₀.₅; offset = mean |i − j| / n_gold over pairs; **match = F · (1 − 0.25·offset)**; pun = max(0, unmatched − 0.5·n_gold) / n_gold; format = 1 iff 2–12 kept steps and exactly one answer line; answer = 1 iff the normalised last-line answer equals the gold answer.

    Bipartite:  R = 5·answer + 2·match + 1·format − 1·pun
    RLVR:       R = 5·answer + 1·format

Gates: a truncated rollout (finish_reason = length) scores R = 0 in both. Hard gates (RLVR) also zero the rollout when no answer is parsable or the answer is not the last line; soft gates (Bipartite) only zero the answer term in those cases.

### 1.4 Optimizer (identical for every run below, both epochs)
verl 0.8.0 GRPO, vLLM rollouts, 2×A100: lr 1e-6, KL loss 0.01 (low_var_kl), clip 0.2 / 0.28, rollout n = 5, batch 32 prompts, max response 2048 tokens, 8,192-token caps, temperature 1.0, std-normalised advantages, no group filtering, seed 42, save every 30 steps. R/R2 additionally run the adaptive entropy controller (target 0.6; coefficient ← clip(coeff + 0.002·(0.6 − entropy), 0, 0.01) after every step, multiplying verl's entropy bonus; `grpo_arms/adaptive_entropy.py`). It engaged only briefly in practice (max coefficient 0.0014 in R, 0.0003 in R2).

### 1.5 Evaluation
- **vpb_dev / vpb_test**: VisualProcessBench single-image questions (2,856) split per source with seed 0 into 400 dev (decisions) and **2,456 test (all numbers here)**: DynaMath 490, MathVerse_MINI_Vision_Only 883, MMMU_DEV_VAL 221, MathVision_MINI 612, WeMath 250 (`grpo_arms/DATA_SPLITS.md`). Greedy, 2,048 tokens, VPB prompt; answer extracted by the reward's cascade (`Final answer:` → `\boxed{}` → `<answer>` → last line), exact / letter / numeric-tolerance match.
- **Statistics**: paired exact McNemar on the discordant pairs (b = row right & reference wrong, c = the reverse). One binomial SE at these accuracies is ≈ 0.0095.
- **OOD**: seven external benchmarks (§5), same decoding and rule-based extraction (no VLMEvalKit on Sol).

### 1.6 Lineage (why the optimizer above)
With lr 1e-5 / KL 0.001 / no entropy bonus (v2, v3; September 11–12) every arm collapsed below cs25 on vpb_test — reward_v2 at 183 steps 0.2406, answer-only control 0.1922 @60 / 0.1539 @90, match without std-normalisation 0.2423 @60, and the answer-only control from the base model 0.1832 @60 — so the degradation was optimizer-driven, not reward-driven. The v4 pair (A, B; September 13) at lr 1e-6 / KL 0.01 / clip 0.2–0.28 fixed it and is the configuration continued here.

---

## 2. Epoch 1 (steps 0–183), vpb_test

| model | step | accuracy | Δ vs cs25 | Δ vs base3b | truncated | McNemar vs cs25 | McNemar vs v4 @60 (B for M, A for R) | McNemar Bipartite vs RLVR, same step |
|---|---|---|---|---|---|---|---|---|
| base3b | – | 0.3025 | −0.0167 | – | 0.032 | b=228 c=269 p=0.073 | – | – |
| **cs25** | 25 | **0.3192** | – | +0.0167 | 0.039 | – | – | – |
| RLVR A (v4, no entropy) | 30 | 0.3384 | +0.0191 | +0.0358 | 0.023 | p=0.041 | – | – |
| RLVR A (v4, no entropy) | 60 | 0.3282 | +0.0090 | +0.0257 | 0.020 | p=0.347 | – | – |
| Bipartite B (v4) | 30 | 0.3176 | −0.0016 | +0.0151 | 0.021 | p=0.887 | – | – |
| Bipartite B (v4) | 60 | 0.3436 | +0.0244 | +0.0411 | 0.013 | p=0.011 | – | – |
| **Bipartite M** | 90 | 0.3335 | +0.0143 | +0.0309 | 0.027 | p=0.129 | p=0.263 | −0.0163, p=0.069 |
| **Bipartite M** | 120 | 0.3465 | +0.0273 | +0.0440 | 0.012 | p=0.003 | p=0.787 | −0.0029, p=0.777 |
| **Bipartite M** | 150 | 0.3599 | +0.0407 | +0.0574 | 0.015 | p<0.001 | p=0.078 | +0.0118, p=0.198 |
| **Bipartite M** | 183 | **0.3632** | **+0.0440** | +0.0607 | 0.031 | p<0.001 | p=0.036 | +0.0069, p=0.471 |
| **RLVR R** | 30 | 0.3265 | +0.0073 | +0.0240 | 0.030 | p=0.439 | p=0.892 | |
| **RLVR R** | 60 | 0.3359 | +0.0167 | +0.0334 | 0.026 | p=0.070 | p=0.422 | |
| **RLVR R** | 90 | 0.3498 | +0.0305 | +0.0472 | 0.019 | p=0.001 | p=0.020 | |
| **RLVR R** | 120 | 0.3493 | +0.0301 | +0.0468 | 0.022 | p=0.001 | p=0.020 | |
| **RLVR R** | 150 | 0.3481 | +0.0289 | +0.0456 | 0.021 | p=0.002 | p=0.031 | |
| **RLVR R** | 183 | **0.3563** | **+0.0371** | +0.0537 | 0.030 | p<0.001 | p=0.003 | |

Per-source accuracy at the end of epoch 1:

| model | step | DynaMath (490) | MMMU (221) | MathVerse VO (883) | MathVision (612) | WeMath (250) |
|---|---|---|---|---|---|---|
| base3b | – | 0.4490 | 0.3529 | 0.2265 | 0.2059 | 0.4760 |
| cs25 | 25 | 0.4571 | 0.3982 | 0.2356 | 0.2075 | 0.5480 |
| Bipartite M | 183 | 0.5163 | 0.4027 | 0.2786 | 0.2500 | 0.6040 |
| RLVR R | 183 | 0.4592 | 0.4163 | 0.2820 | 0.2761 | 0.5600 |

Training dynamics (train-val = the 120 held-out VisualPRM prompts, gated answer accuracy; match/answer from the rollout dumps; KL = actor KL loss to the cs25 reference):

| run | step | train-val acc | vpb_dev | match | answer | entropy | KL | resp. tokens |
|---|---|---|---|---|---|---|---|---|
| Bipartite | 0 | 0.673 | 0.345 | – | – | – | – | – |
| Bipartite | 60 | 0.722 | 0.318 | 0.313 | 0.681 | 0.453 | 0.028 | 295 |
| Bipartite | 120 | 0.750 | 0.355 | 0.341 | 0.694 | 0.563 | 0.038 | 375 |
| Bipartite | 183 | 0.759 | 0.380 | 0.434 | 0.787 | 0.553 | 0.057 | 345 |
| RLVR | 0 | 0.656 | 0.315 | – | – | – | – | – |
| RLVR | 60 | 0.770 | 0.333 | n/a | 0.656 | 0.641 | 0.018 | 330 |
| RLVR | 120 | 0.815 | 0.362 | n/a | 0.731 | 0.643 | 0.026 | 400 |
| RLVR | 183 | 0.806 | 0.380 | n/a | 0.869 | 0.695 | 0.034 | 286 |

Bipartite's entropy fell to 0.39 once (step 146) and averaged 0.54 over the last 30 steps; RLVR settled at 0.65–0.9 with the entropy controller almost never engaging. No truncation-driven length blow-up in either run (truncation rate 0.000 at every logged step after step 20).

**Epoch-1 read.** Both rewards beat cs25 by a wide, significant margin (+4.4 and +3.7 points, p < 0.001) and rise monotonically to step 183. They are not separable from each other: +0.7 points for Bipartite at 183, p = 0.47, and no step where either leads at p < 0.05.

---

## 3. Epoch 2 (steps 184–366), vpb_test

Same 5,880 prompts, reshuffled (seed 43); optimizer state carried over; no configuration change. M2 was stopped at step 330 (its last checkpoint) when the account's fairshare ran out; R2 completed.

| model | step | accuracy | Δ vs cs25 | truncated | McNemar vs own step 183 | McNemar M2 vs R2 |
|---|---|---|---|---|---|---|
| Bipartite M (ref.) | 183 | 0.3632 | +0.0440 | 0.031 | – | +0.0069, p=0.471 |
| Bipartite M2 | 210 | 0.3612 | +0.0419 | 0.013 | −0.0020, p=0.850 | +0.0130, p=0.162 |
| Bipartite M2 | 240 | **0.3660** | +0.0468 | 0.035 | +0.0028, p=0.781 | +0.0037, p=0.719 |
| Bipartite M2 | 270 | 0.3644 | +0.0452 | 0.070 | +0.0012, p=0.925 | +0.0037, p=0.726 |
| Bipartite M2 | 300 | 0.3392 | +0.0200 | 0.130 | −0.0240, p=0.007 | −0.0098, p=0.295 |
| Bipartite M2 | 330 | 0.3160 | −0.0033 | 0.167 | −0.0472, p<0.001 | −0.0590, p<0.001 |
| RLVR R (ref.) | 183 | 0.3563 | +0.0371 | 0.030 | – | |
| RLVR R2 | 210 | 0.3481 | +0.0289 | 0.013 | −0.0081, p=0.362 | |
| RLVR R2 | 240 | 0.3624 | +0.0432 | 0.019 | +0.0061, p=0.513 | |
| RLVR R2 | 270 | 0.3607 | +0.0415 | 0.022 | +0.0045, p=0.651 | |
| RLVR R2 | 300 | 0.3489 | +0.0297 | 0.025 | −0.0073, p=0.435 | |
| RLVR R2 | 330 | **0.3750** | +0.0558 | 0.024 | +0.0187, p=0.039 | |
| RLVR R2 | 360 | 0.3648 | +0.0456 | 0.021 | +0.0085, p=0.356 | |
| RLVR R2 | 366 | 0.3502 | +0.0309 | 0.026 | −0.0061, p=0.523 | |

Per-source accuracy, best and last epoch-2 checkpoints:

| model | step | DynaMath | MMMU | MathVerse VO | MathVision | WeMath |
|---|---|---|---|---|---|---|
| Bipartite M2 | 240 | 0.5061 | 0.3801 | 0.2797 | 0.2582 | 0.6480 |
| Bipartite M2 | 330 | 0.4245 | 0.2986 | 0.2616 | 0.2075 | 0.5760 |
| RLVR R2 | 330 | 0.4939 | 0.3982 | 0.2990 | 0.3007 | 0.5720 |
| RLVR R2 | 366 | 0.4837 | 0.3710 | 0.2967 | 0.2206 | 0.5760 |

Training dynamics in epoch 2:

| run | step | train-val acc | vpb_dev | match | answer | entropy | KL | resp. tokens |
|---|---|---|---|---|---|---|---|---|
| Bipartite M2 | 240 | 0.820 | 0.372 | 0.389 | 0.738 | 0.557 | 0.067 | 382 |
| Bipartite M2 | 270 | 0.792 | 0.343 | 0.412 | 0.750 | 0.665 | 0.079 | 458 |
| Bipartite M2 | 300 | 0.725 | 0.345 | 0.449 | 0.775 | 1.023 | 0.100 | 468 |
| Bipartite M2 | 330 | 0.738 | 0.287 | 0.482 | 0.775 | 1.351 | 0.125 | 383 |
| RLVR R2 | 240 | 0.771 | 0.318 | n/a | 0.725 | 0.718 | 0.043 | 282 |
| RLVR R2 | 300 | 0.767 | 0.328 | n/a | 0.731 | 0.908 | 0.041 | 330 |
| RLVR R2 | 330 | 0.820 | 0.335 | n/a | 0.762 | 0.991 | 0.051 | 280 |
| RLVR R2 | 366 | 0.838 | 0.345 | n/a | 0.800 | 0.633 | 0.050 | 336 |

**Epoch-2 read.** RLVR plateaus: every R2 checkpoint is within noise of R@183 except step 330, the best checkpoint of the whole campaign (0.3750, +0.019 vs R@183, p = 0.039), and it ends at 366 within one SE of where it started. Bipartite peaks at step 240 (0.3660, the best match checkpoint) and then degrades: from step 270 on, the match term keeps rising (0.41 → 0.48) while entropy (0.67 → 1.35) and KL to the reference (0.079 → 0.125) climb, test-set truncation grows from 3.5 % to 16.7 %, and held-out accuracy falls back to the cs25 level (0.3160 at 330, −0.047 vs M@183, p < 0.001; −0.059 vs R2 at the same step). The training-side answer accuracy never shows this; only vpb_dev (0.372 → 0.287) and the test set do.

---

## 4. Out-of-distribution evaluation (M@183 and R@183)

Seven external benchmarks, greedy, 2,048 tokens, the VPB prompt template, rule-based extraction with each benchmark's answer normalisation. MMMU uses image_1 only (43/900 rows have more images). MathVista and MMK12 share no source with vpb_test; the other five overlap the VPB sources.

| benchmark | n | base3b | cs25 | Bipartite M | RLVR R | Δ(M−R) | Δ(M−cs25) | Δ(R−cs25) | McNemar M vs R |
|---|---|---|---|---|---|---|---|---|---|
| MathVista testmini (disjoint) | 1,000 | 0.5620 | 0.5470 | 0.6150 | 0.6300 | −0.0150 | +0.0680 | +0.0830 | p=0.328 |
| MMK12, 1,024 balanced (disjoint) | 1,024 | 0.3975 | 0.3955 | 0.4561 | 0.4893 | −0.0332 | +0.0605 | +0.0938 | p=0.054 |
| MathVerse testmini, Vision-Only + Vision-Dominant | 1,576 | 0.2532 | 0.2963 | 0.3230 | 0.3274 | −0.0044 | +0.0266 | +0.0311 | p=0.726 |
| ↳ Vision-Only | 788 | 0.2398 | 0.2868 | 0.3274 | 0.3160 | +0.0114 | +0.0406 | +0.0292 | p=0.510 |
| ↳ Vision-Dominant | 788 | 0.2665 | 0.3058 | 0.3185 | 0.3388 | −0.0203 | +0.0127 | +0.0330 | p=0.214 |
| MathVision test (full) | 3,040 | 0.1970 | 0.2003 | 0.2299 | 0.2391 | −0.0092 | +0.0296 | +0.0388 | p=0.302 |
| WeMath testmini | 1,740 | 0.5080 | 0.5500 | 0.6362 | 0.6333 | +0.0029 | +0.0862 | +0.0833 | p=0.849 |
| DynaMath, 10 variants × 501 seeds | 5,010 | 0.4192 | 0.4641 | 0.5050 | 0.5040 | +0.0010 | +0.0409 | +0.0399 | p=0.894 |
| ↳ per-seed worst case (all 10 right) | 501 | 0.1118 | 0.1537 | 0.2016 | 0.1896 | +0.0120 | +0.0479 | +0.0359 | p=0.461 |
| MMMU validation | 900 | 0.4411 | 0.4567 | 0.4911 | 0.4911 | 0.0000 | +0.0344 | +0.0344 | p=1.000 |

Pooled over all 14,290 items: Bipartite 0.4457 vs RLVR 0.4509, McNemar b=1473 c=1547, **p = 0.184**; with DynaMath collapsed to per-seed worst case (n = 9,781): 0.4028 vs 0.4103, p = 0.122. Both beat cs25 and base3b at p < 0.001 pooled. MMK12 by subject (256 each): math 0.531 vs 0.551, physics 0.367 vs 0.449, chemistry 0.457 vs 0.453, biology 0.469 vs 0.504 (Bipartite vs RLVR).

---

## 5. What the training runs show

1. Both rewards work as RL signals from cs25: +3.7 to +4.4 points in-distribution and +2.7 to +9.4 points on every external benchmark, all significant.
2. On final-answer accuracy the bipartite match reward does not beat answer-only RLVR anywhere: tie in-distribution (p = 0.47 at 183), tie pooled OOD (p = 0.18, slightly favouring RLVR), and RLVR ahead on the fully disjoint MMK12 at p = 0.054.
3. With continued training the match term is the destabiliser. In the second pass over the same prompts, Bipartite over-optimises match (rising match with rising entropy, KL and output length) while test accuracy falls below cs25; RLVR stays flat. The best checkpoints are Bipartite@240 (0.3660) and RLVR@330 (0.3750).
4. What is *not* measured: whether the bipartite reward improves the reasoning itself. VisualProcessBench has step-level labels, and this campaign used it only for final answers.
5. The most likely mechanism behind (2) and (3) is the judge: §6 shows the deployed NLI model credits a step whose number is wrong two times in three, so the match term can be raised without the reasoning becoming more correct.

---

## 6. Judge benchmark: current NLI model vs DeBERTa-v3

Full report: `reward_redesign/JUDGE_BENCH.md` (job 63722928, 2026-09-20; 64 work units, one A100, 2.5 GPU-hours plus the 4-hour first job).

### 6.1 The two judges
| | current | candidate |
|---|---|---|
| model | `microsoft/deberta-xlarge-mnli` (the model inside the reward today) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| scoring | identical for both: s = clip[0,1](0.5·E(a→b) + 0.5·E(b→a) − max C), 512-token truncation, batch 32, raw 3-class probabilities kept | |

### 6.2 Datasets
- **3a Hard negatives** — `data/visualprm400k/pairs.jsonl` (40,000 rows, built 2026-06-25 from the VisualPRM400K math subset): for each correct solution (`pos`) the negative (`neg`) is byte-identical except that the first number on the right-hand side of an equation is bumped by +1 (e.g. `∠AOB = 76°` → `77°`; the later lines still use the original value); `baseline` is an unrelated solution. Benchmark sample: 4,999 rows, seed 0, stratified by source file; scored at **step level** on the one step that differs (gold = contradict), plus that step against a random step of `baseline` (gold = unrelated). Edited steps are categorised by the edited line: *bare equation* (≤ 2 words; n = 377), *prose with a number* (n = 2,236), *mixed* (n = 2,386).
- **3b Paraphrases** — `data/visualprm400k/paraphrase_pairs.jsonl` (10,000 rows, 2026-06-29): an LLM rewording of `pos` under the instruction to keep every number, variable, equation and the final answer identical; a paraphrase is kept as *faithful* only if the multiset of numeric tokens equals the original's (9,254 of 10,000). Sample: 5,000 faithful rows, seed 0, scored whole-solution vs whole-solution (gold = entail).
- **3c Through the reward** — the same 3a/3b rows run through `reward_v2.score_new`'s actual Hungarian matching (whole steps): pos→neg, pos→baseline, pos→paraphrase.
- **3d EQUATE** (Ravichander et al., 2019): RTE-Quant 166, NewsNLI 968, RedditNLI 250, AWP-NLI 722, Stress Test 7,596 pairs (9,702 total, all five subsets), whole-pair, forward direction for the 3-class accuracy.
- **3e** — 300 real (rollout step, reference step) pairs from run M's rollout dumps, 50 per cell of {steps 1–60, 61–120, 121–183} × {correct, wrong rollout}, dumped with each config's score and a blank label field for hand labelling (`reward_redesign/judge_pairs_300.jsonl` / `.md`).

### 6.3 Segmentation: steps and clauses
- **Steps** (the reward's unit, and 3a's unit): strip `<think>`/`</think>`, split on blank lines, drop answer lines (`Final answer:` / `Answer:` / a terminal `\boxed{}`).
- **Clauses** (the June splitter, reused verbatim from `score_negation_clausesplit.py`): split the text on newlines, then split each line **after sentence punctuation — a period, semicolon or colon followed by whitespace** (`re.split(r"(?<=[.;:])\s+", line)`); empty pieces dropped; at most 40 clauses. It is purely punctuation-based, no parser; an equation on its own line is one clause, and "Given AB = 14, and because …" stays one clause because a comma does not split. (The ScienceQA-era SBERT reward used a spaCy dependency parse instead; the reward on VisualPRM uses whole steps.)
- **Alignment in the clause configs**: *clause-min* aligns the two clause lists with SBERT `all-mpnet-base-v2` cosine + Hungarian assignment (τ = 0.15), scores each aligned pair with NLI, and takes the **minimum** s; *clause-hung-F05 / -mean* take the SBERT top-3 candidate partners per clause (both directions), score those with NLI, Hungarian-assign on the sparse s matrix, and report F₀.₅ of clause precision/recall or the mean s over assigned pairs; *asym-mean* scores the whole step a against each clause of b (and the mirror), forward direction only, and averages. Pairs where both sides are a single clause fall back to `whole` (2.5 % of 3a pairs).

### 6.4 Results

**3a — hard negatives, step level** (FPR = fraction of corrupted steps still scored as matching, s ≥ 0.5; lower is better):

| judge | config | FPR | mean s (edited) | mean s (unrelated step) | bare equation | prose w/ number | mixed | NLI calls / pair |
|---|---|---|---|---|---|---|---|---|
| current | whole step | **0.684** | 0.621 | 0.267 | 0.281 | 0.711 | 0.723 | 2.0 |
| current | clause-min | **0.008** | 0.012 | 0.003 | 0.019 | 0.005 | 0.010 | 38.1 |
| current | clause-hung-F05 | 0.941 | 0.809 | 0.149 | 0.618 | 0.954 | 0.979 | 146.4 |
| current | clause-hung-mean | 0.975 | 0.905 | 0.410 | 0.825 | 0.975 | 1.000 | 146.4 |
| current | asym-mean | 0.121 | 0.212 | 0.044 | 0.064 | 0.130 | 0.123 | 38.1 |
| v3 | whole step | **0.017** | 0.016 | 0.487 | 0.005 | 0.013 | 0.022 | 2.0 |
| v3 | clause-min | **0.000** | 0.001 | 0.003 | 0.003 | 0.000 | 0.000 | 38.1 |
| v3 | clause-hung-F05 | 0.918 | 0.743 | 0.130 | 0.501 | 0.945 | 0.959 | 146.4 |
| v3 | clause-hung-mean | 0.955 | 0.829 | 0.321 | 0.645 | 0.970 | 0.991 | 146.4 |
| v3 | asym-mean | 0.013 | 0.020 | 0.035 | 0.000 | 0.014 | 0.013 | 38.1 |

June reference on the same data (paragraph level, older formula, 256-token truncation): paragraph NLI FPR@0.5 = 0.594; clause-split min-pool FPR@0.5 = 0.015.

**3b — paraphrases, whole solution** (FNR = fraction of faithful paraphrases rejected, s < 0.5; lower is better):

| judge | config | FNR | mean s | NLI calls / pair |
|---|---|---|---|---|
| current | whole | **0.001** | 0.970 | 2.0 |
| current | clause-min | 0.084 | 0.758 | 53.1 |
| current | clause-hung-F05 | 0.000 | 0.896 | 209.5 |
| current | clause-hung-mean | 0.000 | 0.930 | 209.5 |
| current | asym-mean | 0.715 | 0.380 | 53.5 |
| v3 | whole | **0.099** | 0.810 | 2.0 |
| v3 | clause-min | 0.630 | 0.358 | 53.1 |
| v3 | clause-hung-F05 | 0.002 | 0.825 | 209.5 |
| v3 | clause-hung-mean | 0.000 | 0.855 | 209.5 |
| v3 | asym-mean | 0.737 | 0.264 | 53.5 |

**3c — inside the real reward** (Hungarian matching over whole steps, 4,999 / 5,000 solutions):

| judge | mean match pos→neg ↓ | mean match pos→unrelated ↓ | mean match pos→paraphrase ↑ | corrupted step credited (s ≥ 0.5) ↓ |
|---|---|---|---|---|
| current | 0.672 | 0.199 | 0.839 | **0.578** |
| v3 | 0.183 | 0.396 | 0.761 | **0.055** |

**3d — EQUATE** (whole pair; accuracy = argmax of the forward probabilities; AUROC of s for entail vs non-entail; FNR = gold-entail with s < 0.5):

| subset | n | current acc | current AUROC | current FNR | v3 acc | v3 AUROC | v3 FNR |
|---|---|---|---|---|---|---|---|
| RTE-Quant | 166 | 0.614 | 0.858 | 0.686 | 0.747 | 0.931 | 0.271 |
| NewsNLI | 968 | 0.728 | 0.853 | 0.615 | 0.751 | 0.835 | 0.399 |
| RedditNLI | 250 | 0.636 | 0.808 | 0.459 | 0.692 | 0.827 | 0.500 |
| AWP-NLI | 722 | 0.231 | 0.499 | 0.997 | 0.313 | 0.562 | 0.964 |
| Stress Test | 7,596 | 0.800 | 0.633 | 0.735 | 0.867 | 0.709 | 0.850 |

**Throughput** (one A100, batch 32, NLI forward passes per second, measured over the full runs):

| pair type | current | v3 |
|---|---|---|
| 3a step pairs (1.20 M calls) | 137 | 472 |
| 3b whole solutions (0.50 M calls) | 102 | 344 |
| 3c reward matching, neg + unrelated (0.14 M calls) | 88 | 254 |
| 3d EQUATE sentences | 426 | 942 |

### 6.5 Read
- **The judge inside today's reward passes 68 % of one-number corruptions**, and 71 % when the number sits inside a prose step; through the real matching the corrupted step still earns credit 58 % of the time. That is consistent with the epoch-2 behaviour in §3: match can rise without the reasoning getting more correct.
- **Two fixes work, with different costs.** *Model change*: DeBERTa-v3 with the existing whole-step config cuts pass-through to 1.7 % (5.5 % inside the reward) at the same 2 NLI calls per pair and 3.4× the speed, at the cost of rejecting 9.9 % of faithful paraphrases and scoring unrelated steps higher (mean s 0.49 vs 0.27; unrelated whole solutions get match 0.40 vs 0.20), i.e. it is sharper on numbers and looser on topic. *Pooling change*: the current judge with clause-split min-pooling reaches 0.8 % pass-through and 8.4 % paraphrase rejection but needs 38 NLI calls per pair, 19× the reward's current NLI load. Combining both (v3 + clause-min) over-rejects (63 %).
- By the brief's rule (FPR < 10 %, then lowest paraphrase FNR, throughput as tie-break) the formal winner is **current judge + clause-min** (0.008 / 0.084), ahead of v3 + whole (0.017 / 0.099) by 1.5 points of paraphrase FNR; on cost and the reward-level numbers v3 + whole is the practical candidate. Either is a candidate for later arms only after a probe rerun; no training run in this report used anything but the current judge with whole steps.
- **Neither judge does arithmetic**: both are at chance on EQUATE's arithmetic word problems (AUROC 0.50 / 0.56). v3 notices when a stated number changes; it cannot verify a computation.

---

## 7. Compute
Epoch 1 and its evaluations on grp_vgupt140 (RawUsage_CHE 3,832 → 6,346); epoch 2, OOD and the judge benchmark on grp_bshettah (1,518 → 5,067). Training pace: Bipartite ≈ 4.7–4.9 min/step, RLVR ≈ 2.4–2.5 min/step on 2×A100.

## 8. Known gaps
- M2 steps 360/366 never ran (cancelled at 330 for fairshare); the 330 checkpoint is already below cs25, so the verdict would not change.
- Single seed per arm; the Bipartite–RLVR differences are within noise in both directions.
- No step-level (process) evaluation of the trained policies.

## 9. Sources
`grpo_arms/FULL_REPORT.md` (epoch 1), `grpo_arms/E2_REPORT.md` (epoch 2), `grpo_arms/OOD_REPORT.md`, `grpo_arms/V4_REPORT.md` (A/B), `grpo_arms/V3_REPORT.md` (lineage), `reward_redesign/JUDGE_BENCH.md`, `grpo_arms/DATA_SPLITS.md`, `grpo_arms/data/subset_manifest.json`, `reward_redesign/reward_v2.py`, `score_negation_clausesplit.py`, `build_negation_pairs.py`, `build_paraphrase_pairs.py`, `reward_redesign/judge_bench/jb_lib.py`; status logs `grpo_arms/FULL_STATUS.md`, `grpo_arms/E2_STATUS.md`. Per-checkpoint generations: `grpo_arms/evals/vpb_gen_*_2ktest.jsonl`, `grpo_arms/evals/ood/`.
