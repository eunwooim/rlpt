# GRPO results — ScienceQA (in-domain) and MMK12 (OOD)

*RLPT image-description reward track · Qwen2.5-VL-3B-Instruct · GRPO (verl) · ASU Sol.*

**Setup.** Base model **Qwen2.5-VL-3B-Instruct**, trained with GRPO (verl 0.8.0 / vLLM
0.10.1.1, `rollout.n=5` = GRPO group, KL 0.001) on the image+solution subset of ScienceQA
(`derek-thomas/ScienceQA`, 5,678 train rows). Reward = composite
`R = w_f·format + w_m·match + w_a·answer − w_p·pun` (weights 1 / 2 / 5 / 1). The model emits
`<think>` reasoning + `<answer>` choice; `match` is the SBERT bipartite alignment of the
reasoning to the GT `solution`. **Two training arms, identical config/seed/steps:** the **main
arm** (full reward) and a **`w_match=0` ablation** (answer-only; `RLPT_W_MATCH=0` and
`RLPT_W_PUN=0`). Full run = 2 epochs / 176 steps.

---

## 1. ScienceQA — held-out test (in-domain)

Final checkpoint `global_step_176`, evaluated on the **1,836-row** held-out image+solution test
split (`eval_test_qwen2_5vl.sbatch`, val_only):

| metric | value |
|---|---|
| **answer accuracy** | **90.3 %** |
| format validity | 100 % |
| reasoning-match F (vs GT solution) | 0.778 |
| hallucinated clauses / response | 0.091 |
| composite reward | 6.98 |

**Interpretation.** The base model is **79 % zero-shot** on ScienceQA (supervisor-measured,
lenient parse), so RL delivered **+11.3 pts accuracy (79 % → 90.3 %)** on an *already-competent*
model, plus near-zero hallucination and perfect format. The often-quoted "−0.29 composite
baseline" is **not** the base model being bad — it is that the untrained model doesn't emit our
required `<think>/<answer>` format, so the strict harness can't parse its answer. RL's value is
accuracy **and** reasoning faithfulness, not teaching a weak model from scratch.

### 1.1 Learning curve — the decomposition is the real finding

Checkpoint sweep on a fixed 256-row test subset (`match_sweep_*.tsv`). Base row is the untrained
model on the same subset; greedy/strict harness deflates its accuracy vs the 79 % lenient number.

| step | accuracy | reasoning match | format | reward |
|---|---|---|---|---|
| base | 0.645 | 0.371 | 0.480 | −0.31 |
| 20 | 0.824 | 0.625 | 1.000 | 6.30 |
| 40 | 0.848 | 0.682 | 1.000 | 6.53 |
| 60 | 0.859 | 0.705 | 1.000 | 6.64 |
| 80 | 0.883 | 0.723 | 1.000 | 6.75 |
| 100 | 0.910 | 0.737 | 0.996 | 6.50 |
| 120 | 0.879 | 0.741 | 1.000 | 6.74 |
| 140 | 0.902 | 0.751 | 1.000 | 6.92 |
| 160 | 0.891 | 0.752 | 1.000 | 6.90 |
| 176 | 0.891 | 0.766 | 1.000 | 6.91 |

Three components move on different timescales:

- **format saturates immediately** — 1.00 by step 20 and stays there.
- **accuracy is mostly there early** — ~0.82 by step 20, drifting to ~0.89–0.90 (noisy).
- **reasoning match climbs steadily and monotonically — 0.371 → 0.625 (step 20) → 0.766 (176)**,
  roughly *doubling*. RL kept refining *reasoning faithfulness* (alignment to GT solutions)
  long after format and answers were largely solved — exactly the signal the `match` term was
  designed to capture.

So on ScienceQA the headline accuracy understates what training did: the differentiated signal
is in the reasoning channel, which the answer-only metric cannot see.

---

## 2. MMK12 — out-of-distribution hard benchmark

ScienceQA accuracy is near a **ceiling** (base already 62–79 %), which masks whether the `match`
term *causes* accuracy gains. MMK12 (`FanqingM/MMK12`, MM-Eureka; 2,000 K12 exam MCQs — 500 each
math/physics/chem/bio, single image, letter answers) is much harder (base ≈ chance) and the
model was **never trained on it**. Balanced 1,024-row sample (`validation_mmk12.tsv`):

| model | accuracy | format | biology | chemistry | math | physics |
|---|---|---|---|---|---|---|
| base (zero-shot) | 0.311 | 0.18 | 0.336 | 0.270 | 0.363 | 0.277 |
| ablation-176 (answer-only) | 0.397 | 0.99 | 0.465 | 0.332 | 0.465 | 0.328 |
| **main-176 (with `match`)** | **0.439** | 0.99 | **0.535** | 0.336 | **0.508** | **0.375** |

**The match term buys +4.1 pts accuracy at identical format** (main 0.439 vs ablation 0.397).
Paired **McNemar exact p = 0.021** (discordant: 179 main-only-correct vs 137
ablation-only-correct) — the gain is statistically significant, not noise. The per-subject split
shows it concentrates where reasoning matters most: biology +7.0, math +4.3, physics +4.7
(chemistry flat).

**Why this matters.** On ScienceQA the two arms tie on accuracy (the term buys faithfulness, not
accuracy — a ceiling effect). On MMK12, where **base 31 % ≈ near chance** and reasoning is the
bottleneck, *training the reasoning channel converts directly into answer accuracy*. The
"no-accuracy-difference" caveat from ScienceQA was a property of the ceiling, not of the term.

---

## 3. Headline

| claim | evidence |
|---|---|
| RL lifts an already-strong model | ScienceQA 79 % → **90.3 %** (+11.3 pts), format 100 %, hallucination 0.09/resp |
| The gain is in reasoning, not just answers | match **0.37 → 0.77** monotone while format/accuracy saturate early |
| The match term *causes* OOD accuracy | MMK12 main **0.439** vs answer-only ablation 0.397, **McNemar p = 0.021** |

The match term's value is **reasoning faithfulness in-domain** (ScienceQA) that **converts to
answer accuracy out-of-domain** (MMK12), on top of an already-competent base model.

*Sources: `data/logs/validation_results/validation_mmk12.tsv`, `data/logs/match_sweep_results.tsv`,
`data/logs/match_sweep_anchors.tsv`, held-out test log `eval-test` (step 176). Mirrored in
`docs/results/`.*
