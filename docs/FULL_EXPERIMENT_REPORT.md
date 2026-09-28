# RLPT — Complete Experiment Report (2026-05 → 2026-08-22)

Every experiment conducted in this project, from the first reward design to the
six-arm GRPO campaign currently training. Compiled 2026-08-22 from the primary
reports in `docs/`, `chunker/`, `chunk_eval/`, `chunk_canonical/`, `granularity/`,
`data/visualprm_v11_filtered/`, and `grpo_arms/`; each section names its sources.

**Project frame.** RL post-training (GRPO) of Qwen2.5-VL with a reward that scores
the *content of free-text reasoning*, not only the final answer. Stack: verl 0.8.0 /
vLLM 0.10.1.1 / torch 2.7.1+cu126 / transformers 4.56.2 on ASU Sol (A100-80GB,
SLURM, RHEL8 glibc 2.28, BeeGFS /scratch).

**Timeline.**
| period | work |
|---|---|
| late May – 2026-06-05 | Phase 1: reward design evolution (QCVSR → parser → bipartite composite) |
| 06-06 – 06-08 | Phase 2: Sol training infrastructure (8 fixes) |
| 06-08 – 06-09 | Phase 3: ScienceQA GRPO run (176 steps) |
| 06-10 – 06-12 | Phase 4: mimicry-vs-reasoning validation campaign + MMK12 |
| 06-17 – 07-02 | Reward-scorer reliability suite (DCI, SugarCrepe/NegBench, negation, COCO-IoU) |
| 07-10 – 07-19 | VisualPRM400K: noise audit, two-stage filtering, SymPy audit; supervisor baselines + 2×2 |
| 07-20 – 08-10 | DeBERTa chunker: train, evaluate, v2-mask retrain, canonical production chunking |
| 08-13 – 08-14 | Granularity: reference bands + Qwen controllability experiment |
| 08-22 – | Six-arm GRPO campaign (gate 1 done, arm 1 running) |

---

# 1. Phase 1 — Reward design evolution (→ 2026-06-05)

## 1a. QCVSR: typed schema + deterministic verifier (dropped)
Model emits a typed evidence object (bounding-box/count pairs) plus an answer; a
deterministic verifier checks schema, answer, and evidence↔answer consistency
(dual-reward). Robust but narrow — not free-text reasoning. Surviving idea: **a hard
verifiable term alongside a soft term**. Data: `data/qcvsr_gqa_v1/`.

## 1b. Caption → frozen parser → scene-graph match (dropped)
**Parser benchmark (2026-06-02)**: `lizhuang144/flan-t5-base-VG-factual-sg` vs the
large variant. Clean FACTUAL gold tuple-F1: base **0.943** vs large **0.963**; on
noisy VG human-gold graphs **both collapse to ≈0.59** and large ≈ base, at ~2×
latency → **base chosen**. Whole track then dropped: a parser in the reward loop is
a reward-hacking magnet (RL learns parser-friendly phrasing instead of seeing
better). Writeup `data/visual_genome/PARSER_EVAL.md`; tooling kept as legacy.

## 1c. Free-text bipartite match reward (FINAL; `tools/graph_match_reward.py`)
Pipeline: (1) clause-split both sides (deterministic spaCy dependency parse);
(2) embed clauses with SBERT `all-MiniLM-L6-v2` (GT side cached); (3) cosine matrix
→ **Hungarian one-to-one assignment** (`scipy.linear_sum_assignment`), pair counts
iff cosine ≥ **τ = 0.15** (one-to-one blocks "say the same fact five ways");
(4) precision-dominant **Fβ (β = 0.5)** from matched similarity mass M:
P = M/n_pred, R = M/n_GT, Fβ = 1.25·P·R/(0.25·P + R). Hallucination count
n_unmatched = predicted clauses with no GT partner above τ.

**SBERT blindness probe** (the finding that forced the composite): a prediction
stating the *opposite* of the truth matched at cosine **0.903**; under match-only,
confidently-wrong scored **0.41** vs **0.63** for a good answer. With the composite:
wrong-but-fluent **1.8** vs correct **7.3**.

**Composite reward (STAR-R1-shaped):**  **R = 1·format + 2·match + 5·answer − 1·pun**
| term | measures | weight |
|---|---|---|
| format | exactly one `<think>` and one `<answer>` block (hard gate) | ×1 |
| match | bipartite reasoning-match Fβ vs GT solution | ×2 |
| answer | hard MCQ correctness (text/letter/index) | ×5 |
| pun | −1 per hallucinated clause | −1 |

verl drop-in `compute_score(...)` returning `{score, acc, match, format,
n_hallucinated}`; weights env-overridable (`RLPT_W_*`) — that is how the causal
ablation ran with zero code change. SBERT runs as a subprocess server
(`tools/sbert_embed_server.py`) — models cannot load inside verl's reward worker.

## 1d. Dataset pivot
Synthetic GQA → **ScienceQA** image-bearing subset (5,678 train / 1,922 val /
1,836 test) because it has GT *reasoning text* (`lecture`/`solution`).
Converter `src/data_factory/scienceqa_to_verl.py`.

---

# 2. Phase 2 — Training infrastructure on Sol (2026-06-06 → 06-08)

Eight fixes, each found by smoke-testing on a 64-row parquet:
1. **Ray startup hang** (diagnosed with py-spy): verl default `ray_init.num_cpus=null`
   grabs all 48 node cores vs ~12 allocated → BeeGFS import storm, workers never
   register. Fix: `ray_kwargs.ray_init.num_cpus=<cpus-per-task>`.
2. Micro-batch-size configs must be set explicitly.
3. `unset ROCR_VISIBLE_DEVICES` (Sol's AMD var conflicts with CUDA).
4–5. flash-attn/vLLM version maze → final stack verl 0.8.0 + vLLM 0.10.1.1 +
   torch 2.7.1+cu126 + transformers 4.56.2, attn = SDPA.
6. SBERT can't load in verl's reward worker (meta-tensor state) → subprocess server.
7. No real flash-attn on glibc 2.28 → **pure-torch shim** (`src/train/flash_attn_shim/`,
   SDPA-backed, numerically verified).
8. Memory: FSDP CPU offload + `rollout.gpu_memory_utilization=0.4` + host 240G for
   the FSDP↔vLLM handoff on A100-80GB.

Ops conventions: always `--no-requeue`; smoke before full; login node is 1-core.

---

# 3. Phase 3 — ScienceQA GRPO run (2026-06-08 → 06-09)

**Config:** batch 64 questions/step × `rollout.n = 5` = 320 completions/step (group
mean = baseline, no critic); KL coef 0.001 vs frozen base; 88 steps/epoch × 2 =
**176 steps**; checkpoints every 20 (`models/qwen2_5vl-3b-scienceqa-grpo-step*`).

**Held-out test, step 176 (1,836 rows):** accuracy **90.3%** (base zero-shot 79% →
+11.3 pts), format **100%**, reasoning-match F **0.778**, hallucinated clauses
**0.091/response**, composite reward **6.98** (1.000 + 1.556 + 4.515 − 0.091).

**Learning-curve decomposition (fixed 256-row subset):**
| step | accuracy | match | format | reward |
|---|---|---|---|---|
| base | 0.645 | 0.371 | 0.480 | −0.31 |
| 20 | 0.824 | 0.625 | 1.000 | 6.30 |
| 40 | 0.848 | 0.682 | 1.000 | 6.53 |
| 80 | 0.883 | 0.723 | 1.000 | 6.75 |
| 120 | 0.879 | 0.741 | 1.000 | 6.74 |
| 176 | 0.891 | 0.766 | 1.000 | 6.91 |

Format saturates by step 20; accuracy mostly there early (~0.82); **match climbs
monotonically 0.371 → 0.625 → 0.766, roughly doubling long after the others
saturate** — the headline accuracy understates what training did. (90.3% vs 87.5%
reconciliation: full 1,836-row test vs the 256-row strict-greedy controlled subset.)

---

# 4. Phase 4 — Validation campaign: mimicry vs reasoning (2026-06-10 → 06-12)

Design principle: mimicry is style-specific + image-independent; real reasoning is
content-specific + image-dependent. Five controls, all PASS:

| control | method | numbers | conclusion |
|---|---|---|---|
| 1. Shuffled-GT | score `<think>` vs other same-topic GT solutions | own-GT 0.38→0.76; shuffled flat ≈0.22; gap widens 0.22→0.53 | content-specific, not style |
| 2. NLI re-scoring | DeBERTa-MNLI entailment on same generations | own-GT 0.21→0.52; shuffled flat ≈0.07 | propositional content; rules out encoder hacking |
| 3. Image-swap | eval every ckpt with swapped images | accuracy ≈ −0.21 everywhere; match swap-sensitivity grows −0.048 (base) → −0.079 (176) | RL *increased* image-grounding |
| 4. A-OKVQA transfer | zero-shot on differently-styled benchmark | base 0.43 → step-176 0.84 | gains transfer; style can't |
| 5. w_match=0 ablation | identical config/seed, reward = format+answer only | ablation match 0.38→0.46→flat 0.42 vs main 0.38→0.76; NLI never moves; grounding stays at base | everything beyond the step-20 bump is **caused by the match term** |

**MMK12 OOD accuracy test (2026-06-12).** 2,000 K12 exam MCQs (500 each
math/physics/chem/bio), never trained on; balanced 1,024-row eval:
| model | acc | format | bio | chem | math | phys |
|---|---|---|---|---|---|---|
| base | 0.311 | 0.18 | 0.336 | 0.270 | 0.363 | 0.277 |
| ablation-176 (no match) | 0.397 | 0.99 | 0.465 | 0.332 | 0.465 | 0.328 |
| **main-176 (with match)** | **0.439** | 0.99 | 0.535 | 0.336 | 0.508 | 0.375 |

**+4.1 pts from the match term** at identical format; McNemar exact **p = 0.021**
(179 main-only-correct vs 137 ablation-only). Where base ≈ chance and reasoning is
the bottleneck, training the reasoning channel converts into answer accuracy — the
ScienceQA "no accuracy difference" was a ceiling effect.

---

# 5. Reward-scorer reliability suite (2026-06-17 → 07-02)

Scorers under test (from `src/metrics/scorers/`): `sbert` =
`all-mpnet-base-v2` cosine; `nli` = `microsoft/deberta-xlarge-mnli` bidirectional,
composites `coverage = clamp01(0.7·E_ab + 0.3·E_ba − 0.5·max(C))` and
`equiv = clamp01(0.5·E_ab + 0.5·E_ba − 0.5·max(C))`; `bertscore` =
deberta-xlarge-mnli token P/R/F1. Metrics: separation mode FPR@τ on hard negatives
(lower better); ranking mode rank_acc (0.50 = chance).

## 5.1 DCI reward-encoder reliability (2026-06-17)
Testbed: facebookresearch/DCI — 7,805 images, 313,391 submasks, 135,024 hard
negatives each of `swaps` (near-verbatim attribute/entity swap), `layout`,
`basic`; all 7,805 SA-1B images matched, 0 missing. 1,500 units for bi-encoders,
300 for NLI; ≤3 paraphrases + 3 negatives/family per unit; seed 42.

**Test 1 — P(paraphrase > negative):**
| encoder | swaps | layout | basic |
|---|---|---|---|
| MiniLM-L6 (the reward encoder) | **0.183** | 0.519 | 0.573 |
| mpnet-base | 0.272 | 0.553 | 0.581 |
| NLI-DeBERTa-v3 (entail) | 0.381 | **0.920** | **0.830** |

**Test 4 — mean scores:** MiniLM scores the swap (0.881) *above* the faithful
paraphrase (0.758); FNR-vs-swaps 0.895. NLI over-penalizes detail-adding
paraphrases (mean entail 0.43, FNR 0.68).
**Test 2 — agreement:** MiniLM↔mpnet κ 0.617 / ρ 0.865 (paradigm property, not one
model); MiniLM↔NLI κ 0.141 (genuinely different paradigm; complementary failures).
**Embedding-inversion probe (vec2text, GTR-base stand-in):** short units near-
losslessly invertible — token-F1 **0.984**, ADJ/NUM/spatial recall
**0.983/0.972/0.980** → the encoder *retains* the content; cosine cannot *score*
it (geometry problem, not information bottleneck). Long captions degrade (F1 0.465).
Implication set: hybrid SBERT-coverage + NLI-faithfulness reward, multimodal
grounding term, keep clauses short.

## 5.2 Public benchmarks: SugarCrepe / SugarCrepe++ / NegBench (uncapped)
Reproduction of supervisor's harness `src/metrics/reliable/` (commit 0ad601c),
uncapped: 12,268 cases · 17,025 pairs · 51,075 scores (job ①, ~10 min); + NegBench
(job ② 56959065, ~25 min, text-only MCQ CSVs, 5 subcategories).

| dataset | mode | sbert | bertscore f1 | nli (best) |
|---|---|---|---|---|
| SugarCrepe | FPR@0.6 ↓ | 0.974 | 1.000 | **0.069** (coverage) |
| SugarCrepe++ | rank_acc ↑ | 0.725 | **0.140** (worse than chance) | **0.979** (equiv) |
| NegBench | FPR@0.6 ↓ | 0.753 | 0.997 | **0.0013** (coverage) |

Cosine and token-overlap react to word overlap, not meaning (hard negatives share
~95% of tokens); BERTScore actively prefers the corrupted caption. NLI both
separates and ranks. (Supervisor's cap-500 numbers reproduced within noise.)

## 5.3 Numerical sensitivity: AUC + τ sweep
Pure-CPU re-aggregation of 252,693 saved score rows (`sensitivity_analysis.py`).

**ROC-AUC:** nli coverage **0.981** (SugarCrepe++) / **0.994** (pooled match-
detector, 4,757 pos vs 79,474 neg); sbert 0.728 / 0.908; bertscore **0.178**
(inverted — anti-reliable) / 0.775. NLI contradiction alone: AUC 0.0115 → flipped
≈0.99, a near-perfect hard-negative detector.

**τ sweep:** NLI's FPR is stable across the whole 0.30–0.90 grid (SugarCrepe
0.134→0.040); sbert/bertscore need τ* ≈ 0.98–0.99 to cap FPR at 5%, where TPR
collapses (sbert TPR 0.80 at τ=0.90) — **no threshold rescues them**. NegBench NLI
τ*@FPR=1% = 0.000 (hard negatives all cluster at ~0 coverage).

## 5.4 VisualPRM math negation probe — paragraph level (N = 40,000)
Hard negative = correct solution with the first RHS numeric value +1 (single-char
edit, e.g. `∠B = 50°`→`51°`); BASE = unrelated-solution floor. VisualPRM400K v1
math subset (24 files).

| scorer | NEG mean sim | missed@τ=0.6 (FPR ↓) |
|---|---|---|
| SBERT cosine | 0.993 | **1.000** |
| BERTScore f1 | 0.981 | **1.000** |
| NLI equiv (reported) | 0.568 | **0.530** |
| NLI coverage | 0.544 | 0.504 |

Even NLI misses half the single-digit corruptions when buried in ~26-clause
solutions; cosine-family misses all. Worst NLI result across benchmarks:
NegBench 0.001 → SugarCrepe 0.069 → **VisualPRM 0.530** — the dilution effect.

## 5.5 Clause-split + min-pool — negation recall (N = 10,000)
Pipeline mirrors the match reward with NLI atoms: newline+sentence-punct clause
split (≤40 clauses) → SBERT-cosine Hungarian align (τ=0.15) → NLI equiv per
matched pair → **min**-pool (one contradicted clause vetoes). Mean matched
clauses/example 26.6.

| metric | paragraph equiv | clause-split min |
|---|---|---|
| FPR@0.6 on corrupted ↓ | 0.530 | **0.008** |
| mean equiv on corrupted | 0.569 | 0.015 |
| mean on unrelated floor | 0.101 | 0.000 |

Splitting removes the dilution: **66× reduction**. Residual = NLI's numeric-equality
blindness → case for a hard numeric-match term.

## 5.6 Paraphrase control — precision (N = 9,254)
Faithful paraphrases by Qwen2.5-VL-3B (keep every number/equation/answer exactly;
reword prose only); **number-multiset verified** (unfaithful excluded: 746/10,000).
Same pipeline. FPR = correct paraphrase wrongly scored below τ:

| metric | clause-split min |
|---|---|
| FPR@0.6 ↓ | **0.599** |
| FPR@0.5 ↓ | 0.226 |
| mean / median min-equiv | 0.594 / 0.578 |

**Hard min FAILS precision** — one "neutral" clause sinks the whole min. Verdict:
do not ship hard min; needs softened pool (k-th-lowest quantile) + numeric term.
The recall/precision tension: negation 0.008 vs paraphrase 0.599.

## 5.7 COCO bounding-box IoU sensitivity (N = 20,000 triples)
GT box → two jittered candidates; boxes as text `"[x1, y1, x2, y2]"` (0–1000
Qwen-VL convention); criterion: does the scorer rank the higher-IoU candidate
first? Mean IoU: high 0.432 / low 0.260 (margin 0.172).

| scorer | rank acc ↑ | margin | Spearman(sim, IoU) |
|---|---|---|---|
| IoU oracle | 1.000 | +0.172 | 1.000 |
| BERTScore f1 | 0.620 | +0.011 | 0.183 |
| SBERT | 0.577 | +0.015 | 0.026 |
| NLI equiv/coverage | 0.503 | +0.002 | 0.064 |

Text scorers are blind to spatial magnitude. NLI's ≈0 similarity is **confident
contradiction on any digit difference** (contradiction ≈0.93–0.99 for near-right
and wildly-wrong boxes alike), not neutrality. Quantitative case for an explicit
IoU term in any grounding reward.

## 5.8 Adversarial attack taxonomy (A / B / C)
| axis | attack | status | evidence |
|---|---|---|---|
| A-1 dilution (paragraph) | contradicting clause outvoted | MEASURED, FIXED by splitting | 0.530 → 0.008 |
| A-2 truncation / A-3 length-OOD | | fixed by splitting | — |
| B-1 clause fusion | withhold punctuation → dilution returns | SURVIVING, untested | probe queued |
| B-2 cosine misalignment | corrupted clause routed past its partner | SURVIVING, untested | probe queued |
| B-3 locally-true/globally-invalid chains | bipartite is a bag of clauses (permutation-invariant) | SURVIVING, untested | needs chain-inverting negatives |
| C-1 neutral trap | correct rewording marked neutral → min sinks it | MEASURED | paraphrase FPR 0.599 |
| C-2 numeric contradiction | no magnitude sense | MEASURED | COCO 0.503, contra ≈0.93–0.99 |
| C-3/4, C-5/7/8 | cue-free NLI-atom failures | untested | — |

Two root-cause families: **right-answers-failing** (C-1/5/7/8; fix = soften pool +
hard numeric/answer term) and **wrong-answers-passing** (B-1/2/3, C-3/4; fix =
probe sets + structural alignment). Queued: pooling sweep, SBERT/BERTScore
clause-split control (predicted no-rescue), B-probe sets, numeric-match term,
grounding term, hybrid-reward GRPO run.

---

# 6. VisualPRM400K: audit, filtering, symbolic verification (2026-07-10 → 07-19)

Dataset: `OpenGVLab/VisualPRM400K-v1.1-Raw` (snapshot 650e4b2f) —
`annotations.zip` 186 MB / 38 jsonl / **565,149 rows** of
`question/response/answer/steps_with_score` (per-step Monte-Carlo scores);
`images.zip` 21.55 GB / 214,557 images. Image mapping verified on all 565,149 rows
(0 missing; nlvr2 nests one level deeper). Distinct from the v1 zip used only for
the negation-pairs experiment.

## 6.1 Full-corpus noise audit (2026-07-11)
| statistic | count | % |
|---|---|---|
| response final answer contradicts `answer` | 58,802 | 10.4% |
| final answer matches | 180,447 | 31.9% |
| no "Final answer:" parsed (per-source templates) | 325,900 | 57.7% |
| all steps MC ≥ 0.5 | 457,644 | 81.0% |
| **CLEAN (answer-correct AND min step ≥ 0.5)** | **152,611** | **27.0%** |
| rows with gold `analysis` | 134,012 | 23.7% |

Implications: SFT on raw responses imitates 10.4% provably-wrong traces; the match
reward should weight/filter references by MC scores.

## 6.2 Two-stage MC + NLI filtering pipeline (2026-07-16)
Stage 1 (CPU, streaming): keep iff answer-correct (`sws[-1].score > 0.5`) AND
min(body MC) ≥ τ; τ ∈ {0.85, 0.90}; + 15k seed-0 reservoir of the [0.5, 0.85)
band. Stage 2 (GPU): deberta-xlarge-mnli, premise = question + steps 1..k−1,
hypothesis = step k; trace fails iff max_k P(contradiction) > 0.5; τ=0.90 outputs
derived exactly from τ=0.85 scores. Stage 4: dedup (image-key, normalized
question) → SFT; one per unique question → RLVR. Stage 5: invariant verifier.

| stage | τ=0.85 | τ=0.90 |
|---|---|---|
| raw | 565,149 | 565,149 |
| answer-correct | 503,885 | 503,885 |
| stage-1 kept | 338,685 | 289,209 |
| stage-2 kept (NLI) | 319,274 (94.27%) | 272,786 |
| stage-2 rejected | 19,411 (5.73%) | 16,423 |
| **sft_train.jsonl** | **100,194** | 88,555 |
| **rlvr_prompts.jsonl** | **73,704** | 65,034 |

Verification: known numbers reproduced exactly on first run (hard gate); 13/13
invariants PASS incl. verbatim-steps merge-join over 1,408,852 rows (0 mismatch),
0 dupes, 0 unresolved images; TIMEOUT+resume across jobs 59078004→59114290 with
exact count reconciliation. Cost ≈ 6.7 A100-hours.

Per-source NLI rejection extremes: super_clevr 29.5%, cocorem 20.3%, CLEVR_math
18.7% (numeric false-alarm inflation — known NLI digit blindness) vs
mavis_function_sin 0.3%. MC × NLI nearly orthogonal (NLI-fail 5.7% of MC-pass vs
6.3% of MC-fail band). Threshold sensitivity: 0.5→0.45 would reject only +1.6pp.

## 6.3 SymPy symbolic-verification audit (2026-07-19)
Third instrument vs MC and NLI. Extraction hardened over four iterations, each
validated against fabricated-FAIL inspections (kept-sample FAILs 729 v2 → 548 v3
→ 365 v4; the biggest fix: trig/log LaTeX macros were being deleted, fabricating
claims like `(-pi) = 1`).

| population | extractable | PASS | FAIL | FAIL_LC | fail rate of extractable |
|---|---|---|---|---|---|
| stage2-kept sample (5,622) | 2,492 (44%) | 2,326 | 365 | 134 | 20.0% |
| NLI-rejected (19,411, full) | 2,816 (15%) | 6,107* | 210 | 62 | 9.7% |
| MC-fail band (15,000, full) | 6,487 (43%) | 5,793 | 1,836 | 494 | 35.9% |

(*PASS includes claim-free traces; PASS-with-claims = 2,544.)

**Headline: 2,544 NLI-rejected traces verify fully clean symbolically
(1,647 under the strict variant) → NLI-false-alarm candidates.** Tier-matched:
**within every tier the NLI-rejected population fails at about the same rate as
the NLI-passing kept sample — the NLI contradiction signal is essentially
orthogonal to symbolic correctness.** The MC-fail band is elevated in every tier
(~1.6–2× geometry, 2–3× arithmetic). FAIL counts are stated upper bounds
(residual extraction artifacts).

---

# 7. Supervisor baselines + filtered-vs-unfiltered 2×2 (2026-07-12 → 07-18)

## 7.1 First SFT/RLVR baselines (commit 545a4d0; fixed controls)
Controls: Qwen2.5-VL-3B, visualprm, DATA_RATIO=0.01 (5,651 rows), SEED=42,
MAX_SEQ_LEN=4096, TOKEN_BUDGET=1e6, rewards accuracy+format only.

The pristine code needed **5 local edits + 1 env fix** (each crash strictly later
than the last):
| edit | fix |
|---|---|
| 1 | SFT image-token overflow → pixel cap in collator (`SFT_MAX_IMAGE_PIXELS`, 1280·28·28) |
| 2 | nlvr2 zip paths nest one level deeper → `resolve_zip_member` candidate |
| 3 | wrong Hydra key → `actor_rollout_ref.model.enable_gradient_checkpointing` |
| 4 | multi-image rows truncated to one image → keep all |
| 5 | multimodal prompt > 2048 → bound images at materialization (`RLVR_MAX_IMAGE_PIXELS`, 640·28·28) |
| Env A | `VLLM_USE_V1=1` (verl builds the V1 engine off-thread; vLLM's oracle soft-falls to V0) |

Canonical results: **SFT job 58944776** — 245/245 steps, loss 8.24→2.23, 0 dropped
images. **RLVR job 58945708** — 4/4 GRPO steps (2×A100, 21:44), val reward
0.5376→0.5553, entropy 0.80→0.67.

**Degenerate accuracy reward found:** parquet `ground_truth` = the whole response
text (not `answer`) and no "Final answer:" pattern in extraction → falls to
first-number compare (usually the "1." step marker). Empirically every train
reward ∈ {0.25, 0.75} (format ≡ 0.5, accuracy a coin-flip bit). All baseline RLVR
"accuracy" carries this caveat. Proposed fix: ground_truth = answer + add the
pattern. Later edits 6 (filtered-jsonl adapter re-reading raw records byte-
identically) and 7 (scale plumbing + resume) documented in
`docs/local_edits_to_training_code.md`.

## 7.2 Filtered (τ=0.85) vs unfiltered 2×2 (smoke scale, 2026-07-18)
Four arms byte-identical except the filter (jobs 58944776 / 58945708 / 59198050 /
59198051); eval job 59199631 — greedy vLLM, identical prompts, ScienceQA test
(1,836) + MMK12 (1,024).

Boxed-aware scorer (uniform fallback added after diagnosis):
| model | ScienceQA | MMK12 |
|---|---|---|
| base | 0.7369 | 0.3584 |
| sft_unfiltered | 0.6520 | 0.3389 |
| **sft_filtered** | **0.6770** | **0.3750** |
| rlvr_unfiltered | 0.7484 | 0.3633 |
| rlvr_filtered | 0.7500 | 0.3545 |

**Findings:** (1) unfiltered SFT *hurts* the base on both benchmarks — the noisy-
imitation concern confirmed; (2) filtering significantly improves SFT (McNemar
p=0.016 SQA / p=0.004–0.018 MMK12); (3) RLVR arms statistically tied once scoring
is format-robust (both ~+1.1 pt over base after only 4 steps); (4) the apparent
filtered-RLVR MMK12 collapse (0.1895 primary scorer) was an **extraction
artifact** — 82.5% of its outputs ended in `\boxed{...}` with no `<answer>`/"Final
answer:" (template drift; entropy 0.72→1.44), fixed by the uniform boxed fallback.
Lesson inherited by all later scoring: extraction cascades must include \boxed{}.

## 7.3 Scale runs (prepared, not run)
Four `run_scale_*.sbatch` calibrated (~155 GPU-h total; 64×8 rollouts/step, mini
16, 191 steps, resume-capable) — submitted for approval, superseded in priority by
the six-arm campaign.

---

# 8. DeBERTa chunker (2026-07-20 → 08-10)

Token-level chunk-boundary classifier (DeBERTa-v3-small, 2 labels) reproducing the
human step segmentation of VisualPRM400K-v1.1, for segmenting rollouts before a
process-reward model scores each segment. Own env `chunker/env` (torch
2.13.0+cu130, transformers 5.14.1 — **4.56 cannot load the release tokenizer**).

## 8.1 Data profiling (canonical.jsonl: 9,478,315 rows, 18.3 GB)
Multi-step (gold) 495,756 (5.23%); single-step 8,982,559. **Leakage finding:** only
969,152 unique questions — 99.0% of records share a question (max 87,502 rollouts)
→ split grouped by question (a previous split had given val just 31 distinct
questions, inflating val F1 to 0.9147 with val→test recall collapse 0.871→0.535).
Candidate-boundary rule recall over gold: ASCII-only 84.25% → expanded (CJK punct,
math closers, digits, list items) **98.75%**. 17.3% of joined multi-step texts
> 512 tokens → windowing stride 384.

## 8.2 Training
fp32/TF32 (bf16 NaN blow-up caught in smoke — NanGuard added), effective batch 128,
lr 3e-5, seed 42, 3 epochs, early-stop on val boundary F1. Two Gate-4 code-review
catches: a CJK candidate-gate bug (would have crippled raw-CJK inference) and soft
min_tokens (sub-8-token fragment leak) — both fixed; gold selectability 96.5%.

**Arm comparison 1 (v1): plain vs weighted CE** — plain wins (val best F1 0.8154 @
thr 0.25 vs 0.8141); CJK F1 0.856 ≥ English 0.824 → mDeBERTa contingency closed.

**Threshold story:** 0.25 = v1 shipped operating point (boundary-F1 objective,
zero-cut-constrained sweep); 0.35 = training-time threshold — the one-shot test
eval was spent there and 0.35 was also overridden into the production canonical
run; 0.40 = the v2-mask shipped operating point.

**v1 test (one run, thr 0.35, 19,000 traj):** boundary F1 **0.7987** (P 0.844 /
R 0.758); token F1 0.8209; zero-cut 3.02% of multi (453/15,000; short-text
concentrated); over-split on singles 2.43/traj (67.8% affected). Long-text
diagnostics: no seam pathology (F1 within-32-tok-of-seam 0.7393 vs beyond 0.7454);
decode grid flat (±0.002).

## 8.3 v2-mask retrain (single-step negative policy)
Motivating finding: **83% of single-step records contain step markers/lists/blank
lines** — `len(steps)==1` means UNSEGMENTED, not atomic; the 120k singles used as
negative-only examples were suppressing real boundaries. Four arms
(keep/mask/drop/clean), frozen val/test:

| arm | thr | val boundary F1 | zero-cut |
|---|---|---|---|
| keep (baseline) | 0.25 | 0.8017 | 2.09% |
| **mask (winner)** | **0.40** | **0.8205** | **1.35%** |
| drop | 0.45 | 0.8202 | 1.41% |
| clean | 0.45 | 0.8195 | 1.32% |

**Test (mask @ 0.40): boundary F1 0.8163** (P 0.847 / R 0.788) vs v1 0.7987 —
**+0.018, all recall**. Caveats: early stopping silently disabled in all four arms
(identical across arms, so comparison holds; mask's ceiling likely higher);
over-split/single rose 2.43→3.89 but the reference labels themselves are the
unsegmented singles. Release: `chunker-deberta-v3-small-v2-mask` (best_thr 0.40,
min 8 / max 220 tokens).

**Behavioural eval on singles (32,971 records):** marker cut-rates stepN 0.893 /
numlist 0.719 / blank 0.694 / bullet 0.030; cuts at no marker 2.7%; **vs the gold
structural baseline (2.86M gold boundaries) the model cuts mid-sentence and
mid-equation LESS than the human annotation itself** (mid_sentence 0.01% vs 0.12%;
operator-endings 0.14% vs 0.84%); only unbalanced-delimiter flags exceed gold,
concentrated in mavis trig functions hitting the 220-token cap.

## 8.4 Chunk-eval suite (4 experiments, 2026-07-29)
Gold = passthrough multi-step records of the same file; seeds 0.

**Exp 1 — completeness (all 78.0M chunks):** no produced tier catastrophically
worse than gold; produced 1.5–1.8× gold on missing terminal punctuation;
dangling_start BETTER than gold (0.0–0.6% vs 2.5%). **Defect found (1c):** the
marker path splits inside code fences — **187,780 records / 810,053 boundaries
(2.4% of marker records vs gold 0.02%)**; the model path is at 0 (v2 guard
verified). v3 marker-path guard = open decision.

**Exp 2 — corruption-calibrated boundary stats (20k/pop, deberta-xlarge NLI fp32):**
primary statistic s1 (cross-boundary SBERT cosine): gold 0.527, merged 0.590,
fragmented 0.450, **produced 0.528 (Cohen's d = 0.00)** — produced boundaries
statistically indistinguishable from gold, far from both corruptions. s2/s3
deviations consistent with finer chunks; s3 neutral/contradiction uninformative.

**Exp 3 — weld detection (sympy anchors, 200k chunks):** produced fuses LESS than
gold in every tier (function 8.2% vs 9.4%; geometry 3.3% vs 8.9%; arithmetic 0.0%
vs 0.8%).

**Exp 4 — strip-and-resegment (all 495,756 gold records):** full cascade on
`\n\n`-join macroF1 0.569; model-only on space-join **0.796**. Stratified:
**clean subset (no internal markers, 32.5%) cascade macroF1 0.936, WindowDiff
0.030, 74.1% perfect** — beats model-only (0.881); structured subset (67.5%)
0.392, density 1.52, 0% perfect — **the list tier is the culprit: disabling it
alone moves 0.392 → 0.924** (Exp 4b). Exp 4c: a step>para production variant would
drop ~16.6M chunks (78.0M → 61.4M), 99.4% resolving at para, chunk length 32→40
tokens (gold 44.9), 0.6% extra model load. List-tier policy = open decision.

## 8.5 Cross-dataset generalization (2026-07-30)
V2 strip-and-resegment (space join, model path only, thr 0.35):
| corpus | n | macroF1 | coverage | density | note |
|---|---|---|---|---|---|
| VisualPRM (reference) | 495,756 | 0.796 | 0.769 | 0.92 | harness check reproduced exactly |
| **PRM800K** | 10,007 | **0.940** | 0.928 | 0.93 | excluding the typography-borne `# Answer` cut: **99.0%** of reasoning boundaries found at P 0.981 |
| ProcessBench | 3,372 | 0.646 | 0.704 | 1.40 | convention (paragraph-grained gold; 4.1% steps > 220-tok cap force cuts); 79.2% of boundaries within one sentence |

**Transfer confirmed** — better than in-domain on the finer-grained corpus, the
opposite of style-memorization. v2-mask re-run: ProcessBench 0.646→**0.722**,
VisualPRM 0.796→0.823, PRM800K flat (no headroom).

## 8.6 Canonical production chunking (2026-07-25)
Deliverable `canonical_chunked_v2.jsonl` (9,478,315 rows): marker cascade
(step > list > para) 7,674,203 records (81.0%, 73.2M chunks) + model path
1,282,069 (thr 0.35; only 7.9% of model-path records actually split) +
nonprose-skips 26,287 + passthrough 495,756. Full cascade cost: 49 min on one A100.
**v2 fence-guard patch:** model-path in-fence splits (5,024 records / 9,083
boundaries) → targeted 15 s re-run of exactly 7,826 records; zero-in-fence now
verifier-enforced. Verification (both versions): ALL PASS incl. mass conservation
over 8.98M records and byte-identical passthrough. Label policy: single-step
labels replicated across chunks, flagged `label_replicated` for later exclusion.
Granularity: marker chunks mean 32.1 tokens vs gold 44.9 — the triply-corroborated
"finer than gold" finding. Trap recorded: stride sampling is biased on this file
(periodic rows) — seeded random sampling required.

---

# 9. Granularity (2026-08-13 → 08-14)

## 9.1 Reference bands
VisualPRM400K body steps (499,525; final-answer step excluded): mean **30.1**
w/step, p50 **24**; body steps/record 4.81; per-source spread 14.2–46.9.
VPB per-policy:
| policy | w/step | backtracking markers/step |
|---|---|---|
| Claude 3.5 Sonnet | 21.0 | 0.00 |
| GPT-4o | 23.8 | 0.00 |
| InternVL | 24.1 | 0.00 |
| QvQ | 99.1 | **1.10** |

The apparent 1.67× VPB/VisualPRM granularity gap is a QvQ mixture artifact (OVL
0.81–0.85 for the three compact policies vs 0.46 for QvQ). **Target band adopted:
~21–24 w/step.** Band split of VisualPRM (475,285 records, vpb_compact anchor):
low <18 = 20.7%, match 18–28 = 40.7%, high >28 = 38.6% — heavily source-confounded
(recorded per band). Side finding: 25 VPB records reference more `<imageN>` tags
than images supplied (a published-benchmark defect).

## 9.2 Qwen granularity-controllability experiment
200 frozen VPB questions (sha 480aa48a…, stratified 72/49/40/21/18 across five
sources) × 4 instruction conditions × {3B, 7B}; temp 0.7 / top_p 0.9 / max 1024 /
seed 0; jobs 61377406/61377408, 1,600 generations, 0 empty. Conditions:
A unconstrained, B "about 6 steps", C "about 25 words per step", D "number each
step 1) 2) 3)".

**Method finding first: Qwen's step boundaries are not identifiable from
whitespace.** Blank-line vs newline segment counts differ by >2× on **54–72.5%**
of responses in every cell; under blank every condition reads above the 24-word
target, under newline every condition reads below it — same generations. Headline
rule = **struct** (blank line OR line-initial `1.`/`1)`/bullet/`Step N:` opener —
the `Step N:` alternative fixed a regex gap that had mis-scored 46.5% of 3B's
C-cell as zero-split).

Headline (struct): 3B A/B/C/D words-per-step 17.9/18.7/**24.7**/18.9; 7B
17.0/18.0/**14.3**/18.2. Steps/response collapse under C: 3B 12.8→5.6, 7B
14.8→5.0.

**Results:** (1) **B_count fails outright** — neither model moves toward 6 steps
under any rule. (2) **C_length "complies" with the wrong behaviour** — both models
cut step *count* by half to two-thirds and switch formatting convention wholesale
rather than tuning step length; C is also the only source of degeneracy (zero-split
3B 20.0% / 7B 9.5%). (3) **D_format is the cleanest separation: 3B 71.5% vs 7B
98.5% marker emission.** (4) Degeneracy-inflation correction confined to C (3B
24.7→21.4 excluding 40 degenerates; 7B 14.3→13.7); 3B's "on-target" C result is
source-driven (concentrated exactly where it degenerates most). (5) Rule-
independent solids: both models effectively non-backtracking (max 0.019
markers/step vs QvQ 1.10); truncation ≤4%; 7B source-invariant (0.9-word spread vs
3B 5.2). **Verdict: 7B is the more controllable model on every predictability
axis, but neither model has a usable numeric control surface.** Limitations:
absolute w/step is rule-conditional; the 21–24 band is convention-confounded;
MMMU n=18 thin; resolution ±1.5 words (response-level clustered SE 0.59–0.89;
3B C-cell SE 6.16 → essentially unresolved as reported).

This experiment's D-format / marker-emission numbers (57.5% vs 88.5%
unconstrained) are the direct empirical basis for the six-arm campaign's
chunker-vs-marker interaction prediction.

---

# 10. Six-arm GRPO campaign (2026-08-22, IN PROGRESS)

**Question.** Does a bipartite-matching NLI text reward train a better multimodal
reasoner than a process reward model (VisualPRM-8B), and does a learned chunker
beat regex segmentation feeding that match? Three comparisons: chunker vs marker
(arms 1↔2, 4↔5; prediction from §9: the chunker should help 3B more, since 3B
emits markers on only 57.5% of unconstrained responses vs 88.5% for 7B), match vs
PRM (1↔3, 4↔6), and scale (all six). Infra in `grpo_arms/`.

| # | model | reward | segmentation |
|---|---|---|---|
| 1 | Qwen2.5-VL-3B | match | chunker v2-mask, thr 0.40 |
| 2 | 3B | match | marker regex (blank line, `Step N:`, `\d+[.)]`, `[-*•]`) |
| 3 | 3B | VisualPRM-8B min step score | VisualPRM's own `\n\n` split |
| 4 | 7B | match | chunker |
| 5 | 7B | match | marker |
| 6 | 7B | VisualPRM | `\n\n` |

**Reward.** Arms 1/2/4/5: R = 1·format + 2·match + 5·answer, where match =
segment rollout → s(i,j) = 0.5·E(i→j) + 0.5·E(j→i) − max(C(i→j), C(j→i)) clipped
[0,1] (deberta-xlarge-mnli fp32+TF32, label indices from id2label) → Hungarian →
keep pairs s ≥ τ=0.45 → precision-dominant Fβ (β=0.5) on matched mass.
Bidirectional because each direction is blind to one failure mode; max-C so
one-direction entailment can't launder a contradiction; neutral weighted zero.
Arms 3/6: match term → min over VisualPRM-8B `generate_steps_with_soft_score`
(revision 7b7c9c4 pinned), served cross-process from `envs/vprm-judge`. Shared
format term (explicit final-answer marker) and answer term (extraction cascade
"Final answer:" → \boxed{} → <answer> → last line; normalized/letter/numeric-
tolerance compare).

**Data.** Frozen train subset sha `bd15b1f1…`: 6,000 prompts. Funnel: 538,710 rows
(37 files, nlvr2 excluded) → 215,281 eligible (answer-correct, ≥5 body steps) →
114,541 unique (dedup keep-best-min-MC) → cap 300/source (held, 36/37 at 300) →
10,847 → seed-42 sample 6,000. Mix near-uniform (max source 3.0%; mavis 28.2%,
geometry 22.2%). Split: seed-0, 5,880 train / 120 verl-val, identical all arms.
Validation set frozen: 2,856 single-image VPB questions + answers only (sha
7cd6d974…) — VisualPRM never trained on the five VPB sources, so neither reward
has home advantage.

**Architecture.** verl GRPO: batch 32 × rollout n 5, mini 8, micro 1, lr 1e-5, KL
0.001, prompt 2048 / response 1024, image cap 640·28·28, 183 steps = 1 epoch,
save/test every 20, resume_mode auto, 2×A100/200G/48h. Reward fn (stdlib-only) →
Unix socket → scoring server on GPU 1: NLI+chunker under `chunker/env`
(transformers 4.56 cannot load the release chunker tokenizer — all match arms
share the 5.14 stack so segmentation is their only difference), VisualPRM under
`envs/vprm-judge`. Per-component dict returns → per-sample per-step dumps.

**Preflight (jobs 61997442, 61997451): 9/10 PASS** — subset sha + all images;
chunker thr 0.40 segments; NLI id2label {0 contradiction, 1 neutral,
2 entailment}; cross-process match identity 0.977 vs shuffled-gold 0.150
(chunker) / 0.240 vs 0.030 (marker, over-split 19 vs 5 segments); VisualPRM gold
trace scored [0.996–1.0] min 0.996 with matching split count. The one FAIL was
the τ-env-transfer check catching max|Δs| = 0.121 between transformers 4.56 and
5.14 (max 0.334, 4.9% of pairs |Δ|>0.05 on the full sweep; only 7/29,944 verdict
flips at high τ) → the sweep was recalibrated in the serving env.

**Gate 1 — τ sweep (jobs 61997443 / 61997456 authoritative).** Pairs =
granularity/nli_pairs.jsonl: 14,972 identity + 14,972 cross-record hard negatives
+ 5,000 each merge2/merge3/split2/split3 (49,944 pairs, ~100k NLI forwards,
38 min). Identity vs hard-negative essentially solved: **AUC 0.9999** (hn p50 0.00
/ p95 0.27 vs identity p5 0.97); any τ ∈ [0.3, 0.85] gives balanced acc ≥ 0.978.
The decision therefore rests on perturbation survival:

| τ | id TPR | hn FPR | merge2 | merge3 | split2 | split3 |
|---|---|---|---|---|---|---|
| 0.30 | 1.0000 | .0433 | .979 | .972 | .960 | .875 |
| 0.40 | 1.0000 | .0266 | .966 | .951 | .936 | .828 |
| **0.45** | 1.0000 | **.0209** | .953 | .932 | .907 | .785 |
| 0.56 | .9999 | .0099 | .812 | .810 | .623 | .561 |
| 0.81 (Youden) | .9999 | .0011 | .450 | .351 | .275 | .178 |

Asymmetry finding: **merges outlive splits at every τ** (gap peaks +0.22 near
τ=0.6) — fragments degrade both NLI directions, merges only one. A split-survival
cliff sits between 0.45 and 0.56 (split2 0.91→0.62).

**Decision: τ = 0.45.** Identity-vs-hardneg has no discriminating power over the
choice; 0.45 sits below the split cliff with all four kinds ≥ 0.79 at hn FPR
2.1%. Decisive: the cliff is not arm-neutral — over-segmentation is the marker
regex's characteristic failure on unstructured 3B output, so τ ≥ 0.56 would
penalize arms 2/5 for a segmentation artifact rather than a reasoning error,
biasing the chunker-vs-marker comparison before training starts. 0.56 (the
FPR≤1% point) explicitly rejected; real reward-loop negatives are same-trajectory
segments, easier than the sweep's cross-record ones. Recorded in
`grpo_arms/TAU_DECISION.md` and every match arm's config header.

**Smoke ladder (each failure strictly later):**
| job | outcome | fix |
|---|---|---|
| 61997452 | step-0 val OK (full per-component metrics) → EAGAIN: server backlog-64 overflow under ~160 concurrent reward-loop connects | threaded accept + GPU lock + backlog 1024 + client exponential backoff (300-concurrent hammer test passes) |
| 61997527 | step-0 val OK → CUDA OOM at vLLM wake_up (FSDP↔vLLM handoff + 5 GB scorer on GPU 1) | FSDP param+optimizer+ref CPU offload, rollout util 0.40 |
| 61999523 | **PASS rc=0** — 2/2 GRPO steps, global_step_2 saved, resume marker written; GPU peak 44.8 GB alloc, host 203 GB; ~300 s/step (gen 145 s, reward mean 40 s, update 85 s) | — |

**Status at compile time:** **Arm 1 running as job 62044345**
(runs/arm1_3b_match_chunker, 183 steps, ~15 h projected). Gate-2 deliverables:
reward-component curves; whether match moves independently of answer (per-sample
rollout dumps → within-batch correlation per step + match movement in
answer-constant strata); verl-val curve; VPB accuracy of the merged final
checkpoint vs base 3B. Gates 2 (arm-1 review) and 3 (arms 2–6) await go-ahead.

---

# 11. Open items and queued work

| item | status |
|---|---|
| Six-arm campaign gates 2–3 | arm 1 training; report → wait → arms 2–6 |
| Pooling sweep (min → quantile → mean), negation-FPR vs paraphrase-FPR frontier | queued (needs per-clause equiv vectors saved) |
| SBERT/BERTScore clause-split control | queued (predicted: no rescue) |
| Adversarial probe sets B-1/B-2/B-3 + cue-free NLI-atom families | queued |
| Hard numeric-match term; multimodal grounding (IoU/region-CLIP) term | design case complete (§5.5–5.7), not built |
| v3 marker-path fence guard (810K in-fence boundaries) | user decision pending |
| List-tier granularity policy (step>para variant: −16.6M chunks) | user decision pending |
| Supervisor scale runs (4× run_scale_*.sbatch, ~155 GPU-h) | prepared, superseded in priority |
| Baseline accuracy-reward fix (ground_truth = answer + "Final answer:" pattern) | proposed to supervisor |
| canonical v1 + shards cleanup (~56 GB) | gated on handoff confirmation |

# 12. Source documents
`docs/project_history.md`, `docs/research_brief.md`, `docs/results_scienceqa_mmk12.md`,
`docs/presentation_briefing.md`, `docs/reliability_results.md`,
`docs/sensitivity_results.md`, `docs/negation_clausesplit_results.md`,
`docs/paraphrase_clausesplit_results.md`, `docs/visualprm_negation_results.md`,
`docs/coco_iou_results.md`, `docs/reward_reliability_dci.md`,
`docs/visualprm_filtering_2026-07-16.md`, `docs/filtered_vs_unfiltered_2x2_2026-07-18.md`,
`docs/local_edits_to_training_code.md`, `docs/progress_2026-07-11.md`,
`data/visualprm_v11_filtered/sympy_audit/REPORT.md`, `chunker/README.md`,
`chunker/MORNING_REPORT.md`, `chunker/SESSION_LOG.md`, `chunk_eval/EVAL_REPORT.md`,
`chunk_eval/xdataset/REPORT.md`, `chunk_canonical/REPORT.md`,
`granularity/REPORT.md`, `grpo_arms/{README,LAUNCH_PLAN,TAU_DECISION}.md`.
