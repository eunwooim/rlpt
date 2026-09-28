# RLPT — Project History: Everything Done So Far

*Written 2026-07-07. A chronological + thematic account of all work completed on this project to
date. Companion to `docs/research_brief.md` (the forward-looking research dossier); this document
is the record of what was actually done, in order, with pointers to the code and results.*

**Project:** RL post-training (GRPO) of a multimodal model — Qwen2.5-VL-3B-Instruct — with a
reward that scores the **content of its free-text reasoning**, not just its final answer, plus an
ongoing investigation into whether that reward's text scorer is itself **reliable** (resistant to
hard negatives, negation, numeric corruption, and reward hacking).

**Stack:** Qwen2.5-VL-3B-Instruct · GRPO via verl 0.8.0 / vLLM 0.10.1.1 / torch 2.7.1+cu126 ·
ASU Sol cluster (A100s, SLURM).

---

## Phase 1 — Reward design evolution (late May → 2026-06-05)

The reward went through three designs; each drop explains the final shape.

### 1a. QCVSR: typed schema + deterministic verifier (dropped)
A synthetic GQA-derived dataset (`data/qcvsr_gqa_v1/`) where the model emits a **typed evidence
object** (bounding-box pairs / count pairs) plus an answer, with a deterministic verifier checking
schema correctness, answer correctness, and consistency between them (dual-reward design).
Robust but narrow — not free-text reasoning. Kept as a design ancestor: the "hard verifiable term
alongside a soft term" idea survives in the final composite reward.

### 1b. Caption → frozen parser → scene-graph match (dropped)
Plan: model emits a caption, a frozen `flan-t5` parser converts it to a scene graph, the graph is
matched against GQA oracle scene graphs. Work completed before the drop:
- **Parser benchmark (2026-06-02):** `lizhuang144/flan-t5-base-VG-factual-sg` chosen over the
  large variant — on clean FACTUAL gold, base tuple-F1 0.943 vs large 0.963, but both collapse to
  ~0.59 on noisy VG human-gold graphs and large ≈ base there, at ~2× latency. Writeup:
  `data/visual_genome/PARSER_EVAL.md`.
- **Why dropped:** a parser *in the reward loop* is a reward-hacking magnet — RL can raise reward
  by learning parser-friendly phrasing instead of seeing better. Tooling (`tools/caption_to_graph.py`,
  `tools/eval_parser_on_vg.py`, `tools/vg_query_to_graph.py`, `tools/eval_sgg_vs_gt.py`) kept as
  legacy/instrumentation.

### 1c. Free-text reasoning scored by soft bipartite matching (FINAL, built 2026-06-05)
`tools/graph_match_reward.py`. The model emits `<think>reasoning</think><answer>choice</answer>`;
the `<think>` text is scored against the ground-truth explanation with **no parser in the loop**:
1. **Clause-split** both sides (deterministic spaCy dependency parse, `tools/clause_split.py`).
2. **Embed** every clause with SBERT `all-MiniLM-L6-v2` (GT side cached).
3. Build the cosine-similarity matrix, solve **Hungarian one-to-one assignment**
   (`scipy.linear_sum_assignment`); pairs count only if cosine ≥ τ = 0.15. One-to-one blocks the
   "say the same fact five ways" exploit.
4. Read off **precision-dominant Fβ (β = 0.5)** from the matched similarity mass.

**Key empirical finding that shaped the reward:** SBERT cosine is **blind to entity swaps and
negation** — a prediction stating the *opposite* of the truth matched at 0.903 cosine; a
confidently-wrong answer scored 0.41 vs 0.63 for a good one. Fix: the **composite reward**
(STAR-R1-shaped):

> **R = 1·format + 2·match + 5·answer − 1·pun**

| term | what | weight |
|---|---|---|
| format | exactly one `<think>` + one `<answer>` (hard gate) | ×1 |
| match | bipartite reasoning-match Fβ vs GT solution | ×2 |
| answer | hard MCQ correctness (text/letter/index) | ×5 |
| pun | −1 per reasoning clause matching no GT fact (hallucination) | −1 |

With the composite, wrong-but-fluent scores 1.8 vs 7.3 for correct — the hard answer term fixes
the swap-blindness. verl drop-in adapter: `compute_score(...)` returning
`{score, acc, match, format, n_hallucinated}`; weights env-overridable via `RLPT_W_*` (this is how
the ablation ran with no code change).

**Dataset pivot:** synthetic GQA → **ScienceQA** (image-bearing subset: 5,678 train / 1,922 val /
1,836 test), because it has ground-truth *reasoning text* (`lecture`/`solution`) to match against.
Conversion: `src/data_factory/scienceqa_to_verl.py` → `data/scienceqa_verl/`.

---

## Phase 2 — Training infrastructure on ASU Sol (2026-06-06 → 06-08)

Getting verl + vLLM + Qwen2.5-VL GRPO to run on Sol (RHEL8, glibc 2.28, BeeGFS `/scratch`) took a
documented chain of eight fixes, each found by smoke-testing on a 64-row parquet:

1. **Ray startup hang** (the big one, diagnosed with py-spy): verl's default
   `ray_init.num_cpus=null` makes Ray grab all 48 node cores while SLURM allocated ~12; 48 workers
   importing torch/vllm simultaneously from BeeGFS = IO storm, workers never register. Fix:
   `ray_kwargs.ray_init.num_cpus=<cpus-per-task>`. (The initial "too-new pip versions" theory was
   a red herring, though the pinned env was kept anyway.)
2. Required micro-batch-size configs set explicitly.
3. `unset ROCR_VISIBLE_DEVICES` (Sol sets the AMD var; conflicts with CUDA_VISIBLE_DEVICES).
4–5. flash-attn / vLLM version maze → final stack **verl 0.8.0 + vllm 0.10.1.1 + torch
   2.7.1+cu126 + transformers 4.56.2**, attn = SDPA.
6. SBERT can't load inside verl's reward worker (meta-tensor state) → **subprocess embed server**
   (`tools/sbert_embed_server.py`).
7. No real flash-attn on glibc 2.28 → **pure-torch shim** on PYTHONPATH
   (`src/train/flash_attn_shim/`: SDPA-backed flash_attn_interface + bert_padding, numerically
   verified).
8. Memory: FSDP CPU offload + `rollout.gpu_memory_utilization=0.4` + host `--mem 240G` to fit
   FSDP↔vLLM handoff on A100-80GB.

Ops conventions established: always `#SBATCH --no-requeue`; always smoke-test before full runs;
`srun --overlap --jobid=<J>` to inspect running jobs; login node is 1-core (never build/run there).

---

## Phase 3 — The GRPO run (2026-06-08 → 06-09)

- **Config:** batch 64 questions/step, `rollout.n = 5` (GRPO group of 5, group-mean baseline, no
  critic), KL coef 0.001 vs frozen base. 88 steps/epoch × 2 epochs = **176 steps**, checkpoints
  every 20 (`models/qwen2_5vl-3b-scienceqa-grpo-step*`).
- **ScienceQA held-out test (1,836 rows), step 176:** accuracy **90.3%** (base 79% zero-shot →
  +11.3 pts), format 100%, reasoning-match F **0.778**, hallucinated clauses **0.09/response**,
  composite reward 6.98.
- **The real finding is the decomposition:** format saturates by step 20 (0.48→1.00); accuracy is
  mostly there early (~0.82 by step 20); but **reasoning match climbs monotonically 0.371 → 0.625
  (step 20) → 0.766 (step 176)** — roughly doubling long after format/answers saturated. The
  accuracy headline understates what training did.

Detailed results: `docs/results_scienceqa_mmk12.md`; presentation Q&A: `docs/presentation_briefing.md`.

---

## Phase 4 — Validation campaign: mimicry vs reasoning (2026-06-10 → 06-12)

The match climb is partly circular (we optimized it); SBERT cosine could reward *style mimicry* of
ScienceQA solutions. Design principle: mimicry is style-specific + image-independent; real
reasoning is content-specific + image-dependent. **Five tests, all PASS:**

1. **Shuffled-GT control** — own-GT match climbs 0.38→0.76 while same-topic shuffled-GT stays flat
   ~0.22; the gap *widens*. Mimicry would lift both.
2. **NLI re-scoring** (DeBERTa-MNLI, a different scorer) — own-GT entailment 0.21→0.52, shuffled
   flat → the gain is propositional content, not phrasing; also rules out encoder hacking.
3. **Image-swap probe** — accuracy collapses ~−0.21 with swapped images at every checkpoint;
   reasoning-match swap-sensitivity *grows* with training (−0.048 → −0.079) → RL increased
   image-grounding.
4. **A-OKVQA transfer** — zero-shot on a differently-styled benchmark: 0.43 → **0.84**.
5. **w_match=0 ablation (causal keystone)** — identical config/seed/steps, reward =
   format+answer only (`models/qwen2_5vl-3b-scienceqa-nomatch-step*`). Ablation match goes
   0.38→0.46 then flat (0.42 at 176) vs main arm 0.38→0.76; NLI never moves; image-grounding stays
   at base. Everything beyond a small step-20 bump is **caused by the match term**.

**MMK12 OOD accuracy test (2026-06-12):** ScienceQA is at a ceiling, so accuracy causality was
tested on MMK12 (2,000 K12 exam MCQs, base ≈ chance). Balanced 1,024-row sample: base 0.311 →
answer-only ablation 0.397 → **with-match 0.439** — **+4.1 pts from the match term** at identical
format, paired McNemar exact **p = 0.021**. Where reasoning is the bottleneck, training the
reasoning channel converts to answer accuracy.

**Publishable claim, three legs + one:** content-not-style (1,2) · image-grounded (3,4) ·
match-term-causal (5) · OOD accuracy gains (MMK12). Eval TSVs: `data/logs/validation_results{,_nomatch}/`;
generations under `outputs/validation*`.

---

## Phase 5 — Reward reliability investigation (2026-06-13 → present)

The validation showed the *model* improves, but exposed that the reward's *scorer* is unreliable
on adversarial text. This became the active track.

### 5a. DCI internal probe
Using Densely Captioned Images (paraphrases + aligned hard negatives), measured whether scorers
rank a correct paraphrase above a corruption (`docs/reward_reliability_dci.md`):
- SBERT is consistent but structurally blind: ~chance on content corruptions, **worse than chance
  on swaps** (scores the lexical near-copy swap 0.88 above the faithful paraphrase 0.76).
- NLI far better on content changes (0.92/0.83 discrimination) but fooled by near-verbatim swaps.
- Embedding-inversion probe: GT embeddings *retain* swap-relevant content → it's a
  **cosine-geometry problem, not an information bottleneck**.
- Attribute-swaps resist **both** text methods → the case for a multimodal grounding term.

### 5b. Supervisor's benchmark reproduced at full scale (jobs 56951317, 56959065)
The supervisor productionized the question onto public benchmarks (`src/metrics/reliable/`, commit
`0ad601c` "codex metric test v1.0"). Reproduced on Sol uncapped (his lab-box env can't be rebuilt
on Sol — glibc/CUDA wall — but the framework runs as-is on `rlpt-train`). Results
(`docs/reliability_results.md`):

| dataset | mode | SBERT | BERTScore | NLI |
|---|---|---|---|---|
| SugarCrepe | separation FPR@0.6 ↓ | 0.974 ❌ | 1.000 ❌ | **0.069** ✅ |
| SugarCrepe++ | ranking acc ↑ | 0.725 | 0.140 ❌ (below chance) | **0.979** ✅ |
| NegBench | separation FPR@0.6 ↓ | ~0.75 ❌ | ~0.997 ❌ | **~0.001** ✅ |

**Convergent conclusion (DCI + public benchmarks): cosine and BERTScore cannot be trusted on hard
negatives/negation; bidirectional NLI entailment can.**

### 5c. Sensitivity analysis / τ calibration (`sensitivity_analysis.py`, `docs/sensitivity_results.md`)
Pure-CPU re-aggregation of the 252,693 saved score rows: ROC-AUC (NLI 0.98 on SugarCrepe++ vs
SBERT 0.73, BERTScore 0.18), full FPR(τ) sweeps, TPR/specificity curves, and operating thresholds
τ* at target FPRs — SBERT/BERTScore have no usable operating point (τ* ≈ 0.99).

### 5d. VisualPRM400K math-negation benchmark (job 57348788/57370177; `docs/visualprm_negation_results.md`)
New benchmark built in-repo (`build_negation_pairs.py`, `score_negation.py`, `aggregate_negation.py`):
40,000 pairs from the VisualPRM400K math subset; hard negative = correct step-by-step solution with
**one digit changed** in the first equation (`AB = 14` → `15`). Paragraph-level results:
SBERT FPR@0.6 **1.000**, BERTScore **1.000**, NLI equiv **0.530** — even the entailment model
misses half the single-digit corruptions when buried in long reasoning. Hardest negation setting
across all four benchmarks (NegBench 0.001 → SugarCrepe 0.069 → VisualPRM 0.530).

### 5e. Clause-split + min-pool NLI (job 57761933, 2026-06-29; `docs/negation_clausesplit_results.md`)
`score_negation_clausesplit.py`: split each solution into atomic clauses → SBERT-cosine Hungarian
**alignment** (corruption-blind matching is a feature) → NLI equiv per matched pair → **min-pool**
(one contradicted clause vetoes the solution). Result: FPR@0.6 on the corrupted copy
**0.530 → 0.008**. Paragraph-level *dilution* (plus length truncation/OOD — confounded) was what
defeated NLI; with clause scaffolding it does catch `14` vs `15`.

### 5f. Paraphrase precision control (job 57861882, 2026-06-30; `docs/paraphrase_clausesplit_results.md`)
The negation test only checked recall on byte-identical near-copies. Precision control:
9,254 number-faithful paraphrases of correct solutions (Qwen2.5-VL-3B paraphraser,
`build_paraphrase_pairs.py`), same pipeline. **Hard min FAILS precision: FPR@0.6 = 0.599** of
correct rewordings are false-flagged — one clause NLI calls "neutral" sinks the whole min (median
min-equiv 0.578, right on the 0.6 boundary). **Verdict: do not ship hard min** — needs a softened
pool (k-th-lowest quantile) and/or a hard numeric-match term.

### 5g. COCO bounding-box IoU ranking (job 57515972; re-framed 2026-07-01; `docs/coco_iou_results.md`)
Spatial analogue of numeric blindness (`build_coco_iou_pairs.py`, `score_coco_iou.py`,
`aggregate_coco_iou.py`): 20,000 triples, GT box + two jittered candidates rendered as text
`[x1,y1,x2,y2]`. Canonical framing = **threshold-free rank accuracy** (does sim respect IoU
order?): IoU oracle 1.000 (by construction) vs BERTScore **0.620**, SBERT **0.577**, NLI **0.503**
(chance); Spearman(sim, IoU) ≈ 0 for all three. Mechanism (recomputed live 2026-07-02): NLI's ≈0
is **confident contradiction on any digit difference with no magnitude sense** (a nearly-right box
and a wildly-wrong box get the same contradiction ≈0.93–0.99) — not neutrality. Quantitative case
for an explicit IoU term in any grounding reward.

### 5h. Adversarial catalogue (walked through 2026-07-02)
Full taxonomy of attacks on the clause-split+NLI reward across three axes — **A** paragraph-level
(dilution/truncation/length-OOD, all fixed by splitting), **B** pipeline-structural, **C** NLI-atom
failures. Measured or mechanism-confirmed: A-1 dilution (FPR 0.530), C-1 neutral trap (drove the
paraphrase 0.599 failure), C-2 indiscriminate numeric contradiction (confirmed by COCO). Surviving
untested attacks: **B-1 clause fusion** (withhold punctuation → dilution returns), **B-2 cosine
misalignment** (corrupted clause routed past its true partner → veto never fires), **B-3
locally-true/globally-invalid chains** (bipartite is a bag of clauses; order/dependency discarded).
Two root-cause families: right-answers-failing (C-1/5/7/8: NLI treats not-obviously-entailed as
worthless + min amplifies) → fix by softening the pool + hard numeric/answer term;
wrong-answers-passing (B-1/2/3, C-3/4: structure discarded + NLI lacks cue-free semantics) → fix
by probe sets + structural alignment.

---

## Current status & queued next steps

**Established:** the reasoning-match reward trains faithful, image-grounded reasoning that converts
to accuracy where reasoning is the bottleneck; but its scorer must move from cosine toward an
NLI/hybrid design, and the naive hard-min veto is not shippable (recall 0.008 vs precision 0.599).

Queued (rough leverage order):
1. **Pooling sweep** (min → k-th-lowest quantile → mean), plotting negation-FPR vs paraphrase-FPR
   to find an operating point — needs a re-run saving full per-clause equiv vectors.
2. **SBERT/BERTScore clause-split control** — prove the clause-split win is NLI-specific
   (prediction: no rescue; their blindness is per-clause, not dilution).
3. **Adversarial probe sets** for the B-1/B-2/B-3 survivors and cue-free NLI-atom families —
   measure each attack before putting the reward in an RL loop.
4. **Hard numeric-match term** (parse numbers, compare exactly) — exempts numbers from NLI's
   no-magnitude-sense contradiction reflex; plus consider `coverage`→`equiv` as the per-clause edge.
5. **Multimodal grounding term** (box-IoU + region-CLIP blended into the similarity) — the COCO
   result is the quantitative case.
6. **Hybrid-reward GRPO run** vs the current arm on MMK12 — does a more reliable scorer train a
   better model? Plus the `w_answer=0` stretch ablation.

---

## Artifact map

| artifact | what |
|---|---|
| `tools/graph_match_reward.py` | composite reward (bipartite match + verl adapter) |
| `tools/clause_split.py`, `tools/sbert_embed_server.py` | clause splitter; subprocess SBERT server |
| `src/train/` | GRPO launch scripts + flash-attn shim |
| `src/data_factory/scienceqa_to_verl.py` | ScienceQA → verl parquet |
| `src/metrics/reliable/` | supervisor's reliability harness (commit 0ad601c) |
| `build_/score_/aggregate_{negation,coco_iou}*.py`, `*_clausesplit.py`, `build_paraphrase_pairs.py` | in-repo reliability benchmarks (VisualPRM negation, paraphrase control, COCO IoU) |
| `sensitivity_analysis.py` | AUC + τ-sweep re-aggregation |
| `run_*.sbatch` | SLURM launchers for all of the above |
| `models/qwen2_5vl-3b-scienceqa-{grpo,nomatch}-step*` | main-arm and ablation checkpoints (20…176) |
| `chunk_canonical/` + `canonical_chunked_v2.jsonl` | canonical.jsonl step-segmentation pipeline (marker cascade + release DeBERTa chunker + fence guard) and its verified 9.48M-row deliverable — see `docs/canonical_chunking_2026-07-25.md` |
| `docs/` | all result writeups; `research_brief.md` is the forward-looking dossier |
| `data/logs/validation_results{,_nomatch}/` | validation-campaign eval TSVs |

### Key numbers to remember

- Base Qwen2.5-VL-3B: 79% ScienceQA zero-shot; 31% MMK12 (≈ chance).
- Trained step 176: ScienceQA **90.3%**, match F 0.778, hallucination 0.09/resp; match trajectory 0.37→0.77.
- MMK12: 0.311 → 0.397 (answer-only) → **0.439** (with match), McNemar p = 0.021.
- Scorer reliability: SBERT/BERTScore FPR ≈ 1.0 on hard negatives; NLI ranking 0.98, negation FPR 0.001 — but 0.530 on buried numeric flips.
- Clause-split min-pool: negation FPR 0.530 → **0.008**, but paraphrase precision FPR **0.599** — don't ship hard min.
- COCO IoU rank accuracy: oracle 1.000 vs BERTScore 0.620 / SBERT 0.577 / NLI 0.503 (chance).
