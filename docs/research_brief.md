# RLPT — Research Brief: Reasoning-Faithfulness Rewards for Multimodal RL

*Self-contained dossier of everything done and planned on this project, written so it can be
dropped into an LLM as research context. Project: RL post-training of a multimodal model with a
reward that scores the **content of its reasoning**, not just its final answer.
Stack: Qwen2.5-VL-3B-Instruct · GRPO (verl) · ASU Sol cluster.*

---

## 1. The research question

**Does rewarding a multimodal model on the *content of its free-text reasoning* (not only its
final answer) make it a better image analyzer — and can we build a reward signal for that which
is faithful (rewards correct reasoning) rather than gameable (rewards stylistic mimicry of the
training references)?**

Two sub-questions that organize the whole project:
1. **Reward design** — how to score a free-text reasoning trace against a ground-truth
   explanation in a way that resists reward hacking.
2. **Reward reliability** — does that scorer actually distinguish a correct statement from a
   minimally-corrupted wrong one (hard negatives, negation, attribute/relation swaps)?

---

## 2. How we got here (project arc)

The reward design went through three stages; the dropped stages explain why the final design
looks the way it does.

1. **QCVSR (typed schema + verifier)** — early track. A synthetic GQA dataset where the model
   emitted a *typed evidence object* (bounding-box pairs / count pairs) + an answer, checked by a
   deterministic verifier. Robust but narrow; not free-text reasoning.
2. **Caption → frozen parser → scene-graph match** — brief intermediate plan. Dropped because a
   parser *in the reward loop* is a reward-hacking magnet: RL can raise reward by learning
   *parser-friendly phrasing* instead of seeing better.
3. **Free-text description scored by soft bipartite matching (FINAL).** The model emits a
   natural-language reasoning trace; the reward soft-matches it against ground-truth facts with
   **no parser in the loop**. Removing the parser removed the biggest confound. This is the
   current design.

Dataset also pivoted: from synthetic GQA to **ScienceQA** (real multimodal exam questions with
free-text `lecture`/`solution` explanations), because we needed ground-truth *reasoning text* to
match against, not just object annotations.

---

## 3. The reward (current design)

The model is trained to emit `<think>reasoning</think><answer>choice</answer>`. The reward is a
**composite**:

> **R = w_f·format + w_m·match + w_a·answer − w_p·pun**, weights **1 / 2 / 5 / 1**
> (STAR-R1-shaped: format gate + dense partial credit + hard verifiable term + per-mistake penalty).

| term | what it measures | weight |
|---|---|---|
| **format** | output has exactly one `<think>` and one `<answer>` (hard gate) | ×1 |
| **match** | **bipartite reasoning-match** of `<think>` vs the GT `solution` (the novel term) | ×2 |
| **answer** | hard MCQ correctness (exact text / letter / index match) | ×5 |
| **pun** | −1 per reasoning clause that matches **no** GT fact (hallucination penalty) | −1 |

### 3.1 The bipartite reasoning-match term (the core contribution)

Both the model's `<think>` text and the GT `solution` are **free text**. The match term:
1. **Clause-splits both sides** into near-atomic claims with a deterministic spaCy dependency
   parser (no LLM in the loop — it splits boundaries, not truth).
2. **Embeds** every clause with **SBERT** (`all-MiniLM-L6-v2`). GT-side embeddings are **cached**
   (fixed); only the prediction side runs per step.
3. Builds the **cosine-similarity matrix S** (predicted clauses × GT clauses).
4. Solves the **one-to-one optimal assignment** with the **Hungarian algorithm**
   (`scipy.linear_sum_assignment` on −S). One-to-one (not greedy) blocks the "say the same fact
   five ways" exploit — each GT fact can be claimed by at most one predicted clause. A pair only
   counts if cosine ≥ **τ = 0.15**.
5. Reads off a **precision-dominant Fβ** (β = 0.5) from the matched similarity mass M:
   - **Precision = M / (#predicted clauses)** — of what I said, how much is supported.
   - **Recall = M / (#GT clauses)** — of the GT, how much I covered.
   - **Fβ = (1+β²)·P·R / (β²·P + R)**, β=0.5 → precision dominates (prefer fewer well-grounded
     claims over padded coverage; recall is a guardrail).

"Graph" = the bipartite *matching* graph (two columns of clause-nodes, edges weighted by
embedding similarity), **not** a scene graph.

### 3.2 Why the hard answer term (×5) is necessary

SBERT cosine is **blind to entity-swaps and negation**: a fluent-but-wrong answer scored ~0.41
vs a correct 0.63 under match-only — nearly indistinguishable. The hard answer-correctness term
fixes that and dominates the reward; the composite separated those cases to **1.8 vs 7.3**. This
blindness is the seed of the whole reliability investigation (§6–7).

### 3.3 Implementation notes (verl integration)

- `compute_score(data_source, solution_str, ground_truth, extra_info)` is a verl drop-in
  (cached singleton). `ground_truth = {answer, solution, choices}`. Returns a **dict**
  `{score, acc, match, format, n_hallucinated}` so verl logs each component separately (it uses
  `score` as the reward, logs the rest).
- Reward weights are **env-overridable** (`RLPT_W_{FORMAT,MATCH,ANSWER,PUN}`) — this is how the
  causal ablation (§5.3) was run without code changes.
- SBERT runs in a **subprocess** (`sbert_embed_server.py`) because verl leaves its reward worker
  in a meta/fast-init torch state that breaks in-process model load.

---

## 4. Training setup

- **Base model:** Qwen2.5-VL-3B-Instruct (3B chosen for iteration speed; supervisor-measured
  **79% zero-shot** on ScienceQA with lenient parsing).
- **Algorithm:** GRPO (Group-Relative Policy Optimization) via **verl 0.8.0 / vLLM 0.10.1.1 /
  torch 2.7.1+cu126 / transformers 4.56.2**.
- **GRPO group:** `rollout.n = 5` — 5 sampled completions ("attempts") per question; each
  rollout's advantage is its reward minus the group mean (the group is the baseline; no separate
  critic). KL coef **0.001** (loose anchor to the frozen base model — enough to prevent
  reward-hacking collapse, loose enough to allow real change).
- **One step** = one batch (64 questions) → generate 5 rollouts each (320 completions) → score →
  compute advantages → FSDP weight update → sync weights to vLLM. **88 steps ≈ 1 epoch**;
  trained **2 epochs = 176 steps**. Checkpoints every 20 steps.
- **Cluster reality (ASU Sol):** RHEL8 glibc 2.28, BeeGFS `/scratch`, A100s. No real flash-attn
  (needs glibc ≥ 2.32) → a pure-torch/SDPA **shim** on PYTHONPATH. FSDP CPU offload + low vLLM
  GPU fraction to fit FSDP+vLLM on one 80GB A100 at the train↔rollout handoff. Always
  `#SBATCH --no-requeue`.

---

## 5. Results

### 5.1 ScienceQA (in-domain), final step 176, 1,836-row held-out test

| metric | value |
|---|---|
| **answer accuracy** | **90.3%** (base 79% zero-shot → **+11.3 pts**) |
| format validity | 100% |
| reasoning-match F | 0.778 |
| hallucinated clauses / response | 0.091 |
| composite reward | 6.98 |

**The decomposition is the real finding.** Across the run (256-row tracking subset):
- **format saturates immediately** — 0.48 (base) → 1.00 by step 20.
- **accuracy is mostly there early** — ~0.82 by step 20 → ~0.89–0.90 (noisy).
- **reasoning match climbs steadily and monotonically — 0.371 → 0.625 (step 20) → 0.766 (176)**,
  roughly *doubling*, long after format/answers saturated.

So the headline accuracy *understates* what training did: the differentiated learning signal is
in the reasoning channel, which an answer-only metric cannot see.

> Note on two ScienceQA accuracy numbers: **90.3%** = full 1,836-row held-out test (benchmark
> headline). **87.5%** = the same model on a fixed **256-row** subset used for controlled
> base/ablation/ours comparisons (strict greedy parse). Same model, different eval set + ~±2-pt
> sampling noise — not a contradiction.

### 5.2 MMK12 (OOD hard benchmark, never trained on it)

ScienceQA accuracy is near a **ceiling**, masking whether the match term *causes* accuracy gains.
MMK12 (2,000 K12 exam MCQs, base ≈ chance) separates the arms. Balanced 1,024-row sample:

| model | accuracy | format |
|---|---|---|
| base (zero-shot) | 0.311 | 0.18 |
| answer-only ablation (w_match=0) | 0.397 | 0.99 |
| **ours (with match)** | **0.439** | 0.99 |

**+4.1 pts from the match term at identical format**, paired **McNemar exact p = 0.021**
(significant). Where reasoning is the bottleneck (base ≈ chance), training the reasoning channel
**converts directly into answer accuracy**. The "no accuracy difference on ScienceQA" was a
ceiling effect, not a property of the term.

### 5.3 The validation campaign — ruling out "stylistic mimicry"

The match climb is partly circular (we optimized it) and SBERT cosine could be rewarding
*style mimicry* of ScienceQA solutions rather than better image analysis. **Design principle:
mimicry is style-specific + image-independent; real reasoning is content-specific +
image-dependent.** Five tests, all PASS:

1. **Shuffled-GT control** — score `<think>` against GT solutions of *other* same-topic questions
   (style-matched, content-mismatched). Own-GT match 0.38 → 0.76 while shuffled-GT stays flat
   ~0.22. The gap *widens* (0.22 → 0.53). Mimicry would lift both; it didn't.
2. **Style-insensitive re-scoring** — replace SBERT cosine with **NLI entailment** (DeBERTa-MNLI).
   Own-GT 0.21 → 0.52, shuffled flat ~0.07 → the gain is **propositional content**, not phrasing.
   A *different* embedding model tracks the reward encoder ~1:1 → no encoder hacking.
3. **Image-swap probe** — eval with swapped images. Accuracy collapses ~−0.21 at every checkpoint
   (answers are image-dependent). Reasoning-match swap-sensitivity *grows* with training
   (−0.048 base → −0.079 step 176) → RL **increased image-grounding** of the reasoning.
4. **A-OKVQA transfer** — differently-styled benchmark, zero training: base 0.43 → step 176
   **0.84** accuracy. Gains transfer; style can't.
5. **w_match=0 ablation (the causal keystone)** — identical config/seed/steps, reward =
   format+answer only. **Ablation own-GT match: 0.38 → 0.46 then flat (0.42 at step 176)**, vs
   main arm 0.38 → 0.76. NLI entailment **never moves** in the ablation; image-grounding stays at
   base level. So everything beyond a small step-20 answer-only bump is **caused by the match
   term**. Honest caveat: on ScienceQA (ceiling) the term buys *faithfulness + grounding, not
   accuracy* — but on MMK12 (§5.2) it **does** buy accuracy OOD.

**Publishable claim, three legs + one:** content-not-style (1,2) · image-grounded (3,4) ·
match-term-causal for faithfulness (5) · **and OOD accuracy gains** (MMK12).

---

## 6. Reward reliability — the open problem

The validation showed the *trained model* improves, but it also exposed that **the reward's text
scorer (SBERT cosine) is itself unreliable** on adversarial cases. This is now an active
investigation track.

### 6.1 DCI analysis (our internal probe)

Using **Densely Captioned Images** (each caption unit has ~10 correct paraphrases + aligned
hard negatives `{swaps, layout, basic}`), we measured whether the scorer ranks a correct
paraphrase above a corruption. Findings:
- **SBERT is *consistent* but structurally blind**: ~chance on genuine content corruptions
  (layout/basic ≈ 0.52–0.57 discrimination), and **worse than chance on swaps** — it scores the
  swap (mean cos 0.88) *above* a faithful paraphrase (0.76), because swaps are lexical near-copies
  of the anchor while paraphrases reword.
- **NLI entailment is far better on content changes** (0.92/0.83 discrimination) — it catches
  contradictions cosine can't — but is itself fooled by near-verbatim swaps and over-penalizes
  detail-adding paraphrases.
- **Embedding-inversion probe**: GT embeddings *retain* the swap-relevant content (round-trip
  recovers adjectives/numbers) → it's a **scoring-geometry problem (cosine throws the
  information away), not an information-bottleneck**.
- **Attribute-swaps resist BOTH text methods** → argues for a **multimodal grounding term**
  (which object has the attribute is a pixel question, not a text question).

### 6.2 Supervisor's benchmark (`src/metrics/reliable/`, "codex metric test v1.0")

The supervisor productionized the reliability question onto **standard public benchmarks** so
results are reviewer-recognizable. A self-contained harness that normalizes captions into
`Case(anchor, positive, negatives)` and runs two modes:
- **Separation mode** (SugarCrepe, NegBench): positive vs hard-negatives; diagnostic =
  **false-positive rate** = fraction where `score(positive, negative)` exceeds τ (over a τ grid).
- **Ranking mode** (SugarCrepe++): pass if `score(anchor, positive) > score(anchor, negative)`;
  reports ranking accuracy.

Three scorers (raw `transformers`, bigger models than our reward): **sbert** (mpnet cosine),
**nli** (deberta-large-mnli, **bidirectional** entailment with `coverage`/`equiv` composites),
**bertscore** (deberta-xlarge-mnli token P/R/F1). τ calibration deliberately deferred; saves all
raw values. **v1 results (500 cases/subcat) corroborate DCI on standard data:**
- **Cosine (SBERT) unreliable on hard negatives** — SugarCrepe separation FPR@0.6 ≈ **0.99**;
  SugarCrepe++ ranking accuracy only **0.63**.
- **BERTScore equally fooled in separation** (FPR ≈ 1.0; ranking 0.14) — lexical overlap.
- **NLI dramatically better** — SugarCrepe++ ranking **≈ 0.97**; NegBench negation separation
  FPR collapses to **~0.003**. Bidirectional entailment catches contradictions/negation.

**Convergent conclusion (DCI + standard benchmarks): cosine and BERTScore cannot be trusted on
hard negatives / negation; NLI (bidirectional entailment) can.**

---

## 7. What we're trying to do next (open directions)

Ordered roughly by leverage:

1. **Hybrid reward scorer** — SBERT cosine for *coverage* + NLI entailment for *faithfulness* on
   content changes, with an SBERT prefilter to bound the cross-encoder cost (one NLI pass per
   matched clause, not the full P×G matrix). Add an NLI-matcher variant to the reward, re-run the
   DCI 1/4 tests, re-score the existing validation generations, then **one hybrid-reward GRPO run
   vs the current arm on MMK12** to test whether a more reliable scorer trains a better model.
2. **τ calibration + hyperparameter search** — the supervisor's harness deliberately defers this;
   it's the next script (turn the raw-value tables into a calibrated decision threshold per
   scorer/mode).
3. **Multimodal grounding term** — attribute-swaps resist all text methods, so blend image
   grounding into the similarity: `S = w1·text + w2·box-IoU + w3·region-CLIP(GT-crop, span)`.
   GT-side image embeddings are fixed → precompute/cache; only the description side runs per step.
   This is the strongest defense against rephrasing exploits because it grounds reward in pixels.
4. **`w_answer=0` stretch ablation** — does the match term *alone* (no answer reward) lift MCQ
   accuracy? If yes, strongest evidence the term trains real reasoning, not just faithfulness.
5. **Scale / transfer** — bigger base model; more OOD benchmarks; check whether the
   faithfulness→accuracy conversion (ScienceQA ceiling vs MMK12) holds at scale.

**The throughline:** we have evidence the reasoning-match reward trains *faithful, image-grounded*
reasoning that converts to accuracy where reasoning is the bottleneck (legs 1–5 above) — but the
*reward's own scorer* is unreliable on adversarial text, so the frontier is **making the scorer
faithful** (NLI/hybrid/multimodal) and proving a better scorer trains a better model.

---

## 8. Key facts / numbers to anchor research

- Base Qwen2.5-VL-3B: **79%** ScienceQA zero-shot (lenient); **31%** MMK12 (≈ chance).
- Trained (step 176): ScienceQA **90.3%**, format 100%, match F **0.778**, hallucination
  **0.09/resp**, composite **6.98**.
- Reasoning-match trajectory: **0.37 → 0.63 (step 20) → 0.68 (40) → 0.77 (176)**.
- MMK12: base 0.311 → answer-only 0.397 → **with-match 0.439** (McNemar p=0.021).
- Reward: composite `1·format + 2·match + 5·answer − 1·pun`; match = Hungarian one-to-one over
  SBERT-MiniLM clause cosines, precision-dominant **Fβ (β=0.5, τ=0.15)**.
- Reliability: cosine FPR@0.6 ≈ 0.99 on SugarCrepe hard negatives; NLI ranking ≈ 0.97, negation
  separation FPR ≈ 0.003.

*(Companion docs in this repo: `docs/results_scienceqa_mmk12.md`, `docs/reward_reliability_dci.md`,
`docs/presentation_briefing.md`; reward code `tools/graph_match_reward.py`; supervisor benchmark
`src/metrics/reliable/`. Full chronology in `CLAUDE.md`.)*
