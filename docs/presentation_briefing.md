# Presentation briefing — RLPT GRPO reward & ScienceQA/MMK12 results

*Consolidated Q&A for the presentation. Qwen2.5-VL-3B-Instruct · GRPO (verl) · ASU Sol.*

---

## 0. The big picture (one sentence)

We GRPO-train Qwen2.5-VL-3B so that for each image+question it emits
`<think>reasoning</think><answer>choice</answer>`, and we reward it with a **composite**
of four terms — is the format valid, does the reasoning *match the ground-truth solution*
(via bipartite matching), is the answer right, and a penalty for made-up reasoning.

---

## 1. Rollout (what GRPO generates)

- A **rollout** = one sampled completion (one `<think>…</think><answer>…</answer>`) from the
  policy for a given prompt.
- `rollout.n = 5` → for **each** training question the model produces **5 rollouts**
  ("attempts"). That group of 5 *is* the GRPO group.
- GRPO scores all 5, computes each one's advantage **relative to the group mean** (no separate
  critic/value network — the group is the baseline), and pushes probability toward the
  above-average rollouts.
- Engine: vLLM generates rollouts → verl computes the reward → FSDP updates weights → weights
  sync back to vLLM. Group size 5, KL coef 0.001, train batch 64.

**Soundbite:** "Rollout = one sampled reasoning+answer attempt. We sample 5 per question and
let them compete; the reward decides which attempts get reinforced."

---

## 2. What is one "step" / iteration? (176 steps)

One **step** = one full GRPO update on one batch of questions:

1. Take **64 questions** (the batch).
2. Generate **5 rollouts each** → 64 × 5 = **320 completions** (vLLM).
3. **Score all 320** with the composite reward.
4. **Compute GRPO advantages** — within each group of 5, each rollout vs its group mean.
5. **Update the policy weights** (FSDP) toward above-average rollouts.
6. **Sync** new weights back into vLLM for the next step.

That whole cycle = one step (`step:1`, `step:2`, …).

**How 176 falls out:** steps per epoch = 5,678 rows ÷ 64 batch ≈ **88**; epochs = 2 →
88 × 2 = **176 steps** (the model saw the whole training set twice). Checkpoints saved every
20 steps; step 176 = final checkpoint.

| term | what it is | our number |
|---|---|---|
| rollout | one sampled completion (one attempt) | 5 per question |
| step (iteration) | one batch → generate → score → weight update | 176 total |
| epoch | one full pass over all training data | 2 (= 176 steps) |

---

## 3. KL coefficient (= 0.001)

A penalty weight that keeps the policy from drifting too far from a **frozen reference model**
(the base model before RL).

- Reward effectively becomes `reward − KL_coef · KL(new ‖ reference)`.
- **Why:** without a leash, GRPO can collapse into a degenerate high-reward shortcut (repetition,
  format hacking, gibberish that scores) and forget general language ability.
- **Why 0.001 (small/loose):** we *want* substantial behavior change (learn our format + better
  reasoning), so we don't over-constrain — but it's still enough to prevent collapse.

**Soundbite:** "KL coef is how hard we anchor the training model to the original base model.
0.001 is a loose anchor — enough to stop reward-hacking collapse, loose enough to let it change."

---

## 4. ScienceQA format validity

- `format_ok` is a hard gate: output must contain **exactly one** `<think>…</think>` **and exactly
  one** `<answer>…</answer>`. 1 point if yes, 0 if no.
- This is why the base model looks "bad" under our strict harness — it *answers correctly* but
  doesn't wrap it in our tags, so the parser can't extract the answer (base is actually 79%
  zero-shot).
- **Result: format saturates almost instantly** — 0.48 (base) → **1.00 by step 20**, stays there.

---

## 5. Reasoning matching (the heart of the contribution)

Compare the model's **`<think>` text** to the ScienceQA **`solution`** text — both *free text*,
**no parser, no scene graph**:

1. **Clause-split both sides** (deterministic spaCy dep-parse, no LLM) → near-atomic claim chunks.
2. **Embed** every clause with **SBERT (`all-MiniLM-L6-v2`)**. GT side **cached**; only prediction
   side runs per step.
3. Build the **cosine-similarity matrix S** (rows = predicted clauses, cols = GT clauses).
4. **Bipartite one-to-one matching** (Hungarian).
5. Read off **precision / recall / Fβ**.

**Why bipartite not greedy:** one-to-one assignment blocks the "say the same fact five ways"
exploit — each GT fact claimed by at most one predicted clause.

**Result — the headline finding:** reasoning match climbs **monotonically 0.371 (base) → 0.625
(step 20) → 0.766 (step 176)** — roughly *doubles* — while format and accuracy saturate early.
RL kept improving *reasoning faithfulness* long after it solved format and answers.

---

## 6. Bipartite matching — nodes, vertices, edges

**Node = vertex** (the dots). **Edge** = a line connecting two nodes. **Bipartite** = nodes split
into two groups; edges only go *between* groups, never within.

In our reward:
- **Left-side nodes** = clauses from the model's `<think>` reasoning.
- **Right-side nodes** = clauses from the ground-truth `solution`.
- **Edge weight** between left i and right j = **cosine similarity S[i][j]** of their SBERT
  embeddings.
- **The matching** = pick a subset of edges so each node touches at most one chosen edge
  (one-to-one), maximizing total weight (Hungarian algorithm, `scipy.linear_sum_assignment`).
- A pair only counts if cosine ≥ **τ = 0.15**.

```
  PREDICTED clauses          GT clauses
   (left nodes)             (right nodes)
      p1  ●───────0.81──────● g1
      p2  ●───────0.74──────● g2
      p3  ●        (no edge ≥0.15 → unmatched = hallucination)
```

"Graph" here = the **bipartite matching graph**, NOT a scene graph (the scene-graph design was
dropped to remove the parser reward-hacking confound).

---

## 7. The F score / Fβ

An **F-score** combines **precision** and **recall** into one number. **Fβ** tilts the balance
with a knob β.

- **Precision** = M / (number of predicted clauses) → of what I said, how much is supported.
- **Recall** = M / (number of GT clauses) → of the GT, how much I covered.
  (M = matched similarity **mass** = sum of cosines of the kept pairs.)
- Formula: `Fβ = (1 + β²) · P · R / (β²·P + R)`.
- β = 1 → equal (F1); β > 1 → recall-heavy; **β < 1 → precision-dominant.** We use **β = 0.5**.

**Why β=0.5:** we'd rather the model say **fewer, well-grounded** things than pad with
unsupported coverage. Precision is the priority; recall is a guardrail so it can't win by saying
almost nothing.

**Worked example:** P = 0.8, R = 0.4, β² = 0.25 →
`Fβ = 1.25·0.8·0.4 / (0.25·0.8 + 0.4) = 0.4 / 0.6 = 0.667` (F1 would be 0.533 — Fβ<1 pulls toward
the higher precision). This Fβ ∈ [0,1] is the `match` term.

---

## 8. Rollout segments not matching GT (the punishment term)

- After matching, any **predicted clause with no GT partner** (above τ) = an **unmatched /
  hallucinated clause** (`n_unmatched = n_pred − matched`).
- **Subtract 1 per unmatched clause** (`w_pun = 1`) — penalizes invented reasoning.
- **Result:** trained model averages just **0.09 hallucinated clauses per response** on the
  held-out test.

**What 0.09 per response means:** about **9 of every 100 answers** contain a single hallucinated
clause; the other ~91 have zero. It is **per response, already averaged** over all test responses.

**Is it 0.09 × 5 because of 5 rollouts? NO — it's 0.09.**
1. Each rollout is scored **independently**; the 5 in a GRPO group are never summed into one reward.
2. 0.09 is **already a per-response average** — averaging five ~0.09 numbers is still ~0.09.
3. The 6.98 / 0.09 numbers are from the held-out **test** (`val_only`), which uses **one greedy
   generation per question**, not 5. `rollout.n=5` is a **training-time** setting only.

---

## 9. The composite reward

**R = w_f·format + w_m·match + w_a·answer − w_p·pun**, weights **1 / 2 / 5 / 1** (STAR-R1-shaped):

| term | what | weight |
|---|---|---|
| format | valid `<think>`/`<answer>` (gate) | ×1 |
| match | precision-dominant bipartite F (reasoning vs GT solution) | ×2 |
| answer | hard MCQ correctness (exact text / letter / index) | ×5 |
| pun | −1 per hallucinated clause | −1 |

**Why a hard answer term (×5):** SBERT cosine is **blind to entity-swaps and negation** — a
fluent-but-wrong answer scored ~0.41 vs a correct 0.63 under match-only. The hard
answer-correctness term fixes that; composite separated those cases to **1.8 vs 7.3**.

### How the held-out composite = 6.98 (step 176)

Each term = mean of that metric over all evaluated responses:

| term | metric value | weight | contribution |
|---|---|---|---|
| format | 1.000 | ×1 | +1.000 |
| match | 0.778 | ×2 | +1.556 |
| answer (accuracy) | 0.903 | ×5 | +4.515 |
| pun (hallucinated) | 0.091 | ×1 | −0.091 |
| | | **total** | **6.980** |

`1.000 + 1.556 + 4.515 − 0.091 = 6.98`. The accuracy term (×5) dominates by design.

---

## 10. Results

### ScienceQA (in-domain), step 176, 1,836-row held-out test
- **Accuracy 90.3%** (base 79% zero-shot → **+11.3 pts**), format 100%, reasoning-match F 0.778,
  hallucination 0.09/resp, composite reward 6.98.

**The decomposition is the real story:** format saturates by step 20, accuracy mostly there early
(~0.82 → ~0.90, noisy), but **reasoning match climbs steadily 0.37 → 0.77**. The accuracy headline
*understates* what training did — the differentiated signal is in the reasoning channel.

Learning curve (256-row test subset):

| step | accuracy | reasoning match | format | reward |
|---|---|---|---|---|
| base | 0.645 | 0.371 | 0.480 | −0.31 |
| 20 | 0.824 | 0.625 | 1.000 | 6.30 |
| 40 | 0.848 | 0.682 | 1.000 | 6.53 |
| 80 | 0.883 | 0.723 | 1.000 | 6.75 |
| 120 | 0.879 | 0.741 | 1.000 | 6.74 |
| 176 | 0.891 | 0.766 | 1.000 | 6.91 |

### MMK12 (OOD, never trained on it) — proof the match term *causes* gains

| model | accuracy | format |
|---|---|---|
| base | 0.311 | 0.18 |
| RLVR / answer-only ablation (w_match=0) | 0.397 | 0.99 |
| **ours (with match)** | **0.439** | 0.99 |

**+4.1 pts from the match term at identical format**, paired **McNemar p = 0.021** (significant).
On ScienceQA both arms tie on accuracy (ceiling effect); on MMK12 where base ≈ chance and reasoning
is the bottleneck, training the reasoning channel **converts to answer accuracy**.

**The three-leg claim:** the gain is (1) content not style [shuffled-GT + NLI controls],
(2) image-grounded [image-swap + A-OKVQA transfer], (3) **caused by the match term** [w_match=0
ablation], and it transfers to OOD accuracy [MMK12].

---

## 11. Likely Q&A traps

- **"Why 90.3% sometimes, 87.5% other times?"** 90.3% = full 1,836-row held-out test (the benchmark
  number). 87.5% = a fixed 256-row subset used for the *controlled* base/RLVR/ours comparison (only
  set where all three arms were scored identically). Same model, different eval set + ~±2-pt
  sampling noise.
- **"Isn't optimizing your own match metric circular?"** Yes — that's why we ran shuffled-GT, NLI
  re-scoring, image-swap, A-OKVQA, and the ablation controls. The climb survives a *different*
  scorer (NLI) and a *different* encoder, and the ablation shows it's causal.
- **"Why bipartite not greedy / why β=0.5?"** One-to-one blocks repetition farming; β<1 prefers
  grounded precision over padded coverage.
- **"Parser in the loop?"** No — free-text on both sides; dropping the parser removed the biggest
  reward-hacking confound.
