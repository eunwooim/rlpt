# Text-Scorer Reliability Test — Results

Reproduction of the supervisor's reliability harness (`src/metrics/reliable/`, commit `0ad601c`
"codex metric test v1.0") on ASU Sol. Goal: measure whether candidate text scorers (SBERT cosine,
NLI, BERTScore) can reliably tell a correct caption from a **hard negative** (a near-identical
caption with one object/attribute/relation swapped or negated). This is the scorer-faithfulness
question underneath any process-reward / reasoning-match signal.

- **Env:** `rlpt-train` (torch 2.7.1+cu126, transformers 4.56.2, datasets 5.0.0). No verbatim build
  of the supervisor's lab-box `requirements.txt` (flash-attn glibc wall on Sol).
- **Hardware:** 1× A100-80GB, `public` partition, `--no-requeue`.
- **Command** (uncapped — dropped `--max_cases_per_subcategory`):
  ```
  python src/metrics/reliable/run_experiment.py \
    --datasets sugarcrepe,sugarcrepepp,negbench \
    --scorers sbert,nli,bertscore \
    --batch_size 32 --device auto --seed 42
  ```
  Identical to the README "overnight" command except the per-subcategory cap is removed (full data).

---

## How to read the metrics

Two evaluation modes, with **opposite** "good" directions:

- **Separation mode** (SugarCrepe, NegBench) → metric **`FPR@0.6` = false-positive rate at
  threshold τ=0.6. LOWER IS BETTER.** A "positive" = the scorer calls the pair a *match* (score ≥
  0.6). For a hard-negative pair that is the *wrong* call, so FPR = fraction of hard negatives the
  scorer wrongly accepts as matching. `0.97` = blind to the corruption; `0.07` = catches it.
- **Ranking mode** (SugarCrepe++) → metric **`rank_acc` = fraction of cases where the true positive
  is ranked above the hard negative. HIGHER IS BETTER. 0.50 = chance.** Below 0.50 means the scorer
  systematically prefers the *wrong* caption.

✅ = reliable for that role · ❌ = unreliable (would give a misleading reward).

---

## Job ① — SugarCrepe + SugarCrepe++ (uncapped) — COMPLETE

`src/outputs/reliable_full/` · 12,268 cases · 17,025 pairs · 51,075 scores · 0 scorer errors ·
runtime ~10 min. (NegBench skipped in this run — CSVs not yet present.)

### Headline

| dataset | mode | scorer (best field) | metric | value | verdict |
|---|---|---|---|---|---|
| sugarcrepe | separation (↓) | sbert `raw_score` | FPR@0.6 | **0.974** | ❌ blind |
| sugarcrepe | separation (↓) | bertscore `f1` | FPR@0.6 | **1.000** | ❌ blind |
| sugarcrepe | separation (↓) | nli `coverage` | FPR@0.6 | **0.069** | ✅ catches it |
| sugarcrepe++ | ranking (↑) | sbert `raw_score` | rank_acc | **0.725** | ~ mediocre |
| sugarcrepe++ | ranking (↑) | bertscore `f1` | rank_acc | **0.140** | ❌ worse than chance |
| sugarcrepe++ | ranking (↑) | nli `equiv` | rank_acc | **0.979** | ✅ reliable |

**Reading:** cosine (SBERT) and BERTScore react to *word overlap*, not meaning — hard negatives
share ~95% of tokens, so both wave them through (FPR 0.97–1.0) and BERTScore even ranks the wrong
caption higher (0.14 < 0.50 chance). **NLI**, which explicitly models entailment vs. contradiction,
separates them (FPR 0.07) and ranks correctly (0.98). Implication: a faithful reasoning-match /
PRM-style reward should be **entailment-based, not embedding-cosine-based.**

### Full scorer-overall table

| dataset | scorer | score_field | rank_acc | FPR@0.6 | mean_pos | mean_neg | mean_margin |
|---|---|---|--:|--:|--:|--:|--:|
| sugarcrepe | bertscore | bertscore_f1 |  | 1.000 |  | 0.9397 | -0.3397 |
| sugarcrepe | bertscore | bertscore_precision |  | 1.000 |  | 0.9327 | -0.3327 |
| sugarcrepe | bertscore | bertscore_recall |  | 1.000 |  | 0.9473 | -0.3473 |
| sugarcrepe | nli | C_ab |  | 0.5154 |  | 0.5201 | 0.0799 |
| sugarcrepe | nli | C_ba |  | 0.4902 |  | 0.4954 | 0.1046 |
| sugarcrepe | nli | E_ab |  | 0.0811 |  | 0.0919 | 0.5081 |
| sugarcrepe | nli | E_ba |  | 0.4645 |  | 0.4709 | 0.1291 |
| sugarcrepe | nli | nli_score_coverage |  | 0.0691 |  | 0.1787 | 0.4213 |
| sugarcrepe | nli | nli_score_equiv |  | 0.0691 |  | 0.2511 | 0.3489 |
| sugarcrepe | sbert | raw_score |  | 0.9744 |  | 0.8582 | -0.2582 |
| sugarcrepe++ | bertscore | bertscore_f1 | 0.1402 |  | 0.9008 | 0.9434 | -0.0426 |
| sugarcrepe++ | bertscore | bertscore_precision | 0.1278 |  | 0.8930 | 0.9429 | -0.0498 |
| sugarcrepe++ | bertscore | bertscore_recall | 0.1818 |  | 0.9089 | 0.9440 | -0.0351 |
| sugarcrepe++ | nli | C_ab | 0.0124 |  | 0.0119 | 0.8019 | -0.7900 |
| sugarcrepe++ | nli | C_ba | 0.0183 |  | 0.0093 | 0.7693 | -0.7600 |
| sugarcrepe++ | nli | E_ab | 0.9830 |  | 0.9644 | 0.1362 | 0.8283 |
| sugarcrepe++ | nli | E_ba | 0.9752 |  | 0.9606 | 0.1850 | 0.7756 |
| sugarcrepe++ | nli | nli_score_coverage | 0.9777 |  | 0.9576 | 0.1146 | 0.8430 |
| sugarcrepe++ | nli | nli_score_equiv | 0.9788 |  | 0.9566 | 0.1199 | 0.8367 |
| sugarcrepe++ | sbert | raw_score | 0.7248 |  | 0.9293 | 0.8402 | 0.0891 |

NLI field key: `E_ab/E_ba` = entailment (a→b / b→a), `C_ab/C_ba` = contradiction, `coverage` =
`clamp01(0.7·E_ab + 0.3·E_ba − 0.5·max(C))`, `equiv` = `clamp01(0.5·E_ab + 0.5·E_ba − 0.5·max(C))`.

### Comparison to supervisor's published 500-cap table

Uncapped numbers track his capped run closely (same conclusion, slightly more data):

| metric | his (cap 500) | ours (uncapped) |
|---|--:|--:|
| sugarcrepe sbert FPR@0.6 | 0.986 | 0.974 |
| sugarcrepe nli coverage FPR@0.6 | 0.100 | 0.069 |
| sugarcrepe++ sbert rank_acc | 0.627 | 0.725 |
| sugarcrepe++ nli equiv rank_acc | 0.968 | 0.979 |

---

## Job ② — + NegBench (uncapped) — COMPLETE (job 56959065)

`src/outputs/reliable_full_negbench/` · all three datasets · 0 scorer errors · runtime ~25 min.
NegBench CSVs sourced from the authors' Google Drive
([m1k2zoo/negbench](https://github.com/m1k2zoo/negbench)) into `data/negbench/` (text-only MCQ CSVs;
images not needed since the harness scores text pairs). Loads with 0 errors, 5 subcategories
(COCO / MSR-VTT / VOC2007 / HardNeg-Syn = 3 negs/case, CheXpert binary = 1). SugarCrepe and
SugarCrepe++ numbers are identical to Job ① (same seed/data) — see that table.

### NegBench headline (separation mode, FPR@0.6 ↓ lower better)

| scorer (best field) | FPR@0.6 | verdict |
|---|--:|---|
| nli `coverage` | **0.0013** | ✅ near-perfect |
| nli `equiv` | **0.0012** | ✅ near-perfect |
| sbert `raw_score` | **0.7531** | ❌ weak |
| bertscore `f1` | **0.9967** | ❌ blind |

**NegBench is the cleanest separation of the three datasets:** its hard negatives use explicit
*negation* ("a photo with **no** dog"), which NLI is purpose-built to catch (FPR 0.001 — essentially
zero false accepts). SBERT does better here than on SugarCrepe (0.75 vs 0.97) because negation words
perturb the embedding more than a quiet object-swap does — but 0.75 is still a failing score.
BERTScore stays blind (0.997): the negated caption shares nearly all tokens with the positive.

### NegBench full scorer-overall table

| scorer | score_field | FPR@0.6 | mean_neg | mean_margin |
|---|---|--:|--:|--:|
| bertscore | bertscore_f1 | 0.9967 | 0.8373 | -0.2373 |
| bertscore | bertscore_precision | 0.9967 | 0.8393 | -0.2393 |
| bertscore | bertscore_recall | 0.9959 | 0.8362 | -0.2362 |
| nli | C_ab | 0.9748 | 0.9697 | -0.3697 |
| nli | C_ba | 0.9648 | 0.9587 | -0.3587 |
| nli | E_ab | 0.0024 | 0.0072 | 0.5928 |
| nli | E_ba | 0.0084 | 0.0164 | 0.5836 |
| nli | nli_score_coverage | 0.0013 | 0.0021 | 0.5979 |
| nli | nli_score_equiv | 0.0012 | 0.0021 | 0.5979 |
| sbert | raw_score | 0.7531 | 0.6983 | -0.0983 |

(Separation mode has no positive pairs, so `mean_pos`/`rank_acc` are blank by design.)

### NegBench vs supervisor's published 500-cap table

| metric | his (cap 500) | ours (uncapped) |
|---|--:|--:|
| negbench nli coverage FPR@0.6 | 0.0025 | 0.0013 |
| negbench sbert FPR@0.6 | 0.6385 | 0.7531 |
| negbench bertscore f1 FPR@0.6 | 0.9968 | 0.9967 |

Same verdict on all three; the small NegBench shifts are because uncapped pulls the full per-subcat
counts (e.g. COCO/MSR-VTT/VOC2007/HardNeg up to ~1500 negative pairs each vs his 500 cap).

---

## Job ③ — Numerical sensitivity analysis (AUC + τ sweep)

Re-aggregation of the saved scores (`sensitivity_analysis.py`, 252,693 score rows, pure-CPU, no
models re-run). Where the reliability tables report **one number at one operating point** (FPR@0.6),
the sensitivity analysis asks **how robust that verdict is to the threshold** — via threshold-free
ROC-AUC and a full τ sweep. Outputs: `sensitivity_auc.csv`, `sensitivity_tau_sweep.csv`,
`sensitivity_operating_tau.csv` (full version: `docs/sensitivity_results.md`).

**Metrics.** ROC-AUC = P(a true match scores above a hard negative); threshold-free, 1.0 perfect,
0.5 chance, **<0.5 inverted**. Sensitivity (TPR) = fraction of true matches kept. Specificity = 1 − FPR.
True AUC needs both classes, so it's computed on **sugarcrepe++** (within-case positives vs hard
negatives) and a **pooled match-detector** (positives = SugarCrepe++ genuine matches; negatives = all
hard negatives). Separation datasets have no positives → characterized by the τ sweep.

### ROC-AUC (threshold-free)

| scorer (primary field) | AUC on hard negatives (sugarcrepe++) | AUC as match-detector (pooled) |
|---|--:|--:|
| **nli** `coverage` | **0.981** ✅ | **0.994** ✅ |
| sbert `raw_score` | 0.728 | 0.908 |
| bertscore `f1` | **0.178** ❌ inverted | 0.775 |

- **BERTScore 0.178 < 0.5 → anti-reliable**: it systematically ranks the *corrupted* caption above the
  true one (more shared tokens).
- **SBERT is a fine *general* match-detector (0.908) but collapses to 0.728 on within-case hard
  negatives** — it separates apples/oranges but not "red bench"/"blue bench." That gap is the
  reasoning-faithfulness failure.
- NLI's *contradiction* field gives AUC 0.01 → flipped (1−0.01) = **0.99 near-perfect hard-negative
  detector** on its own.

### τ sweep — FPR(τ) on hard negatives (lower better)

| dataset | scorer | τ=0.3 | τ=0.5 | τ=0.6 | τ=0.7 | τ=0.9 |
|---|---|--:|--:|--:|--:|--:|
| negbench | nli `coverage` | 0.003 | 0.002 | 0.001 | 0.001 | 0.0005 |
| negbench | sbert | 0.952 | 0.833 | 0.753 | 0.605 | 0.082 |
| negbench | bertscore `f1` | 1.000 | 1.000 | 0.997 | 0.986 | 0.069 |
| sugarcrepe | nli `coverage` | 0.134 | 0.078 | 0.069 | 0.062 | 0.040 |
| sugarcrepe | sbert | 1.000 | 0.993 | 0.974 | 0.918 | 0.440 |
| sugarcrepe | bertscore `f1` | 1.000 | 1.000 | 1.000 | 1.000 | 0.882 |

**NLI's low FPR holds across the entire τ range** — you cannot break it by moving the threshold.
SBERT/BERTScore only approach low FPR at τ≥0.9, and at that τ they also destroy sensitivity (next table).

### The trade-off — sugarcrepe++ ROC points (sensitivity vs specificity)

| scorer | τ | sensitivity (TPR ↑) | specificity (↑) | FPR (↓) |
|---|--:|--:|--:|--:|
| nli `coverage` | 0.70 | 0.957 | 0.908 | 0.092 |
| sbert `raw_score` | 0.90 | 0.800 | 0.622 | 0.378 |
| bertscore `f1` | 0.90 | 0.531 | 0.113 | 0.888 |

NLI holds **both** TPR and specificity high simultaneously. The others can only buy specificity by
sacrificing sensitivity — **there is no τ that makes them work.**

### Operating threshold τ\* to cap FPR at 5%

| dataset | nli `coverage` | sbert | bertscore `f1` |
|---|--:|--:|--:|
| negbench | 0.000 | 0.919 | 0.905 |
| sugarcrepe | 0.820 | 0.982 | 0.985 |
| sugarcrepe++ | 0.935 | 0.985 | 0.988 |

NLI reaches a 5% false-accept rate at a **reachable** τ that still keeps most true matches; SBERT and
BERTScore need τ≈0.98–0.99, where almost nothing passes (TPR collapses). NegBench NLI τ\*=0.000 means
hard negatives all cluster at ~0 coverage — essentially perfect separation.

---

## Bottom line

Convergent with the DCI analysis and the supervisor's benchmark: **embedding-cosine and token-overlap
scorers are unreliable on hard negatives; NLI/entailment is reliable.** The sensitivity analysis
strengthens this from a single operating point to a **threshold-free, τ-robust** conclusion: NLI's
advantage is not an artifact of the τ=0.6 cutoff (AUC 0.98–0.99, stable across the whole sweep), while
**no choice of τ rescues SBERT or BERTScore** — any threshold low enough to catch hard negatives also
rejects the true matches. A generalizable, cross-domain reasoning evaluator (PRM) should be built on an
entailment signal.
