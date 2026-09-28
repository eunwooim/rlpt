# Reward-Scorer Reliability Benchmark Suite — Consolidated Notes

Compiled 2026-08-24 from: `docs/{reliability_results,sensitivity_results,negation_clausesplit_results,paraphrase_clausesplit_results,visualprm_negation_results,coco_iou_results,reward_reliability_dci}.md`, `data/visualprm_v11_filtered/sympy_audit/REPORT.md`, `.claude/skills/reward-reliability/SKILL.md`, `docs/project_history.md` (taxonomy). The cross-experiment index beside the per-experiment reports.

Known gaps flagged at compile time: attacks C-3 through C-8 exist only as bare numbers in `docs/project_history.md:226-228` (no on-disk descriptions; kept as untested placeholders). Scorer-ID discrepancy: `docs/research_brief.md:238` says `deberta-large-mnli`, but `src/metrics/scorers/nli.py:14` says `microsoft/deberta-xlarge-mnli` — the code value is used throughout here.

## 0. Shared scorer definitions (used across all experiments)

Model IDs from `src/metrics/scorers/`:

| scorer | model | scoring |
|---|---|---|
| `sbert` | `sentence-transformers/all-mpnet-base-v2` | cosine of L2-normalized embeddings |
| `nli` | `microsoft/deberta-xlarge-mnli` | bidirectional entailment, composites below |
| `bertscore` | `microsoft/deberta-xlarge-mnli` | token-level P / R / F1 |
| `cross_nli` (present, not in headline tables) | `cross-encoder/nli-deberta-v3-large` | — |

NLI composites (from `docs/reliability_results.md:86-87`):
- `E_ab / E_ba` = entailment a→b / b→a; `C_ab / C_ba` = contradiction
- `coverage = clamp01(0.7·E_ab + 0.3·E_ba − 0.5·max(C))`
- `equiv = clamp01(0.5·E_ab + 0.5·E_ba − 0.5·max(C))`

Metric conventions:
- **Separation mode** (SugarCrepe, NegBench): `FPR@τ` = fraction of hard negatives scored ≥ τ. **Lower better.**
- **Ranking mode** (SugarCrepe++): `rank_acc` = fraction of cases where true positive outranks hard negative. **Higher better; 0.50 = chance.**

---

## 1. Public-benchmark reliability harness (SugarCrepe / SugarCrepe++ / NegBench)

**Goal.** Can candidate text scorers (SBERT cosine, NLI, BERTScore) reliably tell a correct caption from a hard negative (near-identical caption with one object/attribute/relation swapped or negated)?

**Method.** Reproduction of the supervisor's harness `src/metrics/reliable/` (commit `0ad601c`, "codex metric test v1.0") on ASU Sol. Uncapped (per-subcategory cap `--max_cases_per_subcategory` dropped):

```
python src/metrics/reliable/run_experiment.py \
  --datasets sugarcrepe,sugarcrepepp,negbench \
  --scorers sbert,nli,bertscore \
  --batch_size 32 --device auto --seed 42
```
Env `rlpt-train` (torch 2.7.1+cu126, transformers 4.56.2, datasets 5.0.0); 1× A100-80GB, `public` partition, `--no-requeue`.

### Job ① — SugarCrepe + SugarCrepe++ (uncapped, COMPLETE)
`src/outputs/reliable_full/` · **12,268 cases · 17,025 pairs · 51,075 scores · 0 scorer errors · ~10 min**.

Headline:

| dataset | mode | scorer (best field) | metric | value | verdict |
|---|---|---|---|--:|---|
| sugarcrepe | separation ↓ | sbert `raw_score` | FPR@0.6 | 0.974 | blind |
| sugarcrepe | separation ↓ | bertscore `f1` | FPR@0.6 | 1.000 | blind |
| sugarcrepe | separation ↓ | nli `coverage` | FPR@0.6 | 0.069 | catches |
| sugarcrepe++ | ranking ↑ | sbert `raw_score` | rank_acc | 0.725 | mediocre |
| sugarcrepe++ | ranking ↑ | bertscore `f1` | rank_acc | 0.140 | worse than chance |
| sugarcrepe++ | ranking ↑ | nli `equiv` | rank_acc | 0.979 | reliable |

Full scorer-overall table:

| dataset | scorer | score_field | rank_acc | FPR@0.6 | mean_pos | mean_neg | mean_margin |
|---|---|---|--:|--:|--:|--:|--:|
| sugarcrepe | bertscore | bertscore_f1 | | 1.000 | | 0.9397 | −0.3397 |
| sugarcrepe | bertscore | bertscore_precision | | 1.000 | | 0.9327 | −0.3327 |
| sugarcrepe | bertscore | bertscore_recall | | 1.000 | | 0.9473 | −0.3473 |
| sugarcrepe | nli | C_ab | | 0.5154 | | 0.5201 | 0.0799 |
| sugarcrepe | nli | C_ba | | 0.4902 | | 0.4954 | 0.1046 |
| sugarcrepe | nli | E_ab | | 0.0811 | | 0.0919 | 0.5081 |
| sugarcrepe | nli | E_ba | | 0.4645 | | 0.4709 | 0.1291 |
| sugarcrepe | nli | nli_score_coverage | | 0.0691 | | 0.1787 | 0.4213 |
| sugarcrepe | nli | nli_score_equiv | | 0.0691 | | 0.2511 | 0.3489 |
| sugarcrepe | sbert | raw_score | | 0.9744 | | 0.8582 | −0.2582 |
| sugarcrepe++ | bertscore | bertscore_f1 | 0.1402 | | 0.9008 | 0.9434 | −0.0426 |
| sugarcrepe++ | bertscore | bertscore_precision | 0.1278 | | 0.8930 | 0.9429 | −0.0498 |
| sugarcrepe++ | bertscore | bertscore_recall | 0.1818 | | 0.9089 | 0.9440 | −0.0351 |
| sugarcrepe++ | nli | C_ab | 0.0124 | | 0.0119 | 0.8019 | −0.7900 |
| sugarcrepe++ | nli | C_ba | 0.0183 | | 0.0093 | 0.7693 | −0.7600 |
| sugarcrepe++ | nli | E_ab | 0.9830 | | 0.9644 | 0.1362 | 0.8283 |
| sugarcrepe++ | nli | E_ba | 0.9752 | | 0.9606 | 0.1850 | 0.7756 |
| sugarcrepe++ | nli | nli_score_coverage | 0.9777 | | 0.9576 | 0.1146 | 0.8430 |
| sugarcrepe++ | nli | nli_score_equiv | 0.9788 | | 0.9566 | 0.1199 | 0.8367 |
| sugarcrepe++ | sbert | raw_score | 0.7248 | | 0.9293 | 0.8402 | 0.0891 |

Vs supervisor's published cap-500 run:

| metric | his (cap 500) | ours (uncapped) |
|---|--:|--:|
| sugarcrepe sbert FPR@0.6 | 0.986 | 0.974 |
| sugarcrepe nli coverage FPR@0.6 | 0.100 | 0.069 |
| sugarcrepe++ sbert rank_acc | 0.627 | 0.725 |
| sugarcrepe++ nli equiv rank_acc | 0.968 | 0.979 |

### Job ② — + NegBench (uncapped, COMPLETE, job 56959065)
`src/outputs/reliable_full_negbench/` · all three datasets · 0 scorer errors · ~25 min. NegBench CSVs from `m1k2zoo/negbench` Google Drive into `data/negbench/` (text-only MCQ CSVs; images unneeded). 0 load errors, 5 subcategories: COCO / MSR-VTT / VOC2007 / HardNeg-Syn = 3 negs/case; CheXpert binary = 1 neg. SugarCrepe / SugarCrepe++ numbers identical to Job ① (same seed/data).

NegBench headline: nli `coverage` **0.0013**, nli `equiv` **0.0012**, sbert **0.7531**, bertscore f1 **0.9967** (FPR@0.6).

NegBench full table:

| scorer | score_field | FPR@0.6 | mean_neg | mean_margin |
|---|---|--:|--:|--:|
| bertscore | bertscore_f1 | 0.9967 | 0.8373 | −0.2373 |
| bertscore | bertscore_precision | 0.9967 | 0.8393 | −0.2393 |
| bertscore | bertscore_recall | 0.9959 | 0.8362 | −0.2362 |
| nli | C_ab | 0.9748 | 0.9697 | −0.3697 |
| nli | C_ba | 0.9648 | 0.9587 | −0.3587 |
| nli | E_ab | 0.0024 | 0.0072 | 0.5928 |
| nli | E_ba | 0.0084 | 0.0164 | 0.5836 |
| nli | nli_score_coverage | 0.0013 | 0.0021 | 0.5979 |
| nli | nli_score_equiv | 0.0012 | 0.0021 | 0.5979 |
| sbert | raw_score | 0.7531 | 0.6983 | −0.0983 |

(Separation mode has no positives → `mean_pos`/`rank_acc` blank by design.)

NegBench vs cap-500: nli coverage 0.0025 → 0.0013; sbert 0.6385 → 0.7531; bertscore f1 0.9968 → 0.9967. Shifts due to uncapped pulling full per-subcat counts (COCO/MSR-VTT/VOC2007/HardNeg up to ~1500 negative pairs each vs 500 cap).

**Conclusion.** Cosine and token-overlap react to word overlap, not meaning; hard negatives share ~95% of tokens so both wave them through. BERTScore even ranks the corrupted caption higher (0.14 < 0.50). NLI separates (FPR 0.07) and ranks correctly (0.98). A faithful reasoning-match/PRM reward should be entailment-based, not embedding-cosine-based.

---

## 2. Numerical sensitivity analysis — AUC + τ sweep (Job ③)

**Goal.** The reliability tables give one number at one operating point (FPR@0.6). How robust is the verdict to threshold choice?

**Method.** `sensitivity_analysis.py`, pure-CPU re-aggregation of **252,693 saved score rows**; no models re-run. Outputs `sensitivity_auc.csv`, `sensitivity_tau_sweep.csv`, `sensitivity_operating_tau.csv`. True AUC needs both classes → computed on **sugarcrepe++** (within-case positives vs hard negatives, n_pos = 4,757 / n_neg = 4,757) and a **POOLED match-detector** (positives = SugarCrepe++ genuine matches 4,757; negatives = ALL hard negatives 79,474). Separation datasets have no positives → characterized by τ sweep only.

### ROC-AUC (threshold-free) — primary fields

| scorer (primary field) | AUC sugarcrepe++ | AUC pooled match-detector |
|---|--:|--:|
| nli `nli_score_coverage` | 0.9810 | 0.9938 |
| sbert `raw_score` | 0.7277 | 0.9084 |
| bertscore `bertscore_f1` | 0.1775 (inverted) | 0.7750 |

All fields:

| scope | scorer | field | AUC |
|---|---|---|--:|
| sugarcrepepp | nli | nli_score_equiv | 0.9812 |
| POOLED | nli | nli_score_equiv | 0.9942 |
| sugarcrepepp | nli | E_ab | 0.9835 |
| POOLED | nli | E_ab | 0.9970 |
| sugarcrepepp | nli | E_ba | 0.9744 |
| POOLED | nli | E_ba | 0.9848 |
| sugarcrepepp | nli | C_ab | 0.0115 |
| POOLED | nli | C_ab | 0.0052 |
| sugarcrepepp | nli | C_ba | 0.0188 |
| POOLED | nli | C_ba | 0.0111 |
| sugarcrepepp | bertscore | bertscore_precision | 0.1585 |
| POOLED | bertscore | bertscore_precision | 0.7246 |
| sugarcrepepp | bertscore | bertscore_recall | 0.2153 |
| POOLED | bertscore | bertscore_recall | 0.7900 |

Notes: BERTScore 0.178 < 0.5 → anti-reliable (systematically ranks corrupted caption above true one). SBERT is a fine general match-detector (0.908) but collapses to 0.728 on within-case hard negatives. NLI contradiction AUC 0.0115 → flipped = 0.99 near-perfect hard-negative detector on its own.

### τ sweep — FPR(τ) on hard negatives (lower better)

**negbench**

| scorer | τ=0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| nli `coverage` | 0.0028 | 0.0023 | 0.0017 | 0.0013 | 0.0010 | 0.0007 | 0.0005 |
| sbert `raw_score` | 0.9520 | 0.9023 | 0.8334 | 0.7531 | 0.6054 | 0.3722 | 0.0815 |
| bertscore `f1` | 1.0000 | 1.0000 | 0.9996 | 0.9967 | 0.9858 | 0.8482 | 0.0688 |

**sugarcrepe**

| scorer | τ=0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| nli `coverage` | 0.1339 | 0.0863 | 0.0776 | 0.0691 | 0.0618 | 0.0521 | 0.0403 |
| sbert | 0.9999 | 0.9988 | 0.9929 | 0.9744 | 0.9175 | 0.7386 | 0.4399 |
| bertscore `f1` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9937 | 0.8820 |

**sugarcrepe++**

| scorer | τ=0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| nli `coverage` | 0.1408 | 0.1259 | 0.1144 | 0.1030 | 0.0921 | 0.0780 | 0.0578 |
| sbert | 0.9998 | 0.9983 | 0.9895 | 0.9601 | 0.8833 | 0.6571 | 0.3775 |
| bertscore `f1` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9941 | 0.8875 |

### Full ROC points — sugarcrepe++ (both classes)

**nli `coverage`**

| τ | TPR | specificity | FPR |
|--:|--:|--:|--:|
| 0.10 | 0.9899 | 0.8301 | 0.1699 |
| 0.20 | 0.9878 | 0.8396 | 0.1604 |
| 0.30 | 0.9844 | 0.8592 | 0.1408 |
| 0.40 | 0.9783 | 0.8741 | 0.1259 |
| 0.50 | 0.9748 | 0.8856 | 0.1144 |
| 0.60 | 0.9704 | 0.8970 | 0.1030 |
| 0.70 | 0.9567 | 0.9079 | 0.0921 |
| 0.80 | 0.9359 | 0.9220 | 0.0780 |
| 0.90 | 0.9159 | 0.9422 | 0.0578 |

**sbert `raw_score`**

| τ | TPR | specificity | FPR |
|--:|--:|--:|--:|
| 0.10 | 1.0000 | 0.0000 | 1.0000 |
| 0.20 | 1.0000 | 0.0000 | 1.0000 |
| 0.30 | 1.0000 | 0.0002 | 0.9998 |
| 0.40 | 1.0000 | 0.0017 | 0.9983 |
| 0.50 | 0.9998 | 0.0105 | 0.9895 |
| 0.60 | 0.9994 | 0.0399 | 0.9601 |
| 0.70 | 0.9960 | 0.1167 | 0.8833 |
| 0.80 | 0.9779 | 0.3429 | 0.6571 |
| 0.90 | 0.7999 | 0.6225 | 0.3775 |

**bertscore `f1`**

| τ | TPR | specificity | FPR |
|--:|--:|--:|--:|
| 0.10–0.60 | 1.0000 | 0.0000 | 1.0000 |
| 0.70 | 0.9994 | 0.0002 | 0.9998 |
| 0.80 | 0.9912 | 0.0059 | 0.9941 |
| 0.90 | 0.5314 | 0.1125 | 0.8875 |

Trade-off summary points cited in `reliability_results.md`: nli coverage @τ=0.70 → TPR 0.957 / spec 0.908 / FPR 0.092; sbert @τ=0.90 → 0.800 / 0.622 / 0.378; bertscore @τ=0.90 → 0.531 / 0.113 / 0.888.

### Operating threshold τ* for target FPR

| dataset | scorer | field | τ*@FPR=1% | τ*@5% | τ*@10% |
|---|---|---|--:|--:|--:|
| negbench | sbert | raw_score | 0.9519 | 0.9185 | 0.8912 |
| negbench | nli | coverage | 0.0000 | 0.0000 | 0.0000 |
| negbench | bertscore | f1 | 0.9458 | 0.9048 | 0.8893 |
| sugarcrepe | sbert | raw_score | 0.9928 | 0.9824 | 0.9722 |
| sugarcrepe | nli | coverage | 0.9902 | 0.8203 | 0.3123 |
| sugarcrepe | bertscore | f1 | 0.9916 | 0.9853 | 0.9800 |
| sugarcrepe++ | sbert | raw_score | 0.9932 | 0.9845 | 0.9749 |
| sugarcrepe++ | nli | coverage | 0.9931 | 0.9345 | 0.6396 |
| sugarcrepe++ | bertscore | f1 | 0.9928 | 0.9875 | 0.9836 |

**Conclusion.** NLI's advantage is not an artifact of τ=0.6 (AUC 0.98–0.99, FPR stable across whole sweep). SBERT/BERTScore need τ ≈ 0.98–0.99 to cap FPR at 5%, where TPR collapses — **no choice of τ rescues them**. NegBench NLI τ*=0.000 means hard negatives all cluster at ~0 coverage (essentially perfect separation).

---

## 3. VisualPRM400K math negation probe (paragraph level)

**Goal.** Recall on numeric corruptions: does a scorer notice a single flipped digit buried in long math reasoning?

**Method.** `build_negation_pairs.py` → `score_negation.py` → `aggregate_negation.py`, launcher `run_negation_score.sbatch`. Dataset: VisualPRM400K **math subset** (24 PRM files, geometry/function/visual-math), read from **v1 zip** `data/visualprm400k/annotations.zip`. Math source keys: geometry3k, geoqa, geo170k, geos_en, unigeo, mavis, geomverse, clevr_math, mathv360k, super_clevr, iconqa. **N = 40,000 pairs.** Hard negative = correct step-by-step solution with the **first numeric value on the RHS of an equation changed by +1** (regex `=\s*(-?\d+\.?\d*)`; e.g. `∠B = 50°` → `51°`). Two conditions per example: **NEG** = sim(pos, neg); **BASE** = sim(pos, unrelated shuffled solution) as random-similarity floor. Batch size 64.

Example (geometry3k, `14→15`): the negative differs from the positive in a **single character**, and downstream lines still say `14`, so the negative is also internally inconsistent.

### Similarity distributions

| scorer (field) | NEG mean | NEG median | NEG p10–p90 | BASE mean | gap (NEG−BASE) |
|---|--:|--:|--:|--:|--:|
| SBERT cosine | 0.993 | 0.997 | 0.980–1.000 | 0.302 | +0.691 |
| NLI equiv | 0.568 | 0.647 | 0.000–0.984 | 0.100 | +0.468 |
| NLI coverage | 0.544 | 0.607 | 0.000–0.984 | 0.101 | +0.443 |
| BERTScore f1 | 0.981 | 0.989 | 0.958–0.996 | 0.671 | +0.310 |

### Missed-corruption rate vs τ (fraction of NEG ≥ τ; lower better)

| scorer | τ=0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| SBERT cosine | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.991 |
| NLI equiv | 0.704 | 0.653 | 0.594 | 0.530 | 0.465 | 0.401 | 0.316 | 0.246 |
| NLI coverage | 0.669 | 0.612 | 0.555 | 0.504 | 0.449 | 0.391 | 0.310 | 0.243 |
| BERTScore f1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.975 | 0.917 |

### Paper number — FPR@0.6

| scorer (field) | FPR@0.6 |
|---|--:|
| **NLI (equiv)** — reported field | **0.530** |
| NLI (coverage) | 0.504 |
| SBERT (raw) | 1.000 |
| BERTScore (F1) | 1.000 |

Reported field = **NLI equiv** (strict mutual equivalence is the principled criterion for numeric correctness).

### NLI contradiction probability (max of both directions)

| condition | mean C | median C |
|---|--:|--:|
| NEG | 0.309 | 0.180 |
| BASE | 0.248 | 0.192 |

### Surface-overlap reliance: AUC = P(NEG sim > BASE sim)

| scorer | AUC(NEG>BASE) |
|---|--:|
| SBERT cosine | 1.000 |
| NLI equiv | 0.816 |
| NLI coverage | 0.793 |
| BERTScore f1 | 1.000 |

### NEG mean by source (top by count)

| source | n | SBERT NEG | NLI equiv NEG |
|---|--:|--:|--:|
| mavis_function_poly_prm.jsonl | 8041 | 0.990 | 0.491 |
| mavis_function_abs_prm.jsonl | 4786 | 0.992 | 0.508 |
| mavis_function_log_prm.jsonl | 4760 | 0.992 | 0.365 |
| CLEVR_math_en_20240402_extracted_prm.jsonl | 3771 | 0.998 | 0.519 |
| geometry3k_en_20240402_extracted_prm.jsonl | 2781 | 0.993 | 0.757 |
| MathV360K_prompts_prm.jsonl | 2157 | 0.997 | 0.640 |
| geoqa+_extracted_en_version_prm.jsonl | 1751 | 0.991 | 0.783 |
| geomverse_extracted_prm.jsonl | 1749 | 0.993 | 0.751 |
| geoqa+_en_20240402_extracted_open_ended_only_prm.jsonl | 1618 | 0.994 | 0.696 |
| iconqa_train_extracted_prm.jsonl | 1470 | 0.995 | 0.481 |
| mavis_function_sin_prm.jsonl | 1207 | 0.993 | 0.548 |
| mavis_function_cos_prm.jsonl | 1119 | 0.992 | 0.504 |

**Conclusion.** Even the entailment model misses just over half the single-digit corruptions; cosine-family scorers miss all of them. Worst NLI result across four benchmarks: NegBench 0.001 → SugarCrepe 0.069 → **VisualPRM 0.530**. A wrong number buried in long math reasoning is the hardest negation to catch — the dilution effect motivating clause splitting.

---

## 4. Clause-split + min-pool pipeline — negation (RECALL)

**Goal.** Is the paragraph-level 0.530 caused by **dilution** (one contradicting clause outvoted by ~26 identical ones)? If so, splitting should rescue NLI.

**Method.** `score_negation_clausesplit.py`, launcher `run_negation_clausesplit.sbatch --n 10000`. **N = 10,000 examples** (subset of the 40k pairs). Pipeline mirrors the bipartite-match reward with the per-clause scorer swapped to NLI + min-pool:
1. Split pos and candidate into atomic clauses: split on newlines, then `re.split(r"(?<=[.;:])\s+")` (sentence punctuation `.` `;` `:`); `MAX_CLAUSES = 40`.
2. Embed clauses with SBERT (mpnet), cosine matrix, **Hungarian one-to-one assignment** (`scipy.optimize.linear_sum_assignment` on −sim), gate at **TAU = 0.15** cosine.
3. NLI **equiv** on each matched (pos_clause, cand_clause) pair.
4. `solution_equiv = MIN` over matched pairs (one contradicted clause vetoes the solution).

Conditions: `neg` (corrupted copy, want LOW), `base` (unrelated floor). **Mean matched clauses/example = 26.6.** Paragraph-level numbers are the existing whole-solution NLI equiv on the **same idxs**.

| metric | paragraph equiv | clause-split min-equiv |
|---|--:|--:|
| FPR@0.6 on neg (corrupted) ↓ | 0.530 | **0.008** |
| FPR@0.5 on neg (corrupted) ↓ | 0.594 | 0.015 |
| mean equiv on neg (corrupted) | 0.569 | 0.015 |
| median equiv on neg | 0.647 | 0.000 |
| mean equiv on base (unrelated floor) | 0.101 | 0.000 |

**Conclusion.** Splitting removed the dilution; NLI now catches the isolated numeric flip (FPR 0.530 → 0.008, a 66× reduction). Residual FPR is what NLI still misses on bare numeric equality (e.g. neutral on `=14` vs `=15`) — the case for a hard numeric-match term.

Sample inspection (`docs/negation_clauses_sample.md`, 20 of 10,000): typical example has 17–23 clauses with exactly **1 differing** clause, confirming the recall test is "flattered" (26/27 clauses get a free equiv≈1).

---

## 5. Paraphrase control — clause-split + min-pool (PRECISION)

**Goal.** The negation test only measured recall on byte-identical near-copies. Does min-pool wrongly veto a **correct** but reworded solution? Real model outputs are reworded-but-correct.

**Method.** `build_paraphrase_pairs.py` → `score_paraphrase_clausesplit.py`, launcher `run_paraphrase_clausesplit.sbatch --n 10000`. Paraphraser = **Qwen2.5-VL-3B**, prompted to reword step-by-step solutions with strict rules: keep every number/variable/equation/final answer exactly the same; only change explanatory prose wording; keep same step-by-step structure (one step per line); no adding/removing/merging/reordering steps.

**Confound guardrail:** the multiset of numeric tokens (regex `-?\d+\.?\d*`, via `Counter`) is extracted from pos and paraphrase; if they differ the paraphrase is marked **unfaithful and EXCLUDED**. **N = 9,254 number-multiset-verified faithful paraphrases** kept (of 10,000 attempted). Same pipeline as negation: split → SBERT-cosine Hungarian align (τ=0.15) → NLI **equiv** per matched clause-pair → **min**-pool. Mean matched clauses/example = 26.6.

**FPR here = fraction of CORRECT paraphrases wrongly scored BELOW threshold (false alarm). Lower = better.**

| metric | clause-split min-equiv |
|---|--:|
| FPR@0.6 (paraphrase wrongly flagged) ↓ | **0.599** |
| FPR@0.5 (paraphrase wrongly flagged) ↓ | 0.226 |
| mean min-equiv on paraphrase (want HIGH) | 0.594 |
| median min-equiv on paraphrase | 0.578 |

**Conclusion.** **Hard min FAILS precision.** 59.9% of correct rewordings are false-flagged at τ=0.6; median min-equiv 0.578 sits right on the 0.6 boundary. One clause NLI calls "neutral" sinks the whole min. Verdict: **do not ship hard min** — needs a softened pool (k-th-lowest quantile / soft-min) and/or a hard numeric-match term. Recall/precision tension: negation FPR **0.008** vs paraphrase FPR **0.599**.

---

## 6. COCO bounding-box IoU numerical/spatial sensitivity

**Goal.** Do text scorers respect spatial overlap magnitude — the spatial analogue of numeric blindness?

**Method.** `build_coco_iou_pairs.py` → `score_coco_iou.py` → `aggregate_coco_iou.py`, launcher `run_coco_iou_score.sbatch`. **N = 20,000 triples.** Construction per GT object box:
1. Normalize COCO bbox `[x,y,w,h]` (absolute px) by image size, scale ×1000, convert to integer **corner** coords `[x1,y1,x2,y2]` in a 0–1000 space (Qwen-VL box convention).
2. Generate **two** candidate boxes by jittering the GT (random shift + scale) so IoU spans a useful range (fully-random boxes would both be ~0 IoU → degenerate).
3. Compute IoU of each candidate vs GT.
4. Label the HIGHER-IoU candidate `neg`, the LOWER-IoU one `NEG`. `pos` = GT box string (the reference).

Boxes emitted as **text strings** `"[x1, y1, x2, y2]"` so the same SBERT/NLI/BERTScore text scorers apply. Batch size 128. Pairs scored: HI = sim(pos, neg), LO = sim(pos, NEG).

Ground-truth criterion (threshold-free): **sim(pos, neg) > sim(pos, NEG)**. `rank acc` = fraction of triples where that holds (1.0 = always, 0.5 = chance).

Mean IoU: neg (high) = **0.432**, NEG (low) = **0.260**, margin = **0.172**.

| scorer (field) | rank acc ↑ | mean sim(hi) | mean sim(lo) | margin | Spearman(sim, IoU) |
|---|--:|--:|--:|--:|--:|
| **IoU (oracle)** | **1.000** | 0.432 | 0.260 | +0.172 | 1.000 |
| SBERT cosine | 0.577 | 0.802 | 0.788 | +0.015 | 0.026 |
| NLI equiv | 0.503 | 0.002 | 0.001 | +0.002 | 0.064 |
| NLI coverage | 0.503 | 0.002 | 0.001 | +0.002 | 0.064 |
| BERTScore f1 | 0.620 | 0.867 | 0.856 | +0.011 | 0.183 |

**Conclusion.** Rank acc near 0.5 and Spearman near 0 ⇒ text-scorer similarity is **blind to IoU** — it cannot tell which box overlaps GT more, because it only sees digit strings. Mechanism note (recomputed live 2026-07-02, `project_history.md`): NLI's ≈0 similarity is **confident contradiction on any digit difference with no magnitude sense** — a nearly-right box and a wildly-wrong box both get contradiction ≈0.93–0.99. This is *not* neutrality. Quantitative case for an explicit IoU term in any grounding reward.

---

## 7. DCI reward-encoder reliability (analysis run 2026-06-17)

**Goal.** Is the encoder underneath the GRPO `match` term reliable — does cosine score a faithful description *above* a minimally-corrupted one? Context: composite reward `R = w_f·format + w_m·match + w_a·answer − w_p·pun` with `w = 1 / 2 / 5 / 1`; `match` = soft bipartite (Hungarian) alignment of reasoning clauses vs GT solution clauses, per-pair cost = SBERT cosine with `all-MiniLM-L6-v2`.

### Testbed — DCI (Densely Captioned Images), `facebookresearch/DCI`, CC-BY-NC

| property | value |
|---|---|
| images (complete JSON) | 7,805 (train 7,599 / val 98 / test 108) |
| total submasks | 313,391 (mean 40.2/image, ok-quality 270,219) |
| images with LLaMA2 summaries | 7,805 (100%) |
| images with generated negatives | 7,805 (100%) |
| summaries/image (flattened) | mean 91.5 (median 86) |
| hard negatives generated | `basic` 135,024 · `layout` 135,024 · `swaps` 135,024 |
| matchable GT units/image | mean 36.6 (short + extra + ok-mask captions) |
| total matchable GT units | 285,829 |

Negative families: `swaps` = attribute/entity swapped between two objects, **near-verbatim** minimal edit; `layout` = spatial arrangement altered, full reword; `basic` = other content change, full reword. Provenance: `dci.tar.gz` (831 MB, sha256-verified); images from license-gated SA-1B shard `sa_000138.tar` (IDs 1543972–1554261, ~11 GB) via signed Meta links-file; `match_sa1b.py` → **7,805 / 7,805 matched, 0 missing**.

### Method (Tests 1 / 2 / 4)

`data/DCI/reliability_test.py`, seed=42, **1,500 units for bi-encoders, 300 for NLI** (cross-encoder runs ~P×G pairwise). Per unit: anchor = `summaries[k][0]`; positives (paraphrases) = `summaries[k][1:4]` (≤3); negatives = `negatives[k][{swaps,layout,basic}][:3]` (≤3 each). Only units with ≥2 summaries and all three negative families present kept. Flags: `--bi-units 1500 --nli-units 300 --paras 3 --negs 3`.

Encoders: **MiniLM-L6 (reward)** `all-MiniLM-L6-v2` bi-encoder cosine; **mpnet-base (alt)** `all-mpnet-base-v2` bi-encoder cosine (second independent SBERT); **NLI-DeBERTa-v3** `cross-encoder/nli-deberta-v3-base`, P(entailment) anchor→candidate (softmax, entail label auto-detected).

Metrics: Test 1 pairwise accuracy = fraction of (pos, neg-of-type-t) pairs with score(pos) > score(neg); AUROC treating positives=1 vs that type's negatives=0. Test 4 mean scores by class + **FNR** = fraction of paraphrases scoring below the *median negative* of that type. Test 2 agreement vs reward encoder: Cohen's κ on binary `pos > neg` decisions and Spearman ρ of decision margins `score(pos) − score(neg)`, on the *shared* `swaps` comparisons over the common 300-unit subsample.

### Test 1 — discrimination P(paraphrase > negative)

| encoder | swaps acc | swaps AUROC | layout acc | layout AUROC | basic acc | basic AUROC |
|---|--:|--:|--:|--:|--:|--:|
| **MiniLM-L6 (reward)** | **0.183** | 0.192 | 0.519 | 0.525 | 0.573 | 0.578 |
| mpnet-base (alt) | 0.272 | 0.287 | 0.553 | 0.562 | 0.581 | 0.587 |
| NLI-DeBERTa-v3 (entail) | 0.381 | 0.399 | **0.920** | 0.914 | **0.830** | 0.832 |

### Test 4 — paraphrase robustness (mean score by class)

| encoder | mean paraphrase | mean swaps | mean layout | mean basic | FNR vs swaps |
|---|--:|--:|--:|--:|--:|
| **MiniLM-L6 (reward)** | 0.758 | **0.881** | 0.742 | 0.725 | 0.895 |
| mpnet-base (alt) | 0.795 | 0.873 | 0.762 | 0.761 | 0.783 |
| NLI-DeBERTa-v3 (entail) | 0.426 | 0.637 | 0.031 | 0.095 | 0.684 |

### Test 2 — encoder agreement (vs MiniLM reward encoder, swaps subsample)

| encoder pair | Cohen κ | Spearman(margin) |
|---|--:|--:|
| MiniLM-L6 vs mpnet-base | 0.617 | 0.865 |
| MiniLM-L6 vs NLI-DeBERTa-v3 | 0.141 | 0.303 |

### Mechanistic probe — embedding-inversion roundtrip

`data/DCI/invert_roundtrip.py`, vec2text, 20 refinement steps, beam 4, `--invert-bs 16`, GPU (1×A100, htc, `--no-requeue`). **Caveat:** vec2text ships a public inverter for **GTR-base (768-d)**, not MiniLM — a same-family bi-encoder stand-in. GTR corrector trained for ~32-token inputs → short units probed as in-distribution.

| unit set | n | token-F1 | MiniLM cos | ADJ recall | NUM recall | spatial recall |
|---|--:|--:|--:|--:|--:|--:|
| short (≤28 w) | 400 | **0.984** | 0.991 | **0.983** | **0.972** | **0.980** |
| extra_caption (long) | 100 | 0.465 | 0.776 | 0.284 | 0.291 | 0.372 |

### Findings

1. **SBERT is consistent but structurally blind.** κ=0.62 / ρ=0.87 with an independent SBERT → property of the bi-encoder-cosine *paradigm*, not one model. ~chance on real content corruption (layout 0.52, basic 0.57); **worse than chance on swaps (0.18)** — scores the swap (0.881) above the faithful paraphrase (0.758). FNR-vs-swaps 0.895.
2. **NLI far stronger on content, own blind spot.** layout 0.92 / basic 0.83, but fooled by near-verbatim swaps (0.38, below chance) and over-penalizes detail-adding paraphrases (mean entail 0.43; FNR 0.68). κ=0.14 with SBERT → genuinely different paradigm. SBERT and NLI fail on **complementary** axes.
3. **The encoder retains the content; cosine can't score it.** Short embeddings near-losslessly invertible (ADJ 0.98 / NUM 0.97 / spatial 0.98, token-F1 0.984) → Tests 1/4 failures are a **scoring/geometry** problem, not an information bottleneck. A better read-out can recover it.
4. **Long-caption detail loss real but secondary** (F1 0.47, ADJ/NUM/spatial 0.28–0.37) — partly corrector OOD, partly mean-pool blurring. Argues for keeping clauses short.
5. **Honest confound.** `swaps` are minimal edits → lexically inflated for *every* text method; the clean signal is `layout`/`basic` (SBERT 0.52–0.57, NLI 0.92/0.83). Qualitative case: "silver-trimmed" reattached window→instruments — SBERT scores the wrong **swap 0.92** vs the correct **reword 0.71**.

**Takeaway.** Reward encoder *preserves* swap-relevant content but *cannot score it apart* with cosine; NLI scores content well but is blind to near-verbatim swaps; attribute-swaps resist both text methods and are fundamentally a pixel question.

**Reward-design implications (proposed, not built):** (1) hybrid text reward — SBERT cosine for coverage/recall + NLI entailment for faithfulness, bounded by an SBERT prefilter (entailment only on SBERT-matched clauses, not full P×G); (2) multimodal grounding term (box-IoU / region-CLIP); (3) keep clauses short. Validation order: add NLI-matcher variant to `graph_match_reward.py` → re-run DCI 1/4; re-score existing validation generations under hybrid; one hybrid-reward GRPO run vs current arm on MMK12.

---

## 8. Adversarial attack taxonomy (A / B / C axes)

Walked through 2026-07-02; recorded in `docs/project_history.md:219-229`, `docs/progress_2026-07-11.md:92`, and the `reward-reliability` skill. Attacks on the clause-split + NLI + min-pool reward.

| axis | attack | description | status | evidence |
|---|---|---|---|---|
| **A — paragraph-level** | A-1 dilution | contradicting clause outvoted in a long otherwise-identical solution | **MEASURED, FIXED** by clause splitting | paragraph FPR **0.530** → clause-split min **0.008** |
| **A** | A-2 truncation | — | fixed by splitting | — |
| **A** | A-3 length-OOD | — | fixed by splitting | — |
| **B — pipeline-structural** | **B-1 clause fusion** | withhold punctuation → clauses fuse → dilution returns | **SURVIVING, UNTESTED** | probe set queued |
| **B** | **B-2 cosine misalignment** | corrupted clause routed past its true partner by the Hungarian/cosine align → veto never fires | **SURVIVING, UNTESTED** | probe set queued |
| **B** | **B-3 locally-true / globally-invalid chains** | bipartite match is a bag of clauses; order/dependency discarded (permutation-invariant) | **SURVIVING, UNTESTED** | probe set queued; needs chain-inverting negatives, not shuffles (e.g. MMathCoT-1M order-variance probe) |
| **C — NLI-atom failures** | **C-1 neutral trap** | NLI marks a correctly-reworded clause "neutral" → min-pool sinks the whole solution | **MEASURED** | paraphrase FPR@0.6 **0.599**, median min-equiv 0.578 |
| **C** | **C-2 indiscriminate numeric contradiction** | NLI confidently contradicts on any digit difference with no magnitude sense | **MEASURED / mechanism-confirmed** | COCO rank acc **0.503**, Spearman 0.064, contradiction ≈0.93–0.99 regardless of IoU |
| **C** | C-3, C-4 | cue-free NLI-atom failures (wrong-answers-passing family) | untested; no on-disk description (referenced by number only) | — |
| **C** | C-5, C-7, C-8 | NLI treats not-obviously-entailed as worthless (right-answers-failing family) | untested; no on-disk description (referenced by number only) | — |

**Two root-cause families:**
- **right-answers-failing** (C-1/5/7/8): NLI treats not-obviously-entailed as worthless + min amplifies → fix by softening the pool + hard numeric/answer term.
- **wrong-answers-passing** (B-1/2/3, C-3/4): structure discarded + NLI lacks cue-free semantics → fix by probe sets + structural alignment.

### Queued work (leverage order)
1. Pooling sweep (min → k-th-lowest quantile → mean) plotting negation-FPR vs paraphrase-FPR — needs a rerun saving full per-clause equiv vectors.
2. SBERT/BERTScore clause-split control (prediction: **no rescue** — their blindness is per-clause, not dilution).
3. Adversarial probe sets for B-1 / B-2 / B-3 and cue-free NLI-atom families.
4. Hard numeric-match term (parse numbers, compare exactly); consider `coverage` → `equiv` as the per-clause edge.
5. Multimodal grounding term (box-IoU + region-CLIP blended into similarity) — COCO is the quantitative case.
6. Hybrid-reward GRPO run vs the current arm on MMK12.

Also noted: supervisor's six-scorer table + NLI logistic calibration (85/15 case-level split, τ via FPR@TPR≥0.96) understood; his fitted weights not committed — refit possible on cached `src/outputs/reliable_full/scores_top3.jsonl` (CPU-only).

---

## 9. SymPy symbolic-verification audit (v2 filtering research)

`data/visualprm_v11_filtered/sympy_audit/REPORT.md`, generated 2026-07-19. sympy 1.14.0, interpreter `/scratch/sghos104/envs/rlpt-train/bin/python`. Code: `src/data/sympy_extract_audit.py`, `src/data/sympy_verify.py`. All inputs read-only; outputs confined to the audit dir.

**Goal.** Independently verify (with an exact symbolic checker) whether the NLI-based filter's decisions correspond to actual mathematical correctness — a third instrument alongside MC scores and NLI.

### Part 1 — Extractability audit (150 traces/source, seed=0; v4 extraction)

Extraction iterations, each validated on logged failures / FAIL inspections:
- **v1** (3.3% parse-fail) → decimal-period, paren-balance, `x`-as-times, bare-sqrt fixes → **2.9%**.
- **v2**: `\text{}`-placeholder symbols, uppercase identifiers (AB/SA/AOB), unicode angle/triangle mapping → geo sources **8.3–10% → 3.1–6.5%** parse-fail; claim yield roughly doubled (geo170k **0.66 → 1.44** claims/step).
- **v3** (verifier-precision pass): sympy namespace singletons neutralized (E/S2/I/Q parsed as Symbols); `|x|` → `Abs`; juxtaposition guard (≥2-space gap without operator = removed separator → side dropped, not glued); trailing unit-noun placeholders stripped (`25547 tonnes_` → `25547`); `\text{or}` treated as chain break; `f(x)`-style claims skipped as unverifiable (FUNC_SKIP); multi-branch anchoring (constraint passes if ANY asserted solution branch satisfies it — pre-fix, `x = 6 or x = -1` traces fabricated FAILs).
- **v4** (largest single fabrication source): `\tan/\sin/\cos/\log`/greek macros were being DELETED by the generic latex-command strip, turning `\tan(-\pi) = 1` into the false claim `(-pi) = 1`; now preserved as functions/symbols. Failing claims additionally record a RELATIVE diff.

**Kept-sample FAIL count across iterations: 729 (v2) → 548 (v3) → 365 (v4).**

### Part 2/3 — Verifier verdict counts per population

Scope (condition 2): only tiers {mavis_function, geometry family, CLEVR/MathV360K/dvqa arithmetic} are verified; all other sources emit SKIPPED_NO_SURFACE (never PASS). FAIL_LOW_CONFIDENCE (condition 3) = every failing claim in the trace came from a prose-trimmed side.

**stage2-kept sample (150/source stratified, seed=0)**
- traces **5,622**; in-tier **3,222**; extractable (≥1 parsed claim, scored) **2,492 (44.3%)**
- verdicts: PASS **2,326** | FAIL **365** | FAIL_LOW_CONFIDENCE **134** | EXPLORATORY **397** | SKIPPED_NO_SURFACE **2,400** | ERROR **0**
- of extractable: PASS-clean **1,993**
- claim-level: **21,925** claims, **159** parse-fail, **2,215** func-call-skipped, **0** timeouts, **3,301** unresolved-constraint
- failing-claim relative-diff split: **99** <1% (rounding-scale) / **56** in [1%,20%) / **855** ≥20% (gross), of **1,010** with magnitude

**NLI-rejected (stage2_rejected_tau0.85, full)**
- traces **19,411**; in-tier **6,605**; extractable **2,816 (14.5%)**
- verdicts: PASS **6,107** | FAIL **210** | FAIL_LC **62** | EXPLORATORY **226** | SKIPPED_NO_SURFACE **12,806** | ERROR **0**
- of extractable: PASS-clean **2,544**
- claim-level: **15,475** claims, **94** parse-fail, **1,111** func-call-skipped, **0** timeouts, **3,163** unresolved-constraint
- relative-diff split: **23** <1% / **37** [1%,20%) / **479** ≥20%, of **539** with magnitude

**MC-fail band (stage1_band_sample, full)**
- traces **15,000**; in-tier **8,891**; extractable **6,487 (43.2%)**
- verdicts: PASS **5,793** | FAIL **1,836** | FAIL_LC **494** | EXPLORATORY **768** | SKIPPED_NO_SURFACE **6,109** | ERROR **0**
- of extractable: PASS-clean **4,157**
- claim-level: **84,621** claims, **738** parse-fail, **15,783** func-call-skipped, **1** timeout, **5,105** unresolved-constraint
- relative-diff split: **94** <1% / **201** [1%,20%) / **5,146** ≥20%, of **5,441** with magnitude

### Part 3a — NLI false-alarm candidates (headline count)

Of **19,411** NLI-rejected traces: **2,816** have extractable claims; **2,544 verify fully clean (PASS with ≥1 claim) → NLI_FALSE_ALARM_CANDIDATE**; **1,647** under the strict variant (additionally zero parse-fails / timeouts / unresolved constraints). Full list: `nli_false_alarm_candidates.jsonl`.

Per-source breakdown of the NLI-rejected population (traces | extractable | PASS(extr.) | FAIL | FAIL_LC | EXPLOR | SKIP):

| source | traces | extr. | PASS | FAIL | FAIL_LC | EXPLOR | SKIP |
|---|--:|--:|--:|--:|--:|--:|--:|
| cocorem_exist_yorn_en_20241016 | 3099 | 0 | 0 | 0 | 0 | 0 | 3099 |
| CLEVR_math_en_20240402 | 2618 | 712 | 703 | 1 | 8 | 19 | 0 |
| dvqa_en_20240402_extracted_int_only | 2182 | 489 | 487 | 1 | 1 | 115 | 0 |
| super_clevr_en_20240402_int | 1791 | 0 | 0 | 0 | 0 | 0 | 1791 |
| scienceqa_multi_choice_en_20240402 | 1211 | 0 | 0 | 0 | 0 | 0 | 1211 |
| nlvr2_en_20240910 | 1098 | 0 | 0 | 0 | 0 | 0 | 1098 |
| iconqa_train | 1019 | 0 | 0 | 0 | 0 | 0 | 1019 |
| vqav2_en_20240402_int | 910 | 0 | 0 | 0 | 0 | 0 | 910 |
| docvqa_train_56k_en_20240402 | 778 | 0 | 0 | 0 | 0 | 0 | 778 |
| m3cot_train | 651 | 0 | 0 | 0 | 0 | 0 | 651 |
| chartqa_trainval_30k_w_csv_en_20240402 | 474 | 0 | 0 | 0 | 0 | 0 | 474 |
| ai2d_train_12k_en_20240410 | 359 | 0 | 0 | 0 | 0 | 0 | 359 |
| super_clevr_en_20240402_yorn | 339 | 0 | 0 | 0 | 0 | 0 | 339 |
| SROIE_information_extraction_multi_turn | 329 | 0 | 0 | 0 | 0 | 0 | 329 |
| infographics_20240403_qa_20240407_v2 | 321 | 0 | 0 | 0 | 0 | 0 | 321 |
| figureqa_en_20240402 | 283 | 0 | 0 | 0 | 0 | 0 | 283 |
| geoqa+_en_20240402_extracted_open_ended_only | 188 | 163 | 143 | 19 | 1 | 22 | 0 |
| geoqa+_extracted_en_version | 187 | 182 | 170 | 9 | 3 | 2 | 0 |
| mavis_function_poly | 187 | 178 | 92 | 65 | 21 | 9 | 0 |
| geometry3k_en_20240402 | 182 | 176 | 152 | 19 | 5 | 3 | 0 |
| geo170k_extracted_full | 171 | 168 | 158 | 5 | 5 | 0 | 0 |
| mavis_geo_depth3_text_dominant_vision_dominant | 146 | 142 | 134 | 5 | 3 | 0 | 0 |
| geometry3k_en_20240402_extracted_open_ended_only | 130 | 120 | 96 | 24 | 0 | 7 | 0 |
| unigeo_calc_en_20240402_extracted_open_ended | 117 | 91 | 84 | 7 | 0 | 25 | 0 |
| mavis_geo_depth0_… | 98 | 90 | 88 | 2 | 0 | 5 | 0 |
| mavis_geo_depth1_… | 82 | 81 | 79 | 1 | 1 | 0 | 0 |
| MathV360K_prompts | 77 | 5 | 5 | 0 | 0 | 3 | 0 |
| mapqa_suv_en_20240402 | 76 | 0 | 0 | 0 | 0 | 0 | 76 |
| mavis_geo_depth2_… | 75 | 73 | 69 | 3 | 1 | 0 | 0 |
| koniq10k_en_20240403 | 68 | 0 | 0 | 0 | 0 | 0 | 68 |
| mavis_function_abs | 61 | 58 | 30 | 21 | 7 | 3 | 0 |
| geomverse | 29 | 29 | 21 | 6 | 2 | 0 | 0 |
| mavis_function_log | 21 | 14 | 7 | 6 | 1 | 7 | 0 |
| geos_en_20240402 | 16 | 11 | 11 | 0 | 0 | 2 | 0 |
| mavis_function_cos | 16 | 15 | 4 | 9 | 2 | 1 | 0 |
| mavis_function_sin | 10 | 7 | 4 | 2 | 1 | 3 | 0 |
| mavis_function_tan | 8 | 8 | 4 | 4 | 0 | 0 | 0 |
| geos_en_20240402_extracted_open_ended_only | 4 | 4 | 3 | 1 | 0 | 0 | 0 |

**Cross-reference: the 20 `mcpass_nlifail` agreement examples** (NLI max P(contra) | SymPy verdict | claims parsed/true):

| # | example | P(contra) | SymPy | claims |
|--:|---|--:|---|---|
| 01 | CLEVR_math…L1573 | 0.6766 | PASS | 0/0 |
| 02 | CLEVR_math…L6315 | 0.7879 | PASS | 0/0 |
| 03 | CLEVR_math…L16289 | 0.6011 | PASS | 0/0 |
| 04 | ai2d_train_12k…L1226 | 0.5680 | SKIPPED_NO_SURFACE | 0/0 |
| 05 | cocorem_exist_yorn…L963 | 0.6799 | SKIPPED_NO_SURFACE | 0/0 |
| 06 | cocorem_exist_yorn…L8513 | 0.7259 | SKIPPED_NO_SURFACE | 0/0 |
| 07 | docvqa_train_56k…L11 | 0.5377 | SKIPPED_NO_SURFACE | 0/0 |
| 08 | dvqa…int_only_L655 | 0.9107 | PASS | 0/0 |
| 09 | dvqa…int_only_L5486 | 0.8882 | PASS | 0/0 |
| 10 | dvqa…int_only_L10010 | 0.6827 | PASS | 1/1 |
| 11 | dvqa…int_only_L12363 | 0.7042 | PASS | 0/0 |
| 12 | dvqa…int_only_L20435 | 0.5185 | PASS | 0/0 |
| 13 | dvqa…int_only_L24912 | 0.9154 | PASS | 0/0 |
| 14 | geo170k_extracted_full_L2597 | 0.5513 | PASS | 17/10 |
| 15 | nlvr2_en_20240910_L5777 | 0.8080 | SKIPPED_NO_SURFACE | 0/0 |
| 16 | nlvr2_en_20240910_L23591 | 0.6770 | SKIPPED_NO_SURFACE | 0/0 |
| 17 | scienceqa_multi_choice…L | 0.5917 | SKIPPED_NO_SURFACE | 0/0 |
| 18 | scienceqa_multi_choice…L | 0.7234 | SKIPPED_NO_SURFACE | 0/0 |
| 19 | unigeo_calc…open_e_L | 0.6500 | PASS | 14/10 |
| 20 | vqav2_en_20240402_int_L3594 | 0.7215 | SKIPPED_NO_SURFACE | 0/0 |

### Part 4 — Verdicts by source tier (all three populations)

Populations have very different source mixes (kept sample is stratified 150/source → overweights claim-dense mavis_function; NLI-rejected extractable mass is mostly easy CLEVR/dvqa arithmetic) → compare WITHIN tier.

**MC-fail band**

| tier | traces | extractable | FAIL | FAIL_LC | FAIL% of extractable |
|---|--:|--:|--:|--:|--:|
| mavis_function | 4179 | 3715 | 1551 | 426 | 53.2% |
| geometry family | 2352 | 2085 | 279 | 60 | 16.3% |
| arithmetic (CLEVR/MathV360K/dvqa) | 2360 | 687 | 6 | 8 | 2.0% |
| no-surface | 6109 | 0 | 0 | 0 | 0.0% |

**stage2-kept sample**

| tier | traces | extractable | FAIL | FAIL_LC | FAIL% |
|---|--:|--:|--:|--:|--:|
| mavis_function | 900 | 721 | 245 | 82 | 45.4% |
| geometry family | 1872 | 1626 | 120 | 51 | 10.5% |
| arithmetic | 450 | 145 | 0 | 1 | 0.7% |
| no-surface | 2400 | 0 | 0 | 0 | 0.0% |

**NLI-rejected**

| tier | traces | extractable | FAIL | FAIL_LC | FAIL% |
|---|--:|--:|--:|--:|--:|
| mavis_function | 303 | 280 | 107 | 32 | 49.6% |
| geometry family | 1425 | 1330 | 101 | 21 | 9.2% |
| arithmetic | 4877 | 1206 | 2 | 9 | 0.9% |
| no-surface | 12806 | 0 | 0 | 0 | 0.0% |

### Part 5 — Three-instrument summary (MC × NLI × SymPy)

| population | MC | NLI | SymPy |
|---|---|---|---|
| stage2-kept sample | pass | pass | extractable 2492/5622 (44%); of extractable 1993 PASS / 365 FAIL / 134 FAIL_LC → **fail rate 20.0%** |
| NLI-rejected | pass | fail | extractable 2816/19411 (15%); 2544 PASS / 210 FAIL / 62 FAIL_LC → **fail rate 9.7%** |
| MC-fail band | fail (band) | n/a | extractable 6487/15000 (43%); 4157 PASS / 1836 FAIL / 494 FAIL_LC → **fail rate 35.9%** |

Tier-matched: MC-fail band is elevated over MC-passing populations in every tier, but by different margins — ~1.6–2× in geometry (16.3% vs 10.5% kept / 9.2% rejected), 2–3× in arithmetic (2.0% vs 0.7% / 0.9%), only modestly in mavis_function (53.2% vs 45.4% / 49.6%). **Within every tier the NLI-rejected population fails at about the SAME rate as the NLI-passing kept sample — the NLI contradiction signal is essentially orthogonal to symbolic correctness, consistent with the false-alarm reading.** mavis_function absolute rates likely inflated by residual extraction artifacts, but the inflation applies to all three populations alike.

### Part 6 — Example FAIL traces (first failing claim)

- [kept] `geo170k_extracted_full.jsonl` L4280 step 1: `1 = ((70)*pi/180)` → diff `1 − 7*pi/18` (|diff|~0.2217, rel~0.1815)
- [kept] `geo170k_extracted_full.jsonl` L5863 step 6: `((30)/(10 − x)) = ((30 − x)/(x))` → diff `−773/12081227` (|diff|~6.398e−05, rel~6.398e−05); anchors {BF:['10'], CD:['40'], FC:['30'], B:['pi/2'], x:['6541/100','459/100']}
- [kept] `geo170k_extracted_full.jsonl` L6026 step 4: `((BD)/(5)) = ((8)/(5))` → diff `−1` (rel~0.625); anchors {p:['8'], q:['5'], BD:['3']}
- [kept] `geometry3k_en_20240402_extracted.jsonl` L625 step 0: `Area_of_sector_ = ((theta)/(360)) * pi r**2` → diff `49*pi*(180 − pi)/8100` (|diff|~3.361, rel~0.9825)
- [kept] `geometry3k_en_20240402_extracted.jsonl` L3967 step 3: `((((180)*pi/180))/(50)) = 3.6` → diff `−18/5 + pi/50` (|diff|~3.537, rel~0.9825)
- [NLI-rej] `CLEVR_math_en_20240402_extracted.jsonl` L1814 step 5: `New_Total_ = Current_Total_ + 3` → diff `−3` (rel~0.25)
- [NLI-rej] `dvqa_en_20240402_extracted_int_only.jsonl` L3530 step 2: `70 − 50 = ((20)/100)` → diff `99/5` (|diff|~19.8, rel~0.99)
- [NLI-rej] `geo170k_extracted_full.jsonl` L2057 step 1: `1 = ((25)*pi/180)` → diff `1 − 5*pi/36` (rel~0.5637)
- [NLI-rej] `geo170k_extracted_full.jsonl` L2310 step 1: `A = ((x)/(360)) * pi r**2` → diff `pi*(180 − pi)/540` (|diff|~1.029, rel~0.9825)
- [NLI-rej] `geo170k_extracted_full.jsonl` L2349 step 2: `((1)/(2))(((180)*pi/180) − ((50)*pi/180)) = ((65)*pi/180)/2` → diff `13*pi/72` (rel~0.5)
- [MC-band] `MathV360K_prompts.jsonl` L3616 step 1: `Perimeter_ = 4 * side_length_` → diff `−15` (rel~0.75)
- [MC-band] `MathV360K_prompts.jsonl` L6555 step 3: `((72)*pi/180) = 2` → diff `−2 + 2*pi/5` (rel~0.3717)
- [MC-band] `MathV360K_prompts.jsonl` L7136 step 1: `Area_of_sector_ = ((theta)/(360)) * pi r**2` → diff `583/400 − 1859*pi**2/720000` (rel~0.9825)
- [MC-band] `MathV360K_prompts.jsonl` L9310 step 0: `10 * 1((1)/(4)) = 10 * 1.25` → diff `−10` (rel~0.8)
- [MC-band] `MathV360K_prompts.jsonl` L10162 step 2: `P = 4s` → diff `−12` (rel~0.75)

### Part 7 — Limitations (as stated)

- **Coverage:** only ~36% of the kept pool is in a verifiable tier; within tiers only the extractable fraction is checked. A PASS on a trace with few claims is weak evidence. SKIPPED_NO_SURFACE says nothing about correctness.
- **Prose-trimming fabrication risk:** ~half of claims had prose trimmed adjacent to `=`; traces whose every failing claim is trim-derived are quarantined as FAIL_LOW_CONFIDENCE, but trimmed claims that PASS could still be misextracted.
- **Exact arithmetic, no float tolerance** (per spec): rounded intermediates (e.g. `sqrt(429) = 20.71`) fail exactly; relative-diff split separates rounding-scale (<1%) from gross (≥20%); verdicts stay exact.
- **Residual fabrication risk in FAILs:** v4 kept-sample FAIL inspection still shows a minority of extraction artifacts (bare parenthesized-number remnants like `(4) = 1/5`, placeholder symbols juxtaposed with parens). **FAIL counts are upper bounds** on genuinely broken math.
- **Exploratory traces** excluded from FAIL counts via heuristic marker detection.
- **Method-2 anchoring:** unanchored unknowns leave constraints UNRESOLVED (skipped, counted); a symbol asserted with several values contributes all as candidate branches (constraint passes if any branch fits); numeric answer field anchors a single leftover unknown, with a degree-interpreted variant when the trace used degrees.
- Parse-fail / timeout / unresolved counts are never counted as FAIL.

---

## 10. Cross-experiment convergent conclusions

1. Cosine/BERTScore cannot be trusted on hard negatives or negation — blind by **geometry, not information loss** (the DCI inversion probe showed embeddings retain the swapped content: ADJ 0.98 / NUM 0.97 / spatial 0.98 recall).
2. Bidirectional NLI is the only viable text scorer, **but** (a) needs clause-level scaffolding (paragraph dilution defeats it: 0.530), (b) has no numeric magnitude sense (COCO 0.503), (c) has a neutral trap on paraphrase (min-pool 0.599).
3. Pooling recall/precision tension: min-pool gives negation FPR **0.008** but paraphrase FPR **0.599**. Fix direction = softened pool (k-th-lowest quantile) + a **hard numeric-match term** exempting numbers from NLI.
4. Attribute-swaps resist both text methods → strongest argument for a multimodal grounding term (box-IoU + region-CLIP); the COCO result is its quantitative case.
5. The SymPy audit adds an independent instrument showing the NLI contradiction signal is roughly **orthogonal to symbolic correctness** within tier (2,544 NLI-rejected traces verify fully clean; 1,647 strictly clean).
