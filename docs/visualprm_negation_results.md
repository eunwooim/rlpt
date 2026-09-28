# VisualPRM400K — Math negation sensitivity (SBERT / NLI / BERTScore)

Dataset: VisualPRM400K **math subset** (24 PRM files, geometry/function/visual-math). Hard negative = correct step-by-step solution with the **first equation RHS number changed by +1** (e.g. `∠B = 50°`→`51°`). N = 40,000 pairs.

**NEG** = sim(correct, negated). **BASE** = sim(correct, unrelated solution). A math-faithful scorer drives NEG **down** (it noticed the number changed); a scorer blind to the edit leaves NEG ≈ 1.0, far above BASE.

## 0. Example pair (geometry3k, edit `14->15`)

The hard negative changes **one number on the RHS of the first equation** and leaves everything else
byte-for-byte identical (note the downstream lines still say `14`, so the negative is also internally
inconsistent — exactly what a faithful scorer should penalize).

**POS (correct):**
```
Given that AB = 14, and because AB is a side of the rhombus and all sides of a rhombus are equal:
- Side CD = AB = 14
- Side AD = AB = 14 (since all sides are equal)
- Side BC = AB = 14 (since all sides are equal)
Thus, the length of BC is 14.
Final answer: D
```

**NEG (corrupted, `AB = 14` → `AB = 15`):**
```
Given that AB = 15, and because AB is a side of the rhombus and all sides of a rhombus are equal:
- Side CD = AB = 14
- Side AD = AB = 14 (since all sides are equal)
- Side BC = AB = 14 (since all sides are equal)
Thus, the length of BC is 14.
Final answer: D
```

The two texts differ in a single character (`14`→`15`). SBERT/BERTScore score this ≈0.98–0.99
(blind); NLI drops it but still often rates it above an unrelated solution.

## 1. Similarity distributions

| scorer (field) | NEG mean | NEG median | NEG p10–p90 | BASE mean | gap (NEG−BASE) |
|---|--:|--:|--:|--:|--:|
| SBERT cosine | 0.993 | 0.997 | 0.980–1.000 | 0.302 | +0.691 |
| NLI equiv | 0.568 | 0.647 | 0.000–0.984 | 0.100 | +0.468 |
| NLI coverage | 0.544 | 0.607 | 0.000–0.984 | 0.101 | +0.443 |
| BERTScore f1 | 0.981 | 0.989 | 0.958–0.996 | 0.671 | +0.310 |

## 2. Missed-corruption rate vs τ  (fraction of NEG pairs scoring ≥ τ — lower = better)

| scorer (field) | τ=0.3 | τ=0.4 | τ=0.5 | τ=0.6 | τ=0.7 | τ=0.8 | τ=0.9 | τ=0.95 |
|---|---|---|---|---|---|---|---|---|
| SBERT cosine | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.991 |
| NLI equiv | 0.704 | 0.653 | 0.594 | 0.530 | 0.465 | 0.401 | 0.316 | 0.246 |
| NLI coverage | 0.669 | 0.612 | 0.555 | 0.504 | 0.449 | 0.391 | 0.310 | 0.243 |
| BERTScore f1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.975 | 0.917 |

### Paper number — FPR@0.6 (= missed-corruption rate at τ=0.6; lower better)

Reported field per scorer, for the cross-benchmark table (alongside NegBench / SugarCrepe / SugarCrepe++).
For VisualPRM (math) we report **NLI equiv** — strict mutual equivalence is the principled criterion for
numeric correctness (a single wrong number must break equivalence).

| scorer (field) | FPR@0.6 |
|---|--:|
| **NLI (equiv)** | **0.530** |
| NLI (coverage) | 0.504 |
| SBERT (raw) | 1.000 |
| BERTScore (F1) | 1.000 |

Reading: even the entailment model misses just over half the single-digit corruptions; the cosine-family
scorers miss all of them. This is the worst NLI result across the four benchmarks (NegBench 0.001 →
SugarCrepe 0.069 → VisualPRM 0.530): a wrong number buried in long math reasoning is the hardest negation
to catch.

## 3. NLI contradiction probability (max of both directions)

| condition | mean C | median C |
|---|--:|--:|
| NEG | 0.309 | 0.180 |
| BASE | 0.248 | 0.192 |

## 4. Surface-overlap reliance: AUC = P(NEG sim > BASE sim)

High = scorer ranks a *corrupted copy of the same problem* as more similar than an unrelated solution, i.e. it keys on surface overlap and is blind to the math error. (1.0 = always; 0.5 = chance.)

| scorer (field) | AUC(NEG>BASE) |
|---|--:|
| SBERT cosine | 1.000 |
| NLI equiv | 0.816 |
| NLI coverage | 0.793 |
| BERTScore f1 | 1.000 |

## 5. NEG mean by source (top by count)

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
