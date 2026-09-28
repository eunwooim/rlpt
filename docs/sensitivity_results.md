# Numerical Sensitivity Analysis — AUC + tau sweep

Computed from saved reliability scores (252,693 score rows). Pure-CPU re-aggregation; no models re-run.

**Metrics.** ROC-AUC = P(a true match scores above a hard negative); threshold-free, 1.0 perfect, 0.5 chance, <0.5 inverted. FPR(tau) = fraction of hard negatives scoring >= tau (lower better). Sensitivity (TPR) = fraction of true matches scoring >= tau. Specificity = 1 - FPR.

## 1. ROC-AUC (threshold-free)

Primary similarity field per scorer. `sugarcrepe++` = within-dataset (genuine positives vs hard negatives). `POOLED_match_detector` = positives from SugarCrepe++ vs ALL hard negatives.

| scope | scorer | field | n_pos | n_neg | AUC |
|---|---|---|--:|--:|--:|
| sugarcrepepp | sbert | raw_score | 4757 | 4757 | 0.7277 |
| POOLED_match_detector | sbert | raw_score | 4757 | 79474 | 0.9084 |
| sugarcrepepp | nli | nli_score_coverage | 4757 | 4757 | 0.9810 |
| POOLED_match_detector | nli | nli_score_coverage | 4757 | 79474 | 0.9938 |
| sugarcrepepp | bertscore | bertscore_f1 | 4757 | 4757 | 0.1775 |
| POOLED_match_detector | bertscore | bertscore_f1 | 4757 | 79474 | 0.7750 |

<details><summary>All score fields (incl. NLI entailment/contradiction)</summary>

| scope | scorer | field | n_pos | n_neg | AUC |
|---|---|---|--:|--:|--:|
| sugarcrepepp | sbert | raw_score | 4757 | 4757 | 0.7277 |
| POOLED_match_detector | sbert | raw_score | 4757 | 79474 | 0.9084 |
| sugarcrepepp | nli | nli_score_coverage | 4757 | 4757 | 0.9810 |
| POOLED_match_detector | nli | nli_score_coverage | 4757 | 79474 | 0.9938 |
| sugarcrepepp | nli | nli_score_equiv | 4757 | 4757 | 0.9812 |
| POOLED_match_detector | nli | nli_score_equiv | 4757 | 79474 | 0.9942 |
| sugarcrepepp | nli | E_ab | 4757 | 4757 | 0.9835 |
| POOLED_match_detector | nli | E_ab | 4757 | 79474 | 0.9970 |
| sugarcrepepp | nli | E_ba | 4757 | 4757 | 0.9744 |
| POOLED_match_detector | nli | E_ba | 4757 | 79474 | 0.9848 |
| sugarcrepepp | nli | C_ab | 4757 | 4757 | 0.0115 |
| POOLED_match_detector | nli | C_ab | 4757 | 79474 | 0.0052 |
| sugarcrepepp | nli | C_ba | 4757 | 4757 | 0.0188 |
| POOLED_match_detector | nli | C_ba | 4757 | 79474 | 0.0111 |
| sugarcrepepp | bertscore | bertscore_precision | 4757 | 4757 | 0.1585 |
| POOLED_match_detector | bertscore | bertscore_precision | 4757 | 79474 | 0.7246 |
| sugarcrepepp | bertscore | bertscore_recall | 4757 | 4757 | 0.2153 |
| POOLED_match_detector | bertscore | bertscore_recall | 4757 | 79474 | 0.7900 |
| sugarcrepepp | bertscore | bertscore_f1 | 4757 | 4757 | 0.1775 |
| POOLED_match_detector | bertscore | bertscore_f1 | 4757 | 79474 | 0.7750 |

</details>

## 2. tau sweep — FPR(tau) on hard negatives (lower better)

### negbench

| scorer (primary field) | tau=0.30 | tau=0.40 | tau=0.50 | tau=0.60 | tau=0.70 | tau=0.80 | tau=0.90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| nli `nli_score_coverage` | 0.0028 | 0.0023 | 0.0017 | 0.0013 | 0.0010 | 0.0007 | 0.0005 |
| sbert `raw_score` | 0.9520 | 0.9023 | 0.8334 | 0.7531 | 0.6054 | 0.3722 | 0.0815 |
| bertscore `bertscore_f1` | 1.0000 | 1.0000 | 0.9996 | 0.9967 | 0.9858 | 0.8482 | 0.0688 |

### sugarcrepe

| scorer (primary field) | tau=0.30 | tau=0.40 | tau=0.50 | tau=0.60 | tau=0.70 | tau=0.80 | tau=0.90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| nli `nli_score_coverage` | 0.1339 | 0.0863 | 0.0776 | 0.0691 | 0.0618 | 0.0521 | 0.0403 |
| sbert `raw_score` | 0.9999 | 0.9988 | 0.9929 | 0.9744 | 0.9175 | 0.7386 | 0.4399 |
| bertscore `bertscore_f1` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9937 | 0.8820 |

### sugarcrepepp

| scorer (primary field) | tau=0.30 | tau=0.40 | tau=0.50 | tau=0.60 | tau=0.70 | tau=0.80 | tau=0.90 |
|---|--:|--:|--:|--:|--:|--:|--:|
| nli `nli_score_coverage` | 0.1408 | 0.1259 | 0.1144 | 0.1030 | 0.0921 | 0.0780 | 0.0578 |
| sbert `raw_score` | 0.9998 | 0.9983 | 0.9895 | 0.9601 | 0.8833 | 0.6571 | 0.3775 |
| bertscore `bertscore_f1` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9998 | 0.9941 | 0.8875 |

## 3. Full ROC points — sugarcrepe++ (has both classes)

Sensitivity (TPR, want high) and Specificity (1-FPR, want high) vs tau.

**nli `nli_score_coverage`**

| tau | sensitivity (TPR) | specificity | FPR |
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

| tau | sensitivity (TPR) | specificity | FPR |
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

**bertscore `bertscore_f1`**

| tau | sensitivity (TPR) | specificity | FPR |
|--:|--:|--:|--:|
| 0.10 | 1.0000 | 0.0000 | 1.0000 |
| 0.20 | 1.0000 | 0.0000 | 1.0000 |
| 0.30 | 1.0000 | 0.0000 | 1.0000 |
| 0.40 | 1.0000 | 0.0000 | 1.0000 |
| 0.50 | 1.0000 | 0.0000 | 1.0000 |
| 0.60 | 1.0000 | 0.0000 | 1.0000 |
| 0.70 | 0.9994 | 0.0002 | 0.9998 |
| 0.80 | 0.9912 | 0.0059 | 0.9941 |
| 0.90 | 0.5314 | 0.1125 | 0.8875 |

## 4. Operating threshold tau* for a target FPR

The tau you'd set to cap false-accepts at 1% / 5% / 10% of hard negatives (from the negative-score quantiles). A scorer whose tau* is unreachable/“too high” has no usable operating point.

| dataset | scorer | field | tau*@FPR=1% | tau*@FPR=5% | tau*@FPR=10% |
|---|---|---|--:|--:|--:|
| negbench | sbert | raw_score | 0.9519 | 0.9185 | 0.8912 |
| negbench | nli | nli_score_coverage | 0.0000 | 0.0000 | 0.0000 |
| negbench | bertscore | bertscore_f1 | 0.9458 | 0.9048 | 0.8893 |
| sugarcrepe | sbert | raw_score | 0.9928 | 0.9824 | 0.9722 |
| sugarcrepe | nli | nli_score_coverage | 0.9902 | 0.8203 | 0.3123 |
| sugarcrepe | bertscore | bertscore_f1 | 0.9916 | 0.9853 | 0.9800 |
| sugarcrepepp | sbert | raw_score | 0.9932 | 0.9845 | 0.9749 |
| sugarcrepepp | nli | nli_score_coverage | 0.9931 | 0.9345 | 0.6396 |
| sugarcrepepp | bertscore | bertscore_f1 | 0.9928 | 0.9875 | 0.9836 |
