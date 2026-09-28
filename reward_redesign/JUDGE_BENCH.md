# JUDGE BENCH — current judge (microsoft/deberta-xlarge-mnli) vs MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (generated 2026-09-20 21:30 by reward_redesign/judge_bench/jb_aggregate.py)

Every pair is scored with the reward's formula s = clip(0.5·E(a→b) + 0.5·E(b→a) − max C, 0, 1) at 512-token truncation; raw 3-class
probabilities are kept in the shard files (reward_redesign/judge_bench/out/). Configs: whole (the reward today), clause-min (June's split
+ SBERT-mpnet Hungarian alignment, tau 0.15, min-pool), clause-hung-F05 / clause-hung-mean (clauses both sides, SBERT top-3 candidate
pairs → NLI → Hungarian on the sparse s matrix; F_0.5 of clause precision/recall, resp. mean s over assigned pairs), asym-mean (whole a vs
each b-clause and whole b vs each a-clause, forward direction each; s = clip(0.5·mean E + 0.5·mean E − max C)). Single-clause pairs fall
back to `whole` in the three clause configs. Samples: 3a = 4,999 rows of data/visualprm400k/pairs.jsonl (seed 0, stratified by source;
3 rows without exactly one differing blank-line step dropped); 3b = 5,000 of the 9,254 faithful==True paraphrase rows (seed 0; whole
solution as built); 3c = the same rows through reward_v2.score_new's Hungarian matching (whole steps, soft gates); 3d = EQUATE (9,702
pairs, five subsets); 3e = 300 real (rollout step, reference step) pairs from run M's dumps.

## 3a. VisualPRM hard negatives, step level (edited step of neg vs the same step of pos → gold contradict; FPR = fraction with s ≥ 0.5)

| model | config | n | FPR (s ≥ 0.5) ↓ | mean s (edited) ↓ | baseline mean s (unrelated step) | fallback (single clause) | NLI calls / pair |
|---|---|---|---|---|---|---|---|
| cur | whole | 4999 | **0.684** | 0.621 | 0.267 | 0.000 | 2.0 |
| cur | clause-min | 4999 | **0.008** | 0.012 | 0.003 | 0.025 | 38.1 |
| cur | clause-hung-F05 | 4999 | **0.941** | 0.809 | 0.149 | 0.025 | 146.4 |
| cur | clause-hung-mean | 4999 | **0.975** | 0.905 | 0.410 | 0.025 | 146.4 |
| cur | asym-mean | 4999 | **0.121** | 0.212 | 0.044 | 0.025 | 38.1 |
| v3 | whole | 4999 | **0.017** | 0.016 | 0.487 | 0.000 | 2.0 |
| v3 | clause-min | 4999 | **0.000** | 0.001 | 0.003 | 0.025 | 38.1 |
| v3 | clause-hung-F05 | 4999 | **0.918** | 0.743 | 0.130 | 0.025 | 146.4 |
| v3 | clause-hung-mean | 4999 | **0.955** | 0.829 | 0.321 | 0.025 | 146.4 |
| v3 | asym-mean | 4999 | **0.013** | 0.020 | 0.035 | 0.025 | 38.1 |

June reference (docs/visualprm_negation_results.md, docs/negation_clausesplit_results.md; June's equiv formula used −0.5·max C and 256-token truncation, on whole solutions):
paragraph NLI equiv FPR@0.5 on neg = 0.594 (FPR@0.6 = 0.530, mean 0.569); clause-split min-equiv FPR@0.5 = 0.015 (FPR@0.6 = 0.008, mean 0.015).

3a by edited-step category (FPR at s ≥ 0.5; n in the header):

| model | config | bare equation (n=377) | prose with number (n=2236) | mixed (n=2386) |
|---|---|---|---|---|
| cur | whole | 0.281 | 0.711 | 0.723 |
| cur | clause-min | 0.019 | 0.005 | 0.010 |
| cur | clause-hung-F05 | 0.618 | 0.954 | 0.979 |
| cur | clause-hung-mean | 0.825 | 0.975 | 1.000 |
| cur | asym-mean | 0.064 | 0.130 | 0.123 |
| v3 | whole | 0.005 | 0.013 | 0.022 |
| v3 | clause-min | 0.003 | 0.000 | 0.000 |
| v3 | clause-hung-F05 | 0.501 | 0.945 | 0.959 |
| v3 | clause-hung-mean | 0.645 | 0.970 | 0.991 |
| v3 | asym-mean | 0.000 | 0.014 | 0.013 |

## 3b. Paraphrase set (paraphrase vs pos, whole solution → gold entail; FNR = fraction with s < 0.5)

| model | config | n | FNR (s < 0.5) ↓ | mean s ↑ | fallback | NLI calls / pair |
|---|---|---|---|---|---|---|
| cur | whole | 5000 | **0.001** | 0.970 | 0.000 | 2.0 |
| cur | clause-min | 5000 | **0.084** | 0.758 | 0.000 | 53.1 |
| cur | clause-hung-F05 | 5000 | **0.000** | 0.896 | 0.000 | 209.5 |
| cur | clause-hung-mean | 5000 | **0.000** | 0.930 | 0.000 | 209.5 |
| cur | asym-mean | 5000 | **0.715** | 0.380 | 0.000 | 53.5 |
| v3 | whole | 5000 | **0.099** | 0.810 | 0.000 | 2.0 |
| v3 | clause-min | 5000 | **0.630** | 0.358 | 0.000 | 53.1 |
| v3 | clause-hung-F05 | 5000 | **0.002** | 0.825 | 0.000 | 209.5 |
| v3 | clause-hung-mean | 5000 | **0.000** | 0.855 | 0.000 | 209.5 |
| v3 | asym-mean | 5000 | **0.737** | 0.264 | 0.000 | 53.5 |

June reference: clause-split min-pooling raised paraphrase FNR (paraphrase precision FPR 0.60 in the memory note; see docs) — the gap this table tests.

## 3c. Full-solution level through the reward (reward_v2.score_new matching, whole steps only, soft gates)

| model | n (neg/base) | mean match pos→neg ↓ | mean match pos→baseline | n (para) | mean match pos→paraphrase ↑ | edited step credited (s ≥ 0.5 in the assignment) ↓ | edited step assigned to its own index | edited step deduped |
|---|---|---|---|---|---|---|---|---|
| cur | 4999 | 0.672 | 0.199 | 5000 | 0.839 | **0.578** | 0.681 | 0.001 |
| v3 | 4999 | 0.183 | 0.396 | 5000 | 0.761 | **0.055** | 0.023 | 0.001 |

## 3d. EQUATE (whole-step config; 3-class accuracy = argmax of the forward p→h probabilities; AUROC of s for entail vs non-entail; FNR = gold-entail with s < 0.5)

| subset | n | cur acc | cur AUROC | cur FNR | v3 acc | v3 AUROC | v3 FNR |
|---|---|---|---|---|---|---|---|
| RTE-Quant | 166 | 0.614 | 0.858 | 0.686 | 0.747 | 0.931 | 0.271 |
| NewsNLI | 968 | 0.728 | 0.853 | 0.615 | 0.751 | 0.835 | 0.399 |
| RedditNLI | 250 | 0.636 | 0.808 | 0.459 | 0.692 | 0.827 | 0.500 |
| AWP-NLI | 722 | 0.231 | 0.499 | 0.997 | 0.313 | 0.562 | 0.964 |
| StressTest | 7596 | 0.800 | 0.633 | 0.735 | 0.867 | 0.709 | 0.850 |

## Throughput (one GPU, batch 32 = the reward's Scorers batch; pairs/s = NLI forward passes per second)

| model | pair type | jobs | mean pairs/s | min–max | GPUs seen |
|---|---|---|---|---|---|

## 3e. 300 real matched pairs dumped to reward_redesign/judge_pairs_300.jsonl / .md (blank `label` field) — cells: 1-60 acc=0: 50, 1-60 acc=1: 50, 121-183 acc=0: 50, 121-183 acc=1: 50, 61-120 acc=0: 50, 61-120 acc=1: 50

## Read (selection rule: among configs with 3a FPR < 10 %, lowest 3b FNR wins; throughput as tiebreak)

| model | config | 3a FPR | 3b FNR | calls/pair | eligible (FPR < 0.10) |
|---|---|---|---|---|---|
| cur | whole | 0.684 | 0.001 | 2.0 | no |
| cur | clause-min | 0.008 | 0.084 | 38.1 | yes |
| cur | clause-hung-F05 | 0.941 | 0.000 | 146.4 | no |
| cur | clause-hung-mean | 0.975 | 0.000 | 146.4 | no |
| cur | asym-mean | 0.121 | 0.715 | 38.1 | no |
| v3 | whole | 0.017 | 0.099 | 2.0 | yes |
| v3 | clause-min | 0.000 | 0.630 | 38.1 | yes |
| v3 | clause-hung-F05 | 0.918 | 0.002 | 146.4 | no |
| v3 | clause-hung-mean | 0.955 | 0.000 | 146.4 | no |
| v3 | asym-mean | 0.013 | 0.737 | 38.1 | yes |

**Winner: cur / clause-min** — 3a FPR 0.008, 3b FNR 0.084, 38.1 NLI calls per pair; versus today's judge (cur / whole: FPR 0.684, FNR 0.001). This is a pooling change only. It is a **candidate for arms 2–6 after a probe rerun** (reward_redesign/redteam_reward.py). Epoch 2 (M2/R2) stays on the current judge and the `whole` config regardless.
