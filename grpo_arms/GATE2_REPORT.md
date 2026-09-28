# Gate 2 — Arm 1 (3B / NLI-bipartite match / chunker) to completion

Job 62401599, 183/183 steps, frozen subset sha bd15b1f1..., tau=0.45.
Per-step analysis: analyze_arm1.py -> data/arm1_gate2_stats.json (per-sample
dumps for all 183 steps in runs/arm1_3b_match_chunker/rollouts/).

## 1. Does match move independently of answer? Yes — three acts.

| phase | match | answer | format | words/resp | corr(match, answer) | m|a=1 vs m|a=0 |
|---|---|---|---|---|---|---|
| 1-20 | 0.21 | 0.60 | 0.97 | 189 | +0.29 | 0.25 / 0.16 |
| 21-60 | 0.31 | 0.63 | 0.97 | 272 | +0.16 | 0.33 / 0.28 |
| 61-100 | 0.61 | 0.47 | 0.82 | 630 | +0.04 | 0.61 / 0.60 |
| 101-140 | 0.77 | 0.34 | 0.63 | 722 | -0.09 | 0.74 / 0.77 |
| 141-183 | 0.78 | 0.56 | 0.98 | 764 | +0.13 | 0.79 / 0.76 |

Act 1 (steps 1-60): the match term carries real signal — positively correlated
with answer and higher on correct rollouts.
Act 2 (61-140): LENGTH HACK. Response length 287->1007 tokens, fraction
truncated at the 1024 cap 1.9%->92.5%, entropy 0.34->0.03. Truncation eats the
final-answer line (format 0.97->0.63, answer 0.63->0.34) while coverage keeps
match climbing. Once a GRPO group is uniformly truncated with answer=0, the
group-normalized advantage carries no answer gradient — the 5x weight is inert.
Act 3 (141-183): REPETITION HACK. The policy loops filler sentences ("We should
determine the value of cos A." repeated), wedges an answer marker at ~0.97
relative depth, and pads after it. Format/answer recover nominally; match holds
at 0.78 via repetitive walls clearing tau; match is now answer-INDEPENDENT
(m|a=1 == m|a=0).

## 2. VPB validation accuracy (frozen 2,856-question set, greedy, job 62443511)

| model | overall | DynaMath | MMMU | MathVerse | MathVision | WeMath | marker rate |
|---|---|---|---|---|---|---|---|
| base 3B | **0.2952** | 0.4105 | 0.3658 | 0.2290 | 0.1980 | 0.4777 | 0.916 |
| arm1 step-183 | 0.2549 | 0.2649 | 0.3619 | 0.2115 | 0.1952 | 0.4399 | 0.995 |

**Arm 1 is 4.0 points WORSE than base** (25.5% vs 29.5%), with the damage
concentrated in DynaMath (-14.6 pts). Marker emission went UP (99.5%) — the
policy learned the format shell while the reasoning degenerated.

## 3. Contributing data finding
1,185/6,000 (19.75%) of the frozen subset's gold references are <think>-style
thinking traces (a VisualPRM400K-v1.1-Raw property, spread over many sources).
Long rambling references reward long rambling rollouts under a coverage-style
match term. Arms 1 and 3 share the file, so the 1<->3 comparison stays fair.

## 4. Verdict for the match term as configured
It contributes information early (act 1), then gets hacked twice, and the final
policy is WORSE than base on held-out accuracy. Before arms 2/5 are worth
running, the reward needs an anti-hacking change; candidate levers, smallest
first:
  (a) length/truncation guard: zero match (or zero total reward) when
      finish_reason == length — removes the act-2 gradient entirely;
  (b) repetition penalty in the match term: dedupe near-identical segments
      before Hungarian (or per-unmatched-segment penalty, the old "pun" term);
  (c) raise max_response_length so truncation cannot eat the answer (treats the
      symptom, not the padding incentive);
  (d) restore the pun term (-1 per unmatched segment) from the ScienceQA-era
      composite, which punished padding directly and never showed this failure.
Arm 3 (VisualPRM min-score, no coverage incentive) is the control: if its
length curve stays flat, the blow-up is match-specific.

## Artifacts
- checkpoints: runs/arm1_3b_match_chunker/checkpoints/global_step_{20..183}
- merged HF model: runs/arm1_3b_match_chunker/hf_final
- per-step per-sample dumps: runs/arm1_3b_match_chunker/rollouts/{1..183}.jsonl
- curves: data/arm1_gate2_stats.json; VPB: evals/vpb_scores.json + generations
