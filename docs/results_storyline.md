# RLPT ScienceQA GRPO — Results Storyline

*(as of 2026-06-10; ablation + validation jobs still running)*

## Chapter 1 — The setup

The goal: test whether RL on *free-text reasoning*, scored by soft bipartite matching
against ground-truth solutions, makes a multimodal model genuinely better at analysing
images. Dataset: ScienceQA (5,678 train rows with image+solution). Model:
Qwen2.5-VL-3B-Instruct. Reward: `R = 1·format + 2·match + 5·answer − 1·hallucination-penalty`,
where `match` is the SBERT-Hungarian bipartite F-score between the model's `<think>` clauses
and the GT solution's clauses. The hard `answer` term exists because SBERT cosine alone
couldn't tell a fluent wrong answer from a right one (0.41 vs 0.63 — basically
indistinguishable; the composite separates them 1.8 vs 7.3).

## Chapter 2 — The bring-up war

Twelve distinct failures stood between the env build and the first training step: Ray
hanging on BeeGFS, the `ROCR_VISIBLE_DEVICES` clash, glibc-2.28 killing flash-attn (solved
with a hand-built SDPA shim), SBERT refusing to load inside verl's meta-tensor reward worker
(solved by exiling it to a subprocess pipe server), and finally host-RAM OOM at step 21 of
the first full run — which turned out not to be a leak but a one-time +50 GB optimizer-state
materialization that put steady state at ~270 GB, just over the 240 GB request. Bumped to
400 GB and it never flinched again.

## Chapter 3 — The training run

The relaunched 2×A100 run completed cleanly: **2 epochs, 176 steps, 6h56m**. Reward climbed
from −2.0 to ~+5.9 and plateaued near the ceiling, with checkpoints saved every 20 steps.

## Chapter 4 — The headline result

Held-out test set (1,836 rows, final checkpoint `global_step_176`):

| Metric | Value |
|---|---|
| Accuracy | **90.3%** |
| Format compliance | 100% |
| Reasoning match F | 0.778 |
| Hallucinated clauses | 0.091 per response |
| Composite reward | 6.98 |

Base Qwen2.5-VL-3B is already 79% zero-shot on ScienceQA (supervisor-measured), so **RL
delivered +11.3 accuracy points on an already-competent model**. The base model's "−0.31
composite" isn't incompetence — it just doesn't emit the `<think>/<answer>` format, so the
strict harness can't credit its answers.

## Chapter 5 — The real finding (the checkpoint sweep)

The checkpoint sweep (256-row test subset; `data/logs/match_sweep_results.tsv` +
`match_sweep_anchors.tsv`) decomposes *what* RL learned, *when*:

- **Format**: 0.48 → 1.00 by step 20. Solved almost instantly.
- **Accuracy**: 0.645 → 0.82 by step 20, drifting to ~0.89 with noise. Mostly solved early.
- **Reasoning match**: **0.371 → 0.625 (step 20) → 0.682 (40) → 0.737 (100) → 0.766 (176)**
  — a steady, monotonic climb across the *entire* run, long after format and answers
  saturated.

So RL roughly **doubled reasoning faithfulness** and kept refining it after the easy reward
terms were exhausted — exactly the signal the bipartite match term was designed to capture.
(Caveat on the 0.37 base anchor: no `<think>` block means the matcher scores terse raw
output, so it partly reflects brevity.)

| step | acc | match | format | reward |
|---|---|---|---|---|
| base | 0.645 | 0.371 | 0.480 | −0.31 |
| 20 | 0.824 | 0.625 | 1.00 | 6.30 |
| 40 | 0.848 | 0.682 | 1.00 | 6.53 |
| 60 | 0.859 | 0.705 | 1.00 | 6.64 |
| 80 | 0.883 | 0.723 | 1.00 | 6.75 |
| 100 | 0.910 | 0.737 | 1.00 | 6.50 |
| 120 | 0.879 | 0.741 | 1.00 | 6.74 |
| 140 | 0.902 | 0.751 | 1.00 | 6.92 |
| 160 | 0.891 | 0.752 | 1.00 | 6.90 |
| 176 | 0.891 | 0.766 | 1.00 | 6.91 |

## Chapter 6 — The skeptic's question (in progress)

The match climb is partly circular — we optimized it — and could reflect stylistic mimicry
of ScienceQA solutions rather than better image analysis. The validation plan addresses
this: shuffled-GT control, NLI re-scoring, image-swap probe, transfer eval, and the
`w_match=0` ablation.

Currently running (2026-06-10):

- **`rlpt-grpo-nomatch`** — the `w_match=0` ablation, the causal keystone. Training
  healthily (~140 s/step, score ~4.5–4.8 of max 6). Early hint: entropy falling
  (0.99→0.84) and responses getting *shorter* (119→90 mean tokens) — without the match
  term the model has no incentive to elaborate its reasoning.
- **`rlpt-gen-valid`** — generation pass over validation rows (vLLM warm, generating).
- **`rlpt-score-valid`** — queued behind gen-valid; scores the generations.

The comparison to watch: if the no-match run reaches ~90% accuracy but its match score
*doesn't* climb past ~0.6, that isolates the bipartite term as the cause of the
reasoning-faithfulness gains — the third leg of the publishable claim
(content-not-style · image-grounded · caused-by-match-term).
