# Chunker — morning report

_Generated 2026-07-23T06:34:06-07:00 by Slurm (session-independent chain)._

## Bottom line

- **Operating threshold selected: `0.25`** (boundary F1 0.8078, token F1 0.8699, zero-cut 2.09%, 2.8577 false splits/1-step).
- A token-F1 objective would instead have picked `0.3` (boundary F1 0.8076) — recorded so the choice can be overridden.
- Release packaged: `/scratch/sghos104/rlpt/chunker/release/chunker-deberta-v3-small-v1` (threshold 0.25).

## Arm comparison (validation, corrected split)

| metric | plain (winner) | weighted |
|---|---|---|
| best_f1 | 0.8154 | 0.8141 |
| threshold | 0.35 | 0.8 |
| precision | 0.7733 | 0.7647 |
| recall | 0.8623 | 0.8703 |
| f1_at_0.5 | 0.8064 | 0.7786 |
| final_val_loss | 0.09459236264228821 | 0.1479572355747223 |

Per-language token split-F1:

| bucket | gold boundaries | plain | weighted |
|---|---|---|---|
| ALL | 88,754 | 0.8252 | 0.8239 |
| en | 86,320 | 0.8243 | 0.823 |
| cjk | 2,434 | 0.8559 | 0.8567 |

**CJK outperforms English** — the mDeBERTa-v3-base contingency is closed.

## Threshold sweep (validation) + decision rule

Rule: disqualify if zero-cut > **0.05** or > **2.0x** the 0.35 baseline (baseline zero-cut = 0.0287); then maximise **boundary F1**.

| thr | token F1 | boundary F1 | **zero-cut %** | false splits /1-step | frac<min | eligible |
|---|---|---|---|---|---|---|
| 0.2 | 0.8665 | 0.8057 | 1.70% | 3.1063 | 9e-06 | yes |
| 0.25 **<-- chosen** | 0.8699 | 0.8078 | 2.09% | 2.8577 | 1e-05 | yes |
| 0.3 | 0.8703 | 0.8076 | 2.43% | 2.62 | 1e-05 | yes |
| 0.35 | 0.8678 | 0.805 | 2.87% | 2.414 | 1e-05 | yes |
| 0.4 | 0.8629 | 0.8001 | 3.41% | 2.1862 | 1.1e-05 | yes |
| 0.45 | 0.8553 | 0.7935 | 3.95% | 1.9877 | 1.1e-05 | yes |
| 0.5 | 0.8454 | 0.7854 | 4.63% | 1.8285 | 1.2e-05 | yes |
| 0.55 | 0.8334 | 0.7758 | 5.29% | 1.686 | 1.2e-05 | **NO** |
| 0.6 | 0.8192 | 0.7655 | 5.88% | 1.5637 | 1.2e-05 | **NO** |
| 0.65 | 0.8022 | 0.7532 | 6.78% | 1.4505 | 1.3e-05 | **NO** |
| 0.7 | 0.7826 | 0.739 | 7.66% | 1.348 | 1.3e-05 | **NO** |
| 0.75 | 0.7584 | 0.7223 | 8.79% | 1.2615 | 1.4e-05 | **NO** |
| 0.8 | 0.7273 | 0.7011 | 10.14% | 1.1505 | 1.4e-05 | **NO** |
| 0.85 | 0.6852 | 0.6742 | 11.91% | 1.0457 | 1.5e-05 | **NO** |
| 0.9 | 0.6314 | 0.6413 | 14.53% | 0.9657 | 1.6e-05 | **NO** |

Disqualified:

- `0.55`: zero_cut 0.0529 > 0.05
- `0.6`: zero_cut 0.0588 > 0.05; zero_cut 0.0588 > 2.0x baseline (0.0287)
- `0.65`: zero_cut 0.0678 > 0.05; zero_cut 0.0678 > 2.0x baseline (0.0287)
- `0.7`: zero_cut 0.0766 > 0.05; zero_cut 0.0766 > 2.0x baseline (0.0287)
- `0.75`: zero_cut 0.0879 > 0.05; zero_cut 0.0879 > 2.0x baseline (0.0287)
- `0.8`: zero_cut 0.1014 > 0.05; zero_cut 0.1014 > 2.0x baseline (0.0287)
- `0.85`: zero_cut 0.1191 > 0.05; zero_cut 0.1191 > 2.0x baseline (0.0287)
- `0.9`: zero_cut 0.1453 > 0.05; zero_cut 0.1453 > 2.0x baseline (0.0287)

## Zero-cut analysis (test, at threshold 0.35)

- **453 of 15000 multi-step trajectories (3.02%) received ZERO cuts** — they reach the reward model as one undivided blob.

| language | zero-cut | total | rate |
|---|---|---|---|
| en | 443 | 14597 | 3.03% |
| cjk | 10 | 403 | 2.48% |

- Zero-cut trajectories average **117.1 tokens** vs **328.9** for normally-segmented ones — SHORTER, so the slice is concentrated in short texts.
- Per-trajectory boundary recall: mean 0.7574, 57.5% of trajectories at >=0.8 recall, 3.87% at zero.

Examples of zero-cut trajectories:

- `en` 5 steps, 119 tok, 4 gold — "To determine how many objects are left after subtracting 0 purple balls, let's c"
- `en` 5 steps, 113 tok, 4 gold — 'To determine the number of objects left after subtracting 0 purple balls, we nee'
- `en` 2 steps, 145 tok, 1 gold — '1. The question asks about adding 7 big blue metal things and how many big blue '
- `en` 7 steps, 175 tok, 6 gold — 'To determine how many objects exist once we add 1 small gray cube to the origina'
- `en` 6 steps, 128 tok, 5 gold — 'To determine how many objects are left after subtracting 0 gray balls, follow th'

## Test-set eval (winner only, ONE run)

Threshold 0.35, 19,000 trajectories (15,000 multi / 4,000 single).

| bucket | gold | token F1 | P | R |
|---|---|---|---|---|
| ALL | 88,256 | 0.8209 | 0.7787 | 0.868 |
| en | 85,762 | 0.82 | 0.7778 | 0.8671 |
| cjk | 2,494 | 0.8529 | 0.8109 | 0.8994 |

- min-8 fragments: **2 of 94,273** (frac 2e-05).
- over-split on 1-step: 2.4312 mean false splits, 67.8% of trajectories affected.

_Note: this eval used threshold 0.35. If the sweep selected a different threshold, these test numbers correspond to 0.35, NOT to the packaged threshold — test was deliberately spent once and not re-run._

## Split health (why everything was rebuilt)

| split | distinct multi-step questions | rollouts/question (mean/max) | CJK multi |
|---|---|---|---|
| train | 193,114 | 2.41 / 2217 | 12,301 |
| val | 7,374 | 2.03 / 20 | 400 |
| test | 7,410 | 2.02 / 20 | 403 |

Flags: ['none -- all split-health checks passed']

The previous split gave val **31** and test **72** distinct questions (~490 near-duplicate rollouts each) while trajectory counts looked perfect, and put all 13,103 CJK multi-step trajectories in train. That inflated val F1 to 0.9147 and produced a val->test recall collapse of 0.871 -> 0.535.

## Known gaps / caveats

- Test numbers above are at threshold 0.35; the packaged threshold may differ (test not re-run).
- `other` language bucket has no multi-step trajectories, so its F1 is an empty denominator, not a result.
- Over-splitting on single-step trajectories is the main weakness; the threshold is the knob.
- The model peaked at step 10,000 of 16,335 and early-stopped at 15,000.

<!-- metrics-pack:start -->
## Extended metrics pack (val @ thr 0.25)

- boundary F1 (multi) 0.8078 [0.8041, 0.8114]; normalized BED 0.3689 [0.3643, 0.3739]; segment mAP@0.5 0.7614 [0.757, 0.7659]; fragmentation 0.9777 [0.9726, 0.9829]
- full report: `release/chunker-deberta-v3-small-v1/METRICS_REPORT.md`
- failures / improvised definitions: 3 (listed in the report)
<!-- metrics-pack:end -->

<!-- longtext-diag:start -->
## Long-text decoder follow-ups (>512-token val multi, thr from release)

### Seam diagnostic (stride 384, 2,972 traj)

| dist to nearest seam (tok) | n gold | n cut | recall | precision | F1 | token share |
|---|---|---|---|---|---|---|
| 0-8 | 1,485 | 1,665 | 0.7778 | 0.6937 | 0.7333 | 5.5% |
| 9-16 | 1,300 | 1,505 | 0.7962 | 0.6877 | 0.738 | 5.0% |
| 17-32 | 2,533 | 2,990 | 0.8105 | 0.6866 | 0.7434 | 9.6% |
| 33-64 | 4,584 | 5,325 | 0.8019 | 0.6903 | 0.742 | 17.7% |
| 65-128 | 7,299 | 8,438 | 0.8037 | 0.6952 | 0.7455 | 28.3% |
| >128 | 8,945 | 10,172 | 0.7983 | 0.702 | 0.7471 | 33.8% |

- F1 within 32 tokens of a seam: **0.7393**; beyond 32: **0.7454**

### Decode grid (2x2), boundary F1 on the same bucket (2,972 traj @ thr 0.25)

| stride | merge | precision | recall | boundary F1 |
|---|---|---|---|---|
| 384 | mean | 0.6953 | 0.8004 | 0.7442 |
| 384 | max | 0.6883 | 0.8077 | 0.7432 |
| 256 | mean | 0.6964 | 0.7993 | 0.7443 |
| 256 | max | 0.6866 | 0.8079 | 0.7423 |

<!-- longtext-diag:end -->

<!-- metrics-pack-multi-step only:start -->
## Extended metrics pack (val @ thr 0.35, tag multi-step only)

- boundary F1 (multi) 0.805 [0.8011, 0.8087]; normalized BED 0.3714 [0.3663, 0.3763]; segment mAP@0.5 0.7446 [0.7399, 0.7493]; fragmentation 0.922 [0.9175, 0.9269]
- full report: `release/chunker-deberta-v3-small-v1/METRICS_REPORT_thr035.md`
- failures / improvised definitions: 4 (listed in the report)
<!-- metrics-pack-multi-step only:end -->
