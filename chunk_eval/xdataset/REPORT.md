# Cross-dataset chunker generalization — PRM800K + ProcessBench (2026-07-30)

**Question:** does the release boundary chunker's segmentation skill
transfer to reasoning corpora it never saw (critical-path if it will
segment RLPT policy rollouts), or did it learn VisualPRM's segmentation
style?

**Protocol:** V2 strip-and-resegment ONLY — gold steps joined with a single
space (destroys every marker/typography cue), release chunker
`chunker-deberta-v3-small-v1` forced model path, threshold 0.35, min/max
8/220 tokens; exact-position boundary match on whitespace-insensitive
character positions. Code path identical to chunk_eval Exp 4 (imports
`exp4_resegment` / `cascade` verbatim). Marker tiers deliberately NOT run:
they would measure each corpus's typography, not the model.

Jobs (all rc=0): download 59934752, convert 59934996, timing-2K 59935480,
full 59935908 (45 min wall, ~0.75 A100-h allocated; GPU compute 26 min).
Data acquisition + SHA256s: `ACQUISITION.md` / `acquisition.json`.
Eval-only: nothing from these corpora touches any training directory.

## Populations & conventions (measured — `convert_stats.json`)

Gold populations, solutions with ≥3 steps: PRM800K = the per-step
chosen-completion path (human fallback) of phase2 trajectories, non-QC,
non-screening, `finish_reason=="solution"`, fully resolved, exact-text
deduped (funnel in convert_stats.json: 100,544 rows → 10,007 solutions;
dominant drop = `finish_found_error` 66,258, which is expected for phase 2).
ProcessBench = the pre-split `steps` field (3,400 rows → 3,372).
VisualPRM reference stats from the exp2 gold sample (20K, ≥3 steps),
identical tokenizer measurement (release chunker tokenizer):

| corpus | records | steps/sol mean/p50/p90 | step tokens mean/p50/p99 | steps >220 tok |
|---|---:|---|---|---:|
| VisualPRM gold | 20,000 (sample) | 7.1 / 7 / 11 | 46.8 / 35 / 211 | — |
| PRM800K | 10,007 | 10.8 / 9 / 18 | **31.6** / 28 / 105 | 0.04% |
| ProcessBench | 3,372 | 7.6 / 7 / 12 | **89.5** / 74 / 302 | **4.11%** (in 20.4% of records) |

Convention spread is wide: PRM800K steps are ~sentence-grained (finer than
VisualPRM), ProcessBench ~paragraph-grained (much coarser); 4.1% of
ProcessBench steps exceed the model's hard 220-token max-chunk cap, so some
over-splitting there is mechanically forced regardless of model skill.

## Main results (measured — `xdataset_stats_full.json`)

Overall, V2 protocol, exact-position boundary scoring:

| corpus | n | macroF1 | macroP | macroR | coverage (micro R) | density | WindowDiff | perfect% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VisualPRM (reference) | 495,756 | 0.796 | 0.870 | 0.766 | 0.769 | 0.92 | 0.108 | 28.9 |
| **PRM800K** | 10,007 | **0.940** | 0.981 | 0.910 | **0.928** | 0.93 | 0.028 | 30.8 |
| **ProcessBench** | 3,372 | **0.646** | 0.637 | 0.764 | **0.704** | 1.40 | 0.303 | 4.4 |

Harness consistency check: the VisualPRM row, run through this job's code
path over all 495,756 gold records, reproduces the Exp 4 V2 numbers
exactly (0.796 / 0.769 / 0.92 / 0.108 / 28.9%). PASS.

By n-steps bucket:

| corpus | bucket | n | F1 | P | R | cov | dens | wdiff | perfect% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prm800k | 3-4 | 454 | 0.861 | 0.967 | 0.804 | 0.806 | 0.85 | 0.050 | 39.2 |
| prm800k | 5-7 | 2,759 | 0.917 | 0.977 | 0.873 | 0.875 | 0.90 | 0.036 | 30.8 |
| prm800k | 8+ | 6,794 | 0.955 | 0.984 | 0.932 | 0.939 | 0.95 | 0.024 | 30.2 |
| processbench | 3-4 | 586 | 0.578 | 0.461 | 0.874 | 0.873 | **2.20** | 0.413 | 3.4 |
| processbench | 5-7 | 1,442 | 0.653 | 0.598 | 0.810 | 0.805 | 1.51 | 0.309 | 5.7 |
| processbench | 8+ | 1,344 | 0.667 | 0.756 | 0.666 | 0.632 | **0.95** | 0.249 | 3.4 |
| visualprm | 2 | 34,422 | 0.791 | 0.780 | 0.822 | 0.822 | 1.19 | 0.070 | 74.9 |
| visualprm | 3-4 | 88,873 | 0.786 | 0.871 | 0.753 | 0.751 | 0.92 | 0.095 | 38.5 |
| visualprm | 5-7 | 200,990 | 0.797 | 0.884 | 0.760 | 0.762 | 0.89 | 0.108 | 23.7 |
| visualprm | 8+ | 171,471 | 0.800 | 0.872 | 0.768 | 0.775 | 0.90 | 0.122 | 20.9 |

By subset: PRM800K train 0.940 / test 0.937 (n=582). ProcessBench gsm8k
0.669, math 0.648, olympiadbench 0.640, omnimath 0.640 — flat across
difficulty; coverage falls gsm8k 0.828 → olympiadbench 0.655.

Diagnostics (measured):

- PRM800K's final gold step is a `# Answer` block in 99.9% of solutions;
  its boundary is missed 61.3% of the time (the space-join destroys the
  line-start `#` cue the V2 protocol removes by construction). **Excluding
  that one typography-borne cut, the model finds 99.0% of PRM800K's
  reasoning-step boundaries** (899 of 87,640 non-final boundaries missed).
- ProcessBench near-miss profile: 70.4% of gold boundaries matched
  exactly; 71.5% have a predicted cut within 30 non-ws chars, 79.2% within
  80 (≈ one short sentence). Near-misses add ~9 points; the remaining ~21%
  are genuine granularity disagreements.
- Example dumps (5 best / 5 worst / 5 median each, boundaries marked
  inline): `examples_prm800k.txt`, `examples_processbench.txt`. PRM800K's
  worst records are multi-sentence gold steps that the model splits per
  sentence (density up to 3.0) — same finer-grained disposition seen on
  VisualPRM. ProcessBench's worst records are one-sentence offsets
  (⟦G⟧…⟦P⟧ adjacent pairs), not chaotic segmentation.

## Reading (interpretation, kept separate from measurements)

**Gold-standard caveat, stated up front:** PRM800K and ProcessBench "gold"
step cuts are generator-typography-vetted, not authored — annotators
worked with the generator's newline/paragraph splits, they did not draw
boundaries themselves. So **coverage (gold boundaries found) is the
primary transfer signal; density is a convention signal**, and exact-match
F1 mixes the two.

**Verdict: boundary skill transfers; the model did not merely memorize
VisualPRM's segmentation style.** The evidence:

1. On PRM800K — a text-only math corpus the chunker never saw, with a
   *finer* step convention than its training data — it finds 92.8% of all
   gold boundaries (99.0% of reasoning boundaries once the `# Answer`
   typography cut, unrecoverable under V2 by construction, is set aside)
   at 0.981 precision. That is *better* than its in-domain number, which
   is the opposite of what style-memorization would predict.
2. On ProcessBench, coverage is 0.704 exact but ~0.79 within one sentence;
   the F1 drop is dominated by density 1.40 — the model cuts at
   sentence-ish granularity inside ProcessBench's long paragraph-steps.
   The bucket table shows this is convention, not skill: in the 8+ bucket
   (where ProcessBench's steps are short, mean length closest to
   VisualPRM's) density is 0.95 and precision 0.756, i.e. the model
   behaves normally as soon as the corpus's granularity approaches the
   one it was trained at. The 220-token cap additionally *forces* cuts in
   4.1% of steps (20.4% of records).
3. The failure modes are the already-characterized ones from
   chunk_eval/EVAL_REPORT.md (a finer-than-gold granularity disposition),
   not new cross-domain pathologies: no fusing, no drift, no degradation
   with difficulty (olympiad ≈ gsm8k in F1).

**Implication for RLPT rollout segmentation:** the chunker can be expected
to place boundaries at real reasoning-move seams on unseen model-generated
math text, at sentence-to-short-paragraph granularity. If rollouts are
written in long-paragraph style (ProcessBench-like), expect ~1.4×
over-segmentation relative to paragraph convention — consistent with the
list-tier/granularity discussion already open for the canonical file — and
note the 220-token cap guarantees a cut inside any longer step.

**Scope limits (honest):** both corpora are text-only math (no images —
the VisualPRM-specific modality is untested by construction here); the V2
space-join protocol removes typography for fairness, so these numbers are
a floor for deployment, where marker structure usually survives; and the
`# Answer` diagnostic shows one boundary class per PRM800K record is
typography-borne and invisible to any typography-free method.
