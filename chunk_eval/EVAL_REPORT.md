# chunk_eval — semantic-quality evaluation of canonical_chunked_v2.jsonl

Four experiments assessing whether the produced chunking (marker + model
cascade) yields step-like units comparable to the human/gold segmentation of
the 495,756 passthrough records. Read-only on all inputs; all code, stats
json, and examples in `/scratch/sghos104/rlpt/chunk_eval/`. Seeds = 0
everywhere. Gold = passthrough multi-step records of the SAME file (same
distribution of sources, same writing style) — the acceptable-baseline
reference for every experiment.

Status: ALL FOUR EXPERIMENTS DONE (2026-07-29), every job rc=0 —
Exp 1 (+1b/1c supplements), Exp 2 (fragment rule v2, scoring job
59861877), Exp 3, Exp 4 (full + stratified + list-tier ablation +
production relevance). One-page summary table at the bottom.

---

## EXP 1 — Completeness audit (ALL 77,998,006 chunks; job 59861315, rc=0)

**measured:** flag rates per bucket (self-tested flag logic; marker chunk
total 73,184,545 and gold step total 3,356,413 exactly match the cascade /
verifier totals — full coverage confirmed).

| bucket | chunks | no_terminal_punct | unbalanced_delims | dangling_start | dangling_end | ends_on_math |
|---|---:|---:|---:|---:|---:|---:|
| gold (baseline) | 3,356,413 | 10.48% | 0.88% | 2.50% | 10.73% | 4.08% |
| marker/step | 3,291,723 | 18.91% | 11.85% | 0.00% | 4.24% | 11.81% |
| marker/list | 64,904,438 | 17.24% | 0.90% | 0.05% | 26.64% | 5.70% |
| marker/para | 4,988,384 | 15.57% | 4.99% | 0.48% | 10.83% | 3.71% |
| model | 1,430,761 | 17.07% | 2.06% | 0.60% | 1.31% | 0.98% |
| model_skipped_nonprose | 26,287 | 17.35% | 11.05% | 0.05% | 0.44% | 20.53% |

Flag definitions (incl. the trailing-closer/emphasis strip and the
ends_on_math whitelist) are in the exp1_completeness.py docstring;
30 examples per (flag, bucket) in `exp1_examples/`. Full numbers incl.
comma-variant dangling starts: `exp1_stats.json`.

**measured (delimiter decomposition, exp1b job 59861343, full file):**
marker/step's 11.85% is NOT dollars or brackets — it is ``` fence parity:

| bucket | ANY | fence_parity | dollar_parity | paren | brace | latex `\[ \]` |
|---|---:|---:|---:|---:|---:|---:|
| gold | 0.88% | 0.01% | 0.21% | 0.33% | 0.04% | 0.22% |
| marker/step | 11.85% | **10.63%** | 0.58% | 0.37% | 0.07% | 0.27% |
| marker/list | 0.90% | 0.01% | 0.40% | 0.18% | 0.04% | 0.21% |
| marker/para | 4.99% | **3.13%** | 0.55% | 0.47% | 1.38% | 0.09% |
| model | 2.06% | 0.01% | 0.99% | 0.84% | 0.26% | 0.05% |

(dollar_parity is partly a detector artifact — currency "$2.47 trillion"
has odd $ counts; gold's own 0.21% bounds that noise. Non-fence delimiter
rates of produced tiers are all within ~2x of gold.)

**measured (exp1c job 59861353 — boundary-parity fence analysis, same
cumulative line-start-``` method as the v2 verifier, separating in-fence
boundaries from source-unbalanced fences):**

| bucket | records w/ fence | source-unbalanced | records w/ in-fence boundary | in-fence boundaries |
|---|---:|---:|---:|---:|
| marker/step | 122,261 | 2,085 | **118,584** | **453,847** |
| marker/para | 87,488 | 4,918 | **66,649** | **342,771** |
| marker/list | 61,532 | 4,563 | 2,547 | 13,435 |
| model | 6,661 | 151 | **0** | **0** |
| model_skipped_nonprose | 7,001 | 875 | 0 | 0 |
| gold | 1,051 | 93 | 105 | 223 |

**reading (⚠ NEW DEFECT — marker-path in-fence splits):** the v2 fence
guard covers ONLY the model path (per the v2 patch spec) and exp1c confirms
it holds there (0 in-fence boundaries — an independent re-verification).
But the MARKER path splits inside fenced code blocks at scale: 187,780
records / 810,053 boundaries (2.4% of marker records; gold baseline
~0.02%). Mechanism: STEP_RE accepts `#`-prefixed markers, so python
comments like `# Step 3: sort the result` inside a fence match the
step-tier splitter; PARA_RE likewise matches blank lines inside code.
97% of fence-containing step-tier records are affected. Whether this
warrants a v3 marker-path fence guard (mask marker matches inside
fence_regions) is a user/Eun-Woo decision; the fix would be the exact
marker-path analogue of the v2 model-path patch.

**reading:**
- No produced tier is catastrophically less "complete" than gold, but all
  exceed gold on missing terminal punctuation (15.6–18.9% vs 10.5%, i.e.
  1.5–1.8x) — consistent with chunks that end mid-clause or on markdown
  headings.
- marker/list's 26.6% dangling_end (vs gold 10.7%) is dominated by list
  lead-in items ending in ":" — an artifact of one-chunk-per-bullet
  granularity: the lead-in ("follow these steps:") becomes its own chunk.
- dangling_start is BELOW gold for all produced tiers (0.0–0.6% vs 2.5%):
  marker splits at explicit markers, which rarely begin with continuation
  tokens, while human gold steps often begin with "Then/Therefore".
- marker/step's unbalanced_delims (11.85% vs gold 0.88%) resolved by the
  exp1b/exp1c decomposition above: it is the marker-path in-fence-split
  defect, not bracket/dollar noise.

---

## EXP 3 — Weld detection via sympy anchors (job 59861316, rc=0)

**measured:** weld_suspect = chunk whose extracted claims contain >= 2
anchor assertions (single-symbol = ground-value) in >= 2 disjoint
components of the chunk's claim graph. Extraction = verbatim
`src/data/sympy_extract_audit.py` (v4 rules). 200K sampled chunks
(reservoirs seed 0, quota proportional to tier populations; produced =
marker+model records, gold = passthrough steps, same source tiers via
image-path source map — `source_map.py`, MathV360K subset list from raw
annos, 0 colliding subsets).

| tier | side | chunks | weld_suspect | rate | >=1 anchor | >=2 anchors |
|---|---|---:|---:|---:|---:|---:|
| function | gold | 36,150 | 3,394 | 9.39% | 50.7% | 28.1% |
| function | produced | 13,445 | 1,105 | 8.22% | 42.9% | 21.7% |
| geometry | gold | 51,999 | 4,623 | 8.89% | 34.3% | 15.9% |
| geometry | produced | 69,597 | 2,324 | 3.34% | 24.3% | 7.3% |
| arithmetic | gold | 11,850 | 96 | 0.81% | 3.5% | 1.5% |
| arithmetic | produced | 16,958 | 0 | 0.00% | 0.03% | 0.01% |

20 examples per (tier, side): `exp3_examples/`. Stats: `exp3_stats.json`.

**reading:**
- Produced chunks have a LOWER weld rate than gold steps in every tier
  (function 8.2% vs 9.4%; geometry 3.3% vs 8.9%; arithmetic 0.0% vs 0.8%).
  By this metric the cascade does not fuse reasoning moves beyond what the
  human segmentation itself does.
- Part of the gap is mechanical: produced chunks are finer than gold
  (Exp 1 granularity finding), so they carry fewer anchors per chunk
  (produced >=2-anchor share is roughly half of gold's in every tier) —
  fewer opportunities to look welded. The direction (produced <= gold) is
  still the relevant conclusion; the magnitude is not size-adjusted.
- Known extractor limitation (unit-tested, documented in exp3_weld.py):
  chains written through prose ("angle B = angle A") lose their linking
  claim to prose-trimming and look disjoint. Same extractor on both sides,
  so the gold baseline absorbs it.
- Arithmetic tier has almost no symbolic anchors (3.5% of gold steps) —
  its weld estimate is correspondingly weak evidence.

---

## EXP 2 — Corruption-calibrated boundary statistics [GATED]

Populations built (job 59861317, rc=0): gold 20,000 (quota 3-4: 3,853 /
5-7: 8,713 / 8+: 7,434, proportional to the 461,334-record census of
gold-with->=3-steps), merged + fragmented derived from that same sample
(mass conservation asserted per record), produced 20,000 with identical
bucket quotas (bucket = n_chunks). Fragmentation split 106,549 of 142,623
steps (74.7%); 36,074 were < 12 tokens (kept whole).

Fragment rule v2 (user-directed tightening after example review): cuts
are additionally rejected immediately after display-math closers
(`\]`, `]`, `\)`) and immediately before list/heading/bold markers.
FRAGMENTED was rebuilt from the SAME gold sample
(exp2_rebuild_fragmented.py, mass conservation asserted per record):
106,537 splits (was 106,549), 12 steps newly unsplittable, and the
residual "clean-looking cut" rate measured with the ORIGINAL census
criteria is **0.000% / 0.000%** (before-marker / after-closer; the v1
rule measured 2.4% / 2.5%) — under the < 0.5% confirmation threshold.
Stats: `exp2_fragment_v2_stats.json`.

### Exp 2 scoring (job 59861877, rc=0, 2h43m one A100)

Population sizes: gold 142,623 chunks / 122,623 boundaries; merged 76,190 /
56,190; fragmented 249,160 / 229,160; produced 170,492 / 150,492.
Single-sentence chunks (skipped for s2): 68,029 / 7,999 / 129,168 / 67,928.
NLI = deberta-xlarge-mnli, plain fp32 + TF32 (every reduced-precision mode
fails in this DeBERTa-v1 implementation — 3 failed jobs; see docstring).

**measured (mean, with p25/p50/p75 in exp2_stats.json; Cohen's d vs gold,
positive = gold higher; informative = |d| >= 0.2 vs either corruption):**

| statistic | gold | merged | fragmented | produced | d(merged) | d(frag) | d(produced) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| s1 cross-boundary SBERT cos | 0.527 | 0.590 | 0.450 | **0.528** | −0.30 | +0.38 | **−0.00** | informative; produced == gold |
| s2 within-chunk dispersion | 0.326 | 0.360 | 0.294 | 0.248 | −0.22 | +0.20 | +0.52 | weakly informative; see below |
| s3 P(entail) at boundary | 0.186 | 0.263 | 0.147 | 0.123 | −0.36 | +0.20 | +0.32 | informative; produced finer-side |
| s3 P(neutral) | 0.651 | 0.604 | 0.692 | 0.642 | +0.19 | −0.16 | +0.04 | UNINFORMATIVE (fails gold-vs-corruption) |
| s3 P(contradiction) | 0.163 | 0.134 | 0.161 | 0.235 | +0.16 | +0.01 | −0.29 | UNINFORMATIVE (fails gold-vs-corruption) |

**reading:**
- The primary boundary statistic (s1) delivers the clean result: the
  corruptions bracket gold in the expected directions (merging removes
  real boundaries -> the surviving cross-boundary pairs are longer, more
  similar, 0.590; fragmenting inserts mid-thought boundaries between
  short half-chunks, 0.450), and **produced boundaries are statistically
  indistinguishable from gold (0.528 vs 0.527, d = 0.00)** — near gold,
  far from both corruptions, exactly the pattern the experiment tests.
- s2 and s3-entail separate gold from the corruptions only weakly
  (|d| ~= 0.2, at the informativeness threshold), and on both, produced
  deviates from gold in the direction consistent with FINER chunks
  (shorter chunks -> fewer/more-homogeneous sentence pairs -> lower
  dispersion; shorter premises -> less material to entail the next chunk
  -> lower P(entail)). Given s1's null and Exp 4's direct boundary
  evidence, these deviations are attributed to chunk-size composition,
  not to misplaced boundaries; they are corroborating, not contradicting.
- s3 neutral/contradiction fail to separate gold from the corruptions and
  are reported as uninformative, used for no conclusions (consistent with
  the project's prior finding that NLI at paragraph granularity is mostly
  neutral).

---

## EXP 4 — Strip-and-resegment (FULL run, all 495,756 gold records; job
59861399, rc=0, 46 min wall / ~0.77 GPU-h)

**measured (overall):** boundary positions canonicalized to non-whitespace
prefix counts (whitespace-insensitive exact match); WindowDiff over
whitespace tokens, k = half mean gold segment length.

| variant | macro F1 | coverage (micro recall) | density ratio | WindowDiff | perfect records |
|---|---:|---:|---:|---:|---:|
| V1 full cascade on "\n\n" join | 0.569 | 0.618 | 1.32 | 0.333 | 24.1% |
| V2 model-only on space join | 0.796 | 0.769 | 0.92 | 0.108 | 28.9% |

V1 method mix: marker/list 336,256 (68%), marker/para 141,772, marker/step
17,728, model 0 — every joined record resolved at a marker tier.
Confusion: both perfect 84,969; V1-only (marker-carried) 34,753; V2-only
58,525; neither 317,509. (The earlier 4,958-record stride timing run gave
the same picture within ~0.005 on every metric.)

**measured (V1 stratified by internal marker typography in the gold
steps — internal = any \n\n inside a step, or a line-start list/step
marker at position > 0; production regexes verbatim):**

| subset | records | V1 macroF1 | V1 cov | V1 dens | V1 wdiff | V1 perfect | V2 control (F1/cov/wdiff) |
|---|---:|---:|---:|---:|---:|---:|---|
| clean (no internal markers) | 161,345 (32.5%) | **0.936** | 0.870 | 0.90 | 0.030 | 74.1% | 0.881 / 0.871 / 0.061 |
| structured (internal markers) | 334,411 (67.5%) | **0.392** | 0.514 | 1.52 | 0.479 | 0.0% | 0.755 / 0.727 / 0.131 |

V1 method mix — clean: para 119,226 / list 38,737 / step 3,382;
structured: list 297,519 / para 22,546 / step 14,346.

**measured (by n_steps bucket / by family, from exp4_stats_full.json):**
V1 wins only the 2-step bucket (0.918 vs 0.791); V2 wins 3-4/5-7/8+
(≈0.79–0.80 vs 0.52–0.57). The `<no-image>` family (88,440 records,
text-only) is near-perfect under both (V1 0.975, V2 0.979); every
image-bearing family scores V2 0.64–0.88 vs V1 0.40–0.52.

**reading:**
- The clean-subset V1 number is the fair cascade score: **where gold-step
  typography is unambiguous, the cascade reproduces human boundaries
  almost exactly (macroF1 0.936, WindowDiff 0.030, 74% perfect), and
  beats model-only (0.881)** — the marker tiers are not broken, and the
  para tier (74% of clean resolutions) is well aligned with human steps.
- The structured-subset gap (0.392, density 1.52, 0.0% perfect) IS the
  granularity disagreement, now quantified: for the 67.5% of gold records
  whose steps contain internal bullets/paragraphs, the list tier splits
  at every bullet inside a human step — 1.5x more boundaries than gold,
  and virtually never the exact human segmentation. On the same records
  the model alone scores 0.755.
- Caveats (encoded): V1's "\n\n" join partially feeds the answer to the
  para tier (V1 is structurally favored — which makes its structured-
  subset loss more notable); V2's space join removes newline evidence the
  model was trained with, so V2 likely understates deployed model
  accuracy. The n=2 bucket is also where the join contamination is
  strongest (one separator, para tier hits it trivially).
- Together with Exp 1 (bullet-lead-in dangling ends) and the chunk-length
  stats in chunk_canonical/REPORT.md, the consistent picture: the cascade
  is FINER than human steps on typographically structured responses, and
  faithful elsewhere. Whether per-bullet granularity is right for the PRM
  is a design decision, not an accuracy failure — the `tier` provenance
  tag supports a coarser re-merge without re-running.

### Exp 4b — list-tier ablation on the structured subset (job 59861607,
rc=0; same "\n\n" join, same guards/model fallback, list tier disabled)

**measured (all 334,411 structured records, per arm):**

| arm (structured subset) | macroF1 | coverage | density | WindowDiff | perfect | resolution mix |
|---|---:|---:|---:|---:|---:|---|
| V1 full cascade (list ON) — reference | 0.392 | 0.514 | 1.52 | 0.479 | 0.0% | list 297,519 / para 22,546 / step 14,346 |
| **A: step > para > model (list OFF)** | **0.924** | 0.971 | 1.15 | 0.075 | 71.4% | para 320,065 (95.7%) / step 14,346 |
| B: step > model (list+para OFF) | 0.754 | 0.723 | 0.89 | 0.129 | 15.4% | model 318,308 (95.2%) / step 14,346 / skips 1,757 |
| V2 model-only — reference | 0.755 | 0.727 | — | 0.131 | — | model 100% |

**reading:**
- Essentially ALL of the structured-subset damage is the list tier:
  disabling it alone moves macroF1 0.392 -> 0.924 (above the clean-subset
  V2 and close to clean-subset V1 0.936). Arm B == V2 within noise
  (0.754 vs 0.755) — with para also off, 95% of records reach the model,
  a consistency check that the harness is behaving.
- Join-inflation caveat CUTS BOTH WAYS here and must temper arm A: the
  gold boundaries in this test ARE "\n\n" positions by construction, so
  the para tier's recall is near-trivially high (0.971). Arm A's real
  content is the precision side: internal \n\n structure inside human
  steps adds only ~15% spurious boundaries (density 1.15), versus ~52%
  for list markers (1.52). Arm A is an upper bound on deployed step>para
  agreement, exactly as V1's structured 0.392 is a lower bound.
- Decision-relevant summary: a coarser production variant (or post-hoc
  re-merge of list-tier chunks to paragraph granularity) recovers most
  human-convention agreement; dropping to model-only instead buys 0.755
  and loses the marker tiers' clean-subset advantage (0.936 vs 0.881).

### Exp 4c — production relevance of a list-tier policy change (job
59861608, rc=0; measurement only, no file changes)

**measured:** marker/list records in the shipped v2: 6,370,908 (exact
match with cascade totals). On a 20K reservoir sample (seed 0) of them,
re-splitting the ORIGINAL canonical.jsonl blobs with step > para (list
off): 99.4% resolve at the para tier (0.6% would fall to the model
path); chunks/record 10.11 -> 7.50 (p50 8 -> 7, p90 19 -> 12); after-
chunk token length mean 39.7, p50 30, p90 81, p99 184, 0.56% > 220
tokens. Extrapolated to all list-tier records: ~16.6M fewer chunks
file-wide (78.0M -> ~61.4M); list-population mean chunk length rises
~32 -> ~40 tokens, i.e. closer to but still below the gold-step mean
(44.9).

**reading:** the coarsening is real but modest — many bullet lists
separate items with blank lines, so the para tier still splits per item
in those records. A step>para production variant lands between current
list-tier granularity and human-step granularity, with a negligible
model-path cost increase (0.6% of 6.37M ≈ 41K extra model records) and
essentially no >220-token risk (0.56%).

### Caveats for this whole section (encoded per instruction)
1. The "\n\n" join both inflates V1's structured-subset damage (list
   markers inside steps get line starts they may not have in single-blob
   production text) and inflates arm A's para agreement (gold boundaries
   are \n\n by construction) — all joined-text numbers are bounds, not
   deployed values. Exp 4c is the production-side measurement.
2. The structured share differs between populations: 67.5% of gold
   multi-step records vs an unknown (and different) share of the
   production single-blob population — extrapolate proportions, not
   absolute record counts, from Exp 4 to production.

---

## Summary table (one page)

| exp | metric | gold baseline | produced | verdict |
|---|---|---:|---:|---|
| 1 | no_terminal_punct rate | 10.5% | 15.6–18.9% by tier | produced 1.5–1.8x gold — mild, tier-dependent |
| 1 | dangling_start rate | 2.50% | 0.00–0.60% | produced BETTER than gold |
| 1 | dangling_end rate | 10.7% | 26.6% (list) / 1.3–10.8% (rest) | list-tier lead-in artifact; others <= gold |
| 1 | unbalanced delims (non-fence) | 0.87% | 0.9–2.1% | comparable to gold |
| 1c | in-fence split boundaries | 223 (0.02% of recs) | **810,053 (2.4% of marker recs)** | ⚠ DEFECT: marker path splits inside code fences (model path: 0); v3 guard decision pending |
| 2 | s1 cross-boundary SBERT cos | 0.527 (corruptions 0.590/0.450) | 0.528 (d=0.00) | **produced == gold; far from both corruptions** |
| 2 | s2 dispersion / s3 entail | 0.326 / 0.186 | 0.248 / 0.123 | deviations consistent with finer chunks; statistics only weakly informative |
| 2 | s3 neutral / contradiction | — | — | uninformative (fail gold-vs-corruption separation) |
| 3 | weld_suspect rate (function/geometry/arithmetic) | 9.4% / 8.9% / 0.8% | 8.2% / 3.3% / 0.0% | produced fuses LESS than gold in every tier |
| 4 | V1 cascade boundary macroF1 (clean subset, fair score) | 1.0 by defn | **0.936** (74% perfect, wdiff 0.030) | cascade ~reproduces human boundaries when typography is unambiguous |
| 4 | V1 macroF1 (structured subset) | 1.0 by defn | 0.392 (density 1.52, 0% perfect) | granularity disagreement: list tier splits every bullet |
| 4b | structured subset, list tier OFF | — | 0.924 | virtually all structured-subset damage is the list tier |
| 4 | V2 model-only macroF1 (uncontaminated) | 1.0 by defn | 0.796 | model alone is good but below clean-subset cascade (0.936) |
| 4c | step>para production variant | gold steps ~44.9 tok mean | chunks 10.1->7.5/rec, ~40 tok, 0.56% >220 | coarser option quantified: −16.6M chunks file-wide, 0.6% extra model load |

**Bottom line:** the produced chunking is structurally sound — boundary
placement is indistinguishable from human boundaries where they can be
compared (Exp 2 s1; Exp 4 clean subset), it does not fuse reasoning moves
(Exp 3), and its known weaknesses are precisely characterized: (1) the
list tier imposes per-bullet granularity finer than human step
conventions (Exp 4 structured subset; fixable by tier policy or post-hoc
re-merge, options quantified in 4b/4c), and (2) the marker path can split
inside code fences (Exp 1c; the v3-guard analogue of the shipped v2
model-path fix is the pending decision).
