# VisualPRM400K-v1.1-Raw Two-Stage Trace Filtering — Observations & Tests

Date: 2026-07-16. Companion to `data/visualprm_v11_filtered/REPORT.md` (counts-focused);
this doc records the full design rationale, every test that ran, throughput/behavior
observations, and how to rerun or extend the pipeline.

Goal: build clean GT reference traces for (a) the bipartite-match RLVR reward and
(b) SFT baseline data, from the noisy VisualPRM400K-v1.1-Raw rollouts
(supervisor's SFT baseline imitates ~10.4% wrong-answer traces; MC step scores
exist but were unused).

---

## 1. Pipeline overview

```
raw (565,149 traces, 38 jsonl)                          data/visualprm_v11_raw/  [IMMUTABLE]
  │
  ├─ Stage 1  MC trace filter (CPU, streaming)          src/data/filter_visualprm_stage1.py
  │     keep iff answer-correct AND min(body MC) >= tau; taus 0.85 & 0.90
  │     + 15k seed=0 reservoir of the [0.5,0.85) "plausible victim" band
  │
  ├─ Stage 2  NLI contradiction screen (GPU, sbatch)    src/data/filter_visualprm_stage2_nli.py
  │     microsoft/deberta-xlarge-mnli; per step k: premise = question + steps 1..k-1,
  │     hypothesis = step k; trace FAILS iff max_k P(contradiction) > 0.5
  │     tau=0.90 outputs DERIVED exactly from tau=0.85 scores (no 2nd GPU pass)
  │
  ├─ Stage 3  MC × NLI agreement table + examples       src/data/make_agreement_table.py
  │
  ├─ Stage 4  dedup + emission                          src/data/emit_training_files.py
  │     group = (image content key, normalized question); best trace per group → SFT;
  │     one per unique question → RLVR
  │
  └─ Stage 5  invariant verification                    src/data/verify_filtered_output.py
```

Hard constraints honored throughout: raw dir never modified; traces kept/dropped
WHOLE (only the trailing "Final answer" boilerplate step is stripped from `steps`);
`min` over body scores (mean only as tie-breaker); off-the-shelf NLI only;
everything streamed (no full-corpus loads); seed=0 + sorted (source_file,
line_index) ordering → byte-reproducible; GPU work via Slurm `public` partition.

### Semantics lifted from `$RAW/check_step_scores.py` (must match to reproduce known numbers)
- answer-correct = `steps_with_score[-1]["score"] > 0.5` (strict >)
- empty `steps_with_score` counts in totals, never passes
- body = `sws[:-1]` if `len(sws) > 1` else `sws` (single-step traces keep their one step)
- keep iff `min(body scores) >= tau`

### Image path translation (validated 100% earlier; replicated torch-free in stage 4/5)
- `VisualPRM400K-v1.1-Raw/<x>` → `images/<x>`
- bare `images/train-*.png` (nlvr2 short style) → try as-is, then `images/nlvr2/<ref>`
- anything else → `images/<ref>`

---

## 2. Final numbers

| stage | tau=0.85 | tau=0.90 |
|---|---:|---:|
| raw rows | 565,149 | 565,149 |
| answer-correct | 503,885 | 503,885 |
| Stage 1 kept | 338,685 | 289,209 |
| Stage 2 kept | 319,274 (94.27%) | 272,786 (94.32%) |
| Stage 2 rejected | 19,411 (5.73%) | 16,423 (5.68%) |
| NLI pairs scored | 1,357,633 | (derived) 1,094,805 |
| multi-image dropped at emission | 23,124 | 22,308 |
| sft_train.jsonl | 100,194 | 88,555 |
| rlvr_prompts.jsonl | 73,704 | 65,034 |

Band: population 118,806 (answer-correct, min body MC in [0.5, 0.85)); sample
15,000; NLI on sample → 14,057 kept / 943 rejected (6.29%), 88,788 pairs.

Dedup ratio ~3:1 (296,150 single-image kept traces → 100,194 image×question
groups) — VisualPRM sampled multiple rollouts per question. 26,490 groups at
tau=0.85 share a question string with another image (RLVR keeps one per unique
question per spec).

---

## 3. Tests that ran (all passed)

### Stage 1
- **File count gate**: annotations.zip has 39 members but one is the bare
  `annotations/` directory entry → exactly 38 jsonl. Script exits 2 on any
  other count.
- **Known-number reproduction (hard gate)**: 565,149 total / 503,885
  answer-correct / 338,685 kept @ tau=0.85 — matched EXACTLY on first run.
  Instruction was to STOP on mismatch; never triggered.

### Stage 2
- **In-job smoke test** (both jobs): 50 band traces end-to-end before the full
  run. Both runs produced identical results — 37 kept / 13 rejected / 238
  pairs — confirming model + tokenizer + batching determinism across nodes.
- **Derive-mode synthetic test** (pre-launch): tiny handcrafted donor/input →
  success path (kept=1 rejected=1 pairs=4) and missing-trace failure path
  (rc 2, FATAL) both behaved.
- **Resume test (real, unplanned)**: job 59078004 TIMEOUT at 6 h with 322,693
  traces durably written (fsync every 10k). Resubmitted identical sbatch as
  59114290; `[resume] 322693 traces already scored, appending`; finished the
  remaining 15,992 traces + band + derive in 39:52. Final kept+rejected sums
  match Stage-1 counts exactly at both taus → no loss/duplication across the
  resume boundary.
- **Derive consistency**: 272,786 + 16,423 = 289,209 = stage1 tau=0.90 kept;
  pair count 1,094,805 consistent with the stricter subset.

### Stage 5 (invariants) — `verify_filtered_output.py`, 13/13 PASS
1. **Verbatim steps** for all 7 outputs (stage1 kept ×2, stage2 kept ×2, band
   sample, sft_train, rlvr_prompts) vs raw annotations via ONE streaming
   merge-join (exploits the global (source_file, line_index) write order):
   0 mismatched, 0 missing across 1,408,852 checked rows.
2. **No duplicate** (image-content-key, normalized question) in sft_train:
   100,194 rows, 0 dupes.
3. **All image refs resolve** in images.zip (sft + rlvr): 0 unresolved.
4. **Count consistency**: stage1 kept == stage2 kept+rejected at both taus;
   sft == group count ≤ stage2 kept; rlvr == unique-question count ≤ sft.
5. **Known numbers** exact.

Plus a targeted verifier for `emit_tau0.90/` (same checks 1–4 scoped to that
dir): 5/5 PASS (88,555 / 65,034 rows, 0 mismatches, 0 dupes, 0 unresolved).

---

## 4. Observations

### NLI rejection is dominated by counting/numeric sources
Top-5 by rejection rate (tau=0.85, full run):

| source | input | rejected | rate |
|---|---:|---:|---:|
| super_clevr_en_20240402_int | 6,078 | 1,791 | 29.5% |
| cocorem_exist_yorn_en_20241016 | 15,258 | 3,099 | 20.3% |
| geoqa+_en_..._open_ended_only | 1,005 | 188 | 18.7% |
| CLEVR_math_en_20240402_extracted | 14,039 | 2,618 | 18.7% |
| unigeo_calc_en_..._open_ended_only | 876 | 117 | 13.4% |

Lowest: mavis_function_sin 0.3%, mavis_function_tan 0.5%, MathV360K 0.6%,
koniq10k 0.8%. This matches the previously-benchmarked NLI
numeric-contradiction blindness (any digit difference → confident
contradiction, no magnitude sense): part of the NLI-fail cell is false alarms
on benign arithmetic restatements. Concrete example:
`agreement_examples/mcpass_nlifail_01_CLEVR_math_..._L1573.txt` — P(contra)
0.68 on "Subtracting 0 from 7 has no effect on the number of objects" after a
premise stating "There are 7 objects". **Implication: do NOT treat
`stage2_rejected_*` as a pure label-noise set, and don't tighten the 0.5
threshold without re-inspecting numeric sources.**

### MC and NLI screens are nearly orthogonal
| | NLI pass | NLI fail |
|---|---:|---:|
| MC pass (full 338,685) | 94.3% | 5.7% |
| MC fail (band sample 15,000) | 93.7% | 6.3% |

NLI barely distinguishes MC-pass from MC-band-fail traces (5.7% vs 6.3%
rejection). Reading: (a) the NLI screen removes a slice MC sampling cannot see
(textual contradictions in traces whose every step still reached a correct
answer in ≥85% of rollouts); (b) most band traces (mid MC score, 0.5–0.85)
contain no textual contradiction — consistent with MC harshness (correct step,
unlucky rollouts) rather than genuine step errors. Band caveat: sample covers
only the answer-correct [0.5, 0.85) band, not min<0.5 or wrong-answer traces.

### max-P(contradiction) distribution is heavily right-skewed
tau=0.85 histogram: 105,703 traces in [0, 0.05); monotone decay; only 594 in
[0.95, 1.00]. 5,395 traces sit in [0.45, 0.50) just under the threshold —
threshold sensitivity is mild (moving 0.5 → 0.45 would reject +5,395 ≈ +1.6pp).

### Throughput (A100-80GB, fp16, batch 128, xlarge model)
- Early sources: ~47 traces/s; slow tail (long premises, e.g. SROIE multi-turn,
  scienceqa): ~5–15 traces/s. Overall full pass ≈ 6.4 h — the naive early-rate
  ETA (~2 h) was wrong by 3×. Budget ~7 h for a rerun, or split inputs.
- Band sample is disproportionately slow (long traces): 15k traces ≈ 27 min.
- Derive step: 14 s (CPU-bound join).

### Premise token budget (512, question dropped first, then oldest steps)
- Full-set counters only cover the post-resume ~16k traces (processing counters
  aren't reconstructible from outputs — see caveat below): 2,478 pairs dropped
  the question, 2,231 dropped oldest steps, 77 hypotheses truncated.
- Band (complete, single run): 17,015/88,788 pairs (19.2%) dropped the
  question, 13,761 (15.5%) dropped oldest steps, 211 hypotheses truncated —
  band traces skew long. If premise fidelity matters for a future rerun,
  consider a 1024-token window or per-source inspection of SROIE/scienceqa.

### Multi-image (nlvr2)
23,124 stage-2-kept traces at tau=0.85 have list-typed `image` (nlvr2 pairs);
excluded from SFT/RLVR by `--drop_multi_image` (default). They pass all
upstream filters and remain in `stage2_kept_tau0.85.jsonl` — recoverable via
`--no-drop_multi_image` if the training stack later supports multi-image.

---

## 5. Caveats / anomalies (flagged, none data-corrupting)

1. **TIMEOUT + resume (jobs 59078004 → 59114290)**: outputs-as-checkpoint
   design validated in production; kept/rejected/hist/per-source stats are
   rebuilt from existing output rows on resume, and a crash-partial final line
   is byte-truncated. BUT pure processing counters
   (`premise_question_dropped`, `premise_oldest_steps_dropped`,
   `hypotheses_truncated`) restart from 0 — the tau=0.85 stats undercount them
   (~16k of 338,685 traces covered).
2. **Band row is a sample estimate** (15,000 of 118,806), labeled as such in
   `agreement_table.md`; no coverage of min<0.5 or wrong-answer MC-fail traces.
3. **NLI numeric false alarms** inflate the MC-pass/NLI-fail cell for counting
   sources (section 4).
4. **`per_source_rejection` in stage2 stats is complete and exact** (rebuilt
   from output rows) — safe to use, unlike the counters in (1).

---

## 6. File inventory (`/scratch/sghos104/rlpt/data/visualprm_v11_filtered/`)

| file | rows | what |
|---|---:|---|
| stage1_kept_tau0.85.jsonl | 338,685 | MC-pass traces (image, question_orig, answer, steps, min/mean_score, n_steps) |
| stage1_kept_tau0.90.jsonl | 289,209 | stricter variant |
| stage1_band_sample.jsonl | 15,000 | seed=0 reservoir of the [0.5,0.85) band (+step_scores) |
| stage1_stats_tau{0.85,0.90}.json | — | totals, per-source {total, kept}, band_population |
| stage2_kept_tau0.85.jsonl | 319,274 | + max_p_contra, p_contra_per_step |
| stage2_rejected_tau0.85.jsonl | 19,411 | NLI-flagged (same schema) |
| stage2_{kept,rejected}_tau0.90.jsonl | 272,786 / 16,423 | derived exactly from 0.85 scores |
| stage2_band_{kept,rejected}.jsonl | 14,057 / 943 | band sample NLI results |
| stage2_stats_*.json / stage2_band_stats.json | — | hist, per-source rejection, counters |
| agreement_table.md | — | 2×2 MC × NLI table |
| agreement_examples/ | 40 files | 20 mcpass_nlifail + 20 mcfail_nlipass, per-step MC \| contra |
| sft_train.jsonl | 100,194 | PRIMARY (tau=0.85): image, question_orig, steps, answer, source_file, line_index |
| rlvr_prompts.jsonl | 73,704 | PRIMARY: one row per unique question; `steps` = bipartite-match reference |
| emission_stats.json | — | group counts, per-source composition |
| emit_tau0.90/ | 88,555 / 65,034 | stricter emission variant + its emission_stats.json |
| REPORT.md | — | counts-focused final report |

GPU cost: one 6 h A100 job (timed out at ~95%) + one 40 min resume ≈ 6.7 A100-hours.

---

## 7. How to rerun / extend

- **Full rerun**: `sbatch src/scripts/run_stage2_nli.sbatch` after Stage 1
  (`filter_visualprm_stage1.py` twice, `--tau 0.85` and `--tau 0.90`). Stage 2
  resumes from existing outputs automatically — delete
  `stage2_*_tau0.85.jsonl` first for a from-scratch pass. Budget 7 h.
- **New (stricter) tau, no GPU**: run Stage 1 at the new tau, then stage 2 with
  `--derive_from_kept/--derive_from_rejected` pointing at the tau=0.85 outputs.
  Works for any tau ≥ 0.85 only (thresholds nest upward). Fails loudly (rc 2)
  if any input trace is missing from the donor or step counts mismatch.
- **Different NLI threshold**: no rescoring needed — `max_p_contra` and
  `p_contra_per_step` are stored on every row; re-split kept/rejected with a
  stream filter.
- **Re-verify after any change**: `python src/data/verify_filtered_output.py`
  (exit 0 iff all pass). For non-root emission dirs, adapt the targeted
  verifier `src/data/verify_emit_tau090.py` (verbatim merge-join + dedup +
  image resolution + counts).
- **Determinism check**: any rerun of stage 1 / stage 3 / stage 4 must be
  byte-identical (`cmp` old vs new); stage 2 smoke test (first 50 band traces)
  reproduced 37/13/238 across two different nodes.

Pending upstream decision: `--strategy per_source_topk` / `best_per_question`
in stage 1 are stubbed (NotImplementedError) awaiting supervisor's choice on
sampling strategy; `flat` (keep all passing) is what shipped.
