# Granularity controllability: Qwen2.5-VL 3B vs 7B

**Question.** Can Qwen2.5-VL hold a target reasoning-step granularity when instructed,
and does 7B hold it better than 3B? This is about *controllability*, not intrinsic step size.

**Run.** 200 VisualProcessBench questions x 4 instruction conditions x 2 models =
1,600 generations. Two separate sbatch jobs (3B `61377406`, 7B `61377408`), both rc=0,
800 rows each, `empty=0`. Frozen question set `qwen_question_set.json`
(sha256 `480aa48a245542b2...`), read not re-sampled, so every model x condition hit
byte-identical inputs. temperature 0.7 / top_p 0.9 / max_tokens 1024 / seed 0.

**Reference band.** VisualPRM400K body steps: median 24 words/step (mean 30.1),
4.81 body steps per record. VisualProcessBench by generating policy: Claude 3.5 Sonnet
21.0 w/step, GPT-4o 23.8, InternVL 24.1, QvQ 99.1 (outlier; 1.10 backtracking
markers/step vs 0.00 for the others). Target ~21-24 words/step.

**Findings are ordered by robustness, not by prominence.** The convention-switching
result holds regardless of any analyst choice; the granularity numbers depend on a
segmentation rule and come last.

---

## 1. Qwen's step boundaries are not identifiable from whitespace

This is the most robust result in the run and it constrains everything below it.

Under a **blank-line** rule every condition sits *above* the 24-word target. Under a
**newline** rule every condition sits *below* it. Under **struct** most sit below and
one sits on it. These are the same 1,600 generations. Three rules, three verdicts.

The cause is that the model **changes its formatting convention in response to the
instruction**. `C_length` ("keep each step to about 25 words") makes both models abandon
paragraph breaks entirely for single-newline lines. The steps are plainly present and
plainly ~15-25 words, but a blank-line rule reads the whole response as one step and
scores it at 43.1 (3B) / 38.5 (7B) words — reporting the length instruction as pushing
granularity *away* from the target at the exact moment the model was complying.

### Convention switching, measured directly

Fraction of responses where blank-line and newline segment counts differ by more than 2x:

| model | A_unconstrained | B_count | C_length | D_format |
|---|---:|---:|---:|---:|
| 3B | 54.0% | 71.0% | 55.5% | 71.0% |
| 7B | 68.5% | 72.5% | 68.0% | 72.0% |

The two whitespace conventions disagree by more than 2x on **the majority of responses
in every condition, for both models**. Convention switching is pervasive, not a
`C_length` artefact — `C_length` is merely where the disagreement flips which rule is
wrong. Delta in switch rate from baseline is +17.0% for 3B under both `B_count` and
`D_format`, versus +4.0% / +3.5% for 7B: the instructions perturb 3B's formatting
convention several times more than 7B's.

**Consequence for the reward pipeline.** Any component that segments policy rollouts on
whitespace inherits a measurement that moves when the prompt changes. Step-level
scoring of Qwen rollouts cannot rely on blank lines alone, and a rollout format change
would silently redefine what a "step" is without any code change.

### The three rules

- **blank** — split on blank lines. Under-splits: packs a bulleted 8-item enumeration
  into one "step", and collapses entirely under `C_length`.
- **newline** — split on every newline. Over-splits: every line of a paragraph-internal
  list becomes its own "step", pinning all cells near 11 words.
- **struct** (headline) — a blank line **or** a line-initial opener
  (`1.` / `1)` / `-` / `*` / `•` / `Step 1:`) starts a step.

The `Step\s*\d+[:.)]` alternative is load-bearing: 3B writes `Step 1:` prefixes under
`C_length`, and without that pattern struct mis-scored 46.5% of that cell as zero-split.
Adding it drops 3B `C_length` zero-split **46.5% -> 20.0%** and 7B **21.5% -> 9.5%**.

The residual is genuine model degeneracy, not a further regex gap: of 3B's 40 remaining
zero-splits, **32 are true single-blob prose** and only 8 are struct misses (unmarked
one-sentence-per-line); for 7B **all 19 are true blobs** (median and max newline-lines
both 1).

---

## 2. Format compliance: the cleanest controllability separation

`D_format` marker emission carries no segmentation assumption — it asks only whether the
model emitted line-initial numbered markers when told to.

| condition | 3B | 7B |
|---|---:|---:|
| D_format ("Number each step: 1) 2) 3)") | **71.5%** | **98.5%** |
| B_count | 73.0% | 96.5% |
| A_unconstrained (no format asked) | 57.5% | 88.5% |

**7B complies with an explicit format instruction essentially always (98.5%); 3B fails
more than one time in four (71.5%).** The gap is not merely instruction-following: 7B
also numbers spontaneously far more often (88.5% vs 57.5% unconstrained), so part of the
separation is a baseline stylistic difference rather than differential obedience. The
*increment* from A to D is +14.0 points for 3B and +10.0 for 7B — similar responsiveness
from different baselines. The reliable statement is about the achieved rate, not the
increment.

---

## 3. Instruction following: B fails outright, C complies with the wrong behaviour

### B_count is an instruction-following failure

`B_count` asked for "about 6 steps". **Neither model moved from its baseline under any
rule.**

| rule | 3B: A -> B steps/response | 7B: A -> B steps/response |
|---|---|---|
| struct | 12.78 -> 13.27 (**+0.49, wrong direction**) | 14.79 -> 13.66 (-1.13) |
| blank | 7.97 -> 7.26 (-0.71) | 8.09 -> 7.34 (-0.75) |
| newline | 19.82 -> 21.59 (**+1.77, wrong direction**) | 23.23 -> 21.36 (-1.87) |

Under struct both models produce roughly 13 steps when asked for 6, and 3B moves *away*
from the request. Under the blank rule B superficially looks compliant at 7.26 / 7.34 —
but the *unconstrained baseline was already* 7.97 / 8.09, so the instruction moved
almost nothing there either; the blank rule merely flatters B by starting the baseline
near 6. Words/step barely shifts (+0.8 3B, +1.0 7B under struct). **A numeric step-count
target is not a control surface for either model.**

### C_length complies, but by changing a different quantity

`C_length` is the only condition that moves granularity materially — and it does so
almost entirely by cutting the **number** of steps rather than by tuning their **length**:

| model | steps/response A -> C | words/step A -> C |
|---|---|---|
| 3B | 12.78 -> **5.60** (-7.18) | 17.9 -> 24.7 (+6.8) |
| 7B | 14.79 -> **5.03** (-9.77) | 17.0 -> 14.3 (-2.7) |

Asked to constrain step *length*, both models cut the step *count* by more than half
(3B to 44% of baseline, 7B to 34%) and restructured the response into terse
newline-delimited lines. Step count was never mentioned in the C prompt. **This is a different behaviour from the one requested** —
the models are responding to a length instruction with a compression-and-reformat
strategy, not by regulating per-step length toward a target. It is also the condition
that produces all of the degeneracy in the run (3B 20.0%, 7B 9.5% zero-split; every
other cell is 0.0-0.5%).

So of the three instructions tested, one (D, format) is followed reliably by 7B and
unreliably by 3B, one (B, count) is followed by neither, and one (C, length) triggers a
substantial but off-target behavioural change in both.

---

## 4. Granularity numbers

Most rule-dependent section; read section 1 first. Headline rule is **struct**, with
blank and newline words/step (mean/p50) adjacent so the sensitivity stays visible.
Distribution stats pool over *segments*; steps/response and ratios are per *response*.
`d24` = struct mean minus 24.

| model | condition | **struct w/step** mean/p50 | IQR | CV | steps/resp | zero-split | **d24** | blank w/step | newline w/step | switch >2x | pack | bt/step | len% | numbered% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B | A_unconstrained | **17.9 / 13** | 17 | 0.84 | 12.78 | 0.5% | **-6.1** | 28.7 / 22 | 11.5 / 8 | 54.0% | 2.49 | 0.001 | 3.0% | 57.5% |
| 3B | B_count | **18.7 / 14** | 17 | 0.81 | 13.27 | 0.0% | **-5.3** | 34.2 / 29 | 11.5 / 8 | 71.0% | 2.97 | 0.001 | 1.5% | 73.0% |
| 3B | C_length | **24.7 / 18** | 17 | 1.63 | 5.60 | 20.0% | **+0.7** | 43.1 / 26 | 17.0 / 11 | 55.5% | 2.54 | 0.000 | 2.0% | 11.5% |
| 3B | D_format | **18.9 / 14** | 17 | 1.00 | 13.18 | 0.0% | **-5.1** | 33.6 / 28 | 11.3 / 8 | 71.0% | 2.98 | 0.019 | 4.0% | 71.5% |
| 7B | A_unconstrained | **17.0 / 13** | 16 | 0.79 | 14.79 | 0.0% | **-7.0** | 31.1 / 26 | 10.8 / 8 | 68.5% | 2.87 | 0.000 | 0.5% | 88.5% |
| 7B | B_count | **18.0 / 14** | 16 | 0.78 | 13.66 | 0.0% | **-6.0** | 33.5 / 28 | 11.5 / 8 | 72.5% | 2.91 | 0.001 | 0.5% | 96.5% |
| 7B | C_length | **14.3 / 12** | 7 | 0.70 | 5.03 | 9.5% | **-9.7** | 38.5 / 36 | 13.4 / 12 | 68.0% | 2.88 | 0.000 | 0.5% | 43.5% |
| 7B | D_format | **18.2 / 14** | 16 | 0.80 | 12.12 | 0.0% | **-5.8** | 37.4 / 31 | 11.9 / 8 | 72.0% | 3.15 | 0.002 | 1.5% | 98.5% |

### Deltas vs A_unconstrained

The delta, not the absolute, isolates controllability. Per-rule deltas expose sign flips.
`d|dist24|` is negative when the instruction moved the model *toward* the target.

| model | delta | dw/step blank | dw/step newline | **dw/step struct** | dsteps | **d\|dist24\|** | dpack | dswitch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3B | A->B_count | +5.5 | -0.0 | **+0.8** | +0.49 | **-0.8** | +0.48 | +17.0% |
| 3B | A->C_length | +14.4 | +5.5 | **+6.8** | -7.18 | **-5.4** | +0.05 | +1.5% |
| 3B | A->D_format | +5.0 | -0.3 | **+1.0** | +0.40 | **-1.0** | +0.50 | +17.0% |
| 7B | A->B_count | +2.4 | +0.7 | **+1.0** | -1.13 | **-1.0** | +0.03 | +4.0% |
| 7B | A->C_length | +7.4 | +2.5 | **-2.7** | -9.77 | **+2.7** | +0.01 | -0.5% |
| 7B | A->D_format | +6.3 | +1.1 | **+1.2** | -2.67 | **-1.2** | +0.27 | +3.5% |

**One sign flip: 7B A->C_length is +2.5 under newline and -2.7 under struct.** That
cell's direction is rule-dependent and must not be quoted without the rule. Every other
delta agrees in sign between struct and newline; blank disagrees in magnitude everywhere
and in sign on two 3B cells, which is the collapse of section 1.

### Degeneracy-inflation correction, applied uniformly

A zero-split response collapses to one long segment, which pulls the segment-pooled mean
up. The correction is applied to **every** cell, not only the one where it bites, so the
reader can see where it does and does not matter. `n excl` is the number of responses
removed out of 200.

| model | condition | struct w/step **incl.** degenerate | struct w/step **excl.** degenerate | delta | n excl | d24 (excl.) |
|---|---|---:|---:|---:|---:|---:|
| 3B | A_unconstrained | 17.9 | 17.9 | -0.0 | 1 | -6.1 |
| 3B | B_count | 18.7 | 18.7 | +0.0 | 0 | -5.3 |
| 3B | C_length | 24.7 | **21.4** | **-3.3** | **40** | **-2.6** |
| 3B | D_format | 18.9 | 18.9 | +0.0 | 0 | -5.1 |
| 7B | A_unconstrained | 17.0 | 17.0 | +0.0 | 0 | -7.0 |
| 7B | B_count | 18.0 | 18.0 | +0.0 | 0 | -6.0 |
| 7B | C_length | 14.3 | **13.7** | **-0.7** | **19** | **-10.3** |
| 7B | D_format | 18.2 | 18.2 | +0.0 | 0 | -5.8 |

**The correction is confined to `C_length`.** Six of the eight cells move by 0.0 words
because they contain at most one degenerate response; only C has enough to shift a mean.
3B moves by -3.3 and 7B by -0.7, so the correction also *narrows* the 3B-vs-7B gap in
that cell from 10.4 words to 7.7.

3B still lands inside the 21-24 reference band, so the qualitative claim survives — but
"dead on target at 24.7" does not. 7B remains far below the band either way.

---

## 5. Per-source breakout (domain effect vs model effect)

The 200 questions are stratified across five VPB sources. Reported under the headline
struct rule; cell = mean words/step / zero-split%.

| model | condition | DynaMath (40) | MMMU_DEV_VAL (18) | MathVerse_MINI (72) | MathVision_MINI (49) | WeMath (21) |
|---|---|---|---|---|---|---|
| 3B | A_unconstrained | 15.6 / 0% | 20.5 / 0% | 17.0 / 0% | 20.7 / 2% | 18.0 / 0% |
| 3B | B_count | 19.6 / 0% | 17.0 / 0% | 18.1 / 0% | 22.2 / 0% | 16.3 / 0% |
| 3B | C_length | 21.4 / 15% | 22.7 / 6% | **27.7 / 25%** | **26.4 / 31%** | 21.5 / 0% |
| 3B | D_format | 18.7 / 0% | 19.8 / 0% | 19.1 / 0% | 19.0 / 0% | 16.7 / 0% |
| 7B | A_unconstrained | 17.4 / 0% | 17.4 / 0% | 16.5 / 0% | 17.1 / 0% | 17.0 / 0% |
| 7B | B_count | 20.2 / 0% | 19.5 / 0% | 17.1 / 0% | 17.6 / 0% | 16.7 / 0% |
| 7B | C_length | 13.8 / 0% | 14.2 / 0% | 14.1 / 14% | 14.7 / 10% | 15.5 / 19% |
| 7B | D_format | 17.1 / 0% | 20.5 / 0% | 18.3 / 0% | 18.6 / 0% | 16.6 / 0% |

Two things this separates out:

**3B's on-target `C_length` result is source-driven.** The two sources that pull it up
(MathVerse 27.7, MathVision 26.4) are precisely the two with the highest zero-split rates
(25%, 31%). On DynaMath, MMMU and WeMath — where 3B barely degenerates — it sits at
21.4-22.7. The apparent accuracy advantage is concentrated in the cells where the
measurement is least trustworthy, which is the same confound as the VisualPRM bands.
7B's `C_length` is flat by comparison: 13.8-15.5 across all five sources.

**7B is markedly more source-invariant.** Unconstrained words/step across the five
sources spans **0.9 words** for 7B (16.5-17.4) versus **5.2** for 3B (15.6-20.7). 3B's
step size depends substantially on which benchmark the question came from; 7B's does not.
MMMU (n=18) is the smallest stratum and its cells should be read with that in mind.

---

## 6. What else is solid, independent of the split rule

- **Both models are non-backtracking.** Max 0.019 markers/step (3B `D_format`);
  0.000-0.002 everywhere else. Firmly in the Claude 3.5 Sonnet / GPT-4o / InternVL
  family, nowhere near QvQ's 1.10. The chunker can treat Qwen rollouts as
  non-backtracking prose.
- **Truncation is not a confound.** `length` finishes peak at 4.0% (3B `D_format`);
  7B never exceeds 1.5%. Too small to drive any cell.
- **7B does not degenerate outside C; 3B does.** Zero-split under struct: 7B
  0.0 / 0.0 / 9.5 / 0.0% across A/B/C/D; 3B 0.5 / 0.0 / 20.0 / 0.0%. Under the blank
  rule the gap widens (3B 3.5 / 5.0 / 47.5 / 28.5% vs 7B 0.0 / 0.0 / 38.0 / 1.5%).
- **List packing moves with condition on 3B, not on 7B.** Pack ratio A->B +0.48 and
  A->D +0.50 on 3B, versus +0.03 and +0.27 on 7B. On 3B the count and format
  instructions are partly changing *formatting style* rather than step length.

---

## 7. Reading

Separated from the measurements above.

**The headline answer is that 7B is the more controllable model, but not because it
tracks a numeric target better.** It follows an explicit format instruction 98.5% of the
time against 3B's 71.5%; it never degenerates outside `C_length`; its step size is nearly
invariant to question source (0.9-word spread vs 5.2); its dispersion under the length
instruction is far tighter (CV 0.70 / IQR 7 vs 1.63 / 17, the widest in the run); and its
formatting convention is perturbed several times less by the instructions
(dswitch +3.5/+4.0% vs +17.0%). Every one of those is a *predictability* property.

**On numeric-target accuracy the ranking inverts, and that result is weak.** 3B's
`C_length` mean of 24.7 sits on the reference band and 7B's 14.3 undershoots it by ten
words — but 3B's figure falls to 21.4 once its 40 degenerate responses are removed, and
the sources driving it are the sources where it degenerates most. The honest statement is
that 3B lands inside the 21-24 band and 7B lands below it, with the margin substantially
smaller than the raw table implies.

**Neither model has a usable numeric control surface.** A step-count instruction (B) does
nothing to either. A step-length instruction (C) produces a large response — but the
quantity that actually moves is step count, cut by half to two thirds, together with a
wholesale formatting switch. If granularity needs to be *set* rather than *nudged*, these
two prompts are not the mechanism, and format instructions (D) are the only lever here
with a reliable effect on 7B.

---

## 8. Limitations

**(a) Every absolute words/step figure is rule-conditional.** Whitespace does not
identify Qwen's step boundaries (section 1). The same generations yield "above target"
under a blank-line rule and "below target" under a newline rule. `struct` is a defensible
choice, not a correct one, and no absolute words/step number here should be quoted
without naming the rule that produced it. Deltas are safer than absolutes, and even one
delta (7B A->C_length) flips sign between rules.

**(b) The 21-24 target band is a weak criterion.** It comes from VisualPRM400K plus three
VisualProcessBench policies (Claude 3.5 Sonnet, GPT-4o, InternVL) whose formatting
conventions differ from Qwen's — the band was measured on text segmented under those
conventions, not under `struct`. Distance-to-target therefore mixes a real granularity
difference with a segmentation-convention mismatch, and cannot separate them. Treat
`d24` as orientation, not as a score.

**(c) MMMU_DEV_VAL is too thin for per-source claims.** n=18 questions per cell, against
72 for MathVerse and 49 for MathVision. Its per-source cells are reported for
completeness and should not carry an argument on their own. The other four strata
(21-72) are usable but none is large.

**(d) Resolution is roughly +/-1.5 words, not sub-word.** Segments within a response are
correlated, so the honest unit is the response (n=200 per cell), not the segment
(~2,500). Response-level clustered SE on the mean is 0.59-0.89 words for seven of the
eight cells, i.e. a 95% interval of about +/-1.2 to +/-1.8. **Differences below ~1.5
words should not be read as real** — that includes most of the A/B/D spread between the
two models (17.0-18.9 across six cells) and the +0.8 to +1.2 struct deltas for A->B and
A->D. The exception is 3B `C_length`: including degenerate responses its clustered SE is
6.16 (95% interval +/-12.1), so that cell's mean is essentially unresolved as reported;
excluding them the SE falls to 0.86. This is a further reason the 3B on-target result
does not carry weight.

**Caveat on the target itself.** Given (a) and (b), absolute distance to 24 is the
weakest number in this report. The load-bearing numbers are the convention-switch rates,
the marker-emission rates, the zero-split rates, the CV, and the per-source spread — all
of which are either rule-independent or comparisons within a single rule.

---

## Files

| path | contents |
|---|---|
| `qwen_question_set.json` | frozen 200-question set, sha256 `480aa48a245542b2...` |
| `vpb_multiimage_defect.json` | 25 VPB records referencing more `<imageN>` tags than images supplied |
| `qwen_generate.py`, `run_qwen_gen.sbatch` | generation |
| `qwen_gen_3b.jsonl`, `qwen_gen_7b.jsonl` | 800 rows each |
| `measure_granularity.py` | three-rule measurement + per-source breakout, self-tested |
| `qwen_granularity_stats.json` | all cells x all rules, per-source cells, deltas |
| `qwen_granularity_stats_perrow.jsonl` | 1,600 per-response rows (see below) |
| `logs/slurm-qgen-61377406.out`, `-61377408.out` | job logs |

**`qwen_granularity_stats_perrow.jsonl` is complete for re-aggregation without
regenerating.** 1,600 rows, one per (question x condition x model). Fields:
`qid`, `data_source`, `condition`, `model`, `tag`, `finish_reason`, `n_gen_tokens`,
`n_words_total`, all three segment counts (`n_blank_segments`, `n_newline_segments`,
`n_struct_segments`), `rules.{blank,newline,struct}.{n_steps,words_per_step,zero_split}`
carrying the full per-segment word vectors, `n_list_items`, `n_backtrack`,
`has_numbered_markers`, `convention_switch`, `blank_newline_ratio`. A self-test in
`measure_granularity.py` asserts the presence of the re-aggregation fields.
