# Reward-encoder reliability on DCI — full report

*RLPT image-description reward track · analysis run 2026-06-17 · ASU Sol.*

---

## 1. Motivation

Our GRPO composite reward is

```
R = w_f·format + w_m·match + w_a·answer − w_p·pun        (w = 1 / 2 / 5 / 1)
```

The **`match`** term is a soft bipartite (Hungarian) alignment between the model's reasoning
clauses and the GT solution clauses, where the per-pair cost is **SBERT cosine similarity**
using `sentence-transformers/all-MiniLM-L6-v2`. Our validation campaign already showed `match`
is *causal* for reasoning faithfulness and OOD accuracy (ablation + MMK12, McNemar p = 0.021).

But causal-for-the-metric is not the same as *trustworthy*. A separate, training-independent
question remains: **is the encoder underneath `match` reliable** — does cosine score a faithful
description *above* a minimally-corrupted one? If not, the reward can be raised by stylistic
mimicry rather than correctness, and the bipartite machinery on top inherits that blindness.

This report answers that question on an external, *labeled* testbed, then probes the mechanism
with embedding inversion, and translates the result into a concrete reward-design proposal.

---

## 2. Testbed — DCI (Densely Captioned Images)

`facebookresearch/DCI` (CC-BY-NC). Why it is the right testbed: it ships, **per caption unit**,
both *correct restatements* and *aligned hard negatives*, which is exactly the (faithful,
corrupted) contrast a reward encoder must separate.

Structural facts (`data/DCI/dci_report.md`, from `analyze_dci.py`):

| property | value |
|---|---|
| images (complete JSON) | 7,805 (train 7,599 / val 98 / test 108) |
| total submasks | 313,391 (mean 40.2/image, ok-quality 270,219) |
| images with LLaMA2 summaries | 7,805 (100%) |
| images with generated negatives | 7,805 (100%) |
| summaries / image (flattened) | mean 91.5 (median 86) |
| hard negatives generated | `basic` 135,024 · `layout` 135,024 · `swaps` 135,024 |
| matchable GT units / image | mean 36.6 (short + extra + ok-mask captions) |
| total matchable GT units | 285,829 |

**Negative families** (the three prompt styles DCI uses to corrupt a caption):

| family | corruption | lexical profile |
|---|---|---|
| `swaps` | an attribute/entity is swapped between two objects | **near-verbatim** — minimal edit, almost all words shared with the anchor |
| `layout` | spatial arrangement altered | full reword |
| `basic` | other content change | full reword |

A reliable reward encoder must rank **paraphrase > every negative**. The decisive design
contrast: paraphrases *reword* (low lexical overlap with the anchor, same meaning); `swaps`
*keep the words* (high lexical overlap, wrong meaning). Cosine of bag-of-subword embeddings
rewards overlap — so `swaps` are the adversarial case **by construction**, and they are the
single most diagnostic family.

### 2.1 Data acquisition (provenance)

- Annotations: `dci.tar.gz` (831 MB, public, sha256-verified), extracted to
  `data/densely_captioned_images/{annotations,complete,photos}`.
- Images: SA-1B is **license-gated**. DCI draws all 7,805 images from a single shard,
  `sa_000138.tar` (IDs 1543972–1554261, ~11 GB), obtained via the user's signed Meta links-file
  URL (time-limited `scontent…fbcdn.net`; the links file pairs each `sa_0000NN.tar` with its own
  opaque URL — we matched the `sa_000138.tar` line specifically). `match_sa1b.py` streamed the
  tar and extracted exactly the referenced images → **7,805 / 7,805 matched, 0 missing.**
- Viewer: `make_dci_viewer.py` → `dci_viewer.html` (self-contained: photo + caption tree with
  hover-bbox + summaries/negatives).

---

## 3. Reliability tests (Tests 1 / 2 / 4)

Script: `data/DCI/reliability_test.py`. Sampling is seeded (`seed=42`, 1,500 units for the
bi-encoders, 300 for NLI — the cross-encoder runs ~P×G pairwise and is slower). Per unit:

- **anchor** = `summaries[k][0]`
- **positives** (paraphrases) = `summaries[k][1:4]` (≤ 3) — *should score high*
- **negatives** = `negatives[k][{swaps,layout,basic}][:3]` (≤ 3 each) — *should score low*

Only units with ≥ 2 summaries and all three negative families present are kept.

**Encoders.**

| label | model | paradigm | scoring |
|---|---|---|---|
| **MiniLM-L6 (reward)** | `all-MiniLM-L6-v2` | SBERT bi-encoder | cosine of L2-normalized embeddings |
| mpnet-base (alt) | `all-mpnet-base-v2` | SBERT bi-encoder | cosine — *second, independent SBERT* |
| NLI-DeBERTa-v3 | `cross-encoder/nli-deberta-v3-base` | cross-encoder | P(entailment), anchor → candidate (softmax, entail-label auto-detected) |

**Metric definitions** (exactly as computed):

- **Test 1 — pairwise accuracy** = over all (positive, negative-of-type-`t`) pairs,
  fraction with `score(pos) > score(neg)`. **AUROC** = ROC area treating positives (label 1)
  vs that type's negatives (label 0). *0.50 = chance; < 0.50 = worse than chance.*
- **Test 4 — mean scores** by class + **FNR** = fraction of paraphrases scoring *below the
  median negative* of that type (the rate at which the reward would reject a correct restatement).
- **Test 2 — agreement** vs the reward encoder: **Cohen's κ** on the binary `pos > neg`
  decisions and **Spearman ρ** of the decision margins `(score(pos) − score(neg))`, computed on
  the *shared* `swaps` comparisons over the common 300-unit subsample (apples-to-apples).

### Test 1 — discrimination P(paraphrase > negative)

| encoder | swaps acc | swaps AUROC | layout acc | layout AUROC | basic acc | basic AUROC |
|---|---|---|---|---|---|---|
| **MiniLM-L6 (reward)** | **0.183** | 0.192 | 0.519 | 0.525 | 0.573 | 0.578 |
| mpnet-base (alt) | 0.272 | 0.287 | 0.553 | 0.562 | 0.581 | 0.587 |
| NLI-DeBERTa-v3 (entail) | 0.381 | 0.399 | **0.920** | 0.914 | **0.830** | 0.832 |

### Test 4 — paraphrase robustness (mean score by class)

| encoder | mean paraphrase | mean swaps | mean layout | mean basic | FNR vs swaps |
|---|---|---|---|---|---|
| **MiniLM-L6 (reward)** | 0.758 | **0.881** | 0.742 | 0.725 | 0.895 |
| mpnet-base (alt) | 0.795 | 0.873 | 0.762 | 0.761 | 0.783 |
| NLI-DeBERTa-v3 (entail) | 0.426 | 0.637 | 0.031 | 0.095 | 0.684 |

### Test 2 — encoder agreement (vs MiniLM reward encoder, swaps subsample)

| encoder pair | Cohen κ | Spearman(margin) |
|---|---|---|
| MiniLM-L6 (reward) vs mpnet-base (alt) | 0.617 | 0.865 |
| MiniLM-L6 (reward) vs NLI-DeBERTa-v3 (entail) | 0.141 | 0.303 |

---

## 4. Mechanistic probe — embedding-inversion roundtrip (option 1)

Tests 1/4 show the reward encoder *can't score* the corruptions apart. Two explanations are
possible: either the embedding **discards** the swap-relevant content (an information
bottleneck), or it **retains** it but cosine geometry can't surface it (a scoring problem). We
distinguish them by inversion: encode → **decode the vector back to text** (vec2text) → measure
what survived.

Script: `data/DCI/invert_roundtrip.py`, 20 refinement steps, beam 4, batched inversion
(`--invert-bs 16`). **Caveat:** vec2text ships a public inverter for **GTR-base** (768-d), not
for MiniLM (no public MiniLM inverter), so this characterizes a *same-family* bi-encoder
stand-in, not MiniLM literally. The GTR corrector is trained for ~32-token inputs → we probe
*short* units (`short_caption` + ok-mask captions ≤ 28 w) as the in-distribution case and report
long `extra_caption`s separately (expected to degrade, OOD for the corrector).

Scores: **token-F1** (content-lemma overlap), **MiniLM cosine** (semantic preservation),
**ADJ / NUM / spatial recall** (did the swap/layout-relevant tokens survive?).

| unit set | n | token-F1 | MiniLM cos | ADJ recall | NUM recall | spatial recall |
|---|---|---|---|---|---|---|
| short (≤ 28 w) | 400 | **0.984** | 0.991 | **0.983** | **0.972** | **0.980** |
| extra_caption (long) | 100 | 0.465 | 0.776 | 0.284 | 0.291 | 0.372 |

Sample reconstructions (short; `data/DCI/invert_report.md`):

```
orig : This is a wall on the far left side of the hillside. It's part of a revetment structure, made of tan brick.
recon: This is a wall on the far left side of the hillside. It's part of a revetment structure, made of of

orig : A rail separates the train from the trees and water below.
recon:          A rail separates the train from the trees and water below.
```

(Near-verbatim, with characteristic trailing-token repetition — a decoding artifact, not lost content.)

---

## 5. Findings

1. **SBERT is consistent but structurally blind.** The reward encoder agrees strongly with a
   *second, independent* SBERT (κ = 0.62, ρ = 0.87 with mpnet) — so its behaviour is a property
   of the **bi-encoder-cosine paradigm**, not a one-model artifact. Yet on real content
   corruptions it is ~chance (`layout` 0.52, `basic` 0.57) and on `swaps` it is **worse than
   chance (0.18)**: it scores the swap (mean 0.881) *above* the faithful paraphrase (0.758).
   The FNR-vs-swaps of 0.895 means it would reject ~90% of correct restatements relative to the
   swap. Cause: swaps are lexical near-copies of the anchor; paraphrases reword; cosine rewards
   overlap.

2. **NLI is far stronger on content, with its own blind spot.** The cross-encoder catches
   contradictions cosine cannot — `layout` 0.92, `basic` 0.83 discrimination — because it reasons
   about entailment rather than surface overlap. But it is **also fooled by near-verbatim
   `swaps` (0.38, below chance)** and *over-penalizes* detail-adding paraphrases (mean entail
   0.43; FNR 0.68). Its κ = 0.14 with SBERT confirms a genuinely different paradigm, not a
   re-skin. NLI and SBERT fail on **complementary** axes.

3. **The encoder retains the content; cosine can't score it.** Inversion is the key mechanistic
   result: short embeddings are **near-losslessly invertible** — ADJ 0.98, NUM 0.97, spatial
   0.98 recall, token-F1 0.984. The swap-relevant information is *present in the vector*. So the
   Tests-1/4 failures are a **scoring/geometry** problem (cosine collapses the
   paraphrase-vs-swap distinction), **not** an information-bottleneck. This is good news: it says
   a better *read-out* (e.g. entailment, or a learned head) can recover what pooling preserved.

4. **Long-caption detail loss is real but secondary.** On long `extra_caption`s the GTR
   embedding genuinely sheds fine detail (F1 0.47; ADJ/NUM/spatial recall 0.28–0.37) — partly
   the corrector going OOD (32-tok training), partly that mean-pooled bi-encoders blur attributes
   once text is long. Our reward operates on **clause-split** chunks (near-atomic), so it lives
   mostly in the favorable short regime — but it argues for keeping clauses short.

5. **Honest confound.** `swaps` are minimal edits, so they are lexically inflated for *every*
   text method — both SBERT and NLI fail on them, and that partly reflects the construction, not
   only encoder weakness. The **clean** signal is `layout`/`basic` (full rewordings): SBERT is
   still only 0.52–0.57 there, NLI a strong 0.92/0.83. Qualitative case (`show_examples.py`):
   "silver-trimmed" reattached from window→instruments — SBERT scores the wrong **swap 0.92** vs
   the correct **reword 0.71**.

**One-line takeaway.** The reward encoder *preserves* the swap-relevant content but *cannot
score it apart* with cosine; NLI scores content well but is blind to near-verbatim swaps;
attribute-swaps resist both text methods and are fundamentally a pixel question.

---

## 6. Reward-design implications (proposed, not yet built)

1. **Hybrid text reward.** Keep SBERT cosine for *coverage/recall*, add NLI entailment for
   *faithfulness* on content changes. Bound the cross-encoder cost with an **SBERT prefilter**:
   run entailment only on the clauses SBERT already matched (one pass per matched clause, not the
   full P × G matrix). The two encoders' failures are complementary, so the union is strictly
   safer than either alone.
2. **Multimodal grounding term.** Attribute-swaps resist *both* text methods — *which* object
   carries an attribute is a pixel question. This is the strongest argument yet for the planned
   box-IoU / region-CLIP grounding cost in the bipartite match.
3. **Keep clauses short.** The inversion result shows fidelity is excellent for short units and
   degrades for long ones; the clause splitter should favor near-atomic chunks.

**Validation order before any training run:**
(1) add an NLI-matcher variant to `graph_match_reward.py`, re-run DCI Tests 1/4;
(2) re-score the existing validation generations under the hybrid (no new generation);
(3) one hybrid-reward GRPO run vs the current arm on MMK12 (the OOD benchmark where the match
term already showed it converts to accuracy).

---

## 7. Reproducibility

All under `data/DCI/` (env `rlpt-data` for the CPU tests, `rlpt-train` for GPU inversion).

| script | produces | notes |
|---|---|---|
| `analyze_dci.py` | `dci_report.md` | structural stats |
| `match_sa1b.py --tar sa_000138.tar` | `photos/` (7,805 imgs) | streams the license-gated shard, basename-match extract |
| `make_dci_viewer.py` | `dci_viewer.html` | self-contained viewer |
| `reliability_test.py` | `reliability_report.md` | Tests 1/2/4; `--bi-units 1500 --nli-units 300 --paras 3 --negs 3` |
| `show_examples.py` | stdout | concrete anchor/paraphrase/swap cases with cos + NLI |
| `invert_roundtrip.py` (+ `.sbatch`) | `invert_report.md` | GPU; `--n 400 --steps 20 --beam 4 --invert-bs 16`; htc 1×A100, `--no-requeue` |

Env notes: bi-encoder tests run CPU offline (`HF_HUB_OFFLINE=1`); inversion needs GPU and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + chunked `invert_embeddings` (un-batched
beam search OOMs an 80 GB A100). vec2text installed `--no-deps` into `rlpt-train` to protect the
training stack.

*DCI is CC-BY-NC; SA-1B images are license-gated and were obtained via the user's signed Meta
download links — not redistributable.*
