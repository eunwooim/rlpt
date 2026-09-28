# Clause-split + min-pool NLI: paraphrase PRECISION control

N = 9,254 faithful paraphrases of correct VisualPRM solutions (number-multiset verified identical to pos). Same pipeline as the negation test: split -> SBERT-cosine Hungarian align (tau=0.15) -> NLI **equiv** per matched clause-pair -> **min**-pool. Mean matched clauses/example = 26.6.

**FPR here = fraction of CORRECT paraphrases wrongly scored BELOW threshold (false alarm). Lower = better.** This is the precision side the negation (recall) test left untested.

| metric | clause-split min-equiv |
|---|--:|
| FPR@0.6 (paraphrase wrongly flagged) ↓ | **0.599** |
| FPR@0.5 (paraphrase wrongly flagged) ↓ | 0.226 |
| mean min-equiv on paraphrase (want HIGH) | 0.594 |
| median min-equiv on paraphrase | 0.578 |

**Reading:** a LOW FPR means min-pool keeps correct rewordings above 0.6 — the veto is safe to ship as-is. A HIGH FPR means one noisy reworded clause routinely sinks a correct solution; min-pool is too brittle and needs softening (soft-min / k-th-lowest quantile) and/or a hard numeric-match term. Compare against the negation recall FPR@0.6 = 0.008 to see the precision/recall trade.

