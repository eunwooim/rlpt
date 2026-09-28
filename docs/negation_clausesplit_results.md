# Clause-split + min-pool NLI on VisualPRM numeric negation

N = 10,000 examples. Each solution split into atomic clauses (newline + sentence punctuation), clauses aligned via SBERT-cosine Hungarian matching (tau=0.15), NLI **equiv** scored per matched clause-pair, then pooled with **min** (one contradicted clause vetoes the solution). Mean matched clauses/example = 26.6.

**Lower FPR = better** (corrupted copy correctly scored low / not fooled). Paragraph-level numbers are the existing whole-solution NLI equiv on the same examples.

| metric | paragraph equiv | clause-split min-equiv |
|---|--:|--:|
| FPR@0.6 on neg (corrupted) ↓ | 0.530 | **0.008** |
| FPR@0.5 on neg (corrupted) ↓ | 0.594 | 0.015 |
| mean equiv on neg (corrupted) | 0.569 | 0.015 |
| median equiv on neg | 0.647 | 0.000 |
| mean equiv on base (unrelated floor) | 0.101 | 0.000 |

**Reading:** if clause-split min-equiv drops FPR@0.6 on the corrupted copy well below the paragraph 0.53, splitting removed the dilution and NLI now catches the isolated numeric flip. Residual FPR is the part NLI still misses on bare numeric equality (e.g. neutral on `=14` vs `=15`) -- the case for a hard numeric-match term.

