# COCO bounding-box IoU ranking — do text scorers respect spatial overlap?

N = 20,000 triples. Each: a GT box (`pos`, the reference) and two jittered candidate boxes — `neg` (HIGHER IoU with GT) and `NEG` (LOWER IoU). Boxes are rendered as text `[x1, y1, x2, y2]` in 0-1000 normalized coords (Qwen-VL convention).

A faithful, IoU-aware reward must satisfy **sim(pos, neg) > sim(pos, NEG)** — the box that overlaps the GT more should be judged more similar. **rank acc** = fraction of triples where that holds (1.0 = always, 0.5 = chance).

Mean IoU: neg (high) = 0.432, NEG (low) = 0.260, margin = 0.172.

## Rank accuracy & IoU correlation

| scorer (field) | rank acc ↑ | mean sim(hi) | mean sim(lo) | margin | Spearman(sim, IoU) |
|---|--:|--:|--:|--:|--:|
| **IoU (oracle)** | **1.000** | 0.432 | 0.260 | +0.172 | 1.000 |
| SBERT cosine | 0.577 | 0.802 | 0.788 | +0.015 | 0.026 |
| NLI equiv | 0.503 | 0.002 | 0.001 | +0.002 | 0.064 |
| NLI coverage | 0.503 | 0.002 | 0.001 | +0.002 | 0.064 |
| BERTScore f1 | 0.620 | 0.867 | 0.856 | +0.011 | 0.183 |

**Reading:** rank acc near 0.5 and Spearman near 0 ⇒ the text scorer's similarity is **blind to IoU** — it cannot tell which box overlaps the GT more, because it only sees digit strings. This is the spatial analogue of the numeric-negation blindness, and the quantitative case for an explicit IoU term in any grounding reward (the supervisor's point).

