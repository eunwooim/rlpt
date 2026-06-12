"""Run a scene-graph generator on region phrases and score it against the GT.

Generator: the FactualSceneGraph text->scene-graph parser (Flan-T5).
Ground truth: the refined VG region scene graphs from vg_query_to_graph.py.
Metric: SPICE-style tuple F1 (objects / attributes / relations as a tuple set),
plus per-type breakdown. Deterministic matching after light normalization.

Usage:
    python tools/eval_sgg_vs_gt.py --input data/visual_genome/qsg_sample.jsonl \
        --n 500 --model lizhuang144/flan-t5-base-VG-factual-sg --device cpu
"""

import argparse
import json
import os
import re

os.environ.setdefault("NLTK_DATA", "/scratch/sghos104/rlpt/data/nltk_data")
from nltk.stem import WordNetLemmatizer

_LEM = WordNetLemmatizer()

# Deterministic canonicalization so semantically-identical graphs match.
IRREGULAR_NOUNS = {"men": "man", "women": "woman", "people": "person",
                   "children": "child", "feet": "foot", "teeth": "tooth",
                   "mice": "mouse", "geese": "goose"}
COPULA = {"be", "is", "are", "was", "were", "been", "being", "am"}
PRED_SYN = {  # conservative, clear equivalences only
    "wearing": "wear", "has": "have", "having": "have", "holding": "hold",
    "sitting on": "sit on", "standing on": "stand on",
    "on top of": "on", "onto": "on", "upon": "on",
    "beside": "next to", "underneath": "under", "below": "under",
}


# ---------- tuple extraction ----------

def _norm(tok):
    """Lemmatize a noun/attribute token to a canonical singular form."""
    tok = str(tok).lower().strip()
    if tok in IRREGULAR_NOUNS:
        return IRREGULAR_NOUNS[tok]
    return _LEM.lemmatize(tok, "n")


def canon_pred(pred):
    """Canonicalize a predicate: strip articles, drop leading copula, verb-lemmatize
    each word, map known synonyms. Returns '' if it reduces to a pure copula."""
    p = re.sub(r"\b(a|an|the)\b", " ", str(pred).lower().strip())
    words = [w for w in p.split() if w]
    while words and words[0] in COPULA:
        words = words[1:]
    words = [_LEM.lemmatize(w, "v") for w in words if w not in COPULA]
    p = " ".join(words).strip()
    return PRED_SYN.get(p, p)


def gt_tuples(sg):
    """{objects[], attributes[(o,a)], relations[(s,p,o)]} -> a set of typed tuples."""
    tup = set()
    for o in sg["objects"]:
        tup.add(("obj", _norm(o)))
    for o, a in sg["attributes"]:
        tup.add(("attr", _norm(o), _norm(a)))
    for s, p, o in sg["relations"]:
        cp = canon_pred(p)
        if cp:
            tup.add(("rel", _norm(s), cp, _norm(o)))
    return tup


TRIPLE_RE = re.compile(r"\(([^()]*)\)")


def pred_tuples(parsed_text):
    """Parse FactualSceneGraph output '( a , rel , b ) , ( c , is , d )' into typed tuples.
    Convention: predicate 'is' -> attribute; otherwise -> relation. Entities -> objects."""
    tup = set()
    for m in TRIPLE_RE.findall(parsed_text):
        parts = [p.strip() for p in m.split(",")]
        if len(parts) != 3:
            continue
        a, rel, b = parts
        if not a or not b:
            continue
        tup.add(("obj", _norm(a)))
        if rel.lower().strip() in ("is", "has attribute", "attribute"):
            tup.add(("attr", _norm(a), _norm(b)))
        else:
            tup.add(("obj", _norm(b)))
            cp = canon_pred(rel)
            if cp:
                tup.add(("rel", _norm(a), cp, _norm(b)))
    return tup


# ---------- scoring ----------

def prf(pred, gt):
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    inter = len(pred & gt)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gt) if gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def by_type(tupset, t):
    return {x for x in tupset if x[0] == t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/visual_genome/qsg_sample.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--model", default="lizhuang144/flan-t5-base-VG-factual-sg")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    recs = []
    for line in open(args.input):
        r = json.loads(line)
        if r["phrase"]:
            recs.append(r)
        if len(recs) >= args.n:
            break
    phrases = [r["phrase"] for r in recs]
    print(f"[eval] {len(recs)} regions | model={args.model} device={args.device}", flush=True)

    from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
    parser = SceneGraphParser(args.model, device=args.device)
    # parse returns formatted triple strings
    preds = parser.parse(phrases, batch_size=args.batch_size, max_output_len=128,
                          beam_size=args.beam, return_text=True)

    agg = {"all": [0, 0, 0], "obj": [0, 0, 0], "attr": [0, 0, 0], "rel": [0, 0, 0]}
    n = 0
    shown = 0
    for r, ptext in zip(recs, preds):
        g = gt_tuples(r["scene_graph"])
        p = pred_tuples(ptext if isinstance(ptext, str) else str(ptext))
        for key, sel in (("all", None), ("obj", "obj"), ("attr", "attr"), ("rel", "rel")):
            gg = g if sel is None else by_type(g, sel)
            pp = p if sel is None else by_type(p, sel)
            pr, rc, f1 = prf(pp, gg)
            agg[key][0] += pr; agg[key][1] += rc; agg[key][2] += f1
        n += 1
        if shown < args.show:
            print(f"\n  phrase: {r['phrase']!r}")
            print(f"    GT  : {sorted(g)}")
            print(f"    PRED: {sorted(p)}")
            shown += 1

    print(f"\n[eval] averaged over {n} regions (SPICE-style tuple F1):")
    for key in ("all", "obj", "attr", "rel"):
        pr, rc, f1 = (v / n for v in agg[key])
        print(f"  {key:5s}  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
