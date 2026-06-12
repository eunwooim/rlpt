"""Evaluate the caption->scene-graph parser against VG's OWN ground-truth graphs.

Non-circular: the reference is VG's human-annotated region graph (not anything we
derived). For each region we take the caption (phrase) and the raw VG region graph
(objects + relationships, plus attributes joined from the image-level graph),
parse the caption with FSG, and score predicted-vs-VG with SPICE-style tuple F1.

Usage:
    python tools/eval_parser_on_vg.py --limit 300 --n 500 \
        --model lizhuang144/flan-t5-base-VG-factual-sg --device cpu
"""

import argparse
import re
import sys

sys.path.insert(0, "tools")
from vg_query_to_graph import build_indexes, iter_json_array, VG
from eval_sgg_vs_gt import gt_tuples, prf, by_type   # reuse metric + canonicalization
from caption_to_graph import CaptionToGraph


def _in_phrase(term, phrase_lc):
    return re.search(r"\b" + re.escape(term) + r"\b", phrase_lc) is not None


def raw_region_graph(reg, phrase, oid_attr, oid_name):
    """VG's graph for a region: objects + relationships (names resolved).
    Attributes are joined by object_id from the image-level graph but scoped to
    the caption (only those the phrase actually states) -- the correct gold for a
    *caption* parser, since image-level attributes the caption never mentions are
    not part of the caption's graph."""
    phrase_lc = phrase.lower()
    local_name = {o["object_id"]: o["name"].lower().strip() for o in reg.get("objects", [])}

    def name_of(oid):
        return local_name.get(oid) or oid_name.get(oid, "?")

    objects = sorted(set(local_name.values()))
    attrs = sorted({(nm, str(a).lower().strip())
                    for oid, nm in local_name.items()
                    for a in oid_attr.get(oid, [])
                    if _in_phrase(str(a).lower().strip(), phrase_lc)})
    rels = [[name_of(r["subject_id"]), str(r["predicate"]).lower().strip(), name_of(r["object_id"])]
            for r in reg.get("relationships", [])]
    return {"objects": objects, "attributes": [list(x) for x in attrs], "relations": rels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="VG images to scan for regions")
    ap.add_argument("--n", type=int, default=500, help="number of regions to evaluate")
    ap.add_argument("--model", default="lizhuang144/flan-t5-base-VG-factual-sg")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    print(f"[eval] indexing first {args.limit} images for attributes...", flush=True)
    oid_attr, oid_name = build_indexes(limit=args.limit)

    captions, golds = [], []
    for rg in iter_json_array(VG / "region_graphs.json", limit=args.limit):
        for reg in rg.get("regions", []):
            phrase = reg.get("phrase", "").strip()
            g = raw_region_graph(reg, phrase, oid_attr, oid_name)
            # need a caption and a non-trivial VG graph to score against
            if phrase and g["objects"] and (g["attributes"] or g["relations"]):
                captions.append(phrase)
                golds.append(g)
        if len(captions) >= args.n:
            break
    captions, golds = captions[:args.n], golds[:args.n]
    print(f"[eval] {len(captions)} regions | parser={args.model} device={args.device}", flush=True)

    parser = CaptionToGraph(model=args.model, device=args.device, beam=args.beam)
    preds = parser.parse(captions, batch_size=args.batch_size)

    agg = {"all": [0, 0, 0], "obj": [0, 0, 0], "attr": [0, 0, 0], "rel": [0, 0, 0]}
    for gold, pred in zip(golds, preds):
        g = gt_tuples(gold)
        p = gt_tuples(pred)
        for key, t in (("all", None), ("obj", "obj"), ("attr", "attr"), ("rel", "rel")):
            gg = g if t is None else by_type(g, t)
            pp = p if t is None else by_type(p, t)
            pr, rc, f1 = prf(pp, gg)
            agg[key][0] += pr; agg[key][1] += rc; agg[key][2] += f1

    n = len(captions)
    print(f"\n[eval] parser vs VG human-gold region graphs, {n} regions (SPICE-style tuple F1):")
    for key in ("all", "obj", "attr", "rel"):
        pr, rc, f1 = (v / n for v in agg[key])
        print(f"  {key:5s}  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
