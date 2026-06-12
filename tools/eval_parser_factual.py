"""Evaluate the caption->graph parser on the FACTUAL benchmark (clean, caption-aligned
gold). Contrast with eval_parser_on_vg.py (noisy VG region graphs) to separate true
parser quality from benchmark noise.

NOTE: only a 'train' split is published and FSG was trained on FACTUAL, so this is an
optimistic upper bound (possible train overlap). Still useful as the clean ceiling.

    python tools/eval_parser_factual.py --n 1000 \
        --model lizhuang144/flan-t5-base-VG-factual-sg --device cpu
"""

import argparse
import os
import sys

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")
sys.path.insert(0, "tools")
from datasets import load_dataset
from caption_to_graph import CaptionToGraph, triples_to_graph
from eval_sgg_vs_gt import gt_tuples, prf, by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model", default="lizhuang144/flan-t5-base-VG-factual-sg")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("lizhuang144/FACTUAL_Scene_Graph")["train"]
    ds = ds.shuffle(seed=args.seed).select(range(args.n))
    caps = list(ds["caption"])
    golds = [triples_to_graph(s) for s in ds["scene_graph"]]
    print(f"[eval] FACTUAL clean gold | {len(caps)} captions | {args.model} {args.device}", flush=True)

    parser = CaptionToGraph(model=args.model, device=args.device, beam=args.beam)
    preds = parser.parse(caps, batch_size=args.batch_size)

    agg = {"all": [0, 0, 0], "obj": [0, 0, 0], "attr": [0, 0, 0], "rel": [0, 0, 0]}
    for gold, pred in zip(golds, preds):
        g, p = gt_tuples(gold), gt_tuples(pred)
        for key, t in (("all", None), ("obj", "obj"), ("attr", "attr"), ("rel", "rel")):
            gg = g if t is None else by_type(g, t)
            pp = p if t is None else by_type(p, t)
            pr, rc, f1 = prf(pp, gg)
            agg[key][0] += pr; agg[key][1] += rc; agg[key][2] += f1

    n = len(caps)
    print(f"\n[eval] parser vs FACTUAL clean gold, {n} captions (SPICE-style tuple F1):")
    for key in ("all", "obj", "attr", "rel"):
        pr, rc, f1 = (v / n for v in agg[key])
        print(f"  {key:5s}  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
