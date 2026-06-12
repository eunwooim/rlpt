"""Caption -> scene graph parser: the *predicted* side of the RLVR reward.

Wraps the FactualSceneGraph Flan-T5 parser and returns a scene graph in the SAME
shape as the GT graphs from vg_query_to_graph.py:
    {"objects": [...], "attributes": [[obj, attr], ...], "relations": [[subj, pred, obj], ...]}

Surface forms are only lowercased here; the final canonicalization (lemmatization,
synonym folding) is applied symmetrically to BOTH predicted and GT at reward time,
so this module stays a faithful parser and not a place that hides mismatches.

Models (FACTUAL VG-trained):
    lizhuang144/flan-t5-base-VG-factual-sg    (fast, decent)
    lizhuang144/flan-t5-large-VG-factual-sg   (slower, more accurate -- the "good" one)

Usage:
    from tools.caption_to_graph import CaptionToGraph
    p = CaptionToGraph(model="lizhuang144/flan-t5-large-VG-factual-sg", device="cuda")
    graphs = p.parse(["a man wearing a red hat is riding a brown horse"])

CLI demo:
    python tools/caption_to_graph.py --device cpu \
        --model lizhuang144/flan-t5-base-VG-factual-sg
"""

import argparse
import os
import re

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")
os.environ.setdefault("NLTK_DATA", "/scratch/sghos104/rlpt/data/nltk_data")

_TRIPLE_RE = re.compile(r"\(([^()]*)\)")
_ATTR_PREDS = {"is", "are", "has attribute", "attribute", "attr", "color", "colour"}


def _norm(tok):
    return str(tok).lower().strip()


def triples_to_graph(parsed_text):
    """FSG output '( a , rel , b ) , ( c , is , d )' -> graph dict (lowercased)."""
    objects, attributes, relations = set(), set(), set()
    for m in _TRIPLE_RE.findall(str(parsed_text)):
        parts = [p.strip() for p in m.split(",") if p.strip()]
        if len(parts) == 1:
            objects.add(_norm(parts[0]))
        elif len(parts) == 2:
            o, a = parts
            objects.add(_norm(o))
            attributes.add((_norm(o), _norm(a)))
        elif len(parts) == 3:
            a, rel, b = parts
            objects.add(_norm(a))
            if _norm(rel) in _ATTR_PREDS:
                attributes.add((_norm(a), _norm(b)))
            else:
                objects.add(_norm(b))
                relations.add((_norm(a), _norm(rel), _norm(b)))
    return {
        "objects": sorted(objects),
        "attributes": sorted(list(x) for x in attributes),
        "relations": sorted(list(x) for x in relations),
    }


class CaptionToGraph:
    def __init__(self, model="lizhuang144/flan-t5-large-VG-factual-sg",
                 device="cpu", beam=5):
        from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
        self.parser = SceneGraphParser(model, device=device)
        self.beam = beam

    def parse(self, captions, batch_size=32, max_output_len=128):
        if isinstance(captions, str):
            captions = [captions]
        texts = self.parser.parse(captions, beam_size=self.beam, return_text=True,
                                  batch_size=batch_size, max_output_len=max_output_len)
        return [triples_to_graph(t if isinstance(t, str) else str(t)) for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lizhuang144/flan-t5-base-VG-factual-sg")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=5)
    args = ap.parse_args()

    demo = [
        "a man wearing a red hat is riding a brown horse",
        "two white dogs are sitting on a wooden bench near a tree",
        "the clock on the tall tower is green",
    ]
    p = CaptionToGraph(model=args.model, device=args.device, beam=args.beam)
    for cap, g in zip(demo, p.parse(demo)):
        print(f"\ncaption: {cap!r}")
        print(f"  objects   : {g['objects']}")
        print(f"  attributes: {g['attributes']}")
        print(f"  relations : {g['relations']}")


if __name__ == "__main__":
    main()
