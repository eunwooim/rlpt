"""Deterministic query -> query-relevant scene graph, built from VG region graphs.

The reliable unit is the **region**: each VG region carries a natural-language
phrase plus a local scene graph (objects / attributes / relations). We treat that
region graph as the ground-truth "query-relevant scene graph". Attributes are
sparse in region graphs, so we enrich each object via an `object_id` join to the
image-level scene graph. A VG question can optionally be attached as the query
(via qa_to_region_mapping); queries may also be generated downstream.

Everything here is deterministic: no learned parser, no embeddings.

The big JSON files (region_graphs 2.6G, scene_graphs 706M) are streamed element
by element with bounded memory.

Usage:
    python tools/vg_query_to_graph.py --limit 300 \
        --out data/visual_genome/qsg_sample.jsonl --attach-qa
"""

import argparse
import json
import os
import re
from json import JSONDecoder, JSONDecodeError
from pathlib import Path

os.environ.setdefault("NLTK_DATA", "/scratch/sghos104/rlpt/data/nltk_data")
from nltk.stem import WordNetLemmatizer

_LEM = WordNetLemmatizer()

VG = Path("/scratch/sghos104/rlpt/data/visual_genome/raw")
IMAGES = Path("/scratch/sghos104/rlpt/data/images")


def iter_json_array(path, limit=None):
    """Yield top-level elements of a (possibly huge) JSON array, streaming."""
    dec = JSONDecoder()
    n = 0
    with open(path) as f:
        buf = f.read(1 << 20)
        i = buf.index("[") + 1
        while True:
            while True:
                while i < len(buf) and buf[i] in " \n\r\t,":
                    i += 1
                if i < len(buf):
                    break
                more = f.read(1 << 20)
                if not more:
                    return
                buf, i = buf[i:] + more, 0
            if buf[i] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf, i)
                    break
                except JSONDecodeError:
                    more = f.read(1 << 20)
                    if not more:
                        raise
                    buf += more
            yield obj
            i = end
            n += 1
            if limit and n >= limit:
                return
            if i > (2 << 20):
                buf, i = buf[i:], 0


def build_indexes(limit=None):
    """From the image-level scene graph, build:
       oid_attr: object_id -> [attributes]   (used to enrich region-object attributes)
       oid_name: object_id -> name
    """
    oid_attr, oid_name = {}, {}
    for sg in iter_json_array(VG / "scene_graphs.json", limit=limit):
        for o in sg.get("objects", []):
            oid = o["object_id"]
            oid_attr[oid] = o.get("attributes", []) or []
            names = o.get("names") or ([o["name"]] if "name" in o else [])
            oid_name[oid] = (names[0] if names else "?").lower().strip()
    return oid_attr, oid_name


def _phrase_has(phrase_lc, term):
    """Word-boundary containment so 'red' doesn't match 'covered'."""
    return re.search(r"\b" + re.escape(term) + r"\b", phrase_lc) is not None


# Function words VG occasionally annotates as "objects".
OBJ_STOPWORDS = {"is", "it", "a", "an", "the", "this", "that", "there",
                 "what", "which", "they", "he", "she", "you", "i", "we"}


def canon_pred(p):
    """Canonicalize a predicate: trim trailing/leading articles so 'on a' -> 'on',
    'has a' -> 'has' (mitigates VG's fragmented predicate vocabulary)."""
    p = p.lower().strip()
    for art in (" a", " an", " the"):
        if p.endswith(art):
            p = p[: -len(art)].strip()
    for art in ("a ", "an ", "the "):
        if p.startswith(art):
            p = p[len(art):].strip()
    return p


# ---- deterministic text<->graph consistency (grounding) ----

def phrase_lemma_set(phrase):
    """All word-lemmas present in the caption, for grounding checks."""
    toks = re.findall(r"[a-z]+", phrase.lower())
    s = set(toks)
    for t in toks:
        s.add(_LEM.lemmatize(t, "n"))
        s.add(_LEM.lemmatize(t, "v"))
    return s


def is_grounded(name, lemmas):
    """An object/name is grounded if its head word (lemma) appears in the caption."""
    words = name.split()
    if not words:
        return False
    head = words[-1]
    return (head in lemmas
            or _LEM.lemmatize(head, "n") in lemmas
            or name in lemmas)


def region_to_graph(reg, phrase, oid_attr, oid_name):
    """Build a GT graph that is *faithful to the caption* (deterministic text<->graph
    consistency): every object/attribute/relation must be grounded in the region
    phrase. Spurious objects (in the bbox but not the caption) and non-caption
    relations are dropped. Returns (graph, drop_stats)."""
    phrase_lc = phrase.lower()
    lemmas = phrase_lemma_set(phrase)
    local_name = {o["object_id"]: o["name"].lower().strip() for o in reg.get("objects", [])}

    def name_of(oid):
        return local_name.get(oid) or oid_name.get(oid, "?")

    raw_objs = {v for v in local_name.values() if v not in OBJ_STOPWORDS}
    # CONSISTENCY: keep only objects whose name is grounded in the caption text.
    objects = {v for v in raw_objs if is_grounded(v, lemmas)}
    dropped_obj = len(raw_objs) - len(objects)

    attrs = set()
    for o in reg.get("objects", []):
        nm = local_name[o["object_id"]]
        if nm not in objects:
            continue
        for a in oid_attr.get(o["object_id"], []):
            al = str(a).lower().strip()
            if _phrase_has(phrase_lc, al):
                attrs.add((nm, al))

    # Relations: only the region's OWN relationships (its parse of the caption),
    # and only when BOTH endpoints are grounded objects. No image-level backfill
    # (that would add facts the caption never states).
    rels = set()
    dropped_rel = 0
    for r in reg.get("relationships", []):
        s, t = name_of(r["subject_id"]), name_of(r["object_id"])
        if s in objects and t in objects:
            rels.add((s, canon_pred(r["predicate"]), t))
        else:
            dropped_rel += 1

    graph = {"objects": sorted(objects), "attributes": sorted(attrs), "relations": sorted(rels)}
    return graph, {"dropped_obj": dropped_obj, "dropped_rel": dropped_rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="number of images to process (default: all)")
    ap.add_argument("--out", default="data/visual_genome/qsg_sample.jsonl")
    ap.add_argument("--attach-qa", action="store_true",
                    help="attach VG questions to regions via qa_to_region_mapping")
    ap.add_argument("--keep-degenerate", action="store_true",
                    help="keep single-object graphs with no attributes/relations")
    args = ap.parse_args()

    print(f"[qsg] building indexes (limit={args.limit})...", flush=True)
    oid_attr, oid_name = build_indexes(limit=args.limit)
    print(f"[qsg]   indexed {len(oid_attr)} objects", flush=True)

    region_to_qa = {}
    if args.attach_qa:
        qa_map = json.load(open(VG / "qa_to_region_mapping.json"))  # qa_id(str)->region_id
        region_to_qa = {}  # region_id -> list of qa records
        qa_by_id = {}
        for img in iter_json_array(VG / "question_answers.json", limit=args.limit):
            for q in img.get("qas", []):
                qa_by_id[q["qa_id"]] = q
        for qid_s, rid in qa_map.items():
            q = qa_by_id.get(int(qid_s))
            if q:
                region_to_qa.setdefault(rid, []).append(
                    {"question": q["question"], "answer": q["answer"], "qa_id": q["qa_id"]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_regions = n_empty = n_with_qa = n_degenerate = 0
    sum_o = sum_a = sum_r = 0
    tot_drop_obj = tot_drop_rel = 0
    with out.open("w") as fout:
        for rg in iter_json_array(VG / "region_graphs.json", limit=args.limit):
            image_id = rg["image_id"]
            img_path = str(IMAGES / f"{image_id}.jpg")
            for reg in rg.get("regions", []):
                phrase = reg.get("phrase", "").strip()
                g, drops = region_to_graph(reg, phrase, oid_attr, oid_name)
                tot_drop_obj += drops["dropped_obj"]
                tot_drop_rel += drops["dropped_rel"]
                if not g["objects"]:
                    n_empty += 1
                    continue
                # Degenerate: a lone object with no attributes/relations carries no
                # relational signal for the reward.
                if (not args.keep_degenerate and len(g["objects"]) == 1
                        and not g["attributes"] and not g["relations"]):
                    n_degenerate += 1
                    continue
                rec = {
                    "image_id": image_id,
                    "image": img_path,
                    "region_id": reg["region_id"],
                    "region_bbox_xywh": [reg["x"], reg["y"], reg["width"], reg["height"]],
                    # The region phrase is the canonical (deterministic) query.
                    "query": phrase,
                    "phrase": phrase,
                    "scene_graph": g,
                }
                qas = region_to_qa.get(reg["region_id"])
                if qas:
                    rec["vg_queries"] = qas
                    n_with_qa += 1
                fout.write(json.dumps(rec) + "\n")
                n_regions += 1
                sum_o += len(g["objects"]); sum_a += len(g["attributes"]); sum_r += len(g["relations"])

    print(f"[qsg] wrote {n_regions} region scene-graphs -> {out}", flush=True)
    print(f"[qsg]   skipped {n_empty} regions left empty after consistency filtering", flush=True)
    print(f"[qsg]   skipped {n_degenerate} degenerate (single object, no attr/rel)", flush=True)
    print(f"[qsg]   consistency drops: {tot_drop_obj} ungrounded objects, "
          f"{tot_drop_rel} non-caption/ungrounded relations", flush=True)
    if n_regions:
        print(f"[qsg]   avg per region: objects={sum_o/n_regions:.1f} "
              f"attributes={sum_a/n_regions:.1f} relations={sum_r/n_regions:.1f}", flush=True)
    if args.attach_qa:
        print(f"[qsg]   regions with an attached VG query: {n_with_qa}", flush=True)


if __name__ == "__main__":
    main()
