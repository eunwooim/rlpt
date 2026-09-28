#!/usr/bin/env python
"""Map canonical_chunked_v2 records to sympy-verifier tiers via image paths.

canonical.jsonl was built from the *processed* HF dataset
OpenGVLab/VisualPRM400K-v1.1, whose image paths drop the
"VisualPRM400K-v1.1-Raw/<FamilyDir>/" prefix of the Raw repo. The rules
below were derived by cross-referencing three evidence sources:
  1. raw annos (data/visualprm_v11_raw/annos/annotations/*.jsonl) image paths
  2. the v1 processed annotations.zip *_prm.jsonl image paths (same
     stripped convention as canonical)
  3. byte-offset probes of canonical_chunked_v2.jsonl itself
Tier names and source membership mirror src/data/sympy_verify.py TIERS.

Rules are HIGH-PRECISION by design (exp 3 samples anyway; a record that
doesn't match any rule gets tier None and is simply out of scope).
MathV360K subset dirs are loaded from mathv360k_subsets.json, built once
from the raw MathV360K_prompts.jsonl (build_mathv360k_subsets()); any
subset whose name collides with another rule's first component is
excluded there and counted.
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
MATHV_SUBSETS_JSON = os.path.join(_HERE, "mathv360k_subsets.json")
RAW_MATHV = ("/scratch/sghos104/rlpt/data/visualprm_v11_raw/annos/annotations/"
             "MathV360K_prompts.jsonl")

# (tier, source_label, compiled regex) — first match wins, checked in order.
_RULES = [
    # --- function tier: mavis_function_{abs,cos,log,poly,sin,tan} ---
    ("function", "mavis_function",
     re.compile(r"^(?:abs|cos|log|poly|sin|tan)/\d+/func\d+\.png$")),
    # --- geometry tier ---
    ("geometry", "mavis_geo", re.compile(r"^RuleBaseGeo/depth\d/")),
    ("geometry", "geometry3k", re.compile(r"^train/\d+/img_diagram\.png$")),
    ("geometry", "geo170k", re.compile(r"^geoqa_plus/\d+\.png$")),
    ("geometry", "geoqa_plus", re.compile(r"^images/\d+\.png$")),
    ("geometry", "geomverse", re.compile(r"^TRAIN/TRAIN_MIX")),
    ("geometry", "unigeo", re.compile(r"^calculation_images/\d+\.png$")),
    ("geometry", "geos", re.compile(r"^\d{3}\.png$")),
    # --- arithmetic tier: CLEVR_math, dvqa, MathV360K ---
    ("arithmetic", "clevr_math", re.compile(r"^CLEVR_v1\.0/images/")),
    ("arithmetic", "dvqa", re.compile(r"^images/bar_train_\d+\.png$")),
    # MathV360K handled separately (subset-dir list), appended in load_rules().
]


def build_mathv360k_subsets(raw_path=RAW_MATHV, out_json=MATHV_SUBSETS_JSON):
    """One-time: enumerate MathV360K subset dirs from the raw annos file."""
    import orjson
    subsets = set()
    with open(raw_path, "rb") as f:
        for line in f:
            img = orjson.loads(line)["image"]
            # VisualPRM400K-v1.1-Raw/MathV360K/<Subset>/images/xxx
            parts = img.split("/")
            if len(parts) >= 4 and parts[1] == "MathV360K":
                subsets.add(parts[2])
    # drop subsets whose stripped path would collide with another rule
    dropped = sorted(s for s in subsets
                     if s in ("images", "train", "train1", "CLEVR_v1.0",
                              "geoqa_plus", "TRAIN", "RuleBaseGeo",
                              "calculation_images", "iconqa_data", "abc_images"))
    kept = sorted(subsets - set(dropped))
    with open(out_json, "w") as f:
        json.dump({"kept": kept, "dropped_colliding": dropped}, f, indent=2)
    return kept, dropped


class TierMap:
    def __init__(self):
        self.rules = list(_RULES)
        with open(MATHV_SUBSETS_JSON) as f:
            d = json.load(f)
        if d["kept"]:
            alt = "|".join(re.escape(s) for s in d["kept"])
            self.rules.append(
                ("arithmetic", "mathv360k",
                 re.compile(r"^(?:%s)/images/" % alt)))
        self.mathv_dropped = d["dropped_colliding"]

    def classify(self, images):
        """-> (tier, source_label) or (None, None). Uses the first image."""
        if not images:
            return None, None
        img = images[0]
        for tier, label, rx in self.rules:
            if rx.match(img):
                return tier, label
        return None, None


if __name__ == "__main__":
    kept, dropped = build_mathv360k_subsets()
    print(f"MathV360K subsets kept: {len(kept)}, dropped (collision): {dropped}")
    print(kept)
