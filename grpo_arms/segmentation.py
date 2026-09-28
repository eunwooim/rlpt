#!/usr/bin/env python
"""Rollout segmentation rules for the six-arm GRPO campaign.

Three modes:
  marker   - the validated "struct" rule from granularity/measure_granularity.py:
             a new step starts at a blank line OR a line-initial structural opener
             ("Step N:", "N." / "N)", "-" / "*" / "bullet"). Copied verbatim so the
             reward consumes exactly the rule the granularity report characterized.
  vprm     - VisualPRM's own split: response.split("\n\n") (modeling file split_response).
  chunker  - the DeBERTa release chunker; lives in the scoring server (GPU), not here.
"""
import re

# --- verbatim from granularity/measure_granularity.py (struct rule) ---
BLANK_RE = re.compile(r"\n\s*\n")
STRUCT_OPEN_RE = re.compile(
    r"(?m)^[ \t]*(?:[*_]{1,2}[ \t]*)?(?:step[ \t]*\d{1,2}[:.)]|\d{1,2}[.)]|[-*•][ \t])",
    re.I)


def split_blank(text: str):
    return [s for s in (p.strip() for p in BLANK_RE.split(text.strip())) if s]


def split_marker(text: str):
    """Blank line OR a line-initial structural opener starts a new step."""
    segs = []
    for para in split_blank(text):
        cuts = [m.start() for m in STRUCT_OPEN_RE.finditer(para)]
        if not cuts:
            segs.append(para)
            continue
        if cuts[0] > 0:
            cuts = [0] + cuts
        segs += [para[a:b].strip() for a, b in zip(cuts, cuts[1:] + [len(para)])]
    return [s for s in segs if s]


def split_vprm(text: str):
    """VisualPRM-8B's own convention (split_response in its modeling file)."""
    return [s for s in (p.strip() for p in text.split("\n\n")) if s]


def segment(text: str, mode: str):
    if mode == "marker":
        return split_marker(text)
    if mode == "vprm":
        return split_vprm(text)
    raise ValueError(f"segmentation mode {mode!r} not handled here (chunker runs server-side)")


if __name__ == "__main__":
    # self-tests mirroring the granularity script's assertions
    s = split_marker("Step 1: count them.\nAnswer: eight.\nStep 2: halve it.")
    assert len(s) == 2, s
    assert len(split_marker("step 3) lower case\nSTEP 4. upper")) == 2
    assert len(split_marker("intro\n1. a\n2. b\n\noutro")) == 4
    assert len(split_marker("one two\nthree four")) == 1
    assert len(split_marker("Steps 1 and 2 are done\nso are 3 and 4")) == 1
    assert split_vprm("a\n\nb\n\n\nc") == ["a", "b", "c"]
    print("segmentation self-tests OK")
