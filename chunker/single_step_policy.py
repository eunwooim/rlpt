#!/usr/bin/env python3
"""Single-step negative policy for the chunker retrain.

Background: single-step VisualPRM records were used as negative-only training
examples on the assumption that their internal punctuation is human-confirmed
"do not split here". Audit (2026-08-07, 40 strided blocks over canonical.jsonl)
found 83% of them contain explicit step markers, numbered lists, or blank-line
paragraph breaks -- i.e. len(steps)==1 means UNSEGMENTED, not ATOMIC. Labeling
those candidate tokens 0 taught the model to suppress real boundaries.

Four policies, selected by --single-policy:

  keep   all sampled singles, every candidate labeled 0   (BASELINE = current v2
         behaviour; include it so the arms are comparable to the shipped model)
  drop   no single-step records in training at all
  clean  only structurally clean singles (no marker/list/blank) kept as negatives
  mask   all sampled singles kept, but candidates at structural markers forced
         to -100 (ignored by loss); ordinary internal punctuation stays 0

Interface is char-offset based so it composes with the existing labeling code,
which anchors by character offsets into the joined string.

  is_clean_single(text) -> bool
  masked_char_spans(text) -> [(start, end), ...]
  apply_policy(labels, offsets, text, policy) -> labels

Val/test are NOT affected by any of this: they are human multi-step records and
must stay byte-identical to the v2 split, or the arms are not comparable.
"""
import re

# --- structural boundary markers -------------------------------------------
# Line-anchored: these fire at the START of a line, so the boundary they imply
# is the newline immediately BEFORE the match.
STEPN = re.compile(r"^[ \t]*(step\s*\d+|第\s*\d+\s*步)\b", re.I | re.M)
NUMLIST = re.compile(r"^[ \t]*\d+[.)]\s", re.M)
BULLET = re.compile(r"^[ \t]*[-*•]\s", re.M)
# Blank line = paragraph break. Match the whole run so we can mask its edges.
BLANK = re.compile(r"\n[ \t]*\n")

POLICIES = ("keep", "drop", "clean", "mask")


def structural_markers(text):
    """[(kind, line_start_offset), ...] for every structural boundary marker."""
    out = []
    for kind, rx in (("stepN", STEPN), ("numlist", NUMLIST), ("bullet", BULLET)):
        for m in rx.finditer(text):
            if m.start() > 0:                      # a marker at offset 0 is not a boundary
                out.append((kind, m.start()))
    for m in BLANK.finditer(text):
        out.append(("blank", m.start()))
    return sorted(out, key=lambda x: x[1])


def is_clean_single(text):
    """True when nothing in the text looks like a step boundary.

    Audit rate: ~17% of single-step records pass this. Policy 'clean' keeps
    only these as negatives -- they are the ones for which the original
    'human-confirmed do not split here' reading is actually defensible.
    """
    return not structural_markers(text)


def masked_char_spans(text, pad=1):
    """Char ranges to force to -100 under policy 'mask'.

    For each structural marker we mask a small window around the newline that
    precedes it, since that newline is the candidate token the old pipeline
    would have labeled 0. `pad` widens the window to cover tokenizers that
    merge the newline with adjacent whitespace.
    """
    spans = []
    for kind, off in structural_markers(text):
        if kind == "blank":
            m = BLANK.search(text, off)
            end = m.end() if m else off + 1
            spans.append((max(0, off - pad), min(len(text), end + pad)))
        else:
            nl = text.rfind("\n", 0, off)
            if nl != -1:
                spans.append((max(0, nl - pad), min(len(text), nl + pad + 1)))
    return _merge(spans)


def _merge(spans):
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def apply_policy(labels, offsets, text, policy, ignore_index=-100):
    """Post-process one example's token labels.

    labels   list[int]                per-token labels from the normal pass
    offsets  list[tuple[int,int]]     tokenizer offset_mapping, same length
    text     str                      the joined string labels were anchored to
    policy   one of POLICIES

    Only 'mask' changes anything here; 'drop' and 'clean' are record-level
    filters applied upstream at extraction time.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if policy != "mask":
        return labels
    spans = masked_char_spans(text)
    if not spans:
        return labels
    out = list(labels)
    for i, (a, b) in enumerate(offsets):
        if a == b:                                  # special token
            continue
        for s, e in spans:
            if a < e and b > s:                     # overlap
                out[i] = ignore_index
                break
    return out


# --- self-test --------------------------------------------------------------

if __name__ == "__main__":
    import json, sys, itertools, collections

    demo = (
        "To factor `4x^2 + 12xy + 9y^2`, look for binomials.\n"
        "Step 1: Look for common factors. There are none.\n"
        "Step 2: The expression is a trinomial.\n"
        "\n"
        "1. Multiply the first terms.\n"
        "2. Combine like terms. The answer is `(2x + 3y)^2`."
    )
    print("markers:", structural_markers(demo))
    print("clean_single:", is_clean_single(demo))
    print("mask spans:", masked_char_spans(demo))
    for s, e in masked_char_spans(demo):
        print(f"  [{s}:{e}] {demo[max(0,s-25):e+25]!r}")

    # Rate check over the real corpus, if a path is given.
    if len(sys.argv) > 1:
        p = sys.argv[1]
        import os
        size = os.path.getsize(p)
        c = collections.Counter()
        with open(p, "rb") as f:
            for k in range(20):
                f.seek(int(size * k / 20)); f.readline()
                for line in itertools.islice(f, 1000):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    s = r.get("steps") or []
                    if len(s) != 1:
                        continue
                    c["clean" if is_clean_single(s[0]) else "marked"] += 1
        tot = sum(c.values())
        print(f"\nsingle-step sampled: {tot}")
        for k, v in c.most_common():
            print(f"  {k:8s} {v:7d}  {100*v/tot:5.1f}%")
        print("expect ~17% clean / ~83% marked")
