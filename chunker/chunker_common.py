#!/usr/bin/env python
"""Shared definitions for the DeBERTa-v3-small chunk-boundary classifier.

This module is the SINGLE SOURCE OF TRUTH for:
  * the expanded candidate-token rule (imported by both prepare_data.py and
    chunker.py so training and inference can never diverge),
  * gold-boundary label construction (labels in {1, 0, -100}),
  * sliding-window slicing,
  * question hashing for grouped splits.

Label semantics (per token, over the joined trajectory text = "\n".join(steps)):
  1    -> gold boundary token: the token whose char span contains the LAST
          NON-WHITESPACE character of a non-final step. Always labeled 1, even
          if that char is not "punctuation".
  0    -> a candidate token (plausible split point by the expanded rule below)
          that is NOT a gold boundary -> a negative the model must reject.
  -100 -> ignored by the loss: non-candidate non-gold tokens, all special
          tokens, and the token holding the final char of the whole trajectory
          (the end of text is always a boundary, so nothing to learn there).

Metrics are computed only over tokens with label != -100 ("candidate tokens").
"""
import hashlib
import re

MODEL = "microsoft/deberta-v3-small"
MAX_LEN = 512                      # model max sequence length (incl. specials)
WINDOW_CONTENT = MAX_LEN - 2       # room for [CLS]/[SEP] -> 510 content tokens
STEP = 384                         # sliding-window step (advance) in content tokens
LABEL_SPLIT = 1
LABEL_NOSPLIT = 0
LABEL_IGNORE = -100

# ---- expanded candidate rule character sets (approved 2026-07-22) ----
# Boundary recall over gold ends: 84.25% (ASCII-only) -> 98.75% (expanded).
ASCII_PUNCT = set(".!?;:")
CJK_PUNCT = set("。！？；：，．、")          # multilingual (Chinese) sentence enders
MATH_CLOSE = set("]$)}%")                    # LaTeX \] , inline $ , ) , } , percent
CLOSER_CHARS = ASCII_PUNCT | CJK_PUNCT | MATH_CLOSE
LIST_RX = re.compile(r"^\s*(\d+[.)]|[-*•])")  # list-item line start


def qhash(q: str) -> str:
    """Stable hash of a question string, used as the split-grouping key."""
    return hashlib.blake2b((q or "").encode("utf-8", "ignore"), digest_size=12).hexdigest()


# CJK unicode ranges (CJK ideographs, Hiragana/Katakana, Hangul, fullwidth punct).
def _is_cjk(cp):
    o = ord(cp)
    return (0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF
            or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF
            or 0x3000 <= o <= 0x303F)


def lang_bucket(text, cjk_thresh=0.10):
    """Cheap per-trajectory language tag: 'cjk' if >=10% of alpha-ish chars are
    CJK, 'en' if the text is predominantly ASCII letters, else 'other'.
    Heuristic only — used to bucket split-F1, not for any modeling decision."""
    cjk = latin = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            latin += 1
    tot = cjk + latin
    if tot == 0:
        return "other"
    if cjk / tot >= cjk_thresh:
        return "cjk"
    if latin / tot >= 0.90:
        return "en"
    return "other"


def join_steps(steps):
    return "\n".join(steps)


def gold_boundary_char_positions(steps):
    """Char index (into the joined text) of the last non-whitespace char of each
    NON-FINAL step -> the gold boundary positions. Also returns the final-char
    position of the whole trajectory (to be masked -100)."""
    golds = []
    start = 0
    n = len(steps)
    for i, s in enumerate(steps):
        stripped = s.rstrip()
        last = start + len(stripped) - 1 if stripped else start
        if i < n - 1:
            golds.append(last)          # non-final step boundary
        final_char_pos = last           # updated each step; last value = trajectory end
        start += len(s) + 1             # +1 for the "\n" separator
    return golds, final_char_pos


def candidate_token_mask(text, offsets):
    """Return list[bool]: True where a chunk boundary may fall (this token could be
    the LAST token of a chunk). The ONE rule shared by training and inference.

    A token [s,e) is a candidate iff its last content char `text[e-1]` sits at a
    real break (next char is whitespace or end-of-text) AND either:
      - that char is a closer (ASCII/CJK sentence punct, or math closer ] $ ) } %), or
      - that char is a digit (bare-number boundary, e.g. "Final answer: 2"), or
      - the next line begins a list item (boundary before a bullet/enumeration).
    Special tokens (empty span, e<=s) are never candidates.
    """
    n = len(text)
    mask = []
    for (s, e) in offsets:
        if e <= s:                      # special / empty-span token
            mask.append(False)
            continue
        ch = text[e - 1]
        # CJK sentence punctuation ends a sentence with NO trailing space in
        # Chinese/Japanese, so it is a boundary regardless of the next char.
        # (Without this exemption the whitespace gate below would reject nearly
        # every mid-paragraph CJK boundary at inference on raw, space-free text.)
        if ch in CJK_PUNCT:
            mask.append(True)
            continue
        nxt = text[e] if e < n else ""
        # a real break = next char is whitespace, end-of-text, OR a CJK char
        # (CJK text is space-free, so a following CJK char is a valid follow).
        at_break = (e >= n) or nxt.isspace() or _is_cjk(nxt)
        cand = False
        if at_break:
            if ch in CLOSER_CHARS or ch.isdigit():
                cand = True
            elif nxt == "\n":
                # boundary before a list item on the following line
                nxt_line = text[e + 1:e + 48]
                if LIST_RX.match(nxt_line):
                    cand = True
        mask.append(cand)
    return mask


def _token_covering(offsets, pos, hi=None):
    """Index of the token whose span [s,e) contains char `pos`; fallback to the
    last token starting at or before `pos`. offsets are contiguous & sorted."""
    best = None
    hi = len(offsets) if hi is None else hi
    for t in range(hi):
        s, e = offsets[t]
        if e <= s:
            continue
        if s <= pos < e:
            return t
        if s <= pos:
            best = t
    return best


def build_token_labels(steps, tokenizer):
    """Tokenize the joined trajectory (no special tokens) and assign per-token
    labels in {1,0,-100}. Returns dict with text, input_ids, offsets, labels."""
    text = join_steps(steps)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    cand = candidate_token_mask(text, offsets)
    labels = [LABEL_IGNORE] * len(input_ids)
    # negatives: candidate tokens start as 0
    for t, c in enumerate(cand):
        if c:
            labels[t] = LABEL_NOSPLIT

    golds, final_pos = gold_boundary_char_positions(steps)
    for g in golds:
        t = _token_covering(offsets, g)
        if t is not None:
            labels[t] = LABEL_SPLIT     # gold overrides candidate/ignore
    # the final char of the whole trajectory: always a boundary -> nothing to learn
    tf = _token_covering(offsets, final_pos)
    if tf is not None:
        labels[tf] = LABEL_IGNORE

    return {"text": text, "input_ids": input_ids, "offsets": offsets,
            "labels": labels, "gold_char_positions": golds}


def make_windows(input_ids, offsets, labels, cls_id, sep_id):
    """Slice a (possibly long) labeled sequence into <=MAX_LEN windows with
    [CLS]/[SEP] re-added (label -100). Content window = WINDOW_CONTENT, advancing
    STEP tokens each time; overlap tokens (and any gold boundary in them) appear
    in both windows. Returns list of dicts: input_ids, labels, offsets, offset in
    the parent sequence (for inference merge)."""
    T = len(input_ids)
    windows = []
    if T <= WINDOW_CONTENT:
        starts = [0]
    else:
        starts = list(range(0, T - WINDOW_CONTENT + 1, STEP))
        if starts[-1] != T - WINDOW_CONTENT:
            starts.append(T - WINDOW_CONTENT)   # ensure the tail is covered
    for st in starts:
        en = min(st + WINDOW_CONTENT, T)
        w_ids = [cls_id] + input_ids[st:en] + [sep_id]
        w_lab = [LABEL_IGNORE] + labels[st:en] + [LABEL_IGNORE]
        w_off = [(-1, -1)] + offsets[st:en] + [(-1, -1)]
        windows.append({"input_ids": w_ids, "labels": w_lab,
                        "offsets": w_off, "core_start": st, "core_end": en})
    return windows
