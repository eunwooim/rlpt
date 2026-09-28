#!/usr/bin/env python
"""EXP 1 — completeness audit over ALL chunks of canonical_chunked_v2.jsonl.

Per-chunk boolean flags (deterministic, no models):
  a) no_terminal_punct  — stripped chunk doesn't end in . ! ? : ; or a closing
     math delimiter ($, \\], \\), }, ```). Chunks that instead end on a bare
     equation/expression line (last char digit/letter/')' AND last line is
     math-like) are counted as ends_on_math, NOT as violations.
     Before flagging, trailing closers/quotes/markdown emphasis
     (" ' ” ’ ) ] * _) are stripped and the test re-run (so `**...answer.**`
     or `(see fig.)` don't false-positive).
  b) unbalanced_delims  — unbalanced (), [], {}, $...$ (parity), \\[...\\],
     \\(...\\), ``` (parity) within the chunk. Plain-paren counts exclude the
     escaped LaTeX forms.
  c) dangling_start     — after list-marker strip, begins (case-insensitive)
     with "and ", "which ", "so ", "therefore ", "then ", "gives ", "yields ",
     or a math operator + space ("= ", "+ ", "- "). Comma variants
     ("Therefore, ...") are counted separately as dangling_start_comma and are
     NOT part of the main flag (spec is token+space).
  d) dangling_end       — ends on ":" or a binary operator (=, +, *, /, ^)
     after stripping trailing markdown-emphasis runs (**, __) of length >= 2.

Broken out by bucket: marker/step, marker/list, marker/para, model,
gold passthrough (reference baseline), plus model_skipped_nonprose
(informational). 30 reservoir-sampled examples per (flag, bucket) ->
exp1_examples/. Also dumps an image-path prefix census (first two path
components) per bucket, used to validate the exp-3 source map.

Streams the file; multiprocessing over line batches; deterministic
(ordered imap, seeded reservoirs).
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from multiprocessing import Pool

import orjson

FLAGS = ("no_terminal_punct", "unbalanced_delims", "dangling_start", "dangling_end")
INFO = ("ends_on_math", "dangling_start_comma", "one_char_chunk")
BUCKETS = ("marker/step", "marker/list", "marker/para", "model",
           "model_skipped_nonprose", "gold")

TERMINAL = (".", "!", "?", ":", ";", "$", "\\]", "\\)", "}", "```")
_TRAIL_STRIP = "\"'\u201d\u2019)]*_"
LIST_MARK_RE = re.compile(r"^[ \t]{0,3}(?:\d{1,2}[.)][ \t]+|[-*\u2022][ \t]+)")
DANGLE_START_RE = re.compile(
    r"^(?:(?:and|which|so|therefore|then|gives|yields) |= |\+ |- )", re.I)
DANGLE_START_COMMA_RE = re.compile(
    r"^(?:and|which|so|therefore|then|gives|yields),", re.I)
MATH_MACRO_RE = re.compile(
    r"\\(?:frac|sqrt|cdot|times|pi|left|right|sum|int|angle|triangle|approx|le|ge|neq)\b")
MATH_CHARS = set("0123456789+-*/^=()[]{}.,<>|\\$_ \t")
EMPH_RUN_RE = re.compile(r"(?:\*{2,}|_{2,})$")


def _ends_terminal(s):
    return s.endswith(TERMINAL)


def _mathish_line(line):
    line = line.strip()
    if not line:
        return False
    if "=" in line or "$" in line or MATH_MACRO_RE.search(line):
        return True
    nonspace = [c for c in line if not c.isspace()]
    inmath = sum(1 for c in nonspace if c in MATH_CHARS)
    if nonspace and inmath / len(nonspace) >= 0.85:
        return True
    return len(line) <= 15 and any(c.isdigit() for c in line)


def check_chunk(chunk):
    """-> dict of flag booleans for one chunk."""
    out = {}
    s = chunk.strip()
    if len(s) <= 1:
        # degenerate; count separately, exempt from flags
        return {"one_char_chunk": True}

    # (a) terminal punctuation
    if _ends_terminal(s):
        pass
    else:
        s2 = s.rstrip(_TRAIL_STRIP).rstrip()
        if _ends_terminal(s2):
            pass
        else:
            last = s[-1]
            lastline = s.rsplit("\n", 1)[-1]
            if (last.isdigit() or last.isalpha() or last == ")") and _mathish_line(lastline):
                out["ends_on_math"] = True
            else:
                out["no_terminal_punct"] = True

    # (b) delimiter balance
    n_lp, n_rp = s.count("("), s.count(")")
    e_lp, e_rp = s.count("\\("), s.count("\\)")
    n_lb, n_rb = s.count("["), s.count("]")
    e_lb, e_rb = s.count("\\["), s.count("\\]")
    n_lc, n_rc = s.count("{"), s.count("}")
    dollars = s.count("$") - s.count("\\$")
    fences = s.count("```")
    if ((n_lp - e_lp) != (n_rp - e_rp) or (n_lb - e_lb) != (n_rb - e_rb)
            or n_lc != n_rc or e_lp != e_rp or e_lb != e_rb
            or dollars % 2 or fences % 2):
        out["unbalanced_delims"] = True

    # (c) dangling start
    head = LIST_MARK_RE.sub("", s, count=1).lstrip()
    if DANGLE_START_RE.match(head):
        out["dangling_start"] = True
    elif DANGLE_START_COMMA_RE.match(head):
        out["dangling_start_comma"] = True

    # (d) dangling end
    tail = EMPH_RUN_RE.sub("", s).rstrip()
    if tail and (tail[-1] in ":=+*/^"):
        out["dangling_end"] = True
    return out


def bucket_of(rec):
    prov = rec["metadata"]["chunk_provenance"]
    m = prov["method"]
    if m == "passthrough":
        return "gold" if len(rec["steps"]) >= 1 else None
    if m == "marker":
        return "marker/" + prov["tier"]
    if m == "model":
        return "model"
    if m == "model_skipped_nonprose":
        return "model_skipped_nonprose"
    return None


def _multi_bucket(rec):
    # multi-step passthrough records ARE the gold population
    return "gold"


def work(batch):
    """batch: list[bytes]. Returns (counts, examples, prefix_census)."""
    counts = Counter()   # (bucket, flag) -> n ; (bucket, "_chunks") -> n
    examples = []        # (flag, bucket, chunk_text[:800])
    ex_per_key = Counter()  # cap candidates per (flag,bucket) per batch so
    prefixes = Counter()    # rare buckets aren't crowded out by marker/list
    for line in batch:
        rec = orjson.loads(line)
        prov = rec["metadata"]["chunk_provenance"]
        if prov["method"] == "passthrough" and len(rec["steps"]) == 1:
            continue  # empty-blob passthrough, not gold
        b = ("gold" if prov["method"] == "passthrough" else bucket_of(rec))
        if b is None:
            continue
        imgs = rec.get("images") or []
        if imgs:
            prefixes["/".join(imgs[0].split("/")[:2])] += 1
        else:
            prefixes["<no-image>"] += 1
        for ch in rec["steps"]:
            counts[(b, "_chunks")] += 1
            res = check_chunk(ch)
            for k in res:
                counts[(b, k)] += 1
                if k in FLAGS and ex_per_key[(k, b)] < 5:
                    ex_per_key[(k, b)] += 1
                    examples.append((k, b, ch[:800]))
    return counts, examples, prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/scratch/sghos104/rlpt/canonical_chunked_v2.jsonl")
    ap.add_argument("--outdir", default="/scratch/sghos104/rlpt/chunk_eval")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # ---- self-test of the flag logic (hard-fail on regression) ----
    t = check_chunk("Therefore the answer is 7.")
    assert t == {"dangling_start": True}, t   # "therefore " is a spec token
    t = check_chunk("We add the two counts and get 9.")
    assert t == {}, t
    t = check_chunk("x = 5 + 3\nx = 8")
    assert t.get("ends_on_math"), t
    t = check_chunk("and then we add the")            # dangling both ends
    assert t.get("dangling_start") and t.get("no_terminal_punct"), t
    t = check_chunk("First we compute (3 + 4")
    assert t.get("unbalanced_delims"), t
    t = check_chunk("The value is x =")
    assert t.get("dangling_end"), t
    t = check_chunk("Compute the following:")
    assert t.get("dangling_end") and not t.get("no_terminal_punct"), t
    t = check_chunk("**The answer is 42.**")
    assert t == {}, t
    t = check_chunk("Therefore, the sum is 12.")
    assert t.get("dangling_start_comma") and "dangling_start" not in t, t
    t = check_chunk("- 3 apples remain (see above).")
    assert t == {}, t
    t = check_chunk("The area is \\(x^2\\)")
    assert "no_terminal_punct" not in t, t
    print("[exp1] self-test OK", flush=True)

    rng = random.Random(0)
    reservoirs = {}   # (flag,bucket) -> list
    seen = Counter()
    totals = Counter()
    prefix_census = Counter()

    def feed_examples(exs):
        for flag, b, text in exs:
            key = (flag, b)
            seen[key] += 1
            r = reservoirs.setdefault(key, [])
            if len(r) < 30:
                r.append(text)
            else:
                j = rng.randrange(seen[key])
                if j < 30:
                    r[j] = text

    def batches(f):
        buf = []
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            buf.append(line)
            if len(buf) >= args.batch:
                yield buf
                buf = []
        if buf:
            yield buf

    n_rows = 0
    with open(args.input, "rb") as f, Pool(args.workers) as pool:
        for counts, exs, prefixes in pool.imap(work, batches(f), chunksize=1):
            totals.update(counts)
            prefix_census.update(prefixes)
            feed_examples(exs)
            n_rows += args.batch
            if (n_rows // args.batch) % 25 == 0:
                print(f"[exp1] ~{n_rows:,} rows", flush=True)

    # ---- report ----
    stats = {"buckets": {}, "flag_defs": __doc__.strip().split("\n\n")[1]}
    for b in BUCKETS:
        n = totals[(b, "_chunks")]
        row = {"chunks": n}
        for k in FLAGS + INFO:
            row[k] = totals[(b, k)]
            row[k + "_rate"] = totals[(b, k)] / n if n else 0.0
        stats["buckets"][b] = row
    stats["prefix_census_top60"] = prefix_census.most_common(60)
    stats["prefix_census_total_keys"] = len(prefix_census)

    outdir = args.outdir
    with open(os.path.join(outdir, "exp1_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(outdir, "exp1_prefix_census.json"), "w") as f:
        json.dump(sorted(prefix_census.items(), key=lambda kv: -kv[1]), f, indent=2)

    exdir = os.path.join(outdir, "exp1_examples")
    os.makedirs(exdir, exist_ok=True)
    for (flag, b), texts in sorted(reservoirs.items()):
        fname = f"{flag}__{b.replace('/', '-')}.txt"
        with open(os.path.join(exdir, fname), "w") as f:
            for i, t in enumerate(texts):
                f.write(f"--- example {i+1} ({flag}, {b}) ---\n{t}\n\n")

    # console table
    hdr = f"{'bucket':24s} {'chunks':>12s} " + " ".join(f"{k[:14]:>15s}" for k in FLAGS + INFO)
    print(hdr)
    for b in BUCKETS:
        r = stats["buckets"][b]
        cells = " ".join(f"{100*r[k+'_rate']:14.2f}%" for k in FLAGS + INFO)
        print(f"{b:24s} {r['chunks']:>12,d} {cells}")
    grand = sum(totals[(b, "_chunks")] for b in BUCKETS)
    print(f"[exp1] total chunks audited: {grand:,}")
    print("[exp1] DONE")


if __name__ == "__main__":
    sys.exit(main())
