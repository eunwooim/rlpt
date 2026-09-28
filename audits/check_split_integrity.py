#!/usr/bin/env python3
"""Audit chunk boundaries for mid-sentence / mid-equation splits.

Streams a canonical-chunked jsonl. For every record with >=2 chunks, checks
each chunk (balance flags) and each adjacent boundary (cut-quality flags).
Reports violation rates per chunking method, cross-tabbed BY SOURCE DATASET
so produced chunks are compared against gold within the same population.

Usage:
  python3 check_split_integrity.py canonical_chunked_v3.jsonl \
      [--max-lines N] [--examples-per-flag 15] [--outdir split_audit]

Notes:
  - --max-lines counts LINES SCANNED (not multi-chunk records).
  - Read each produced column against the GOLD column WITHIN THE SAME SOURCE.
    Cross-source comparison is confounded by length/template differences.
"""
import argparse, json, os, random, re, sys
from collections import defaultdict

# ---------------------------------------------------------------- constants

CJK_TERMINAL = "。！？；：…"
TERMINAL = tuple(".!?:;" + CJK_TERMINAL)
CLOSERS  = tuple(")]}\"'`" + "」』）】》〉”’")

CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

OPERATOR_END   = re.compile(r"[=+\-*/^×÷]\s*$")     # ends on a binary operator
OPERATOR_START = re.compile(r"^\s*[=+*/^×÷]")       # leading '-' excluded: bullets/negatives
CONTINUATION_START = re.compile(
    r"^\s*(and|which|so|therefore|then|gives|yields|hence|thus|where|because)\b",
    re.IGNORECASE)
LIST_MARKER = re.compile(r"^\s*(step\s*\d+|[-*•]|\d+[.)]|第\s*\d+\s*步)", re.IGNORECASE)

# A "bare equation line" must carry a RELATION or a math delimiter.
# The presence of a digit alone is NOT enough -- "There are 3 balls" is prose.
MATH_TAIL = re.compile(r"[=<>≤≥±≠≈√]|\\\(|\\\)|\\\[|\\\]|\$|\\frac|\\sqrt|\\times")
ENDS_ALNUM_OR_CLOSE = re.compile(r"[A-Za-z0-9\)\]\}]$")

PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]
FENCE = re.compile(r"^```", re.M)


def strip_md(s):
    return s.strip()


def ends_ok(chunk):
    """(ok, reason). A chunk ending is 'ok' if it terminates a sentence
    (ASCII or CJK), closes a bracket, or is a genuine bare equation line."""
    s = strip_md(chunk)
    if not s:
        return False, "empty"
    last = s[-1]
    if last in TERMINAL or last in CLOSERS:
        return True, ""
    if OPERATOR_END.search(s):
        return False, "ends_on_operator"
    tail = s.split("\n")[-1]
    if MATH_TAIL.search(tail) and ENDS_ALNUM_OR_CLOSE.search(s):
        return True, ""                      # e.g. "2W = 6" / "x = 3" / "\\sqrt{2}"
    if re.search(r"[A-Za-z0-9]$", s) or CJK_CHAR.search(s[-1]):
        return False, "no_terminal_punct"    # prose ending mid-air
    return False, "odd_ending"


def starts_midsentence(n):
    """Lowercase start (Latin) or any CJK char -- CJK has no case, so
    islower() alone silently exempts every CJK boundary."""
    if not n:
        return False
    return n[0].islower() or bool(CJK_CHAR.match(n[0]))


def unbalanced(chunk):
    reasons = []
    for o, c in PAIRS:
        if chunk.count(o) != chunk.count(c):
            reasons.append(f"unbalanced_{o}{c}")
    if chunk.count("$") % 2:
        reasons.append("unbalanced_$")
    if (chunk.count("\\(") != chunk.count("\\)")) or (chunk.count("\\[") != chunk.count("\\]")):
        reasons.append("unbalanced_latex")
    if len(FENCE.findall(chunk)) % 2:
        reasons.append("unbalanced_fence")
    return reasons


def boundary_flags(prev, nxt):
    """Flags for the CUT between two adjacent chunks."""
    flags = []
    ok, why = ends_ok(prev)
    n = strip_md(nxt)
    starts_list = bool(LIST_MARKER.match(n))
    if not ok:
        flags.append(f"prev_{why}")
    if OPERATOR_START.match(n):
        flags.append("next_starts_operator")
    elif (not starts_list) and CONTINUATION_START.match(n):
        flags.append("next_starts_continuation")
    elif (not starts_list) and starts_midsentence(n) and not ok:
        flags.append("mid_sentence_pair")
    return flags


SOURCE_KEYS = ("source", "dataset", "source_dataset", "data_source", "subset")


def get_source(meta):
    for k in SOURCE_KEYS:
        v = meta.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            for kk in SOURCE_KEYS:
                vv = v.get(kk)
                if isinstance(vv, str) and vv:
                    return vv
    sid = meta.get("source_sample_id")
    if isinstance(sid, str) and "/" in sid:
        return sid.rsplit("/", 1)[0]
    return "unknown"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-lines", type=int, default=0, help="0 = all; counts LINES scanned")
    ap.add_argument("--examples-per-flag", type=int, default=15)
    ap.add_argument("--outdir", default="split_audit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-source-support", type=int, default=500,
                    help="min boundaries for a source to get its own table row")
    ap.add_argument("--progress-every", type=int, default=500_000)
    ap.add_argument("--assume-method", default=None,
                    help="label all records this method (for files with no chunk_provenance)")
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    counts = defaultdict(lambda: defaultdict(int))          # (method, source) -> flag -> n
    totals = defaultdict(lambda: {"multi_records": 0, "single_records": 0,
                                  "chunks": 0, "boundaries": 0})
    examples = defaultdict(list)
    seen_per_key = defaultdict(int)

    def maybe_keep(key, item):
        """Reservoir sampling -- without this, every example comes from the
        head of the file and you never see anything from record 4,000,000."""
        seen_per_key[key] += 1
        buf = examples[key]
        if len(buf) < a.examples_per_flag:
            buf.append(item)
        else:
            j = random.randrange(seen_per_key[key])
            if j < a.examples_per_flag:
                buf[j] = item

    lines = 0
    bad_json = 0
    probed = False

    with open(a.path, encoding="utf-8", errors="replace") as f:
        for line in f:
            lines += 1
            if a.progress_every and lines % a.progress_every == 0:
                print(f"  ... {lines:,} lines", file=sys.stderr, flush=True)
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            meta = r.get("metadata") or {}
            if not probed:
                print(f"[schema probe] top-level keys: {sorted(r.keys())}",
                      file=sys.stderr)
                print(f"[schema probe] metadata keys: {sorted(meta.keys())}",
                      file=sys.stderr)
                print(f"[schema probe] resolved source -> {get_source(meta)!r}",
                      file=sys.stderr, flush=True)
                probed = True

            prov = meta.get("chunk_provenance") or {}
            method = a.assume_method or prov.get("method", "unknown")
            if method == "passthrough":
                method = "GOLD"
            src = get_source(meta)
            key = (method, src)

            steps = r.get("steps") or []
            if len(steps) < 2:
                # zero-cut record: the other half of the over-splitting question
                totals[key]["single_records"] += 1
                if a.max_lines and lines >= a.max_lines:
                    break
                continue

            totals[key]["multi_records"] += 1
            totals[key]["chunks"] += len(steps)
            totals[key]["boundaries"] += len(steps) - 1

            sid = meta.get("source_sample_id")
            for i, ch in enumerate(steps):
                for reason in unbalanced(ch):
                    counts[key][reason] += 1
                    maybe_keep((method, src, reason),
                               {"sid": sid, "chunk_idx": i, "chunk": ch[:400]})
            for i in range(len(steps) - 1):
                for flag in boundary_flags(steps[i], steps[i + 1]):
                    counts[key][flag] += 1
                    maybe_keep((method, src, flag),
                               {"sid": sid, "boundary_after_chunk": i,
                                "prev_tail": steps[i][-200:],
                                "next_head": steps[i + 1][:200]})

            if a.max_lines and lines >= a.max_lines:
                break

    # ------------------------------------------------------------ aggregate

    def agg_by_method():
        m_counts = defaultdict(lambda: defaultdict(int))
        m_totals = defaultdict(lambda: {"multi_records": 0, "single_records": 0,
                                        "chunks": 0, "boundaries": 0})
        for (method, src), fl in counts.items():
            for k, v in fl.items():
                m_counts[method][k] += v
        for (method, src), t in totals.items():
            for k, v in t.items():
                m_totals[method][k] += v
        return m_counts, m_totals

    m_counts, m_totals = agg_by_method()
    all_flags = sorted({fl for d in counts.values() for fl in d})
    methods = sorted(m_totals, key=lambda x: (x != "GOLD", x))

    def rate(c, fl, t):
        denom = t["chunks"] if fl.startswith("unbalanced") else t["boundaries"]
        return (100.0 * c / denom) if denom else 0.0

    print(f"\nLines scanned: {lines:,}   unparseable: {bad_json:,}")
    print("\n" + "=" * 78)
    print("AGGREGATE BY METHOD  (confounded across sources -- read the per-source")
    print("tables below for the comparison that actually controls population)")
    print("=" * 78)
    hdr = f"{'flag':34s}" + "".join(f"{m:>22s}" for m in methods)
    print(hdr)
    print("-" * len(hdr))
    denom_row = f"{'(chunks / boundaries)':34s}"
    for m in methods:
        t = m_totals[m]
        denom_row += f"{t['chunks']:,}c {t['boundaries']:,}b".rjust(22)
    print(denom_row)
    zc_row = f"{'(zero-cut records)':34s}"
    for m in methods:
        t = m_totals[m]
        tot = t["multi_records"] + t["single_records"]
        pct = 100.0 * t["single_records"] / tot if tot else 0.0
        zc_row += f"{t['single_records']:,} ({pct:.2f}%)".rjust(22)
    print(zc_row)
    print("-" * len(hdr))
    for fl in all_flags:
        row = f"{fl:34s}"
        for m in methods:
            c = m_counts[m].get(fl, 0)
            row += f"{c:,} ({rate(c, fl, m_totals[m]):5.2f}%)".rjust(22)
        print(row)

    # ------------------------------------------------------------ per source

    sources = sorted({src for (_, src) in totals})
    print("\n" + "=" * 78)
    print(f"PER-SOURCE  (sources with >= {a.min_source_support} boundaries in some method)")
    print("=" * 78)
    for src in sources:
        present = [m for m in methods if (m, src) in totals]
        support = max((totals[(m, src)]["boundaries"] for m in present), default=0)
        if support < a.min_source_support:
            continue
        print(f"\n--- {src} ---")
        hdr = f"{'flag':34s}" + "".join(f"{m:>22s}" for m in present)
        print(hdr)
        d_row = f"{'(chunks / boundaries)':34s}"
        for m in present:
            t = totals[(m, src)]
            d_row += f"{t['chunks']:,}c {t['boundaries']:,}b".rjust(22)
        print(d_row)
        for fl in all_flags:
            if not any(counts[(m, src)].get(fl) for m in present):
                continue
            row = f"{fl:34s}"
            for m in present:
                c = counts[(m, src)].get(fl, 0)
                row += f"{c:,} ({rate(c, fl, totals[(m, src)]):5.2f}%)".rjust(22)
            print(row)

    # ------------------------------------------------------------ dumps

    summary = {
        "path": a.path,
        "lines_scanned": lines,
        "unparseable": bad_json,
        "by_method": {m: {"totals": m_totals[m], "counts": dict(m_counts[m])}
                      for m in methods},
        "by_method_source": {f"{m}||{s}": {"totals": totals[(m, s)],
                                           "counts": dict(counts[(m, s)])}
                             for (m, s) in totals},
    }
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    n_ex = 0
    with open(os.path.join(a.outdir, "examples.jsonl"), "w", encoding="utf-8") as f:
        for (method, src, fl), exs in examples.items():
            for e in exs:
                e.update({"method": method, "source": src, "flag": fl})
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
                n_ex += 1

    print(f"\nWrote {a.outdir}/summary.json and {a.outdir}/examples.jsonl ({n_ex} examples).")
    print("Eyeball mid_sentence_pair and unbalanced_latex first, comparing each")
    print("produced method against GOLD within the same source.")


if __name__ == "__main__":
    main()
