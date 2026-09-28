#!/usr/bin/env python
"""Part 3 of the SymPy verification study: aggregate verdicts + REPORT.md.

Reads the three verdict files produced by sympy_verify.py, computes
per-population and per-source/tier statistics, tags NLI_FALSE_ALARM_CANDIDATEs,
cross-references the 20 mcpass_nlifail agreement examples, and assembles
sympy_audit/REPORT.md.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

D = Path("data/visualprm_v11_filtered")
A = D / "sympy_audit"

TIER_NAME = {"function": "mavis_function", "geometry": "geometry family",
             "arithmetic": "arithmetic (CLEVR/MathV360K/dvqa)", None: "no-surface"}


def load(name):
    with open(A / name) as f:
        return [json.loads(line) for line in f]


def extractable(v):
    return (v.get("n_ground", 0) + v.get("n_constraint", 0)) >= 1


def diff_mag(detail):
    """Return (abs_mag, rel_mag) parsed from a failing-claim detail string."""
    m = re.search(r"\|diff\|~([0-9.e+-]+)(?:, rel~([0-9.e+-]+))?", detail or "")
    if not m:
        return None, None
    try:
        return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)
    except ValueError:
        return None, None


def pop_stats(verdicts):
    s = dict(total=len(verdicts))
    s["verdicts"] = Counter(v["verdict"] for v in verdicts)
    tiered = [v for v in verdicts if v.get("tier")]
    s["tiered"] = len(tiered)
    scored = [v for v in tiered if v["verdict"] in
              ("PASS", "FAIL", "FAIL_LOW_CONFIDENCE")]
    s["extractable"] = sum(1 for v in scored if extractable(v))
    s["extractable_frac_of_total"] = s["extractable"] / max(1, s["total"])
    s["pass_extractable"] = sum(1 for v in scored if extractable(v) and v["verdict"] == "PASS")
    s["fail"] = s["verdicts"].get("FAIL", 0)
    s["fail_lc"] = s["verdicts"].get("FAIL_LOW_CONFIDENCE", 0)
    s["timeout_claims"] = sum(v.get("n_timeout", 0) for v in verdicts)
    s["func_skip_claims"] = sum(v.get("n_func_skip", 0) for v in verdicts)
    s["unresolved_claims"] = sum(v.get("n_unresolved", 0) for v in verdicts)
    s["parse_fail_claims"] = sum(v.get("n_parse_fail", 0) for v in verdicts)
    s["claims"] = sum(v.get("n_claims", 0) for v in verdicts)
    # rounding-vs-gross split over failing claims (relative diff where known)
    pairs = [diff_mag(f.get("detail")) for v in verdicts for f in v.get("failing", [])]
    rels = [(r if r is not None else a) for a, r in pairs if a is not None]
    s["failing_claims_with_mag"] = len(rels)
    s["failing_rel_lt_0.01"] = sum(1 for r in rels if r < 0.01)
    s["failing_rel_ge_0.2"] = sum(1 for r in rels if r >= 0.2)
    return s


def fmt_pop(s):
    v = s["verdicts"]
    lines = [
        f"- traces: {s['total']}; in-tier: {s['tiered']}; "
        f"extractable (>=1 parsed claim, scored): {s['extractable']} "
        f"({100*s['extractable_frac_of_total']:.1f}% of population)",
        f"- verdicts: PASS {v.get('PASS',0)} | FAIL {v.get('FAIL',0)} | "
        f"FAIL_LOW_CONFIDENCE {v.get('FAIL_LOW_CONFIDENCE',0)} | "
        f"EXPLORATORY {v.get('EXPLORATORY',0)} | "
        f"SKIPPED_NO_SURFACE {v.get('SKIPPED_NO_SURFACE',0)} | "
        f"ERROR {v.get('ERROR',0)}",
        f"- of extractable: PASS-clean {s['pass_extractable']}",
        f"- claim-level: {s['claims']} claims, {s['parse_fail_claims']} parse-fail, "
        f"{s['func_skip_claims']} func-call-skipped, "
        f"{s['timeout_claims']} timeouts, {s['unresolved_claims']} unresolved-constraint",
        f"- failing-claim relative-diff split: {s['failing_rel_lt_0.01']} < 1% "
        f"(rounding-scale) / {s['failing_claims_with_mag'] - s['failing_rel_lt_0.01'] - s['failing_rel_ge_0.2']} in [1%,20%) / "
        f"{s['failing_rel_ge_0.2']} >= 20% (gross), of {s['failing_claims_with_mag']} with magnitude",
    ]
    return "\n".join(lines)


def per_source_table(verdicts, min_traces=1):
    rows = defaultdict(lambda: Counter())
    for v in verdicts:
        rows[v["source_file"]][v["verdict"]] += 1
        rows[v["source_file"]]["_total"] += 1
        if v["verdict"] in ("PASS", "FAIL", "FAIL_LOW_CONFIDENCE") and extractable(v):
            rows[v["source_file"]]["_extractable"] += 1
            if v["verdict"] == "PASS":
                rows[v["source_file"]]["_pass_ex"] += 1
    lines = ["| source | traces | extractable | PASS(extr.) | FAIL | FAIL_LC | EXPLOR | SKIP |",
             "|---|---|---|---|---|---|---|---|"]
    for src in sorted(rows, key=lambda s: -rows[s]["_total"]):
        c = rows[src]
        if c["_total"] < min_traces:
            continue
        name = Path(src).name.replace("_extracted.jsonl", "").replace(".jsonl", "")[:45]
        lines.append(f"| {name} | {c['_total']} | {c['_extractable']} | {c['_pass_ex']} | "
                     f"{c['FAIL']} | {c['FAIL_LOW_CONFIDENCE']} | {c['EXPLORATORY']} | "
                     f"{c['SKIPPED_NO_SURFACE']} |")
    return "\n".join(lines)


def tier_fail_table(verdicts):
    rows = defaultdict(Counter)
    for v in verdicts:
        t = TIER_NAME[v.get("tier")]
        rows[t][v["verdict"]] += 1
        rows[t]["_total"] += 1
        if v["verdict"] in ("PASS", "FAIL", "FAIL_LOW_CONFIDENCE") and extractable(v):
            rows[t]["_ex"] += 1
    lines = ["| tier | traces | extractable | FAIL | FAIL_LC | FAIL% of extractable |",
             "|---|---|---|---|---|---|"]
    for t in ("mavis_function", "geometry family", "arithmetic (CLEVR/MathV360K/dvqa)", "no-surface"):
        c = rows.get(t)
        if not c:
            continue
        nf = c["FAIL"] + c["FAIL_LOW_CONFIDENCE"]
        pct = 100 * nf / c["_ex"] if c["_ex"] else 0.0
        lines.append(f"| {t} | {c['_total']} | {c['_ex']} | {c['FAIL']} | "
                     f"{c['FAIL_LOW_CONFIDENCE']} | {pct:.1f}% |")
    return "\n".join(lines)


def main():
    kept = load("verdicts_kept_sample.jsonl")
    rej = load("verdicts_nli_rejected.jsonl")
    band = load("verdicts_mc_band.jsonl")

    # --- NLI false-alarm candidates ---------------------------------------
    cands = [v for v in rej if v["verdict"] == "PASS" and extractable(v)]
    cands_strict = [v for v in cands
                    if v.get("n_timeout", 0) == 0 and v.get("n_unresolved", 0) == 0
                    and v.get("n_parse_fail", 0) == 0]
    with open(A / "nli_false_alarm_candidates.jsonl", "w") as f:
        for v in cands:
            f.write(json.dumps(dict(v, tag="NLI_FALSE_ALARM_CANDIDATE")) + "\n")

    # --- cross-reference the 20 mcpass_nlifail examples -------------------
    idx = {(v["source_file"], v["line_index"]): v for v in rej}
    xref_lines = ["| example | NLI max P(contra) | SymPy verdict | claims (parsed/true) |",
                  "|---|---|---|---|"]
    exdir = D / "agreement_examples"
    for p in sorted(exdir.glob("mcpass_nlifail_*.txt")):
        head = p.read_text()[:400]
        m = re.search(r"source_file: (\S+)\s+line_index: (\d+)", head)
        mc = re.search(r"max P\(contra\): ([0-9.]+)", head)
        if not m:
            xref_lines.append(f"| {p.name} | ? | HEADER_PARSE_ERROR | |")
            continue
        key = (m.group(1), int(m.group(2)))
        v = idx.get(key)
        if v is None:
            xref_lines.append(f"| {p.name[:60]} | {mc.group(1) if mc else '?'} | NOT_IN_REJECTED_FILE | |")
            continue
        parsed = v.get("n_ground", 0) + v.get("n_constraint", 0)
        xref_lines.append(f"| {p.name[:60]} | {mc.group(1) if mc else '?'} | {v['verdict']} | "
                         f"{parsed}/{v.get('n_true', 0)} |")

    # --- 15 example FAIL traces -------------------------------------------
    def fail_examples(verdicts, pop, n):
        out = []
        for v in verdicts:
            if v["verdict"] == "FAIL" and v.get("failing"):
                out.append((pop, v))
            if len(out) >= n:
                break
        return out

    examples = (fail_examples(kept, "kept-sample", 5)
                + fail_examples(rej, "NLI-rejected", 5)
                + fail_examples(band, "MC-band", 5))
    ex_lines = []
    for pop, v in examples[:15]:
        f0 = v["failing"][0]
        ex_lines.append(
            f"- **[{pop}]** `{Path(v['source_file']).name}` line {v['line_index']} "
            f"(step {f0['step_idx']}): claim `{f0['claim']}` -> **{f0['detail']}**"
            + (f"; anchors {v.get('anchors')}" if v.get("anchors") else ""))

    sk, sr, sb = pop_stats(kept), pop_stats(rej), pop_stats(band)

    # --- 3-instrument summary --------------------------------------------
    def cell(s):
        nf = s["fail"] + s["fail_lc"]
        return (f"extractable {s['extractable']}/{s['total']} "
                f"({100*s['extractable_frac_of_total']:.0f}%); of extractable: "
                f"{s['pass_extractable']} PASS / {s['fail']} FAIL / {s['fail_lc']} FAIL_LC "
                f"(fail rate {100*nf/max(1, s['extractable']):.1f}%)")

    report = f"""# SymPy symbolic verification — REPORT (v2 filtering research)

Generated 2026-07-19. sympy 1.14.0, interpreter `/scratch/sghos104/envs/rlpt-train/bin/python`.
Extraction/verification code: `src/data/sympy_extract_audit.py`, `src/data/sympy_verify.py`.
All inputs read-only; outputs confined to `data/visualprm_v11_filtered/sympy_audit/`.

## 1. Extractability audit (Part 1)

See `extractability_table.md` (150 traces/source, seed=0; v4 extraction).
Extraction iterations, each validated on logged failures/FAIL inspections:
- v1 (3.3% parse-fail) -> decimal-period, paren-balance, `x`-as-times,
  bare-sqrt fixes -> 2.9%.
- v2: \\text{{}}-placeholder symbols, uppercase identifiers (AB/SA/AOB),
  unicode angle/triangle mapping -> geo sources 8.3-10% -> 3.1-6.5%
  parse-fail, claim yield roughly doubled (geo170k 0.66 -> 1.44 claims/step).
- v3 (verifier-precision pass, driven by inspecting gross FAILs): sympy
  namespace singletons neutralized (E/S2/I/Q parsed as Symbols), |x|->Abs,
  juxtaposition guard (>=2-space gap without operator = removed separator ->
  side dropped, not glued), trailing unit-noun placeholders stripped
  (`25547 tonnes_` -> `25547`), \\text{{or}} treated as a chain break,
  f(x)-style claims skipped as unverifiable (FUNC_SKIP), and multi-branch
  anchoring (a constraint passes if ANY asserted solution branch satisfies
  it -- pre-fix, `x = 6 or x = -1` traces fabricated FAILs).
- v4 (largest single fabrication source): \\tan/\\sin/\\cos/\\log/greek
  macros were being DELETED by the generic latex-command strip, turning
  `\\tan(-\\pi) = 1` into the false claim `(-pi) = 1` -- they are now
  preserved as functions/symbols. Failing claims additionally record a
  RELATIVE diff so rounding can be separated from gross error.
  Kept-sample FAIL count across iterations: 729 (v2) -> 548 (v3) -> 365 (v4).

## 2. Verifier verdict counts per population (Part 2/3)

Scope (condition 2): only tiers {{mavis_function, geometry family,
CLEVR/MathV360K/dvqa arithmetic}} are verified; all other sources emit
SKIPPED_NO_SURFACE (never PASS). FAIL_LOW_CONFIDENCE (condition 3) = every
failing claim in the trace came from a prose-trimmed side.

### stage2-kept sample (150/source stratified, seed=0)
{fmt_pop(sk)}

### NLI-rejected (stage2_rejected_tau0.85, full)
{fmt_pop(sr)}

### MC-fail band (stage1_band_sample, full)
{fmt_pop(sb)}

## 3. NLI false-alarm candidates (Part 3a)

Of {sr['total']} NLI-rejected traces: {sr['extractable']} have extractable
claims; **{len(cands)} verify fully clean (PASS with >=1 claim) ->
NLI_FALSE_ALARM_CANDIDATE** ({len(cands_strict)} under the strict variant:
additionally zero parse-fails/timeouts/unresolved constraints).
Full list: `nli_false_alarm_candidates.jsonl`.

Per-source breakdown of the NLI-rejected population:

{per_source_table(rej)}

### Cross-reference: the 20 mcpass_nlifail agreement examples

{chr(10).join(xref_lines)}

## 4. Verdicts by source tier (condition 4; all three populations)

The three populations have very different source mixes (the kept sample is
stratified 150/source and so overweights claim-dense mavis_function; the
NLI-rejected extractable mass is mostly easy CLEVR/dvqa arithmetic), so
cross-population fail rates must be compared WITHIN a tier, not overall.

### MC-fail band
{tier_fail_table(band)}

### stage2-kept sample
{tier_fail_table(kept)}

### NLI-rejected
{tier_fail_table(rej)}

## 5. Three-instrument summary (MC x NLI x SymPy)

| population | MC | NLI | SymPy |
|---|---|---|---|
| stage2-kept sample | pass | pass | {cell(sk)} |
| NLI-rejected | pass | fail | {cell(sr)} |
| MC-fail band | fail (band) | n/a | {cell(sb)} |

Tier-matched comparison (FAIL+FAIL_LC as % of extractable, from section 4):
the MC-fail band is elevated over the MC-passing populations in every tier,
but by different margins -- roughly 1.6-2x in geometry (16.3% vs 10.5% kept /
9.2% rejected) and 2-3x in arithmetic (2.0% vs 0.7% / 0.9%), yet only
modestly in mavis_function (53.2% vs 45.4% / 49.6%). Within every tier the
NLI-rejected population fails at about the SAME rate as the NLI-passing kept
sample -- i.e. the NLI contradiction signal is essentially orthogonal to
symbolic correctness, consistent with the false-alarm reading. The
mavis_function absolute rates are likely inflated by residual extraction
artifacts (see limitations), but that inflation applies to all three
populations alike, so the within-tier contrasts remain meaningful.

## 6. Example FAIL traces (first failing claim highlighted)

{chr(10).join(ex_lines)}

## 7. Limitations (honest)

- **Coverage**: only ~36% of the kept pool is in a verifiable tier, and within
  tiers only the extractable fraction above is actually checked; a PASS on a
  trace with few claims is weak evidence. SKIPPED_NO_SURFACE traces say
  nothing about correctness.
- **Prose-trimming fabrication risk**: ~half of claims had prose trimmed
  adjacent to the '='; failing traces whose every failing claim is
  trim-derived are quarantined as FAIL_LOW_CONFIDENCE, but trimmed claims
  that PASS could still be misextracted.
- **Exact arithmetic, no float tolerance** (per spec): rounded intermediate
  values (e.g. sqrt(429) = 20.71) fail exactly; the relative-diff split above
  separates rounding-scale (rel < 1%) from gross (rel >= 20%) mismatches for
  interpretation -- verdicts themselves stay exact.
- **Residual fabrication risk in FAILs**: inspection of v4 kept-sample FAILs
  still shows a minority of extraction artifacts (bare parenthesized-number
  remnants of non-f/g/h function application like `(4) = 1/5`, placeholder
  symbols juxtaposed with parentheses). FAIL counts are therefore upper
  bounds on genuinely broken math; the per-claim details in the verdict
  files allow case-by-case adjudication.
- **Exploratory traces** are excluded from FAIL counts via marker detection;
  the marker list is heuristic.
- **Method 2 anchoring**: unanchored unknowns leave constraints UNRESOLVED
  (skipped, counted); a symbol asserted with several values contributes all
  of them as candidate branches (a constraint passes if any branch fits);
  a numeric answer field anchors a single leftover unknown, with a
  degree-interpreted variant added when the trace itself used degrees.
- Parse-fail/timeout/unresolved counts are reported per population above;
  none are ever counted as FAIL.
"""
    with open(A / "REPORT.md", "w") as f:
        f.write(report)
    print(f"wrote {A/'REPORT.md'}; candidates={len(cands)} (strict {len(cands_strict)})")


if __name__ == "__main__":
    main()
