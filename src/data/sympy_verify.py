#!/usr/bin/env python
"""Part 2 of the SymPy verification study: symbolic verifier prototype.

Verifies extracted equality claims per trace:
- GROUND claims (no free symbols): sympy.simplify(lhs - rhs) == 0, exact
  (decimals become Rationals at parse time; no float tolerance). The numeric
  magnitude of a nonzero difference is recorded so rounding near-misses can
  be distinguished from gross errors in reporting (verdicts stay exact).
- CONSTRAINT claims (free symbols): Method 2 (default) substitutes the
  trace's final asserted value(s) for the unknown(s) into every constraint;
  each must reduce to a true statement. On clean single-variable linear
  chains, Method 1 (solveset equality of adjacent equations) also runs and
  records which transition breaks.

Trace verdicts:
  SKIPPED_NO_SURFACE  source outside the three justified tiers (never PASS)
  EXPLORATORY         <think>-style / self-correcting trace; reported apart
  PASS                all checks true (or nothing extractable, tiered source)
  FAIL                >=1 false claim not explained by prose-trimming
  FAIL_LOW_CONFIDENCE every failing claim came from a prose-trimmed side

Unparseable steps/claims are SKIPPED, never failed. Trace-level only.
Every sympy call runs under a 5 s SIGALRM guard.
"""

import argparse
import json
import re
import signal
from collections import defaultdict
from pathlib import Path

import sympy
from sympy import Eq, S, simplify, solveset

from sympy_extract_audit import (  # noqa: F401  (same-dir import)
    extract_claims, parse_side, normalize, ParseTimeout, _alarm_handler,
)

# --- tier scoping (condition 2): only these sources are verified ------------
TIERS = {
    "function": ("mavis_function_",),
    "geometry": ("geo170k", "geometry3k", "geomverse", "geoqa+", "geos_en",
                 "unigeo_calc", "mavis_geo_depth"),
    "arithmetic": ("CLEVR_math", "MathV360K_prompts", "dvqa_en"),
}


def source_tier(source_file):
    for tier, pats in TIERS.items():
        if any(p in source_file for p in pats):
            return tier
    return None


# --- exploratory-trace detection (known limitation guard) -------------------
EXPLORATORY_RE = re.compile(
    r"<think>|\bwait\b|let me reconsider|let me rethink|let me re-examine|"
    r"i made a mistake|on second thought|let's try again|that's not right|"
    r"let me correct",
    re.IGNORECASE,
)


def is_exploratory(steps):
    return any(EXPLORATORY_RE.search(s) for s in steps)


# --- guarded sympy ops ------------------------------------------------------

def _with_timeout(fn, timeout_s=5):
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        return fn(), None
    except ParseTimeout:
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)


def _diff_detail(diff, lhs, rhs):
    """Absolute + relative magnitude of a nonzero difference, for reporting
    (rounding-vs-gross distinction; the exact verdict is unchanged)."""
    try:
        mag = float(abs(diff.evalf()))
        scale = max(1.0, float(abs(lhs.evalf())), float(abs(rhs.evalf())))
        return f"diff={sympy.sstr(diff)[:80]} (|diff|~{mag:.4g}, rel~{mag/scale:.4g})"
    except Exception:
        return f"diff={sympy.sstr(diff)[:80]}"


def check_ground(lhs, rhs):
    """Return (True/False/None, detail). None = could not decide (timeout/err)."""
    def run():
        diff = lhs - rhs
        if diff.is_Number:
            return diff
        return simplify(diff)
    diff, err = _with_timeout(run)
    if diff is None:
        return None, err
    if diff == 0:
        return True, None
    return False, _diff_detail(diff, lhs, rhs)


# --- per-trace verification -------------------------------------------------

SELF_EQUAL_SKIP = "SELF_EQUAL"  # x = x style tautology after parsing


def collect_claims(steps):
    """Extract + parse claims for all steps.

    Returns list of dicts: step_idx, lhs_s, rhs_s, trimmed, lhs, rhs (exprs or
    None), kind (GROUND/CONSTRAINT/PARSE_FAIL).
    """
    out = []
    for i, step in enumerate(steps):
        claims, _, _ = extract_claims(step)
        for lhs_s, rhs_s, trimmed in claims:
            lhs, e1 = parse_side(lhs_s)
            rhs, e2 = parse_side(rhs_s) if lhs is not None else (None, None)
            if lhs is None or rhs is None:
                kind = "FUNC_SKIP" if "FUNC_CALL" in (e1, e2) else "PARSE_FAIL"
                out.append(dict(step_idx=i, lhs_s=lhs_s, rhs_s=rhs_s, trimmed=trimmed,
                                lhs=None, rhs=None, kind=kind, err=e1 or e2))
                continue
            try:
                free = lhs.free_symbols | rhs.free_symbols
            except Exception:
                out.append(dict(step_idx=i, lhs_s=lhs_s, rhs_s=rhs_s, trimmed=trimmed,
                                lhs=None, rhs=None, kind="PARSE_FAIL",
                                err="free_symbols failed"))
                continue
            out.append(dict(step_idx=i, lhs_s=lhs_s, rhs_s=rhs_s, trimmed=trimmed,
                            lhs=lhs, rhs=rhs,
                            kind="GROUND" if not free else "CONSTRAINT"))
    return out


def final_values(claims, answer_text, used_degrees):
    """Determine asserted value(s) for unknown symbols (Method 2 anchors).

    Every 'x = <numeric>' style claim contributes a candidate value (traces
    with multi-branch solutions, e.g. |...| equations with 'x = 6 or x = -1',
    legitimately assert several); a constraint later passes if ANY candidate
    assignment satisfies it. If exactly one unknown has no candidates and the
    answer field is numeric, the answer anchors it; if the trace used degree
    conversion, the degree-interpreted answer joins the candidate list.
    """
    values = {}  # Symbol -> list of distinct candidate exprs (trace order)

    def add(sym, val):
        values.setdefault(sym, [])
        if val not in values[sym]:
            values[sym].append(val)

    for c in claims:
        if c["kind"] != "CONSTRAINT":
            continue
        lhs, rhs = c["lhs"], c["rhs"]
        if lhs.is_Symbol and not rhs.free_symbols:
            add(lhs, rhs)
        elif rhs.is_Symbol and not lhs.free_symbols:
            add(rhs, lhs)
    unknowns = set()
    for c in claims:
        if c["kind"] == "CONSTRAINT":
            unknowns |= c["lhs"].free_symbols | c["rhs"].free_symbols
    unanchored = unknowns - set(values)
    if len(unanchored) == 1 and answer_text is not None:
        norm = normalize(str(answer_text)).strip()
        expr, err = parse_side(norm)
        if expr is not None and not expr.free_symbols:
            sym = next(iter(unanchored))
            add(sym, expr)
            if used_degrees and "pi" not in norm:
                add(sym, expr * sympy.pi / 180)
    return values


MAX_ANCHOR_COMBOS = 8


def check_constraints(claims, values):
    """Substitute anchor candidates into every CONSTRAINT claim.

    A claim is TRUE if ANY combination of per-symbol candidate values
    (cartesian product, capped at MAX_ANCHOR_COMBOS; falls back to
    last-asserted values beyond the cap) reduces it to a true statement;
    FALSE only if every fully-resolved combination is false.

    Returns list of (claim, result, detail): result in
    TRUE / FALSE / UNRESOLVED (free symbols remain) / TIMEOUT / SELF_EQUAL.
    """
    import itertools

    results = []
    for c in claims:
        if c["kind"] != "CONSTRAINT":
            continue
        lhs, rhs = c["lhs"], c["rhs"]
        if lhs == rhs:
            results.append((c, SELF_EQUAL_SKIP, None))
            continue
        syms = sorted((lhs.free_symbols | rhs.free_symbols) & set(values),
                      key=sympy.sstr)
        cand_lists = [values[s] for s in syms]
        n_combos = 1
        for cl in cand_lists:
            n_combos *= len(cl)
        if n_combos > MAX_ANCHOR_COMBOS:
            cand_lists = [cl[-1:] for cl in cand_lists]

        def run():
            best = None  # (result, detail); prefer TRUE > UNRESOLVED > FALSE
            for combo in itertools.product(*cand_lists) if syms else [()]:
                sub = dict(zip(syms, combo))
                d = (lhs - rhs).subs(sub)
                if d.free_symbols:
                    if best is None or best[0] == "FALSE":
                        best = ("UNRESOLVED", None)
                    continue
                d = d if d.is_Number else simplify(d)
                if d == 0:
                    return "TRUE", None
                if best is None:
                    best = ("FALSE", _diff_detail(d, lhs.subs(sub), rhs.subs(sub)))
            return best if best is not None else ("UNRESOLVED", None)
        res, err = _with_timeout(run)
        if res is None:
            results.append((c, "TIMEOUT", err))
        else:
            results.append((c, res[0], res[1]))
    return results


def method1_linear_chain(claims):
    """On clean single-variable linear chains: solveset equality of adjacent
    constraint equations; returns index (into constraint list) of the first
    breaking transition, or None if chain is consistent / not clean."""
    cons = [c for c in claims if c["kind"] == "CONSTRAINT"]
    if len(cons) < 2:
        return None, False
    syms = set()
    for c in cons:
        syms |= c["lhs"].free_symbols | c["rhs"].free_symbols
    if len(syms) != 1:
        return None, False
    x = next(iter(syms))
    for c in cons:
        for side in (c["lhs"], c["rhs"]):
            try:
                p = sympy.Poly(side, x)
                if p.degree() > 1:
                    return None, False
            except Exception:
                return None, False

    def run():
        prev = None
        for i, c in enumerate(cons):
            sol = solveset(Eq(c["lhs"], c["rhs"]), x, domain=S.Complexes)
            if prev is not None and sol != prev[1]:
                return i
            prev = (i, sol)
        return None
    brk, err = _with_timeout(run)
    return brk, True


def verify_trace(rec):
    """Full verdict for one record. Returns a JSON-serializable dict."""
    src = rec["source_file"]
    tier = source_tier(src)
    base = dict(source_file=src, line_index=rec.get("line_index"), tier=tier)
    for k in ("max_p_contra", "min_score", "mean_score"):
        if k in rec:
            base[k] = rec[k]
    if tier is None:
        return dict(base, verdict="SKIPPED_NO_SURFACE", n_claims=0)
    steps = rec["steps"]
    if is_exploratory(steps):
        return dict(base, verdict="EXPLORATORY", n_claims=0)

    claims = collect_claims(steps)
    n_parse_fail = sum(1 for c in claims if c["kind"] == "PARSE_FAIL")
    n_func_skip = sum(1 for c in claims if c["kind"] == "FUNC_SKIP")
    ok_claims = [c for c in claims if c["kind"] in ("GROUND", "CONSTRAINT")]
    used_degrees = any("pi/180" in c["lhs_s"] + c["rhs_s"] for c in ok_claims)

    failing = []       # (claim, detail, check_kind)
    n_true = 0
    n_timeout = 0
    n_unresolved = 0

    for c in ok_claims:
        if c["kind"] != "GROUND":
            continue
        res, det = check_ground(c["lhs"], c["rhs"])
        if res is True:
            n_true += 1
        elif res is False:
            failing.append((c, det, "ground"))
        else:
            n_timeout += 1

    values = final_values(ok_claims, rec.get("answer"), used_degrees)
    cons_results = check_constraints(ok_claims, values)
    cons_failed = [(c, det) for c, res, det in cons_results if res == "FALSE"]
    for c, res, det in cons_results:
        if res == "TRUE":
            n_true += 1
        elif res == "TIMEOUT":
            n_timeout += 1
        elif res == "UNRESOLVED":
            n_unresolved += 1
    for c, det in cons_failed:
        failing.append((c, det, "constraint"))

    m1_break, m1_ran = method1_linear_chain(ok_claims)

    if not failing:
        verdict = "PASS"
    elif all(c["trimmed"] for c, _, _ in failing):
        verdict = "FAIL_LOW_CONFIDENCE"
    else:
        verdict = "FAIL"
    return dict(
        base, verdict=verdict,
        n_claims=len(claims), n_parse_fail=n_parse_fail, n_func_skip=n_func_skip,
        n_ground=sum(1 for c in ok_claims if c["kind"] == "GROUND"),
        n_constraint=sum(1 for c in ok_claims if c["kind"] == "CONSTRAINT"),
        n_true=n_true, n_timeout=n_timeout, n_unresolved=n_unresolved,
        anchors={sympy.sstr(k): [sympy.sstr(x) for x in v] for k, v in values.items()},
        method1_ran=m1_ran, method1_break_at=m1_break,
        failing=[dict(step_idx=c["step_idx"], claim=f"{c['lhs_s']} = {c['rhs_s']}",
                      trimmed=c["trimmed"], kind=kind, detail=det)
                 for c, det, kind in failing],
    )


# --- runner -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="verdicts jsonl path")
    ap.add_argument("--per-source", type=int, default=0,
                    help="stratified sample size per source (0 = all traces)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    print(f"sympy {sympy.__version__}; input={args.input}", flush=True)
    if args.per_source:
        from sympy_extract_audit import stratified_sample
        samples, _ = stratified_sample(args.input, args.per_source, args.seed)
        records = [r for src in sorted(samples) for r in samples[src]]
    else:
        with open(args.input) as f:
            records = [json.loads(line) for line in f]
    print(f"{len(records)} traces to verify", flush=True)

    counts = defaultdict(int)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for i, rec in enumerate(records):
            try:
                v = verify_trace(rec)
            except Exception as e:
                v = dict(source_file=rec.get("source_file"),
                         line_index=rec.get("line_index"),
                         verdict="ERROR", error=f"{type(e).__name__}: {e}"[:200])
            counts[v["verdict"]] += 1
            out.write(json.dumps(v) + "\n")
            if (i + 1) % args.progress_every == 0:
                print(f"{i+1}/{len(records)} {dict(counts)}", flush=True)
    print(f"DONE {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
