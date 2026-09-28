#!/usr/bin/env python
"""Part 1 of the SymPy verification study: extractability audit.

Stratified-samples traces per source_file from stage2_kept_tau0.85.jsonl,
normalizes each reasoning step (LaTeX/unicode/percent/commas/degrees),
extracts equality-chain claims, classifies them GROUND (no free symbols)
vs CONSTRAINT (free symbols), and reports per-source extractability.

Read-only on inputs; writes only under data/visualprm_v11_filtered/sympy_audit/.
"""

import argparse
import json
import random
import re
import signal
from collections import defaultdict
from pathlib import Path

import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    rationalize,
)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application, convert_xor, rationalize)

# Words allowed inside a math expression (sympy functions/constants).
MATH_WORDS = {
    "sqrt", "sin", "cos", "tan", "sec", "csc", "cot",
    "asin", "acos", "atan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh",
    "log", "ln", "exp", "abs", "pi", "Abs", "floor", "ceil",
}

# Greek letter names survive as symbols (not prose, not sympy functions).
GREEK_WORDS = {
    "alpha", "beta", "gamma", "delta", "theta", "phi", "varphi", "omega",
    "mu", "nu", "rho", "sigma", "tau", "epsilon", "kappa", "eta", "xi",
    "zeta", "iota", "chi", "psi", "lamda",
    "Omega", "Delta", "Theta", "Phi", "Gamma", "Sigma",
}

# \text{...} content that is pure units/currency gets dropped; anything else
# becomes a placeholder symbol (see _text_macro_repl).
UNIT_WORDS = {
    "cm", "mm", "m", "km", "meter", "meters", "metre", "metres",
    "inch", "inches", "in", "ft", "feet", "foot", "yd", "yard", "yards",
    "mi", "mile", "miles", "s", "sec", "secs", "second", "seconds",
    "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour", "hours",
    "kg", "g", "gram", "grams", "mg", "lb", "lbs", "pound", "pounds", "oz",
    "l", "liter", "liters", "litre", "litres", "ml",
    "unit", "units", "square", "sq", "cubic", "cu",
    "degree", "degrees", "deg", "radian", "radians", "rad", "percent",
    "cm2", "cm3", "m2", "m3", "km2", "rm", "usd", "dollar", "dollars",
    "cent", "cents", "sqrt",
}


CONNECTIVE_WORDS = {"or", "and", "either", "otherwise", "if", "then", "so", "thus"}


def _text_macro_repl(m):
    """\\text{BC} -> 'BC_' placeholder symbol; \\text{cm} / \\text{sq units} -> dropped;
    \\text{ or } -> chain break (multi-branch solutions must not be glued).

    The trailing underscore marks the token as math so prose-trimming keeps it.
    """
    words = re.findall(r"[A-Za-z0-9]+", m.group(1))
    if not words or all(w.lower() in UNIT_WORDS for w in words):
        return " "
    if all(w.lower() in CONNECTIVE_WORDS for w in words):
        return " ; "
    return " " + "_".join(words) + "_ "


UNICODE_MAP = {
    "×": "*",   # ×
    "÷": "/",   # ÷
    "−": "-",   # − minus sign
    "⋅": "*",   # ⋅
    "²": "**2",
    "³": "**3",
    "⁴": "**4",
    "√": "sqrt",  # √ (usually √x -> sqrt x, implicit-mult handles sqrt(x) poorly; see below)
    "π": "pi",    # π
    "½": "(1/2)",
    "¼": "(1/4)",
    "¾": "(3/4)",
    "θ": " theta ", "α": " alpha ", "β": " beta ", "γ": " gamma ",
    "φ": " phi ", "ω": " omega ", "μ": " mu ", "λ": " lamda ",
    "σ": " sigma ", "τ": " tau ", "ρ": " rho ", "ε": " epsilon ",
    "Δ": " Delta ",
    "∠": " angle_",   # ∠ABC -> angle_ABC symbol
    "△": " tri_",     # △ABC -> tri_ABC symbol
    "⊙": " circ_",
    "∽": " ; ",       # similarity/congruence/perp/parallel: relation, breaks span
    "≅": " ; ",
    "⊥": " ; ",
    "∥": " ; ",
    "–": "-",   # – en dash (breaks spans harmlessly if misused)
    "—": " ",   # — em dash
}


class ParseTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ParseTimeout()


def _loop_sub(pattern, repl, s, max_iter=10):
    """Apply re.sub repeatedly to handle nested constructs (innermost-first)."""
    for _ in range(max_iter):
        new = re.sub(pattern, repl, s)
        if new == s:
            return new
        s = new
    return s


def normalize(text):
    """Normalize a reasoning-step string toward sympy-parseable notation."""
    s = text
    # LaTeX math delimiters -> plain
    s = s.replace("\\[", " ").replace("\\]", " ")
    s = s.replace("\\(", " ").replace("\\)", " ")
    s = s.replace("$$", " ").replace("$", " ")
    # Macro substitutions to a joint fixpoint so nested constructs like
    # \frac{\sqrt{3}}{2} resolve innermost-first across macro types.
    for _ in range(10):
        prev = s
        s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}", _text_macro_repl, s)
        s = re.sub(r"\\sqrt\[([^\[\]]+)\]\{([^{}]+)\}", r"((\2)**(1/(\1)))", s)
        s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
        if s == prev:
            break
    # \frac ab shorthand (single tokens)
    s = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", r"((\1)/(\2))", s)
    # operators / constants
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\left", "").replace("\\right", "")
    # trig/log macros and greek letters must survive the generic
    # latex-command strip below (else \tan(-\pi)=1 fabricates (-pi)=1)
    s = re.sub(r"\\(arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|sec|csc|cot|log|ln|exp)\b",
               r" \1", s)
    s = re.sub(r"\\lambda\b", " lamda", s)
    s = re.sub(r"\\(alpha|beta|gamma|delta|theta|phi|varphi|omega|mu|nu|rho|sigma|tau"
               r"|epsilon|kappa|eta|xi|zeta|iota|chi|psi|Omega|Delta|Theta|Phi|Gamma|Sigma)\b",
               r" \1 ", s)
    # \approx / ≈ are not equalities: break the span
    s = s.replace("\\approx", " ; ").replace("≈", " ; ")
    # degrees: 120° or 120^\circ or 120^{\circ} -> (120*pi/180)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(?:°|\^\{?\\circ\}?|\\degree)", r"((\1)*pi/180)", s)
    # unicode symbols
    for k, v in UNICODE_MAP.items():
        s = s.replace(k, v)
    # bare "sqrt3" / "sqrt 3" (from unicode √3) -> sqrt(3)
    s = re.sub(r"\bsqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", s)
    # percent: 45% -> (45/100); "45% of 200" -> (45/100)*200
    s = s.replace("\\%", "%")
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"((\1)/100)", s)
    s = re.sub(r"(/100\))\s+of\s+", r"\1*", s)
    # absolute value |x-1| -> Abs(x-1); '=' excluded so markdown table rows
    # (which rarely form claims anyway) don't get glued into one expression
    s = re.sub(r"\|([^|=]{1,40})\|", r"Abs(\1)", s)
    # receipt-style multiplication: "5.90 x 3" -> "5.90 * 3"
    s = re.sub(r"(?<=\d)\s+x\s+(?=\d)", " * ", s)
    # thousands commas: 1,234,567 -> 1234567 (only digit,3-digit-boundary)
    for _ in range(3):
        s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # remaining latex commands -> space; braces -> parens; ^ -> **
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = s.replace("\\", " ")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("^", "**")
    return s


# A candidate math span: run of allowed chars containing '='.
SPAN_RE = re.compile(r"[0-9A-Za-z_+\-*/().=\s]+")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_]+")  # multi-letter words


def _is_prose(w):
    """Prose word = multi-letter, not a math function, not a placeholder symbol.

    Short all-uppercase identifiers (AB, SA, AOB, ADE) are geometry
    segment/angle/triangle names, not prose.
    """
    if w in MATH_WORDS or w in GREEK_WORDS or "_" in w:
        return False
    if w.isupper() and len(w) <= 4:
        return False
    return True


def _strip_annotation_parens(s):
    """Drop parenthesized groups that contain prose words: '1 (initial count) + 7' -> '1  + 7'."""
    def repl(m):
        inner = m.group(1)
        words = [w for w in WORD_RE.findall(inner) if _is_prose(w)]
        return " " if words else m.group(0)
    return _loop_sub(r"\(([^()]*)\)", lambda m: repl(m), s)


def _trim_side(side, keep):
    """Trim prose words off an equality side.

    keep='right': keep rightmost math segment (for LHS - prose is a prefix).
    keep='left' : keep leftmost math segment (for RHS - prose is a suffix).
    Returns '' if nothing math-like remains.
    """
    # positions of prose words (multi-letter, not whitelisted)
    cuts = [m.span() for m in WORD_RE.finditer(side) if _is_prose(m.group(0))]
    if cuts:
        if keep == "right":
            side = side[cuts[-1][1]:]
        else:
            side = side[: cuts[0][0]]
    side = side.strip().strip("+*/= ").strip()
    # sentence-final period (keep decimal points like "6.", fix "17.70.")
    side = re.sub(r"(?<=[^\d])\.$", "", side).strip()
    side = re.sub(r"(\d+\.\d+)\.$", r"\1", side).strip()
    # balance parentheses: spans that cross a paren boundary, e.g. "(s = 0.5)"
    while side.startswith("(") and side.count("(") > side.count(")"):
        side = side[1:].strip()
    while side.endswith(")") and side.count(")") > side.count("("):
        side = side[:-1].strip()
    # a bare function name is not an expression
    if side in MATH_WORDS:
        return ""
    # trailing placeholder after a complete expression is a unit/noun: drop it
    stripped = re.sub(r"(?<=[\d)])\s+[A-Za-z][A-Za-z0-9_]*_\s*$", "", side).strip()
    if stripped:
        side = stripped
    # juxtaposition guard: a >=2-space gap with no operator at the boundary
    # means a separator (comma, \quad, prose) was removed between two
    # independent expressions -- verifying the glued string fabricates claims
    if re.search(r"[^\s+\-*/=(]\s{2,}[^\s+\-*/=)]", side):
        return ""
    # must contain a digit or a letter to be an expression
    if not re.search(r"[0-9A-Za-z]", side):
        return ""
    return side


def extract_claims(step_text):
    """Return (claims, n_spans_with_eq, n_dropped_word_sides).

    claims: list of (lhs_str, rhs_str, trimmed) adjacent pairs from equality
    chains. For the pair around each '=', the left side keeps its trailing
    math segment (prose prefix trimmed) and the right side its leading math
    segment (prose suffix trimmed); `trimmed` flags that prose was removed
    from a side adjacent to this '=' (higher fabrication risk, tracked).
    """
    norm = normalize(step_text)
    claims = []
    n_spans = 0
    n_dropped = 0
    for line in norm.splitlines():
        # markdown list markers ("- 3 (initial) - 1 = 2" bullet, "1." numbering)
        # would otherwise read as a leading minus sign
        line = re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)", " ", line)
        line = _strip_annotation_parens(line)
        for m in SPAN_RE.finditer(line):
            span = m.group(0)
            if "=" not in span:
                continue
            n_spans += 1
            parts = span.split("=")
            for a_raw, b_raw in zip(parts, parts[1:]):
                a = _trim_side(a_raw, "right")
                b = _trim_side(b_raw, "left")
                if a and b:
                    trimmed = (a != a_raw.strip().strip("+*/= ").strip()
                               or b != b_raw.strip().strip("+*/= ").strip())
                    claims.append((a, b, trimmed))
                else:
                    n_dropped += 1
    return claims, n_spans, n_dropped


FUNC_CALL_RE = re.compile(r"\b[fgh]\s*\(")  # f(x)-style application: unverifiable


def parse_side(s, timeout_s=2):
    """Parse one side with sympy under a SIGALRM timeout. Returns (expr, err).

    Every identifier is pre-bound to a plain Symbol (except whitelisted math
    functions and pi) so sympy namespace singletons (E, I, S, N, O, Q, beta,
    gamma, ...) cannot silently evaluate: geometry point names like E or S2
    must stay symbolic.
    """
    if FUNC_CALL_RE.search(s):
        return None, "FUNC_CALL"
    ids = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s))
    local = {i: sympy.Symbol(i) for i in ids if i not in MATH_WORDS and i != "pi"}
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        expr = parse_expr(s, transformations=TRANSFORMS, evaluate=True,
                          local_dict=local)
        return expr, None
    except ParseTimeout:
        return None, "TIMEOUT"
    except Exception as e:  # sympy raises many types incl. TokenError/SyntaxError
        return None, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)


def classify_claim(lhs_s, rhs_s):
    """Return ('GROUND'|'CONSTRAINT'|'PARSE_FAIL', err_or_None)."""
    lhs, e1 = parse_side(lhs_s)
    if lhs is None:
        if e1 == "FUNC_CALL":
            return "FUNC_SKIP", None
        return "PARSE_FAIL", f"lhs `{lhs_s}` -> {e1}"
    rhs, e2 = parse_side(rhs_s)
    if rhs is None:
        if e2 == "FUNC_CALL":
            return "FUNC_SKIP", None
        return "PARSE_FAIL", f"rhs `{rhs_s}` -> {e2}"
    try:
        free = lhs.free_symbols | rhs.free_symbols
    except Exception as e:
        return "PARSE_FAIL", f"free_symbols failed: {e}"
    return ("GROUND" if not free else "CONSTRAINT"), None


def stratified_sample(path, per_source, seed):
    """Two-pass stratified sample: returns {source: [record, ...]}."""
    counts = defaultdict(int)
    with open(path) as f:
        for line in f:
            counts[json.loads(line)["source_file"]] += 1
    rng = random.Random(seed)
    picks = {}
    for src in sorted(counts):
        n = counts[src]
        k = min(per_source, n)
        picks[src] = set(rng.sample(range(n), k))
    seen = defaultdict(int)
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            src = rec["source_file"]
            if seen[src] in picks[src]:
                out[src].append(rec)
            seen[src] += 1
    return out, {s: counts[s] for s in counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/visualprm_v11_filtered/stage2_kept_tau0.85.jsonl")
    ap.add_argument("--outdir", default="data/visualprm_v11_filtered/sympy_audit")
    ap.add_argument("--per-source", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default=None,
                    help="comma-separated substrings; only matching source_files are audited")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "parse_failures").mkdir(parents=True, exist_ok=True)

    print(f"sympy version: {sympy.__version__}", flush=True)
    print(f"sampling {args.per_source}/source (seed={args.seed}) from {args.input}", flush=True)
    samples, totals = stratified_sample(args.input, args.per_source, args.seed)
    if args.sources:
        pats = [p.strip() for p in args.sources.split(",")]
        samples = {s: v for s, v in samples.items() if any(p in s for p in pats)}
    print(f"{len(samples)} sources, {sum(len(v) for v in samples.values())} traces sampled", flush=True)

    table = {}
    for src in sorted(samples):
        stats = {
            "source_total": totals[src], "traces_sampled": len(samples[src]),
            "steps_seen": 0, "steps_with_claim": 0, "claims": 0,
            "ground": 0, "constraint": 0, "parse_fail": 0, "func_skip": 0,
            "spans_with_eq": 0, "dropped_word_sides": 0, "timeouts": 0,
            "claims_trimmed_side": 0,
        }
        failures = []
        for rec in samples[src]:
            for step in rec["steps"]:
                stats["steps_seen"] += 1
                claims, n_spans, n_dropped = extract_claims(step)
                stats["spans_with_eq"] += n_spans
                stats["dropped_word_sides"] += n_dropped
                step_has_claim = False
                for lhs, rhs, trimmed in claims:
                    stats["claims"] += 1
                    if trimmed:
                        stats["claims_trimmed_side"] += 1
                    verdict, err = classify_claim(lhs, rhs)
                    if verdict == "FUNC_SKIP":
                        stats["func_skip"] += 1
                        continue
                    if verdict == "PARSE_FAIL":
                        stats["parse_fail"] += 1
                        if "TIMEOUT" in (err or ""):
                            stats["timeouts"] += 1
                        if len(failures) < 10:
                            failures.append({
                                "line_index": rec.get("line_index"),
                                "step_snippet": step[:400],
                                "claim": f"{lhs} = {rhs}",
                                "error": err,
                            })
                    else:
                        step_has_claim = True
                        stats["ground" if verdict == "GROUND" else "constraint"] += 1
                if step_has_claim:
                    stats["steps_with_claim"] += 1
        table[src] = stats
        fname = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(src).name) + ".json"
        with open(outdir / "parse_failures" / fname, "w") as f:
            json.dump(failures, f, indent=2)
        ok = stats["ground"] + stats["constraint"]
        print(f"[{src}] steps={stats['steps_seen']} claims={stats['claims']} "
              f"ok={ok} fail={stats['parse_fail']}", flush=True)

    with open(outdir / "extractability_stats.json", "w") as f:
        json.dump(table, f, indent=2)

    # markdown table
    lines = [
        "# SymPy extractability audit — stage2_kept_tau0.85, "
        f"{args.per_source}/source stratified sample (seed={args.seed})",
        "",
        f"sympy {sympy.__version__}. Claims = adjacent pairs of equality chains after "
        "notation normalization (LaTeX/unicode/%/degrees/thousands-commas) and "
        "prose-trimming. GROUND = no free symbols; CONSTRAINT = has free symbols. "
        "Parse-failure rate = failed claims / all extracted claims. "
        "`dropped sides` counts chain sides that were pure prose (not counted as claims or failures).",
        "",
        "| source | pool | traces | steps | steps w/ claim | claims | claims/step | %GROUND | %CONSTRAINT | parse-fail % | dropped sides | timeouts |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    agg = defaultdict(int)
    for src in sorted(table):
        s = table[src]
        for k in ("source_total", "traces_sampled", "steps_seen", "steps_with_claim",
                  "claims", "ground", "constraint", "parse_fail", "dropped_word_sides", "timeouts"):
            agg[k] += s[k]
        ok = s["ground"] + s["constraint"]
        pg = 100 * s["ground"] / ok if ok else 0.0
        pc = 100 * s["constraint"] / ok if ok else 0.0
        pf = 100 * s["parse_fail"] / s["claims"] if s["claims"] else 0.0
        cps = s["claims"] / s["steps_seen"] if s["steps_seen"] else 0.0
        name = Path(src).name.replace("_extracted.jsonl", "").replace(".jsonl", "")
        lines.append(
            f"| {name} | {s['source_total']} | {s['traces_sampled']} | {s['steps_seen']} | "
            f"{s['steps_with_claim']} ({100*s['steps_with_claim']/max(1,s['steps_seen']):.0f}%) | "
            f"{s['claims']} | {cps:.2f} | {pg:.0f}% | {pc:.0f}% | {pf:.1f}% | "
            f"{s['dropped_word_sides']} | {s['timeouts']} |")
    ok = agg["ground"] + agg["constraint"]
    lines.append(
        f"| **ALL** | {agg['source_total']} | {agg['traces_sampled']} | {agg['steps_seen']} | "
        f"{agg['steps_with_claim']} ({100*agg['steps_with_claim']/max(1,agg['steps_seen']):.0f}%) | "
        f"{agg['claims']} | {agg['claims']/max(1,agg['steps_seen']):.2f} | "
        f"{100*agg['ground']/max(1,ok):.0f}% | {100*agg['constraint']/max(1,ok):.0f}% | "
        f"{100*agg['parse_fail']/max(1,agg['claims']):.1f}% | "
        f"{agg['dropped_word_sides']} | {agg['timeouts']} |")
    lines += ["", "Per-source parse-failure examples (10 each): `parse_failures/*.json`.", ""]
    with open(outdir / "extractability_table.md", "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {outdir}/extractability_table.md", flush=True)


if __name__ == "__main__":
    main()
