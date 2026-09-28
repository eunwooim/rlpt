#!/usr/bin/env python
"""report_ood.py — score the OOD generations (grpo_arms/evals/ood/vpb_gen_ood_<model>_<bench>.jsonl) and write grpo_arms/OOD_REPORT.md.
Extraction = grpo_arms/arm_reward.extract_answer ("Final answer:" -> \\boxed{} -> <answer> -> last line), i.e. RULE-BASED (no VLMEvalKit / lmms-eval
on Sol). Per-benchmark normalisation (official conventions, rule-based re-implementation):
  multiple choice (MathVista MC, MMK12, MathVerse MC, MathVision with options, WeMath, DynaMath MC, MMMU MC): predicted letter (accepts "B", "(B)",
    "B.", "B) ...") or the exact choice text mapped to its letter; correct iff letter == gold.
  numeric (MathVista integer/float with `precision`, MathVerse free-form, MathVision open, DynaMath float, MMMU open when numeric): first number in the
    prediction (fractions a/b and percentages handled); MathVista: round both to `precision` decimals (official); others: |p-g| <= max(1e-6, 1e-2*|g|).
  text (DynaMath text, MathVista list, MMMU open non-numeric, MathVision open non-numeric): punctuation/space/case-insensitive exact match.
Tables: accuracy for base3b / cs25 / M183 / R183, Δ(M−R), Δ(M−cs25), Δ(R−cs25), finish=length rate, paired McNemar M vs R per benchmark and pooled
across all seven (all items; DynaMath counted per variant) plus a pooled variant with DynaMath collapsed to per-seed worst-case; overlap flag vs vpb_test sources.
"""
import collections
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime

REPO = "/scratch/sghos104/rlpt"
sys.path.insert(0, f"{REPO}/grpo_arms")
from arm_reward import extract_answer  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

E = f"{REPO}/grpo_arms/evals/ood"
SETS = f"{REPO}/data/ood_eval/sets"
MODELS = ["base3b", "cs25", "M183", "R183"]
BENCH = [("mathvista", "MathVista testmini", "disjoint"), ("mmk12", "MMK12 (1,024 balanced)", "disjoint"),
         ("mathverse", "MathVerse testmini VO+VD", "overlaps vpb_test (MathVerse_MINI_Vision_Only)"), ("mathvision", "MathVision test (full)", "overlaps vpb_test (MathVision_MINI ⊂ test)"),
         ("wemath", "WeMath testmini", "overlaps vpb_test (WeMath)"), ("dynamath", "DynaMath 10×501", "overlaps vpb_test (DynaMath)"), ("mmmu", "MMMU validation", "overlaps vpb_test (MMMU_DEV_VAL)")]
LET = "ABCDEFGHIJ"
LETTER_RE = re.compile(r"^\(?\s*([A-Ja-j])\s*[\)\.\:]*(\s|$)")


def norm_text(s):
    s = str(s or "").strip().strip("$").replace("\\text{", "").replace("}", "").replace("{", "")
    return re.sub(r"[^\w]", "", s).lower()


def parse_number(s):
    s = str(s or "").strip().replace(",", "").replace("$", "").replace("\\%", "%")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m and m.start() == 0:
        try:
            return float(m.group(1)) / float(m.group(2))
        except ZeroDivisionError:
            return None
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", s)
    if not m:
        return None
    v = float(m.group(0))
    if s[m.end():m.end() + 1] == "%":
        v = v / 100.0 if abs(v) > 1 else v
    return v


def mc_letter(pred, choices):
    p = str(pred or "").strip()
    m = LETTER_RE.match(p)
    if m:
        return m.group(1).upper()
    if choices:
        np_ = norm_text(p)
        for i, c in enumerate(choices):
            if np_ and np_ == norm_text(c):
                return LET[i]
    return None


def correct(pred, gold, meta, bench):
    qt = meta.get("question_type", "")
    if qt in ("multi_choice", "multi-choice"):
        return mc_letter(pred, meta.get("choices")) == str(gold).strip().upper()
    gn = parse_number(gold)
    pn = parse_number(pred)
    gold_is_num = gn is not None and re.fullmatch(r"[\s\$]*-?\d+(?:\.\d+)?(?:/\d+)?\s*%?[\s\$]*", str(gold)) is not None
    if gold_is_num and pn is not None:
        prec = meta.get("precision")
        if bench == "mathvista" and prec is not None:
            return round(pn, int(prec)) == round(gn, int(prec))
        return abs(pn - gn) <= max(1e-6, 1e-2 * abs(gn))
    return norm_text(pred) == norm_text(gold)


def load_set(bench):
    return {r["qid"]: r for r in map(json.loads, open(f"{SETS}/{bench}.jsonl"))}


def score(model, bench, refs):
    p = f"{E}/vpb_gen_ood_{model}_{bench}.jsonl"
    if not os.path.exists(p):
        return None
    ok, length, nomark, src = {}, 0, 0, collections.defaultdict(list)
    for l in open(p):
        r = json.loads(l)
        ref = refs[r["qid"]]
        pred, marker = extract_answer(r["output"])
        c = int(correct(pred, ref["answer"], ref["meta"], bench))
        ok[r["qid"]] = c
        length += r.get("finish_reason") == "length"
        nomark += not marker
        src[r["data_source"]].append(c)
    n = len(ok)
    if n == 0:
        return None
    return {"n": n, "acc": sum(ok.values()) / n, "len_rate": length / n, "nomark": nomark / n, "ok": ok,
            "per_src": {s: sum(v) / len(v) for s, v in src.items()}}


def mcnemar(x, y, keys=None):
    if not (x and y):
        return None
    qs = set(x["ok"]) & set(y["ok"]) if keys is None else set(keys)
    b = sum(1 for q in qs if x["ok"].get(q) and not y["ok"].get(q))
    c = sum(1 for q in qs if not x["ok"].get(q) and y["ok"].get(q))
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)


def pstr(m):
    return "-" if m is None else f"b={m[0]} c={m[1]} p={m[2]:.3f}"


def f(x, nd=4):
    return "MISSING" if x is None else f"{x:.{nd}f}"


def main():
    out = [f"# OOD REPORT — base3b / cs25 / M@183 / R@183 on seven external benchmarks (generated {datetime.now():%Y-%m-%d %H:%M} by grpo_arms/ood/report_ood.py)", "",
           "Greedy, 2048 tokens, VPB prompt template + pixel cap (grpo_arms/ood/build_ood_sets.py), one 1-h htc gpu:1 job per (model, benchmark) on grp_bshettah.",
           "**Extraction and scoring are rule-based** (no VLMEvalKit / lmms-eval installed on Sol): arm_reward.extract_answer + the per-benchmark normalisation in this",
           "script's docstring. MMMU: image_1 only (43/900 rows have extra images). McNemar = exact binomial on discordant pairs (b = M right/R wrong, c = the reverse).", "",
           "M183 = runs/arm1_3b_matchv4_softgate/hf_step_183 (match + soft gates); R183 = runs/arm1_3b_rlvr_entropy/hf_step_183 (RLVR + adaptive entropy); cs25 = cold-start init; base3b = Qwen2.5-VL-3B-Instruct.", "",
           "## 1. Accuracy per benchmark", "",
           "| benchmark | n | base3b | cs25 | M183 | R183 | Δ(M−R) | Δ(M−cs25) | Δ(R−cs25) | McNemar M vs R | length rate base/cs25/M/R | overlap with vpb_test sources |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    all_scores = {}
    pooled_keys = {m: {} for m in MODELS}
    for bench, label, overlap in BENCH:
        try:
            refs = load_set(bench)
        except FileNotFoundError:
            out.append(f"| {label} | - | SET MISSING | | | | | | | | | {overlap} |"); continue
        sc = {m: score(m, bench, refs) for m in MODELS}
        all_scores[bench] = (refs, sc)
        for m in MODELS:
            if sc[m]:
                pooled_keys[m].update({f"{bench}:{q}": v for q, v in sc[m]["ok"].items()})
        n = next((s["n"] for s in sc.values() if s), None)
        a = {m: (sc[m]["acc"] if sc[m] else None) for m in MODELS}
        d = lambda x, y: f"{a[x] - a[y]:+.4f}" if a[x] is not None and a[y] is not None else "-"  # noqa: E731
        lr = "/".join(f"{sc[m]['len_rate']:.3f}" if sc[m] else "-" for m in MODELS)
        out.append(f"| {label} | {n or '-'} | {f(a['base3b'])} | {f(a['cs25'])} | {f(a['M183'])} | {f(a['R183'])} | {d('M183', 'R183')} | {d('M183', 'cs25')} | {d('R183', 'cs25')} | "
                   f"{pstr(mcnemar(sc['M183'], sc['R183']))} | {lr} | {overlap} |")
        # sub-rows
        if bench == "mathverse" and sc["M183"]:
            for s in ("MathVerse_VisionOnly", "MathVerse_VisionDominant"):
                accs = {m: (sc[m]["per_src"].get(s) if sc[m] else None) for m in MODELS}
                keys = [q for q, r in refs.items() if r["data_source"] == s]
                mm = mcnemar(sc["M183"], sc["R183"], keys) if sc["R183"] else None
                out.append(f"|  ↳ {s} | {len(keys)} | {f(accs['base3b'])} | {f(accs['cs25'])} | {f(accs['M183'])} | {f(accs['R183'])} | "
                           f"{(accs['M183'] - accs['R183']):+.4f} | {(accs['M183'] - accs['cs25']):+.4f} | {(accs['R183'] - accs['cs25']):+.4f} | {pstr(mm)} | | |"
                           if None not in accs.values() else f"|  ↳ {s} | {len(keys)} | {f(accs['base3b'])} | {f(accs['cs25'])} | {f(accs['M183'])} | {f(accs['R183'])} | - | - | - | - | | |")
        if bench == "dynamath" and sc["M183"]:
            worst = {}
            for m in MODELS:
                if not sc[m]:
                    worst[m] = None; continue
                by_seed = collections.defaultdict(list)
                for q, v in sc[m]["ok"].items():
                    by_seed[refs[q]["meta"]["seed_id"]].append(v)
                worst[m] = {s: int(all(v)) for s, v in by_seed.items()}
            wa = {m: (sum(w.values()) / len(w) if w else None) for m, w in worst.items()}
            mm = None
            if worst["M183"] and worst["R183"]:
                mm = mcnemar({"ok": worst["M183"]}, {"ok": worst["R183"]})
            out.append(f"|  ↳ DynaMath per-seed worst-case (all 10 variants right) | 501 | {f(wa['base3b'])} | {f(wa['cs25'])} | {f(wa['M183'])} | {f(wa['R183'])} | "
                       + (f"{wa['M183'] - wa['R183']:+.4f} | {wa['M183'] - wa['cs25']:+.4f} | {wa['R183'] - wa['cs25']:+.4f} | {pstr(mm)} | | |" if None not in wa.values() else "- | - | - | - | | |"))
            all_scores["dynamath_worst"] = worst
    # pooled
    out += ["", "## 2. Pooled M vs R (paired on every item, all seven benchmarks)", ""]
    pm, pr = pooled_keys["M183"], pooled_keys["R183"]
    keys = set(pm) & set(pr)
    if keys:
        mm = mcnemar({"ok": pm}, {"ok": pr}, keys)
        am, ar = sum(pm[k] for k in keys) / len(keys), sum(pr[k] for k in keys) / len(keys)
        out.append(f"- all items (DynaMath per variant): n = {len(keys)}; M {am:.4f} vs R {ar:.4f} (Δ {am - ar:+.4f}); **McNemar {pstr(mm)}**")
        keys2 = {k for k in keys if not k.startswith("dynamath:")}
        w = all_scores.get("dynamath_worst")
        if w and w.get("M183") and w.get("R183"):
            pm2 = {k: pm[k] for k in keys2}; pr2 = {k: pr[k] for k in keys2}
            pm2.update({f"dyna_seed:{s}": v for s, v in w["M183"].items()}); pr2.update({f"dyna_seed:{s}": v for s, v in w["R183"].items()})
            k2 = set(pm2) & set(pr2)
            mm2 = mcnemar({"ok": pm2}, {"ok": pr2}, k2)
            out.append(f"- DynaMath collapsed to per-seed worst-case: n = {len(k2)}; M {sum(pm2[k] for k in k2) / len(k2):.4f} vs R {sum(pr2[k] for k in k2) / len(k2):.4f}; McNemar {pstr(mm2)}")
        for other in ("cs25", "base3b"):
            po = pooled_keys[other]; ko = set(pm) & set(po)
            if ko:
                out.append(f"- M vs {other}: n = {len(ko)}; {pstr(mcnemar({'ok': pm}, {'ok': po}, ko))}; R vs {other}: {pstr(mcnemar({'ok': pr}, {'ok': po}, set(pr) & set(po)))}")
    else:
        out.append("MISSING (no paired M/R generations yet)")
    # extraction diagnostics + per-source
    out += ["", "## 3. Extraction diagnostics (fraction of outputs without an explicit answer marker; those fall back to the last line)", "",
            "| benchmark | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for bench, label, _ in BENCH:
        if bench in all_scores:
            sc = all_scores[bench][1]
            out.append(f"| {label} | " + " | ".join(f"{sc[m]['nomark']:.3f}" if sc[m] else "-" for m in MODELS) + " |")
    out += ["", "## 4. Per-source accuracy (subjects / variants)", ""]
    for bench, label, _ in BENCH:
        if bench not in all_scores:
            continue
        sc = all_scores[bench][1]
        srcs = sorted({s for m in MODELS if sc[m] for s in sc[m]["per_src"]})
        if bench in ("mmk12", "mathvision", "mmmu", "wemath"):   # subject breakdown from meta instead of data_source
            refs = all_scores[bench][0]
            subj = collections.defaultdict(list)
            for q, r in refs.items():
                subj[str(r["meta"].get("subject"))].append(q)
            top = sorted(subj, key=lambda s: -len(subj[s]))[:12]
            out += [f"### {label} — by subject (top {len(top)})", "", "| subject | n | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
            for s in top:
                out.append(f"| {s} | {len(subj[s])} | " + " | ".join(f"{sum(sc[m]['ok'].get(q, 0) for q in subj[s]) / len(subj[s]):.3f}" if sc[m] else "-" for m in MODELS) + " |")
            out.append("")
        elif len(srcs) > 1:
            out += [f"### {label}", "", "| source | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
            for s in srcs:
                out.append(f"| {s} | " + " | ".join(f"{sc[m]['per_src'].get(s, float('nan')):.4f}" if sc[m] else "-" for m in MODELS) + " |")
            out.append("")
    out += ["## 5. Jobs", ""]
    try:
        ids = open(f"{REPO}/grpo_arms/ood/gen_job_ids.txt").read().strip().replace(":", ",")
        sa = subprocess.run(["sacct", "-j", ids, "-o", "JobID,JobName%24,State,Elapsed,ExitCode", "-P", "-n", "-X"], capture_output=True, text=True, timeout=120).stdout
        out += ["```", sa.strip(), "```"]
    except Exception as exc:  # noqa: BLE001
        out.append(f"(sacct unavailable: {exc})")
    open(f"{REPO}/grpo_arms/OOD_REPORT.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
