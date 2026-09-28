#!/usr/bin/env python
"""Granularity-controllability measurement over qwen_gen_{tag}.jsonl.

Segmentation rule is fixed by condition and never mixed:
  A_unconstrained, B_count, C_length -> split on blank lines
  D_format                           -> split on line-initial numbered markers

A response that yields a single segment under its own rule is "zero-split"
(degenerate). For D that specifically means the model ignored the numbering
instruction, which is also reported directly as the marker-emission fraction.

Three diagnostics ride alongside the headline numbers, because each of them
can make a formatting change masquerade as a step-length change:

  list packing   n_newline_segments / n_blank_segments (pooled). Qwen packs
                 enumerations inside one blank-line paragraph, so a bulleted
                 8-item list scores as one long "step". If this ratio MOVES
                 between conditions, the instruction changed formatting style,
                 not step length.
  backtracking   markers/step using the VPB regex. VPB reference: QvQ 1.10,
                 Claude 3.5 Sonnet / GPT-4o / InternVL all 0.00.
  finish_reason  a response truncated at max_tokens has an artificially low
                 final step count, which reads as successful instruction
                 following. Any condition with meaningful "length" finishes
                 must be flagged before its numbers are believed.

Word counts are whitespace tokens. Distribution stats (mean/p10/p50/p90, IQR,
CV) are pooled over SEGMENTS; steps/response and the ratios are per RESPONSE.

Read-only on the generation files. Output: qwen_granularity_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

WORD_TARGET = 24.0  # VisualPRM400K body-step median 24 w/step; VPB non-QvQ 21-24

BLANK_RE = re.compile(r"\n\s*\n")
# line-initial "1)" / "1." / "**1)" / "- 1." -- the numbering the D prompt asks for
NUM_MARK_RE = re.compile(r"(?m)^[ \t]*(?:[*_]{1,2}[ \t]*)?(\d{1,2})[.)]")
LIST_ITEM_RE = re.compile(r"(?m)^[ \t]*(?:\d{1,2}[.)]|[-*•])[ \t]")
# struct openers: a numbered/bulleted marker OR a "Step 1:" prefix. The latter
# is what 3B writes under C_length; without it the struct rule mis-scored 46.5%
# of that cell as zero-split, which was a regex gap and not a model behaviour.
STRUCT_OPEN_RE = re.compile(
    r"(?m)^[ \t]*(?:[*_]{1,2}[ \t]*)?(?:step[ \t]*\d{1,2}[:.)]|\d{1,2}[.)]|[-*•][ \t])",
    re.I)
# VPB backtracking-marker set, verbatim
BACKTRACK_RE = re.compile(
    r"\b(?:wait|alternatively|but that seems|maybe there'?s a better|"
    r"let me reconsider|actually|hmm|on second thought)\b", re.I)

BLANK_CONDS = {"A_unconstrained", "B_count", "C_length"}
MARK_CONDS = {"D_format"}
COND_ORDER = ["A_unconstrained", "B_count", "C_length", "D_format"]


def split_blank(text):
    return [s for s in (p.strip() for p in BLANK_RE.split(text.strip())) if s]


def split_numbered(text):
    """Cut at each line-initial numbered marker. Text before the first marker
    (a lead-in like "Let's analyze the image:") is kept as its own segment so
    its words are not silently dropped from the word-count population."""
    marks = list(NUM_MARK_RE.finditer(text))
    if not marks:
        return [text.strip()] if text.strip() else []
    cuts = [m.start() for m in marks]
    if cuts[0] > 0:
        cuts = [0] + cuts
    segs = [text[a:b].strip() for a, b in zip(cuts, cuts[1:] + [len(text)])]
    return [s for s in segs if s]


def split_newline(text):
    return [s for s in (l.strip() for l in text.strip().split("\n")) if s]


def split_struct(text):
    """Blank line OR a line-initial structural opener starts a new step.
    Sits between blank (which under-splits enumerations) and newline (which
    splits every line of a paragraph-internal list into its own 'step')."""
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


RULES = {"blank": split_blank, "newline": split_newline, "struct": split_struct}
HEADLINE_RULE = "struct"


def split_for(text, condition):
    if condition in MARK_CONDS:
        return split_numbered(text)
    if condition in BLANK_CONDS:
        return split_blank(text)
    raise ValueError(f"no segmentation rule for condition {condition!r}")


def n_words(s):
    return len(s.split())


def measure_row(row):
    text = row["output"]
    cond = row["condition"]
    segs = split_for(text, cond)
    nl_segs = split_newline(text)
    blank_segs = split_blank(text)
    by_rule = {name: fn(text) for name, fn in RULES.items()}
    nb, nn = len(blank_segs), len(nl_segs)
    hi, lo = max(nb, nn), max(min(nb, nn), 1)
    return {
        "rules": {name: {"n_steps": len(g),
                         "words_per_step": [n_words(s) for s in g],
                         "zero_split": len(g) <= 1}
                  for name, g in by_rule.items()},
        # convention switch: the two whitespace conventions disagree by >2x on
        # how many steps this response contains
        "convention_switch": (hi / lo) > 2.0,
        "blank_newline_ratio": hi / lo,
        "qid": row["qid"], "condition": cond, "data_source": row["data_source"],
        "model": row["model"],
        "finish_reason": row["finish_reason"], "n_gen_tokens": row["n_gen_tokens"],
        "n_struct_segments": len(by_rule["struct"]),
        "n_steps": len(segs),
        "words_per_step": [n_words(s) for s in segs],
        "n_words_total": n_words(text),
        "n_blank_segments": len(blank_segs),
        "n_newline_segments": len(nl_segs),
        "n_list_items": len(LIST_ITEM_RE.findall(text)),
        "n_backtrack": len(BACKTRACK_RE.findall(text)),
        "has_numbered_markers": bool(NUM_MARK_RE.search(text)),
        "zero_split": len(segs) <= 1,
    }


def _stats(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {k: None for k in
                ("mean", "p10", "p50", "p90", "q1", "q3", "iqr", "cv", "sd", "n")}
    q1, q3 = np.percentile(a, [25, 75])
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {"mean": mean, "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "cv": (sd / mean) if mean else None, "sd": sd, "n": int(a.size)}


def aggregate(rows):
    """rows: measured rows for ONE (model, condition) cell."""
    wps_pooled = [w for r in rows for w in r["words_per_step"]]
    wps_resp = [float(np.mean(r["words_per_step"])) for r in rows
                if r["words_per_step"]]
    steps = [r["n_steps"] for r in rows]
    sum_blank = sum(r["n_blank_segments"] for r in rows)
    sum_nl = sum(r["n_newline_segments"] for r in rows)
    sum_steps = sum(steps)
    sum_bt = sum(r["n_backtrack"] for r in rows)
    fr = Counter(r["finish_reason"] for r in rows)
    n = len(rows)
    ws = _stats(wps_pooled)
    by_rule = {}
    for name in RULES:
        pooled = [w for r in rows for w in r["rules"][name]["words_per_step"]]
        steps = [r["rules"][name]["n_steps"] for r in rows]
        st = _stats(pooled)
        by_rule[name] = {
            "words_per_step": st,
            "steps_per_response": _stats(steps),
            "zero_split_frac": sum(r["rules"][name]["zero_split"] for r in rows) / n if n else None,
            "dist_from_target_mean": (st["mean"] - WORD_TARGET) if st["mean"] is not None else None,
            "dist_from_target_p50": (st["p50"] - WORD_TARGET) if st["p50"] is not None else None,
        }
    return {
        "n_responses": n,
        "by_rule": by_rule,
        "headline_rule": HEADLINE_RULE,
        "convention_switch_frac": sum(r["convention_switch"] for r in rows) / n if n else None,
        "words_per_step": ws,
        "words_per_step_response_mean": _stats(wps_resp),
        "steps_per_response": _stats(steps),
        "zero_split_frac": sum(r["zero_split"] for r in rows) / n if n else None,
        "numbered_marker_frac": sum(r["has_numbered_markers"] for r in rows) / n if n else None,
        "list_packing_ratio": (sum_nl / sum_blank) if sum_blank else None,
        "list_items_per_response": sum(r["n_list_items"] for r in rows) / n if n else None,
        "backtrack_per_step": (sum_bt / sum_steps) if sum_steps else None,
        "backtrack_per_response": sum_bt / n if n else None,
        "responses_with_backtrack_frac": sum(r["n_backtrack"] > 0 for r in rows) / n if n else None,
        "finish_reason": dict(fr),
        "length_finish_frac": fr.get("length", 0) / n if n else None,
        "gen_tokens_mean": float(np.mean([r["n_gen_tokens"] for r in rows])) if n else None,
        "dist_from_target_mean": (ws["mean"] - WORD_TARGET) if ws["mean"] is not None else None,
        "dist_from_target_p50": (ws["p50"] - WORD_TARGET) if ws["p50"] is not None else None,
    }


def deltas(cells, model):
    """A->B and A->C, per model. These, not the absolute numbers, isolate
    controllability: a model whose unconstrained baseline already sits near the
    target scores well on C without following any instruction."""
    out = {}
    base = cells.get((model, "A_unconstrained"))
    if not base:
        return out
    for cond in ("B_count", "C_length", "D_format"):
        c = cells.get((model, cond))
        if not c:
            continue
        d = {"d_list_packing_ratio": (c["list_packing_ratio"] - base["list_packing_ratio"])
             if (c["list_packing_ratio"] and base["list_packing_ratio"]) else None,
             "d_convention_switch_frac": c["convention_switch_frac"] - base["convention_switch_frac"]}
        # deltas under every rule: if the sign flips between rules, the
        # controllability claim is a segmentation artefact, not a finding
        for name in RULES:
            cb, bb = c["by_rule"][name], base["by_rule"][name]
            d[name] = {
                "d_words_per_step_mean": cb["words_per_step"]["mean"] - bb["words_per_step"]["mean"],
                "d_words_per_step_p50": cb["words_per_step"]["p50"] - bb["words_per_step"]["p50"],
                "d_steps_per_response_mean": cb["steps_per_response"]["mean"] - bb["steps_per_response"]["mean"],
                "d_steps_per_response_p50": cb["steps_per_response"]["p50"] - bb["steps_per_response"]["p50"],
                "d_cv_words_per_step": (cb["words_per_step"]["cv"] - bb["words_per_step"]["cv"])
                                       if (cb["words_per_step"]["cv"] and bb["words_per_step"]["cv"]) else None,
                "d_abs_dist_from_target": abs(cb["dist_from_target_mean"]) - abs(bb["dist_from_target_mean"]),
            }
        out[f"A->{cond}"] = d
    return out


def _self_test():
    assert split_blank("a\n\nb\n\n\nc") == ["a", "b", "c"]
    assert split_blank("one unbroken paragraph") == ["one unbroken paragraph"]
    # D: lead-in kept, three marker cuts
    s = split_numbered("Let's look.\n1) first\n2) second\n**3)** third")
    assert len(s) == 4, s
    assert s[0] == "Let's look." and s[1] == "1) first", s
    # D with no markers -> one segment, i.e. zero-split
    assert split_numbered("no numbers here") == ["no numbers here"]
    assert len(LIST_ITEM_RE.findall("1. a\n- b\n* c\n• d\nplain")) == 4
    # struct: the "Step N:" prefix 3B writes under C_length must open a step
    s = split_struct("Step 1: count them.\nAnswer: eight.\nStep 2: halve it.")
    assert len(s) == 2, s
    assert s[0].startswith("Step 1") and s[1].startswith("Step 2"), s
    assert len(split_struct("step 3) lower case\nSTEP 4. upper")) == 2
    # struct still cuts plain markers and still respects blank lines
    assert len(split_struct("intro\n1. a\n2. b\n\noutro")) == 4
    # a paragraph with no opener stays whole under struct but not under newline
    assert len(split_struct("one two\nthree four")) == 1
    assert len(split_newline("one two\nthree four")) == 2
    # "Steps 1 and 2" must not open (needs the : . or ) terminator)
    assert len(split_struct("Steps 1 and 2 are done\nso are 3 and 4")) == 1
    assert len(BACKTRACK_RE.findall("Wait, actually hmm. On second thought")) == 4
    assert not BACKTRACK_RE.search("waiter waited")   # word boundary holds
    r = measure_row({"qid": "q", "condition": "A_unconstrained", "data_source": "s",
                     "model": "m", "finish_reason": "stop", "n_gen_tokens": 10,
                     "output": "one two three\n\nfour five"})
    assert r["n_steps"] == 2 and r["words_per_step"] == [3, 2], r
    assert r["n_newline_segments"] == 2 and r["n_blank_segments"] == 2
    # per-row file must carry everything needed to re-aggregate without rerunning
    for k in ("data_source", "condition", "model", "finish_reason",
              "n_blank_segments", "n_newline_segments", "n_struct_segments"):
        assert k in r, k
    # numpy percentiles interpolate linearly: on [1,2,3,4] q1=1.75, q3=3.25
    st = _stats([1, 2, 3, 4])
    assert st["p50"] == 2.5 and st["iqr"] == 1.5, st
    assert abs(st["cv"] - 1.2909944 / 2.5) < 1e-6, st
    print("[measure] self-test OK", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="/scratch/sghos104/rlpt/granularity")
    ap.add_argument("--tags", default="3b,7b")
    ap.add_argument("--out", default="qwen_granularity_stats.json")
    args = ap.parse_args()
    _self_test()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    by_cell, models, per_row = defaultdict(list), {}, []
    for tag in tags:
        path = os.path.join(args.dir, f"qwen_gen_{tag}.jsonl")
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                models[tag] = row["model"]
                m = measure_row(row)
                m["tag"] = tag
                per_row.append(m)
                by_cell[(tag, row["condition"])].append(m)

    cells = {k: aggregate(v) for k, v in by_cell.items()}

    # per-source breakout under the headline rule. The 200 questions are
    # stratified across five VPB sources; if MathVerse and MMMU behave
    # differently that is a DOMAIN effect and must not be read as a model
    # effect. Same source-confounding hazard as the VisualPRM bands.
    by_src = defaultdict(list)
    for m in per_row:
        by_src[(m["tag"], m["condition"], m["data_source"])].append(m)
    src_cells = {}
    for (t, c, s), v in by_src.items():
        pooled = [w for r in v for w in r["rules"][HEADLINE_RULE]["words_per_step"]]
        steps = [r["rules"][HEADLINE_RULE]["n_steps"] for r in v]
        st = _stats(pooled)
        src_cells[f"{t}|{c}|{s}"] = {
            "n_responses": len(v), "n_segments": st["n"],
            "words_per_step": st, "steps_per_response": _stats(steps),
            "zero_split_frac": sum(r["rules"][HEADLINE_RULE]["zero_split"] for r in v) / len(v),
            "numbered_marker_frac": sum(r["has_numbered_markers"] for r in v) / len(v),
            "convention_switch_frac": sum(r["convention_switch"] for r in v) / len(v),
            "dist_from_target_mean": st["mean"] - WORD_TARGET,
        }

    payload = {
        "word_target": WORD_TARGET,
        "target_note": "VisualPRM400K body steps median 24 w/step (mean 30.1); "
                       "VPB non-QvQ policies 21.0-24.1 w/step. Band ~21-24.",
        "segmentation": {"blank_line": sorted(BLANK_CONDS),
                         "numbered_marker": sorted(MARK_CONDS)},
        "models": models,
        "cells": {f"{t}|{c}": v for (t, c), v in cells.items()},
        "cells_by_source": src_cells,
        "deltas": {t: deltas(cells, t) for t in tags},
    }
    out_path = os.path.join(args.dir, args.out)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1)
    perrow_path = os.path.join(
        args.dir, os.path.basename(args.out).replace(".json", "") + "_perrow.jsonl")
    with open(perrow_path, "w") as f:
        for m in per_row:
            f.write(json.dumps(m) + "\n")

    # ---- console table ----
    hdr = (f"{'model':>4} {'condition':<16} | {'STRUCT w/st':>12} {'p50':>4} {'IQR':>4} "
           f"{'CV':>5} {'steps':>6} {'zero':>6} {'d24':>6} | {'blank':>11} {'newline':>11} "
           f"| {'switch>2x':>9} {'pack':>5} {'bt/st':>6} {'len%':>5} {'num%':>5}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for t in tags:
        for c in COND_ORDER:
            v = cells.get((t, c))
            if not v:
                continue
            h = v["by_rule"][HEADLINE_RULE]
            w, s = h["words_per_step"], h["steps_per_response"]
            bl, nl_ = v["by_rule"]["blank"], v["by_rule"]["newline"]
            print(f"{t:>4} {c:<16} | {w['mean']:>7.1f}/{w['p50']:>4.0f} {w['p50']:>4.0f} "
                  f"{w['iqr']:>4.0f} {w['cv']:>5.2f} {s['mean']:>6.2f} "
                  f"{h['zero_split_frac']:>6.1%} {h['dist_from_target_mean']:>+6.1f} | "
                  f"{bl['words_per_step']['mean']:>6.1f}/{bl['words_per_step']['p50']:<4.0f} "
                  f"{nl_['words_per_step']['mean']:>6.1f}/{nl_['words_per_step']['p50']:<4.0f} | "
                  f"{v['convention_switch_frac']:>9.1%} {v['list_packing_ratio']:>5.2f} "
                  f"{v['backtrack_per_step']:>6.3f} {v['length_finish_frac']:>5.1%} "
                  f"{v['numbered_marker_frac']:>5.1%}")
        print("-" * len(hdr))
    print(f"\ndeltas vs A_unconstrained (headline rule = {HEADLINE_RULE}; "
          f"blank/newline shown to expose sign flips)")
    for t in tags:
        for k, d in payload["deltas"][t].items():
            per_rule = " ".join(
                f"{name}{d[name]['d_words_per_step_mean']:+6.1f}" for name in RULES)
            h = d[HEADLINE_RULE]
            print(f"{t:>4} {k:<14} dw/step[{per_rule} ] "
                  f"| dsteps {h['d_steps_per_response_mean']:+5.2f} "
                  f"| d|dist24| {h['d_abs_dist_from_target']:+6.1f} "
                  f"| dpack {(d['d_list_packing_ratio'] or 0):+5.2f} "
                  f"| dswitch {d['d_convention_switch_frac']:+6.1%}")
    # per-source table (headline rule), to separate domain effect from model effect
    srcs = sorted({m["data_source"] for m in per_row})
    print(f"\nper-source {HEADLINE_RULE} words/step mean (n questions per source in "
          f"parentheses); domain effect must not be read as model effect")
    n_per_src = {s: sum(1 for m in per_row
                        if m["data_source"] == s and m["tag"] == tags[0]
                        and m["condition"] == COND_ORDER[0]) for s in srcs}
    head = f"{'model':>4} {'condition':<16}" + "".join(
        f"{s[:14] + f'({n_per_src[s]})':>20}" for s in srcs)
    print(head)
    print("-" * len(head))
    for t in tags:
        for c in COND_ORDER:
            line = f"{t:>4} {c:<16}"
            for s in srcs:
                v = src_cells.get(f"{t}|{c}|{s}")
                line += (f"{v['words_per_step']['mean']:>13.1f}/{v['zero_split_frac']:>5.0%}"
                         if v else f"{'-':>20}")
            print(line)
        print("-" * len(head))
    print("(cell = mean words/step / zero-split%)")

    print(f"\n[measure] -> {out_path}")


if __name__ == "__main__":
    main()
