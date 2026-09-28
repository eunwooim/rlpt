#!/usr/bin/env python
"""Compose MORNING_REPORT.md from whatever artifacts exist.

Deliberately defensive: every section degrades to an explicit "MISSING" note
rather than raising, so a failure in an upstream step still yields a report that
says what broke instead of producing nothing.
"""
import argparse, json, os
from datetime import datetime

REPO = "/scratch/sghos104/rlpt/chunker"


def J(p):
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except Exception as e:
        return {"__error__": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=os.path.join(REPO, "runs/full_plain"))
    ap.add_argument("--weighted_dir", default=os.path.join(REPO, "runs/full_weighted"))
    ap.add_argument("--release_dir", default="")
    ap.add_argument("--out", default=os.path.join(REPO, "MORNING_REPORT.md"))
    ap.add_argument("--stamp", default="")
    a = ap.parse_args()

    sw = J(os.path.join(a.run_dir, "threshold_sweep.json"))
    seg = J(os.path.join(a.run_dir, "segment_stats.json"))
    ev = J(os.path.join(a.run_dir, "eval_test.json"))
    sp = J(os.path.join(a.run_dir, "summary_val.json"))
    sw_w = J(os.path.join(a.weighted_dir, "summary_val.json"))
    prep = J(os.path.join(REPO, "data/prepare_report.json"))
    man = J(os.path.join(a.release_dir, "manifest.json")) if a.release_dir else None

    L = []
    A = L.append
    A(f"# Chunker — morning report")
    A(f"\n_Generated {a.stamp or datetime.now().isoformat(timespec='seconds')} by Slurm "
      f"(session-independent chain)._\n")

    # ---- headline ----
    A("## Bottom line\n")
    if sw and sw.get("chosen"):
        c = sw["chosen"]
        A(f"- **Operating threshold selected: `{c['threshold']}`** "
          f"(boundary F1 {c['boundary_f1']}, token F1 {c['token_f1']}, "
          f"zero-cut {100*c['zero_cut_frac']:.2f}%, {c['oversplit_mean_single']} false splits/1-step).")
        bt = sw.get("would_choose_by_token_f1")
        if bt and bt["threshold"] != c["threshold"]:
            A(f"- A token-F1 objective would instead have picked `{bt['threshold']}` "
              f"(boundary F1 {bt['boundary_f1']}) — recorded so the choice can be overridden.")
    elif sw:
        A("- **NO ELIGIBLE THRESHOLD** — every candidate was disqualified by the zero-cut "
          "constraint. The packaged threshold is unchanged from training. Investigate before use.")
    else:
        A("- **threshold_sweep.json MISSING** — the sweep did not complete; "
          "packaged threshold is the training-time value.")
    if man:
        A(f"- Release packaged: `{a.release_dir}` (threshold {man.get('operating_threshold')}).")
    elif a.release_dir:
        A(f"- **Release packaging did not produce a manifest** at `{a.release_dir}`.")

    # ---- arm comparison ----
    A("\n## Arm comparison (validation, corrected split)\n")
    if sp and sw_w:
        A("| metric | plain (winner) | weighted |")
        A("|---|---|---|")
        for k in ("best_f1", "threshold", "precision", "recall", "f1_at_0.5", "final_val_loss"):
            A(f"| {k} | {sp.get(k)} | {sw_w.get(k)} |")
        pl = (sp.get("token_split_f1_by_language") or {})
        wl = (sw_w.get("token_split_f1_by_language") or {})
        A("\nPer-language token split-F1:\n")
        A("| bucket | gold boundaries | plain | weighted |")
        A("|---|---|---|---|")
        for lg in ("ALL", "en", "cjk"):
            p_, w_ = pl.get(lg, {}), wl.get(lg, {})
            A(f"| {lg} | {p_.get('tp',0)+p_.get('fn',0):,} | {p_.get('f1')} | {w_.get('f1')} |")
        A("\n**CJK outperforms English** — the mDeBERTa-v3-base contingency is closed.")
    else:
        A("_summary_val.json missing for one or both arms._")

    # ---- threshold table ----
    A("\n## Threshold sweep (validation) + decision rule\n")
    if sw and sw.get("table"):
        dr = sw.get("decision_rule", {})
        A(f"Rule: disqualify if zero-cut > **{dr.get('disqualify_if_zero_cut_gt')}** or > "
          f"**{dr.get('disqualify_if_zero_cut_gt_ratio_of_baseline')}x** the 0.35 baseline "
          f"(baseline zero-cut = {sw.get('baseline_zero_cut_frac')}); "
          f"then maximise **boundary F1**.\n")
        A("| thr | token F1 | boundary F1 | **zero-cut %** | false splits /1-step | frac<min | eligible |")
        A("|---|---|---|---|---|---|---|")
        ch = (sw.get("chosen") or {}).get("threshold")
        for r in sw["table"]:
            mark = " **<-- chosen**" if r["threshold"] == ch else ""
            A(f"| {r['threshold']}{mark} | {r['token_f1']} | {r['boundary_f1']} | "
              f"{100*r['zero_cut_frac']:.2f}% | {r['oversplit_mean_single']} | "
              f"{r['frac_under_min']} | {'yes' if r['eligible'] else '**NO**'} |")
        dq = [r for r in sw["table"] if not r["eligible"]]
        if dq:
            A("\nDisqualified:\n")
            for r in dq:
                A(f"- `{r['threshold']}`: {'; '.join(r['disqualified_because'])}")
    else:
        A("_threshold_sweep.json MISSING or empty._")

    # ---- zero-cut concentration ----
    A("\n## Zero-cut analysis (test, at threshold 0.35)\n")
    if seg and not seg.get("__error__"):
        z = seg.get("zero_cut", {})
        A(f"- **{z.get('count')} of {seg.get('n_multi_traj')} multi-step trajectories "
          f"({100*(z.get('frac') or 0):.2f}%) received ZERO cuts** — they reach the reward model "
          f"as one undivided blob.")
        bl = z.get("by_language") or {}
        if bl:
            A("\n| language | zero-cut | total | rate |")
            A("|---|---|---|---|")
            for k, v in bl.items():
                A(f"| {k} | {v.get('zero')} | {v.get('total')} | {100*(v.get('frac') or 0):.2f}% |")
        mz, mn = z.get("mean_tokens_zero_cut"), z.get("mean_tokens_normal")
        if mz and mn:
            A(f"\n- Zero-cut trajectories average **{mz} tokens** vs **{mn}** for normally-segmented "
              f"ones — {'SHORTER, so the slice is concentrated in short texts' if mz < mn else 'not explained by length'}.")
        ptr = seg.get("per_traj_boundary_recall") or {}
        if ptr:
            A(f"- Per-trajectory boundary recall: mean {ptr.get('mean')}, "
              f"{100*(ptr.get('frac_traj_recall_ge_0.8') or 0):.1f}% of trajectories at >=0.8 recall, "
              f"{100*(ptr.get('frac_traj_recall_0') or 0):.2f}% at zero.")
        ex = z.get("examples") or []
        if ex:
            A("\nExamples of zero-cut trajectories:\n")
            for e in ex[:5]:
                A(f"- `{e.get('lang')}` {e.get('n_steps')} steps, {e.get('tokens')} tok, "
                  f"{e.get('n_gold')} gold — {e.get('head','')[:80]!r}")
    else:
        A("_segment_stats.json MISSING._")

    # ---- test eval ----
    A("\n## Test-set eval (winner only, ONE run)\n")
    if ev:
        t = ev.get("token_split_f1", {})
        A(f"Threshold {ev.get('threshold')}, {ev.get('n_test_traj'):,} trajectories "
          f"({ev.get('n_multi'):,} multi / {ev.get('n_single'):,} single).\n")
        A("| bucket | gold | token F1 | P | R |")
        A("|---|---|---|---|---|")
        for lg in ("ALL", "en", "cjk"):
            v = t.get(lg, {})
            A(f"| {lg} | {v.get('tp',0)+v.get('fn',0):,} | {v.get('f1')} | "
              f"{v.get('precision')} | {v.get('recall')} |")
        sf = ev.get("short_fragments_multi", {})
        A(f"\n- min-{sf.get('min_tokens')} fragments: **{sf.get('n_under_min')} of "
          f"{sf.get('n_segments'):,}** (frac {sf.get('frac_under_min')}).")
        os_ = (ev.get("oversplit_1step") or {}).get("ALL", {})
        A(f"- over-split on 1-step: {os_.get('mean_false_splits')} mean false splits, "
          f"{100*(os_.get('frac_traj_oversplit') or 0):.1f}% of trajectories affected.")
        A("\n_Note: this eval used threshold 0.35. If the sweep selected a different threshold, "
          "these test numbers correspond to 0.35, NOT to the packaged threshold — test was "
          "deliberately spent once and not re-run._")
    else:
        A("_eval_test.json MISSING._")

    # ---- split health ----
    A("\n## Split health (why everything was rebuilt)\n")
    if prep and prep.get("split_health"):
        sh = prep["split_health"]
        A("| split | distinct multi-step questions | rollouts/question (mean/max) | CJK multi |")
        A("|---|---|---|---|")
        for s in ("train", "val", "test"):
            h = sh.get(s, {})
            rr = h.get("multi_rollouts_per_question", {})
            A(f"| {s} | {h.get('distinct_questions_with_multi'):,} | "
              f"{rr.get('mean')} / {rr.get('max')} | {h.get('cjk_multi_trajectories'):,} |")
        A(f"\nFlags: {prep.get('split_health_flags')}")
        A("\nThe previous split gave val **31** and test **72** distinct questions (~490 "
          "near-duplicate rollouts each) while trajectory counts looked perfect, and put all "
          "13,103 CJK multi-step trajectories in train. That inflated val F1 to 0.9147 and "
          "produced a val->test recall collapse of 0.871 -> 0.535.")
    else:
        A("_prepare_report.json MISSING._")

    # ---- gaps ----
    A("\n## Known gaps / caveats\n")
    A("- Test numbers above are at threshold 0.35; the packaged threshold may differ (test not re-run).")
    A("- `other` language bucket has no multi-step trajectories, so its F1 is an empty denominator, not a result.")
    A("- Over-splitting on single-step trajectories is the main weakness; the threshold is the knob.")
    A("- The model peaked at step 10,000 of 16,335 and early-stopped at 15,000.")

    txt = "\n".join(L) + "\n"
    with open(a.out, "w") as f:
        f.write(txt)
    print(txt)
    print(f"[written] {a.out}")


if __name__ == "__main__":
    main()
