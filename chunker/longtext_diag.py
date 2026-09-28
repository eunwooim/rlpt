#!/usr/bin/env python
"""Decoder-level long-text follow-ups. VAL only, no retraining.

--part seam  (cache only, fast):
    For multi-step trajectories in the >512-token bucket (T >= 512, the same
    bucket as metrics_pack), bucket every boundary event by the distance (in
    core tokens) from the boundary token to the NEAREST interior window edge of
    the sliding-window pass (window content 510, stride 384; interior edges =
    every window start except 0 and every window end except T). Report per
    distance bucket: n gold, n predicted cuts, recall, precision, F1, plus the
    token-exposure share of each bucket so concentration is interpretable.
    Matching replicates the cache convention exactly: greedy first-fit in char
    space, |cut_char - (gold_char+1)| <= TOL=2, golds in order.
    Trajectories with gold_tok_merged > 0 are skipped (gold_char<->gold_tok
    alignment is ambiguous there) and counted.

--part grid  (model pass over the >512 multi trajectories only):
    2x2 decode grid: stride in {384, 256} x merge in {mean, max}. ONE forward
    pass per stride stores per-window probabilities, so both merges come from
    the same pass. Each cell is DP-decoded at the shipped threshold and scored
    with boundary P/R/F1 (char TOL=2) on the same trajectory set. The
    stride=384/mean cell must reproduce metrics_pack's >512 bucket F1 (0.7442)
    -- printed as a sanity check.

Outputs: runs/full_plain/longtext_diag.json (+ copy to the release reports dir)
and an idempotent sentinel block in MORNING_REPORT.md (full runs only).
"""
import argparse, gzip, json, os
import numpy as np

REPO = "/scratch/sghos104/rlpt/chunker"
RELEASE = os.path.join(REPO, "release/chunker-deberta-v3-small-v1")
CACHE = os.path.join(REPO, "runs/full_plain/metrics_cache_val.json.gz")
OUT_JSON = os.path.join(REPO, "runs/full_plain/longtext_diag.json")
TOL = 2
CONTENT = 510            # cc.WINDOW_CONTENT
BUCKET_T = 512           # metrics_pack ">512" bucket: T >= 512
DIST_BUCKETS = [(0, 8, "0-8"), (9, 16, "9-16"), (17, 32, "17-32"),
                (33, 64, "33-64"), (65, 128, "65-128"), (129, 10 ** 9, ">128")]


def interior_edges(T, stride):
    """Interior window-edge token indices for the sliding pass over T tokens."""
    if T <= CONTENT:
        return []
    starts = list(range(0, T - CONTENT + 1, stride))
    if starts[-1] != T - CONTENT:
        starts.append(T - CONTENT)
    edges = set(starts[1:]) | {s + CONTENT for s in starts[:-1]}
    return sorted(e for e in edges if 0 < e < T)


def match_flags(gold_char, cut_char):
    """Cache-identical greedy matching; returns (gold_matched[], cut_matched[])."""
    gm = [False] * len(gold_char)
    cm = [False] * len(cut_char)
    for gi, g in enumerate(gold_char):
        for ci, c in enumerate(cut_char):
            if cm[ci]:
                continue
            if abs(c - (g + 1)) <= TOL:
                gm[gi] = cm[ci] = True
                break
    return gm, cm


def dist_bucket(d):
    for lo, hi, name in DIST_BUCKETS:
        if lo <= d <= hi:
            return name
    return DIST_BUCKETS[-1][2]


# ---------------------------------------------------------------- seam ----
def run_seam(stride):
    payload = json.load(gzip.open(CACHE, "rt"))
    recs = [r for r in payload["records"]
            if r["kind"] == "multi" and r["T"] >= BUCKET_T]
    skipped = [r for r in recs if r["gold_tok_merged"] > 0]
    recs = [r for r in recs if r["gold_tok_merged"] == 0]
    print(f"[seam] {len(recs):,} multi trajectories with T>={BUCKET_T} "
          f"({len(skipped)} skipped: merged gold tokens)", flush=True)

    agg = {name: {"n_gold": 0, "gold_matched": 0, "n_cut": 0, "cut_matched": 0,
                  "token_exposure": 0}
           for _, _, name in DIST_BUCKETS}
    for r in recs:
        T = r["T"]
        edges = np.array(interior_edges(T, stride))
        assert len(edges), f"T={T} has no interior edges?"
        gm, cm = match_flags(r["gold_char"], r["cut_char"])
        for tok, m in zip(r["gold_tok"], gm):
            b = dist_bucket(int(np.abs(edges - tok).min()))
            agg[b]["n_gold"] += 1
            agg[b]["gold_matched"] += int(m)
        for tok, m in zip(r["cut_tok"], cm):
            b = dist_bucket(int(np.abs(edges - tok).min()))
            agg[b]["n_cut"] += 1
            agg[b]["cut_matched"] += int(m)
        # token exposure: how many tokens sit at each distance
        toks = np.arange(T)
        d = np.abs(toks[:, None] - edges[None, :]).min(axis=1)
        for lo, hi, name in DIST_BUCKETS:
            agg[name]["token_exposure"] += int(((d >= lo) & (d <= hi)).sum())

    table = []
    for _, _, name in DIST_BUCKETS:
        a = agg[name]
        tp, fp, fn = a["gold_matched"], a["n_cut"] - a["cut_matched"], a["n_gold"] - a["gold_matched"]
        rec_ = tp / a["n_gold"] if a["n_gold"] else None
        prec = a["cut_matched"] / a["n_cut"] if a["n_cut"] else None
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
        table.append({"dist_to_seam": name, **a,
                      "recall": round(rec_, 4) if rec_ is not None else None,
                      "precision": round(prec, 4) if prec is not None else None,
                      "boundary_f1": round(f1, 4) if f1 is not None else None})

    def pooled(names):
        tp = sum(agg[n]["gold_matched"] for n in names)
        fp = sum(agg[n]["n_cut"] - agg[n]["cut_matched"] for n in names)
        fn = sum(agg[n]["n_gold"] - agg[n]["gold_matched"] for n in names)
        return round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else None

    near = [n for _, _, n in DIST_BUCKETS[:3]]        # <=32 tokens from a seam
    far = [n for _, _, n in DIST_BUCKETS[3:]]
    out = {"stride": stride, "window_content": CONTENT, "tol": TOL,
           "n_traj": len(recs), "n_skipped_merged_gold": len(skipped),
           "table": table,
           "f1_within_32_of_seam": pooled(near),
           "f1_beyond_32_of_seam": pooled(far)}
    print(f"[seam] F1 within 32 tok of a seam: {out['f1_within_32_of_seam']}  "
          f"beyond: {out['f1_beyond_32_of_seam']}", flush=True)
    return out


# ---------------------------------------------------------------- grid ----
def run_grid(model_dir, val_traj, limit):
    import torch
    import chunker as ck
    import chunker_common as cc

    thr = ck.load(model_dir)
    print(f"[grid] threshold={thr} model={model_dir}", flush=True)

    rows = [json.loads(l) for l in open(val_traj)]
    rows = [r for r in rows if r["kind"] == "multi"]
    keep = []
    for r in rows:
        enc = ck._tok(r["text"], add_special_tokens=False,
                      return_offsets_mapping=True, truncation=False)
        if len(enc["input_ids"]) >= BUCKET_T:
            keep.append((r, enc["input_ids"], enc["offset_mapping"]))
        if limit and len(keep) >= limit:
            break
    print(f"[grid] {len(keep):,} multi trajectories with T>={BUCKET_T}", flush=True)

    @torch.no_grad()
    def window_probs(ids, stride):
        """[(core_start, core_end, p_split_core[])] for one stride."""
        T = len(ids)
        starts = list(range(0, T - CONTENT + 1, stride)) if T > CONTENT else [0]
        if T > CONTENT and starts[-1] != T - CONTENT:
            starts.append(T - CONTENT)
        out = []
        for st in starts:
            en = min(st + CONTENT, T)
            wi = torch.tensor([[ck._tok.cls_token_id] + ids[st:en] + [ck._tok.sep_token_id]],
                              device=ck._device)
            logits = ck._model(wi).logits[0].float().cpu().numpy()
            p1 = np.exp(logits - logits.max(axis=-1, keepdims=True))
            p1 = p1[:, 1] / p1.sum(axis=-1)
            out.append((st, en, p1[1:-1]))
        return out

    def merge(wins, T, how):
        if how == "mean":
            s, c = np.zeros(T), np.zeros(T)
            for st, en, p in wins:
                s[st:en] += p
                c[st:en] += 1.0
            c[c == 0] = 1.0
            return s / c
        m = np.zeros(T)
        for st, en, p in wins:
            m[st:en] = np.maximum(m[st:en], p)
        return m

    def decode(prob, cand):
        bnds, _, feasible = ck._dp_boundaries(prob, cand, 8, 220, thr, hard_min=True)
        if not feasible:
            bnds, _, _ = ck._dp_boundaries(prob, cand, 8, 220, thr, hard_min=False)
        return bnds

    cells = [(s, m) for s in (384, 256) for m in ("mean", "max")]
    acc = {c: {"tp": 0, "fp": 0, "fn": 0} for c in cells}
    for i, (r, ids, offs) in enumerate(keep):
        text = r["text"]
        cand = np.array(cc.candidate_token_mask(text, offs), dtype=bool)
        gold_char, _ = cc.gold_boundary_char_positions(r["steps"])
        wins = {s: window_probs(ids, s) for s in (384, 256)}
        for s, how in cells:
            prob = merge(wins[s], len(ids), how)
            cuts = [offs[b][1] for b in decode(prob, cand)]
            gm, cm = match_flags(gold_char, cuts)
            a = acc[(s, how)]
            a["tp"] += sum(gm)
            a["fp"] += len(cuts) - sum(cm)
            a["fn"] += len(gold_char) - sum(gm)
        if (i + 1) % 250 == 0:
            print(f"  ...grid {i+1:,}/{len(keep):,}", flush=True)

    table = []
    for (s, how), a in acc.items():
        p = a["tp"] / (a["tp"] + a["fp"]) if a["tp"] + a["fp"] else 0.0
        r_ = a["tp"] / (a["tp"] + a["fn"]) if a["tp"] + a["fn"] else 0.0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        table.append({"stride": s, "merge": how, **a,
                      "precision": round(p, 4), "recall": round(r_, 4),
                      "boundary_f1": round(f1, 4)})
        print(f"[grid] stride={s} merge={how}: F1={f1:.4f} (P={p:.4f} R={r_:.4f})",
              flush=True)
    return {"threshold": thr, "n_traj": len(keep), "bucket": f"T>={BUCKET_T}",
            "baseline_expected_f1_stride384_mean": 0.7442, "table": table}


# --------------------------------------------------------------- report ----
def write_outputs(seam, grid, morning_md):
    # merge with an existing partial run so seam/grid parts don't clobber each other
    prev = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    seam = seam or prev.get("seam_diagnostic")
    grid = grid or prev.get("decode_grid_2x2")
    out = {"seam_diagnostic": seam, "decode_grid_2x2": grid}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    rel = os.path.join(RELEASE, "reports", "longtext_diag.json")
    with open(rel, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[out] wrote {OUT_JSON} and {rel}", flush=True)

    if not morning_md:
        return
    L = ["<!-- longtext-diag:start -->",
         "## Long-text decoder follow-ups (>512-token val multi, thr from release)", ""]
    if seam:
        L += [f"### Seam diagnostic (stride {seam['stride']}, {seam['n_traj']:,} traj)", "",
              "| dist to nearest seam (tok) | n gold | n cut | recall | precision | F1 | token share |",
              "|---|---|---|---|---|---|---|"]
        tot_exp = sum(r["token_exposure"] for r in seam["table"]) or 1
        for r in seam["table"]:
            L.append(f"| {r['dist_to_seam']} | {r['n_gold']:,} | {r['n_cut']:,} | "
                     f"{r['recall']} | {r['precision']} | {r['boundary_f1']} | "
                     f"{100*r['token_exposure']/tot_exp:.1f}% |")
        L += ["", f"- F1 within 32 tokens of a seam: **{seam['f1_within_32_of_seam']}**; "
                  f"beyond 32: **{seam['f1_beyond_32_of_seam']}**", ""]
    if grid:
        L += [f"### Decode grid (2x2), boundary F1 on the same bucket "
              f"({grid['n_traj']:,} traj @ thr {grid['threshold']})", "",
              "| stride | merge | precision | recall | boundary F1 |", "|---|---|---|---|---|"]
        for r in grid["table"]:
            L.append(f"| {r['stride']} | {r['merge']} | {r['precision']} | "
                     f"{r['recall']} | {r['boundary_f1']} |")
        L.append("")
    L.append("<!-- longtext-diag:end -->")
    block = "\n".join(L)
    s0, e0 = "<!-- longtext-diag:start -->", "<!-- longtext-diag:end -->"
    t = open(morning_md).read()
    if s0 in t and e0 in t:
        t = t[:t.index(s0)] + block + t[t.index(e0) + len(e0):]
    else:
        t = t.rstrip() + "\n\n" + block + "\n"
    with open(morning_md, "w") as f:
        f.write(t)
    print(f"[out] appended to {morning_md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["seam", "grid", "both"], default="both")
    ap.add_argument("--model_dir", default=os.path.join(RELEASE, "model"))
    ap.add_argument("--val_traj", default=os.path.join(REPO, "data/val_traj.jsonl"))
    ap.add_argument("--stride", type=int, default=384, help="seam diagnostic stride")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap grid trajectories")
    args = ap.parse_args()

    seam = run_seam(args.stride) if args.part in ("seam", "both") else None
    grid = run_grid(args.model_dir, args.val_traj, args.limit) if args.part in ("grid", "both") else None
    # smoke runs never touch the release dir or MORNING_REPORT
    if args.limit:
        print(json.dumps({"seam": seam, "grid": grid}, indent=1)[:2000])
        return
    write_outputs(seam, grid, os.path.join(REPO, "MORNING_REPORT.md"))


if __name__ == "__main__":
    main()
