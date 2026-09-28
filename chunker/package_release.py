#!/usr/bin/env python
"""Package the winning chunker into a versioned, self-contained release dir.

Bundles the best checkpoint + operating threshold + the two inference modules
(chunker.py / chunker_common.py -- the SHARED candidate rule must ship with the
weights or train/inference silently diverge) + the val/test reports, and writes
a README documenting the numbers, the operating point, and the known gaps.

Usage: package_release.py --run_dir runs/full_plain --version v1
"""
import argparse, json, os, shutil, hashlib

REPO = "/scratch/sghos104/rlpt/chunker"


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=os.path.join(REPO, "runs/full_plain"))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out_root", default=os.path.join(REPO, "release"))
    ap.add_argument("--arm", default="plain (Arm A, unweighted CE)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the operating threshold (from threshold_sweep.py); "
                         "rewritten into the packaged model/chunker_meta.json")
    args = ap.parse_args()

    out = os.path.join(args.out_root, f"chunker-deberta-v3-small-{args.version}")
    model_out = os.path.join(out, "model")
    os.makedirs(model_out, exist_ok=True)

    best = os.path.join(args.run_dir, "best")
    meta = load_json(os.path.join(best, "chunker_meta.json"))
    summary = load_json(os.path.join(args.run_dir, "summary_val.json"))
    evaltest = load_json(os.path.join(args.run_dir, "eval_test.json"))
    prep = load_json(os.path.join(REPO, "data/prepare_report.json"))
    cjk = load_json(os.path.join(REPO, "cjk_pool_report.json"))

    # ---- weights + tokenizer ----
    for fn in os.listdir(best):
        src = os.path.join(best, fn)
        if os.path.isfile(src) and fn != "training_args.bin":
            shutil.copy2(src, os.path.join(model_out, fn))

    # ---- inference code (must travel with the weights) ----
    code_out = os.path.join(out, "code")
    os.makedirs(code_out, exist_ok=True)
    for fn in ("chunker.py", "chunker_common.py"):
        shutil.copy2(os.path.join(REPO, fn), os.path.join(code_out, fn))

    # ---- reports ----
    rep_out = os.path.join(out, "reports")
    os.makedirs(rep_out, exist_ok=True)
    for src, dst in [(os.path.join(args.run_dir, "summary_val.json"), "summary_val.json"),
                     (os.path.join(args.run_dir, "eval_test.json"), "eval_test.json"),
                     (os.path.join(args.run_dir, "final_val.json"), "final_val.json"),
                     (os.path.join(args.run_dir, "metrics.jsonl"), "metrics.jsonl"),
                     (os.path.join(args.run_dir, "threshold_sweep.json"), "threshold_sweep.json"),
                     (os.path.join(args.run_dir, "segment_stats.json"), "segment_stats.json"),
                     (os.path.join(REPO, "data/prepare_report.json"), "prepare_report.json"),
                     (os.path.join(REPO, "cjk_pool_report.json"), "cjk_pool_report.json")]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(rep_out, dst))

    # best step: read from metrics.jsonl (the v1 template hardcoded 2000, which
    # went stale the moment the split was fixed and the model was retrained)
    best_step = None
    mpath = os.path.join(args.run_dir, "metrics.jsonl")
    if os.path.exists(mpath):
        best_f1 = -1.0
        for line in open(mpath):
            try:
                d = json.loads(line)
            except Exception:
                continue
            f1 = d.get("eval_f1", d.get("eval_best_f1"))
            if f1 is not None and f1 > best_f1:
                best_f1, best_step = f1, d.get("step")

    # operating threshold: sweep-selected value wins over the training-time one,
    # and is rewritten into the packaged meta so chunker.load() picks it up.
    sweep = load_json(os.path.join(args.run_dir, "threshold_sweep.json"))
    thr = args.threshold if args.threshold is not None else meta.get("best_thr")
    if args.threshold is not None:
        meta_out = os.path.join(model_out, "chunker_meta.json")
        m = load_json(meta_out)
        m["best_thr"] = args.threshold
        m["threshold_source"] = "threshold_sweep.py (val, boundary-F1 objective, zero-cut constrained)"
        m["training_time_thr"] = meta.get("best_thr")
        with open(meta_out, "w") as f:
            json.dump(m, f, indent=2)
    manifest = {
        "name": f"chunker-deberta-v3-small-{args.version}",
        "base_model": "microsoft/deberta-v3-small",
        "task": "token classification (2 labels: no_split / split)",
        "winning_arm": args.arm,
        "operating_threshold": thr,
        "val_best_f1": summary.get("best_f1") or meta.get("final_val_best_f1"),
        "val_precision": summary.get("precision"),
        "val_recall": summary.get("recall"),
        "test": evaltest,
        "training": {
            "precision": "fp32 (weights force-upcast; the fp16 checkpoint NaN's otherwise)",
            "effective_batch": 128, "lr": 3e-5, "seed": 42,
            "best_step": best_step,
            "train_windows": (prep.get("split_windows") or {}).get("train"),
        },
        "files_sha256": {f: sha256(os.path.join(model_out, f)) for f in sorted(os.listdir(model_out))},
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # CJK line from the CURRENT split's health block (cjk_pool_report.json is a
    # pre-split-fix scan and quoting it produced a false "CJK unvalidated" claim)
    sh = prep.get("split_health") or {}
    cjk_line = ""
    if sh:
        cjk_line = (f"- CJK multi-step trajectories per split (stratified): "
                    f"train {(sh.get('train') or {}).get('cjk_multi_trajectories')}, "
                    f"val {(sh.get('val') or {}).get('cjk_multi_trajectories')}, "
                    f"test {(sh.get('test') or {}).get('cjk_multi_trajectories')}.\n")
    val_cjk = ((summary.get("token_split_f1_by_language") or {}).get("cjk") or {}).get("f1")
    test_cjk = ((evaltest.get("token_split_f1") or {}).get("cjk") or {}).get("f1")
    val_en = ((summary.get("token_split_f1_by_language") or {}).get("en") or {}).get("f1")
    test_en = ((evaltest.get("token_split_f1") or {}).get("en") or {}).get("f1")

    readme = f"""# chunker-deberta-v3-small-{args.version}

Token-level **chunk-boundary classifier**. Given a raw reasoning trajectory
(plain text, no step markers), it predicts where to split it into steps,
reproducing VisualPRM400K-v1.1 human segmentation. Downstream it segments model
outputs before the DeBERTa process-reward model scores each segment.

Token classification, 2 labels (`no_split`=0 / `split`=1). NOT generation.

## Operating point
- **threshold = {thr}** (swept 0.20-0.90 on validation; stored in `model/chunker_meta.json`)
- winning arm: **{args.arm}**
- validation: best_f1 **{summary.get('best_f1')}**, P {summary.get('precision')} / R {summary.get('recall')}

## Usage
```python
import sys; sys.path.insert(0, "code")
import chunker as ck
ck.load("model")                      # picks up threshold from chunker_meta.json
out = ck.chunk(text, min_tokens=8, max_tokens=220)
out["chunks"]                  # list[str] segments
out["split_char_offsets"]      # char offset where each chunk ends
out["segment_token_lengths"]   # token length per segment
```
`code/chunker_common.py` holds the **shared candidate rule** used by both training
and inference. Ship it with the weights -- if the two ever diverge, the model
scores tokens the decoder never proposes.

## Decoding
DP constrained decoding over candidate tokens: score = P(split) - threshold,
`max_tokens` enforced hard (forced split if no candidate in range), `min_tokens`
hard-when-feasible with a soft fallback for degenerate-short texts.

## Known gaps
{cjk_line}- **CJK is validated on this split** and is the model's BEST language bucket:
  token split-F1 val {val_cjk} / test {test_cjk} vs English {val_en} / {test_en}.
  The mDeBERTa-v3-base contingency is closed.
- `other` language bucket has no multi-step trajectories; its F1 is an empty
  denominator, not a result.
- Over-splitting on single-step trajectories is the main weakness; the
  threshold is the knob (see `reports/threshold_sweep.json`).
- Test-set numbers in `reports/eval_test.json` were measured at threshold 0.35
  (test deliberately spent once); numbers for any other packaged threshold are
  validation-only.
- The model's best checkpoint is step {best_step}; later checkpoints declined
  on validation (early-stopped).

## Provenance
Base `microsoft/deberta-v3-small`. **The published checkpoint is stored in fp16;
weights must be upcast to fp32 before training or the model NaN's** (this
silently voided an earlier full run). Trained fp32/TF32, eff batch 128, lr 3e-5,
seed 42, early-stopped on validation best_f1.

See `reports/` for the full validation summary, test eval, and data prep report,
and `manifest.json` for checksums.
"""
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(readme)

    print(json.dumps(manifest, indent=2))
    print(f"\n[packaged] {out}")
    for root, _, files in os.walk(out):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            print(f"  {os.path.relpath(p, out):<44} {os.path.getsize(p):>12,} bytes")


if __name__ == "__main__":
    main()
