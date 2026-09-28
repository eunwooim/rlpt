#!/usr/bin/env python
"""Post-training VALIDATION summary for one arm (no test-set eval — that waits
until the winner is picked). Writes summary_val.json with:
  * best_f1, operating threshold, precision/recall at it (from final_val.json),
  * the gap curve (mean P(split) gold-vs-nongold) and best_f1 curve over eval
    steps (from metrics.jsonl),
  * token-level split-F1 at the chosen threshold, OVERALL and by language bucket
    (en / cjk / other), computed on the full val_traj.jsonl with the best model.
"""
import argparse, json, os
from collections import defaultdict
import numpy as np

os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import chunker_common as cc


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


@torch.no_grad()
def token_probs(text, tok, model, device):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    T = len(ids)
    ps = np.zeros(T); cnt = np.zeros(T)
    for w in cc.make_windows(ids, offs, [cc.LABEL_IGNORE] * T, tok.cls_token_id, tok.sep_token_id):
        logits = model(torch.tensor([w["input_ids"]], device=device)).logits[0].float().cpu().numpy()
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p1 = e[:, 1] / e.sum(axis=-1)
        ps[w["core_start"]:w["core_end"]] += p1[1:-1]
        cnt[w["core_start"]:w["core_end"]] += 1.0
    cnt[cnt == 0] = 1.0
    return offs, ps / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="the run dir (has best/, metrics.jsonl, final_val.json)")
    ap.add_argument("--val_traj", default="/scratch/sghos104/rlpt/chunker/data/val_traj.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run_dir, "summary_val.json")
    best_dir = os.path.join(args.run_dir, "best")

    final_val = {}
    fv = os.path.join(args.run_dir, "final_val.json")
    if os.path.exists(fv):
        final_val = json.load(open(fv))
    thr = final_val.get("final_val_best_thr", 0.5)

    # curves from metrics.jsonl
    gap_curve, f1_curve = [], []
    mj = os.path.join(args.run_dir, "metrics.jsonl")
    if os.path.exists(mj):
        for l in open(mj):
            d = json.loads(l)
            gap_curve.append({"step": d.get("step"), "gap": d.get("eval_gap"),
                              "gap_pos": d.get("eval_gap_pos"), "gap_neg": d.get("eval_gap_neg")})
            f1_curve.append({"step": d.get("step"), "best_f1": d.get("eval_best_f1"),
                             "best_thr": d.get("eval_best_thr")})

    # per-language token F1 at the chosen threshold, over full val_traj
    tok = AutoTokenizer.from_pretrained(best_dir)
    model = AutoModelForTokenClassification.from_pretrained(best_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    counts = defaultdict(lambda: [0, 0, 0])   # lang -> tp, fp, fn
    rows = [json.loads(l) for l in open(args.val_traj)]
    for r in rows:
        ex = cc.build_token_labels(r["steps"], tok)
        labels = np.array(ex["labels"])
        offs, prob = token_probs(ex["text"], tok, model, device)
        n = min(len(labels), len(prob))
        labels, prob = labels[:n], prob[:n]
        m = labels != cc.LABEL_IGNORE
        y = (labels[m] == cc.LABEL_SPLIT).astype(int)
        pred = (prob[m] >= thr).astype(int)
        tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum()); fn = int(((1 - pred) & y).sum())
        for key in ("ALL", r["lang"]):
            counts[key][0] += tp; counts[key][1] += fp; counts[key][2] += fn

    token_f1 = {k: dict(zip(("precision", "recall", "f1"), prf(*v)), tp=v[0], fp=v[1], fn=v[2])
                for k, v in counts.items()}

    summary = {
        "run_dir": args.run_dir,
        "threshold": thr,
        "best_f1": final_val.get("final_val_best_f1"),
        "precision": final_val.get("final_val_best_precision"),
        "recall": final_val.get("final_val_best_recall"),
        "f1_at_0.5": final_val.get("final_val_f1_at_0.5"),
        "final_val_loss": final_val.get("final_val_loss"),
        "n_val_traj": len(rows),
        "token_split_f1_by_language": token_f1,
        "gap_curve": gap_curve,
        "f1_curve": f1_curve,
        "gap_final": gap_curve[-1] if gap_curve else None,
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
