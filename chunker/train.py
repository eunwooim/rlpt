#!/usr/bin/env python
"""Finetune microsoft/deberta-v3-small as a token-level chunk-boundary classifier.

Metrics (split class = label 1) are computed ONLY over candidate tokens
(label != -100), with a P(split) threshold swept over 0.1..0.9 to maximize F1.
Early stopping on validation best-threshold F1; auto-resume from last checkpoint.
"""
import argparse, json, os
import numpy as np

os.environ.setdefault("HF_HOME", "/scratch/sghos104/.hf_cache")
import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments,
                          Trainer, EarlyStoppingCallback, TrainerCallback)
from transformers.trainer_utils import get_last_checkpoint
import chunker_common as cc

THRESHOLDS = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]


def softmax_p1(logits):
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e[..., 1] / e.sum(axis=-1)


def make_compute_metrics():
    def compute(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits); labels = np.asarray(labels)
        mask = labels != cc.LABEL_IGNORE            # candidate tokens only
        p1 = softmax_p1(logits)[mask]
        y = (labels[mask] == cc.LABEL_SPLIT).astype(np.int64)
        pos = y.sum(); best = {"f1": -1.0, "thr": 0.5, "p": 0.0, "r": 0.0}
        curve = []
        for thr in THRESHOLDS:
            pred = (p1 >= thr).astype(np.int64)
            tp = int((pred & y).sum()); fp = int((pred & (1 - y)).sum()); fn = int(((1 - pred) & y).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            curve.append({"thr": thr, "p": round(prec, 4), "r": round(rec, 4), "f1": round(f1, 4)})
            if f1 > best["f1"]:
                best = {"f1": f1, "thr": thr, "p": prec, "r": rec}
        # f1 at fixed 0.5 for reference
        pred5 = (p1 >= 0.5).astype(np.int64)
        tp = int((pred5 & y).sum()); fp = int((pred5 & (1 - y)).sum()); fn = int(((1 - pred5) & y).sum())
        f1_5 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        # discrimination gap: mean P(split) on gold vs non-gold candidate tokens.
        # A healthy model has gap >> 0; NaN or ~0 means it isn't discriminating.
        gpos = float(p1[y == 1].mean()) if (y == 1).any() else float("nan")
        gneg = float(p1[y == 0].mean()) if (y == 0).any() else float("nan")
        return {"best_f1": round(best["f1"], 4), "best_thr": best["thr"],
                "best_precision": round(best["p"], 4), "best_recall": round(best["r"], 4),
                "f1_at_0.5": round(f1_5, 4), "n_candidate_tokens": int(mask.sum()),
                "n_positives": int(pos),
                "gap_pos": round(gpos, 4), "gap_neg": round(gneg, 4),
                "gap": round(gpos - gneg, 4), "curve": json.dumps(curve)}
    return compute


class JsonlLogger(TrainerCallback):
    def __init__(self, path): self.path = path
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics:
            row = {"step": state.global_step, **{k: v for k, v in metrics.items() if k != "eval_curve"}}
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")


class NanGuard(TrainerCallback):
    """Stop + loudly flag if the training loss goes non-finite, so an overnight
    run fails legibly instead of silently producing a NaN checkpoint."""
    @staticmethod
    def _bad(v):
        try:
            return v != v or v in (float("inf"), float("-inf"))
        except TypeError:
            return False
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs and self._bad(logs["loss"]):
            print(f"[NAN-GUARD] non-finite training loss {logs['loss']} at step {state.global_step} "
                  f"— stopping training.", flush=True)
            control.should_training_stop = True
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        # belt-and-suspenders: also catch a NaN that only surfaces at eval time.
        if metrics and self._bad(metrics.get("eval_loss")):
            print(f"[NAN-GUARD] non-finite eval_loss {metrics.get('eval_loss')} at step "
                  f"{state.global_step} — stopping training.", flush=True)
            control.should_training_stop = True


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy over the split class. Plain CE
    lets the 1:~7 imbalance collapse the model to a trivial constant output;
    up-weighting the positive (split) class forces it to actually discriminate."""
    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k)
        self._cw = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self._cw.to(device=logits.device, dtype=logits.dtype) if self._cw is not None else None
        lf = torch.nn.CrossEntropyLoss(weight=w, ignore_index=cc.LABEL_IGNORE)
        loss = lf(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/scratch/sghos104/rlpt/chunker/data")
    ap.add_argument("--output_dir", default="/scratch/sghos104/rlpt/chunker/runs/deberta_chunker")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--per_device_batch", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=2)      # eff batch ~128 on 1 GPU
    ap.add_argument("--eval_steps", type=int, default=2000)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--eval_subset", type=int, default=20000) # fixed val-window subset for periodic eval
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max_train_samples", type=int, default=0, help=">0 caps train windows (smoke)")
    ap.add_argument("--pos_weight", type=float, default=0.0,
                    help=">0 enables class-weighted CE with this weight on the split class")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="fp32 (TF32 matmul on A100) is the stable choice for DeBERTa-v3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(cc.MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        cc.MODEL, num_labels=2, id2label={0: "no_split", 1: "split"}, label2id={"no_split": 0, "split": 1})
    # CRITICAL: the microsoft/deberta-v3-small safetensors checkpoint is stored in
    # fp16, and transformers loads weights in the checkpoint dtype. Setting
    # bf16/fp16=False in TrainingArguments only disables autocast — it does NOT
    # upcast fp16 weights. Training on fp16 weights NaN's DeBERTa-v3 (and crashes
    # the weighted-CE path on a dtype mismatch). Force real fp32 weights here.
    model = model.float()
    print(f"[dtype] model param dtype after upcast: {next(model.parameters()).dtype}")

    train_ds = load_from_disk(os.path.join(args.data_dir, "train"))
    val_ds = load_from_disk(os.path.join(args.data_dir, "val"))
    if args.max_train_samples and len(train_ds) > args.max_train_samples:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.max_train_samples))
    eval_ds = val_ds
    if args.eval_subset and len(val_ds) > args.eval_subset:
        eval_ds = val_ds.shuffle(seed=args.seed).select(range(args.eval_subset))
    print(f"[data] train windows={len(train_ds):,}  eval windows={len(eval_ds):,}  full val={len(val_ds):,}")

    # precision: fp32 uses TF32 matmul on A100 — stable for DeBERTa-v3 (bf16/fp16
    # NaN'd the smoke run). Enable TF32 for speed at fp32 storage.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = args.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = args.precision == "fp16" and torch.cuda.is_available()
    print(f"[precision] requested={args.precision} -> bf16={use_bf16} fp16={use_fp16} "
          f"(fp32/TF32 if both False)")
    collator = DataCollatorForTokenClassification(tok, label_pad_token_id=cc.LABEL_IGNORE, max_length=cc.MAX_LEN)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr, warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        bf16=use_bf16, fp16=use_fp16,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="best_f1", greater_is_better=True,
        logging_steps=100, report_to="none", seed=args.seed,
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        eval_accumulation_steps=8,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    common = dict(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collator, compute_metrics=make_compute_metrics(),
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience),
                             NanGuard(),
                             JsonlLogger(os.path.join(args.output_dir, "metrics.jsonl"))])
    if args.pos_weight and args.pos_weight > 0:
        cw = torch.tensor([1.0, float(args.pos_weight)])
        print(f"[loss] class-weighted CE, split-class weight = {args.pos_weight}")
        trainer = WeightedTrainer(class_weights=cw, **common)
    else:
        trainer = Trainer(**common)

    last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last:
        print(f"[resume] from {last}")
    trainer.train(resume_from_checkpoint=last)

    # final metrics on the FULL validation split
    final_val = trainer.evaluate(val_ds, metric_key_prefix="final_val")
    final_val.pop("final_val_curve", None)
    print("[final val]", json.dumps({k: v for k, v in final_val.items()}, indent=2))
    trainer.save_model(os.path.join(args.output_dir, "best"))
    tok.save_pretrained(os.path.join(args.output_dir, "best"))
    with open(os.path.join(args.output_dir, "final_val.json"), "w") as f:
        json.dump(final_val, f, indent=2)
    # persist the chosen operating threshold for inference/eval
    with open(os.path.join(args.output_dir, "best", "chunker_meta.json"), "w") as f:
        json.dump({"best_thr": final_val.get("final_val_best_thr", 0.5),
                   "final_val_best_f1": final_val.get("final_val_best_f1")}, f, indent=2)
    print("[done] best model + threshold saved under", os.path.join(args.output_dir, "best"))


if __name__ == "__main__":
    main()
