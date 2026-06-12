"""Offline scoring for the mimicry-vs-reasoning validation plan (CLAUDE.md).

Consumes the JSONL generation files from tools/gen_eval_outputs.py and produces
three TSVs:

  validation_main.tsv   — per checkpoint: acc, bipartite match vs OWN GT vs
                          SAME-TOPIC-SHUFFLED GT (test 1: a widening own-shuffled
                          gap across checkpoints refutes style mimicry), the same
                          pair under NLI entailment (test 2a: style-insensitive)
                          and under a DIFFERENT sentence encoder (test 2b: rules
                          out encoder hacking).
  validation_swap.tsv   — normal vs image-swapped acc/match (test 3: real image
                          analysis collapses under swap; text-prior mimicry
                          survives; the trained model should degrade MORE).
  validation_aokvqa.tsv — A-OKVQA transfer accuracy (test 4).

Pure CPU. SBERT runs in the existing subprocess embed server; NLI uses a
CrossEncoder (DeBERTa-v3-base MNLI).

Usage:
  python tools/validation_scoring.py [--gen-dir outputs/validation]
      [--out-dir data/logs/validation_results] [--skip-nli] [--skip-alt]
"""

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")
os.environ.setdefault("NLTK_DATA", "/scratch/sghos104/rlpt/data/nltk_data")
os.environ.setdefault("RLPT_REWARD_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from clause_split import clause_split
from graph_match_reward import GraphMatchReward, answer_correct, parse_output

CHECKPOINTS = ["base", "20", "40", "60", "80", "100", "120", "140", "160", "176"]
ALT_ENCODER = "sentence-transformers/all-mpnet-base-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
SHUFFLE_SEED = 7


def load_gens(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def pred_text(rec):
    """Mirror CompositeReward: match the <think> span, else the whole output."""
    think, answer, format_ok = parse_output(rec["output"])
    return (think if think else (rec["output"] or "")), answer, format_ok


def same_topic_shuffle(rows, seed=SHUFFLE_SEED):
    """row index -> donor row index with a DIFFERENT question's GT, same topic.

    Within each topic group of size >= 2: cyclic shift of a seeded shuffle (a
    derangement). Topics with a single row are pooled and deranged together
    (still same-dataset style). Deterministic given the row order + seed.
    """
    rng = np.random.default_rng(seed)
    by_topic = {}
    for i, r in enumerate(rows):
        by_topic.setdefault(r.get("topic") or "_none", []).append(i)
    donor = list(range(len(rows)))
    singletons = []
    for idxs in by_topic.values():
        if len(idxs) < 2:
            singletons.extend(idxs)
            continue
        order = rng.permutation(idxs)
        for k in range(len(order)):
            donor[order[k]] = int(order[(k + 1) % len(order)])
    if len(singletons) >= 2:
        order = rng.permutation(singletons)
        for k in range(len(order)):
            donor[order[k]] = int(order[(k + 1) % len(order)])
    return donor


def match_scores(matcher, rows, donor=None):
    """Mean bipartite match F of each prediction vs its (own or donor) GT."""
    vals = []
    for i, rec in enumerate(rows):
        pred, _, _ = pred_text(rec)
        src = rows[donor[i]] if donor is not None else rec
        m = matcher.score(pred, src["solution"], gt_cache_key=src["question_id"])
        vals.append(m["reward"])
    return float(np.mean(vals))


def accuracy(rows):
    vals = []
    for rec in rows:
        _, answer, _ = pred_text(rec)
        vals.append(1.0 if answer_correct(answer, rec["answer"], rec["choices"]) else 0.0)
    return float(np.mean(vals))


class NliScorer:
    """Mean P(GT solution entails clause) over the prediction's clauses."""

    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(NLI_MODEL, device="cpu")
        # label order for cross-encoder/nli-deberta-v3-base:
        # [contradiction, entailment, neutral]
        self.ent_idx = 1
        self._clause_cache = {}

    def clauses(self, text):
        if text not in self._clause_cache:
            self._clause_cache[text] = clause_split(text) or ([text] if text.strip() else [])
        return self._clause_cache[text]

    def score_rows(self, rows, donor=None):
        import scipy.special
        pairs, spans = [], []
        for i, rec in enumerate(rows):
            pred, _, _ = pred_text(rec)
            src = rows[donor[i]] if donor is not None else rec
            cl = self.clauses(pred)
            spans.append((len(pairs), len(pairs) + len(cl)))
            pairs.extend((src["solution"], c) for c in cl)
        if not pairs:
            return 0.0
        logits = self.ce.predict(pairs, batch_size=64, show_progress_bar=False,
                                 apply_softmax=False)
        probs = scipy.special.softmax(np.asarray(logits), axis=-1)[:, self.ent_idx]
        per_row = [float(np.mean(probs[a:b])) if b > a else 0.0 for a, b in spans]
        return float(np.mean(per_row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default="/scratch/sghos104/rlpt/outputs/validation")
    ap.add_argument("--out-dir", default="/scratch/sghos104/rlpt/data/logs/validation_results")
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--skip-alt", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sbert = GraphMatchReward(beta=0.5, tau=0.15)          # the reward's own encoder
    alt = None if args.skip_alt else GraphMatchReward(
        model_name=ALT_ENCODER, beta=0.5, tau=0.15)
    nli = None if args.skip_nli else NliScorer()

    # ---- tests 1 + 2: own vs shuffled GT, three scorers --------------------
    main_path = os.path.join(args.out_dir, "validation_main.tsv")
    with open(main_path, "w") as f:
        f.write("ckpt\tacc\tmatch_own\tmatch_shuf\tmatch_gap"
                "\tnli_own\tnli_shuf\tnli_gap\talt_own\talt_shuf\talt_gap\n")
        for ck in CHECKPOINTS:
            path = os.path.join(args.gen_dir, f"gen_{ck}.jsonl")
            if not os.path.exists(path):
                print(f"[main] {ck}: missing {path}, skipping", flush=True)
                continue
            rows = load_gens(path)
            donor = same_topic_shuffle(rows)
            acc = accuracy(rows)
            mo = match_scores(sbert, rows)
            ms = match_scores(sbert, rows, donor)
            no = nli.score_rows(rows) if nli else float("nan")
            ns = nli.score_rows(rows, donor) if nli else float("nan")
            ao = match_scores(alt, rows) if alt else float("nan")
            as_ = match_scores(alt, rows, donor) if alt else float("nan")
            f.write(f"{ck}\t{acc:.4f}\t{mo:.4f}\t{ms:.4f}\t{mo-ms:.4f}"
                    f"\t{no:.4f}\t{ns:.4f}\t{no-ns:.4f}"
                    f"\t{ao:.4f}\t{as_:.4f}\t{ao-as_:.4f}\n")
            f.flush()
            print(f"[main] {ck}: acc={acc:.3f} match {mo:.3f}/{ms:.3f} "
                  f"nli {no:.3f}/{ns:.3f} alt {ao:.3f}/{as_:.3f}", flush=True)

    # ---- test 3: image swap -------------------------------------------------
    swap_path = os.path.join(args.out_dir, "validation_swap.tsv")
    with open(swap_path, "w") as f:
        f.write("ckpt\tacc_normal\tacc_swap\td_acc\tmatch_normal\tmatch_swap\td_match\n")
        for ck in CHECKPOINTS:
            p_n = os.path.join(args.gen_dir, f"gen_{ck}.jsonl")
            p_s = os.path.join(args.gen_dir, f"gen_{ck}_swap.jsonl")
            if not (os.path.exists(p_n) and os.path.exists(p_s)):
                continue
            rn, rs = load_gens(p_n), load_gens(p_s)
            an, asw = accuracy(rn), accuracy(rs)
            mn, msw = match_scores(sbert, rn), match_scores(sbert, rs)
            f.write(f"{ck}\t{an:.4f}\t{asw:.4f}\t{asw-an:.4f}"
                    f"\t{mn:.4f}\t{msw:.4f}\t{msw-mn:.4f}\n")
            f.flush()
            print(f"[swap] {ck}: acc {an:.3f}->{asw:.3f} match {mn:.3f}->{msw:.3f}",
                  flush=True)

    # ---- test 4: A-OKVQA transfer -------------------------------------------
    aok_path = os.path.join(args.out_dir, "validation_aokvqa.tsv")
    with open(aok_path, "w") as f:
        f.write("ckpt\tacc\tformat\tmatch_rationales\n")
        for ck in CHECKPOINTS:
            path = os.path.join(args.gen_dir, f"aokvqa_{ck}.jsonl")
            if not os.path.exists(path):
                continue
            rows = load_gens(path)
            acc = accuracy(rows)
            fmt = float(np.mean([1.0 if pred_text(r)[2] else 0.0 for r in rows]))
            mr = match_scores(sbert, rows)
            f.write(f"{ck}\t{acc:.4f}\t{fmt:.4f}\t{mr:.4f}\n")
            f.flush()
            print(f"[aokvqa] {ck}: acc={acc:.3f} format={fmt:.3f} match={mr:.3f}",
                  flush=True)

    print("done:", main_path, swap_path, aok_path, flush=True)


if __name__ == "__main__":
    main()
