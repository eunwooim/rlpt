"""Soft bipartite graph-matching reward for the image-description RL track.

Scores a model's free-text reasoning/description against a ground-truth fact set
(e.g. a ScienceQA `solution`) by:

  1. chopping BOTH sides into near-atomic clause units (clause_split, no LLM),
  2. embedding every unit with SBERT (cached on the GT side, recomputed per step
     on the prediction side),
  3. building the pred x gt cosine-similarity matrix S,
  4. solving the one-to-one optimal assignment (Hungarian; scipy on -S) so the
     model can't farm credit by saying the same fact five ways,
  5. reading off a PRECISION-DOMINANT soft P / R / Fbeta (beta < 1) from the
     matched similarity mass.

"Graph" = the bipartite *matching* structure, not a scene graph.

This is the text-only core. Multimodal node costs (box IoU, region-CLIP) plug in
later by blending into S; the matching/scoring code below is unchanged.

CLI demo (reproduces a worked example):
    python tools/graph_match_reward.py
"""

import json
import os
import re
import subprocess
import sys
import threading

os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")
os.environ.setdefault("NLTK_DATA", "/scratch/sghos104/rlpt/data/nltk_data")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import linear_sum_assignment

from clause_split import clause_split

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, good at RL scale


import contextlib


@contextlib.contextmanager
def _real_device(dev):
    """Pop any active torch-function device modes + force the default device.

    verl runs the reward inside a meta/fast-init torch context; this neutralizes it so
    SBERT load/encode use real tensors instead of 'meta'.
    """
    import torch
    stack = []
    try:
        while torch._C._len_torch_function_stack() > 0:
            stack.append(torch._C._pop_torch_function_stack())
    except Exception:
        pass
    try:
        try:
            torch.set_default_device(dev)
        except Exception:
            pass
        yield
    finally:
        for m in reversed(stack):
            try:
                torch._C._push_on_torch_function_stack(m)
            except Exception:
                pass


def _run_clean(fn, dev):
    """Run `fn` in a FRESH thread with a real default device.

    torch device modes are THREAD-LOCAL. verl computes the reward inside a worker thread
    whose default device is 'meta' (fast-init), which makes SBERT load/encode raise
    'cannot copy out of meta tensor'. Running in a brand-new thread (clean mode stack) +
    `_real_device` inside it sidesteps that regardless of how verl set it.
    """
    import threading
    box = {}

    def worker():
        try:
            with _real_device(dev):
                box["v"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def fbeta(p, r, beta):
    """Precision-dominant when beta < 1."""
    if p <= 0 and r <= 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * p + r
    return 0.0 if denom == 0 else (1 + b2) * p * r / denom


class GraphMatchReward:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, beta=0.5, tau=0.0):
        """beta<1 = precision-dominant; tau = min cosine for a match to count.

        device: None -> env RLPT_REWARD_DEVICE (default "cpu"). Keep on CPU inside
        verl workers to avoid contending with the policy model for GPU VRAM; the
        GT-side embeddings are cached so only the small prediction side runs per step.
        """
        if device is None:
            device = os.environ.get("RLPT_REWARD_DEVICE", "cpu")
        self._device = device
        self.beta = beta
        self.tau = tau
        self._gt_cache = {}  # id -> (units, embeddings)
        # SBERT runs in a SUBPROCESS (fresh interpreter): verl's reward-actor process
        # puts transformers in a meta/fast-init state that breaks in-process SBERT load
        # ("cannot copy out of meta tensor"). See tools/sbert_embed_server.py.
        self._model_name = model_name
        self._dim = 384  # all-MiniLM-L6-v2; corrected on first real encode
        self._proc = None
        self._lock = threading.Lock()

    def _ensure_proc(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sbert_embed_server.py")
        self._proc = subprocess.Popen(
            [sys.executable, server, self._model_name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1, env=os.environ.copy(),
        )
        ready = self._proc.stdout.readline().strip()  # "READY"
        if ready != "READY":
            raise RuntimeError(f"SBERT server failed to start: {ready!r}")

    # --- embedding ---------------------------------------------------------
    def encode(self, units):
        """List[str] -> L2-normalized embedding matrix (len(units), d)."""
        units = list(units)
        if not units:
            return np.zeros((0, self._dim), dtype=np.float32)
        with self._lock:
            self._ensure_proc()
            self._proc.stdin.write(json.dumps(units) + "\n")
            self._proc.stdin.flush()
            resp = self._proc.stdout.readline()
        data = json.loads(resp)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"SBERT server: {data['error']}")
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1]:
            self._dim = arr.shape[1]
        return arr

    def gt_units_and_emb(self, gt, gt_is_text=True, cache_key=None):
        """Decompose + embed the GT side, with optional caching (GT is fixed)."""
        if cache_key is not None and cache_key in self._gt_cache:
            return self._gt_cache[cache_key]
        units = clause_split(gt) if gt_is_text else list(gt)
        emb = self.encode(units)
        if cache_key is not None:
            self._gt_cache[cache_key] = (units, emb)
        return units, emb

    # --- scoring -----------------------------------------------------------
    def score(self, prediction, gt, pred_is_text=True, gt_is_text=True,
              gt_cache_key=None, return_pairs=False):
        """Return dict with precision / recall / fbeta (the reward) and metadata.

        prediction : model free text (or list of pred units if pred_is_text=False)
        gt         : GT solution text (or list of GT fact strings if gt_is_text=False)
        """
        pred_units = clause_split(prediction) if pred_is_text else list(prediction)
        gt_units, G = self.gt_units_and_emb(gt, gt_is_text, gt_cache_key)
        P = self.encode(pred_units)

        n_pred, n_gt = len(pred_units), len(gt_units)
        out = {
            "precision": 0.0, "recall": 0.0, "reward": 0.0,
            "n_pred": n_pred, "n_gt": n_gt, "matched": 0,
            "valid": n_gt > 0,  # no GT facts -> nothing to score (no-signal)
        }
        if n_pred == 0 or n_gt == 0:
            if return_pairs:
                out["pairs"] = []
            return out

        S = P @ G.T  # cosine sim (rows=pred, cols=gt); both normalized
        rows, cols = linear_sum_assignment(-S)  # max-weight one-to-one assignment

        mass, pairs = 0.0, []
        for i, j in zip(rows, cols):
            sim = float(S[i, j])
            if sim >= self.tau:
                mass += sim
                pairs.append((pred_units[i], gt_units[j], round(sim, 3)))

        precision = mass / n_pred
        recall = mass / n_gt
        out.update(
            precision=round(precision, 4),
            recall=round(recall, 4),
            reward=round(fbeta(precision, recall, self.beta), 4),
            matched=len(pairs),
        )
        if return_pairs:
            out["pairs"] = pairs
        return out


# ===========================================================================
# Composite GRPO reward:  R = w_f*format + w_m*match + w_a*answer - w_p*pun
# (STAR-R1-shaped: format gate + dense partial credit + per-mistake punishment,
#  with a hard answer-correctness term that cosine sim alone cannot fake.)
# ===========================================================================

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_LETTER_RE = re.compile(r"^\(?([a-z])\)?[.):]?$", re.IGNORECASE)


def parse_output(text):
    """Extract (think_text, answer_text, format_ok) from a model completion.

    format_ok requires exactly one <think>...</think> and one <answer>...</answer>.
    """
    thinks = _THINK_RE.findall(text or "")
    answers = _ANSWER_RE.findall(text or "")
    format_ok = len(thinks) == 1 and len(answers) == 1
    think = thinks[0].strip() if thinks else ""
    answer = answers[-1].strip() if answers else ""
    return think, answer, format_ok


def _norm(s):
    return re.sub(r"[^\w\s]", "", str(s)).strip().lower()


def answer_correct(pred_answer, gold_text, choices=None):
    """Flexible MCQ match: exact text, letter (A/B/..), 0- or 1-based index, or
    unambiguous substring. `gold_text` is the correct choice STRING."""
    if not pred_answer:
        return False
    p, g = _norm(pred_answer), _norm(gold_text)
    if not g:
        return False
    if p == g:
        return True
    if choices:
        norm_choices = [_norm(c) for c in choices]
        gold_idx = norm_choices.index(g) if g in norm_choices else None
        # letter form: "B" / "(b)" / "b."
        m = _LETTER_RE.match(pred_answer.strip())
        if m and gold_idx is not None:
            if (ord(m.group(1).lower()) - ord("a")) == gold_idx:
                return True
        # numeric index (accept 0- and 1-based); isdigit() alone is True for
        # unicode digits like "③" that int() rejects
        if p.isascii() and p.isdigit() and gold_idx is not None:
            if int(p) == gold_idx or int(p) == gold_idx + 1:
                return True
        # the model wrote the full choice text somewhere in the answer span
        if gold_idx is not None and g and g in p and g not in [c for i, c in
                enumerate(norm_choices) if i != gold_idx and c and c in p]:
            return True
    return False


class CompositeReward:
    """Drop-in GRPO reward: format + bipartite reasoning-match + hard answer
    correctness - hallucination punishment."""

    def __init__(self, matcher=None, w_format=1.0, w_match=2.0, w_answer=5.0,
                 w_pun=1.0, **matcher_kw):
        self.matcher = matcher or GraphMatchReward(**matcher_kw)
        self.w_format = w_format
        self.w_match = w_match
        self.w_answer = w_answer
        self.w_pun = w_pun

    def score(self, model_output, gt_solution, gold_answer_text, choices=None,
              gt_cache_key=None, return_breakdown=True):
        think, answer, format_ok = parse_output(model_output)
        # match the reasoning span (fall back to full output if no <think>)
        pred_for_match = think if think else (model_output or "")
        m = self.matcher.score(pred_for_match, gt_solution,
                               gt_cache_key=gt_cache_key)
        f = m["reward"]                      # precision-dominant bipartite F in [0,1]
        n_unmatched = m["n_pred"] - m["matched"]   # hallucinated clauses
        ans_ok = answer_correct(answer, gold_answer_text, choices)

        r_format = self.w_format * (1.0 if format_ok else 0.0)
        r_match = self.w_match * f
        r_answer = self.w_answer * (1.0 if ans_ok else 0.0)
        r_pun = self.w_pun * n_unmatched
        total = r_format + r_match + r_answer - r_pun

        if not return_breakdown:
            return total
        return {
            "reward": round(total, 4),
            "r_format": round(r_format, 4),
            "r_match": round(r_match, 4),
            "r_answer": round(r_answer, 4),
            "r_pun": round(-r_pun, 4),
            "format_ok": format_ok,
            "answer_ok": ans_ok,
            "match_f": f,
            "n_pred": m["n_pred"],
            "n_matched": m["matched"],
            "n_unmatched": n_unmatched,
            "answer": answer,
        }


# --- verl-style adapter ----------------------------------------------------
_SINGLETON = None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl custom-reward entrypoint. `ground_truth` may be a dict with keys
    {answer, solution, choices}, else those are read from `extra_info`."""
    global _SINGLETON
    if _SINGLETON is None:
        # Weights overridable via env for ablation runs (e.g. RLPT_W_MATCH=0).
        _SINGLETON = CompositeReward(
            beta=0.5, tau=0.15,
            w_format=float(os.environ.get("RLPT_W_FORMAT", 1.0)),
            w_match=float(os.environ.get("RLPT_W_MATCH", 2.0)),
            w_answer=float(os.environ.get("RLPT_W_ANSWER", 5.0)),
            w_pun=float(os.environ.get("RLPT_W_PUN", 1.0)),
        )
    gt = ground_truth if isinstance(ground_truth, dict) else {}
    info = extra_info or {}
    gold = gt.get("answer", info.get("answer"))
    gt_sol = gt.get("solution", info.get("solution", ""))
    choices = gt.get("choices", info.get("choices"))
    key = info.get("question_id") or info.get("index")
    r = _SINGLETON.score(solution_str, gt_sol, gold, choices=choices,
                         gt_cache_key=key, return_breakdown=True)
    # verl uses result["score"] as the reward and logs the rest as reward_extra_info,
    # so `acc` (hard MCQ exact-match), `match` (bipartite reasoning F), and `format`
    # show up as separate metrics — not just the blended composite score.
    return {
        "score": r["reward"],
        "acc": 1.0 if r["answer_ok"] else 0.0,
        "match": float(r["match_f"]),
        "format": 1.0 if r["format_ok"] else 0.0,
        "n_hallucinated": float(r["n_unmatched"]),
    }


def _demo():
    print(f"loading {DEFAULT_MODEL} ...", flush=True)
    scorer = CompositeReward(beta=0.5, tau=0.15)

    gt_solution = (
        "The particles in both samples have the same average speed, but each "
        "particle in sample B has more mass than each particle in sample A. So, "
        "the particles in sample B have a higher average kinetic energy. Because "
        "the particles in sample B have the higher average kinetic energy, sample "
        "B must have the higher temperature.")
    choices = ["neither; the samples have the same temperature", "sample A", "sample B"]
    gold = "sample B"

    def wrap(think, ans):
        return f"<think>{think}</think>\n<answer>{ans}</answer>"

    cases = [
        ("GOOD (right answer + reasoning)", wrap(
            "Both gases move at the same speed. Sample B's particles are heavier, "
            "so sample B has higher kinetic energy and a higher temperature.", "sample B")),
        ("LAZY (right answer, no reasoning)", wrap("Sample B looks hotter.", "B")),
        ("WRONG (wrong answer, plausible words)", wrap(
            "Sample A has the higher temperature because it is lighter.", "sample A")),
        ("HALLUCINATED (right answer, made-up facts)", wrap(
            "The sun heats sample B. Magnets pull the particles. Sample B is hotter.", "sample B")),
        ("MALFORMED (no tags)", "i think it is sample B"),
    ]
    for name, out in cases:
        r = scorer.score(out, gt_solution, gold, choices=choices)
        print(f"\n=== {name} ===")
        print(f"  REWARD = {r['reward']:>6}   "
              f"[format {r['r_format']:+} | match {r['r_match']:+} | "
              f"answer {r['r_answer']:+} | pun {r['r_pun']:+}]")
        print(f"  format_ok={r['format_ok']}  answer_ok={r['answer_ok']} ('{r['answer']}')  "
              f"match_F={r['match_f']}  clauses {r['n_matched']}/{r['n_pred']} matched "
              f"({r['n_unmatched']} hallucinated)")


if __name__ == "__main__":
    _demo()
