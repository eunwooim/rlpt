"""jb_lib.py — judge benchmark library (task 3 of the E2 brief, 2026-09-19).

Two NLI judges: cur = microsoft/deberta-xlarge-mnli (the reward's, fp32 + TF32) and v3 = MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli.
Every pair (a = rollout side, b = reference side) is scored with the reward's formula s = clip(0.5*E(a->b) + 0.5*E(b->a) - max(C(a->b), C(b->a)), 0, 1)
(reward_redesign/reward_v2.Scorers.nli_matrix); raw 3-class probabilities are kept. Truncation at the reward's NLI_MAX_TOKENS (512).
Configs (per step pair):
  whole          one NLI call per direction on the whole step                                      calls = 2
  clause-min     June's clause split + SBERT-mpnet Hungarian alignment (tau 0.15) + min over NLI s   calls = 2*|aligned|
  clause-hung-F05 / clause-hung-mean
                 clauses on both sides; candidate clause pairs = SBERT top-3 per clause (both directions, union) -> NLI s on candidates ->
                 Hungarian one-to-one on the sparse s matrix (unscored = 0) -> credit F_0.5(sum s/n_a, sum s/n_b) resp. mean s over assigned
                 pairs with s > 0.    DECISION: a full n_a x n_b NLI matrix costs ~500 calls per step pair (mean 23 clauses per edited step
                 under the reward's blank-line split), so candidates are pre-filtered by SBERT cosine.                  calls = 2*|candidates|
  asym-mean      whole a as premise vs each b-clause (forward only) and whole b vs each a-clause; s = clip(0.5*mean_j E(a->b_j) + 0.5*mean_i E(b->a_i)
                 - max over all those C)                                                             calls = n_a + n_b
  Single-clause pairs (n_a == n_b == 1) fall back to `whole` in the three clause configs (flagged).
Reused June code: split_clauses / align (score_negation_clausesplit.py, executed from source so the module's stale sys.path does not matter),
src/metrics/scorers NLIScorer (label map, loader; subclassed to parameterise model name, max_length, batch size) and SbertCosineScorer (mpnet).
"""
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = "/scratch/sghos104/rlpt"
sys.path.insert(0, f"{REPO}/reward_redesign")
sys.path.insert(0, f"{REPO}/src/metrics")
import reward_v2 as rv  # noqa: E402
from scorers.nli import NLIScorer as JuneNLI  # noqa: E402
from scorers.sbert import SbertCosineScorer  # noqa: E402

MODELS = {"cur": "microsoft/deberta-xlarge-mnli", "v3": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"}
CONFIGS = ["whole", "clause-min", "clause-hung-F05", "clause-hung-mean", "asym-mean"]
TOPK = 3
MAXLEN = rv.NLI_MAX_TOKENS  # 512
OUT = f"{REPO}/reward_redesign/judge_bench/out"

_src = open(f"{REPO}/score_negation_clausesplit.py").read()
_ns = {"linear_sum_assignment": linear_sum_assignment}
exec("import re\n" + _src[_src.index("TAU = 0.15"):_src.index("def score_condition(")], _ns)  # noqa: S102 — June's splitter + align verbatim
split_clauses, june_align, JUNE_TAU = _ns["split_clauses"], _ns["align"], _ns["TAU"]


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def s_formula(e_ab, e_ba, c_ab, c_ba):
    return clip01(0.5 * e_ab + 0.5 * e_ba - max(c_ab, c_ba))


class Judge:
    """NLI judge with a request cache: queue (premise, hypothesis) pairs, run once in length-sorted batches, then read back."""

    def __init__(self, key, batch_size=32, device=None):
        name = MODELS[key]
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        class _S(JuneNLI):
            model_name = name

        self.key, self.name, self.bs, self.dev = key, name, batch_size, dev
        self.june = _S(dev)
        self.tok, self.model, self.lm = self.june.tokenizer, self.june.model, self.june.label_map
        self.cache = {}
        self.calls = 0
        self.secs = 0.0

    @torch.inference_mode()
    def _run(self, pairs):
        """pairs: list of (premise, hypothesis) -> list of (E, C, N). Length-sorted batching."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = [None] * len(pairs)
        t0 = time.time()
        for k in range(0, len(order), self.bs):
            idx = order[k:k + self.bs]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True, truncation=True,
                           max_length=MAXLEN, return_tensors="pt").to(self.dev)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            for r, i in enumerate(idx):
                out[i] = (float(probs[r, self.lm["entailment"]]), float(probs[r, self.lm["contradiction"]]), float(probs[r, self.lm["neutral"]]))
        self.secs += time.time() - t0
        self.calls += len(pairs)
        return out

    def flush(self, requests):
        todo = [p for p in dict.fromkeys(requests) if p not in self.cache]
        if todo:
            for p, r in zip(todo, self._run(todo)):
                self.cache[p] = r
        return {p: self.cache[p] for p in requests}

    def get(self, prem, hyp):
        return self.cache[(prem, hyp)]

    def s(self, a, b):
        e_ab, c_ab, _ = self.get(a, b)
        e_ba, c_ba, _ = self.get(b, a)
        return s_formula(e_ab, e_ba, c_ab, c_ba)


class Embedder:
    def __init__(self, device=None):
        dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.sb = SbertCosineScorer(dev)
        self.cache = {}

    def encode(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            emb = self.sb._encode(todo, 256)
            for t, e in zip(todo, emb):
                self.cache[t] = e
        return torch.stack([self.cache[t] for t in texts]) if texts else torch.empty(0)


# ------------------------------------------------------------------------------------------------------------ config plumbing
class PairPlan:
    """Per (a, b) pair: the clause split, the request lists per config, and the scoring functions that read the judge cache."""

    def __init__(self, a, b, emb):
        self.a, self.b = a, b
        self.A, self.B = split_clauses(a), split_clauses(b)
        self.single = len(self.A) == 1 and len(self.B) == 1
        self.req = {"whole": [(a, b), (b, a)]}
        if self.single:
            for c in ("clause-min", "clause-hung", "asym-mean"):
                self.req[c] = list(self.req["whole"])
            self.june_pairs, self.cand = [(0, 0)], [(0, 0)]
        else:
            ea, eb = emb.encode(self.A), emb.encode(self.B)
            self.june_pairs = june_align(ea, eb)                       # June: Hungarian on cosine, tau gate
            sim = (ea @ eb.T).numpy()
            cand = set()
            for i in range(len(self.A)):
                for j in np.argsort(-sim[i])[:TOPK]:
                    cand.add((i, int(j)))
            for j in range(len(self.B)):
                for i in np.argsort(-sim[:, j])[:TOPK]:
                    cand.add((int(i), j))
            self.cand = sorted(cand)
            self.req["clause-min"] = [p for i, j in self.june_pairs for p in ((self.A[i], self.B[j]), (self.B[j], self.A[i]))]
            self.req["clause-hung"] = [p for i, j in self.cand for p in ((self.A[i], self.B[j]), (self.B[j], self.A[i]))]
            self.req["asym-mean"] = [(a, bj) for bj in self.B] + [(b, ai) for ai in self.A]

    def requests(self):
        return [p for v in self.req.values() for p in v]

    def score(self, J):
        r = {"whole": J.s(self.a, self.b), "fallback": int(self.single), "n_a": len(self.A), "n_b": len(self.B),
             "calls": {"whole": 2, "clause-min": len(self.req["clause-min"]), "clause-hung-F05": len(self.req["clause-hung"]),
                       "clause-hung-mean": len(self.req["clause-hung"]), "asym-mean": len(self.req["asym-mean"])}}
        e_ab, c_ab, n_ab = J.get(self.a, self.b)
        e_ba, c_ba, n_ba = J.get(self.b, self.a)
        r["raw"] = {"E_ab": e_ab, "C_ab": c_ab, "N_ab": n_ab, "E_ba": e_ba, "C_ba": c_ba, "N_ba": n_ba}
        if self.single:
            for c in ("clause-min", "clause-hung-F05", "clause-hung-mean", "asym-mean"):
                r[c] = r["whole"]
            r["n_june"], r["n_cand"], r["n_assigned"] = 1, 1, 1
            return r
        vals = [J.s(self.A[i], self.B[j]) for i, j in self.june_pairs]
        r["clause-min"] = min(vals) if vals else 0.0
        r["n_june"] = len(vals)
        S = np.zeros((len(self.A), len(self.B)))
        for i, j in self.cand:
            S[i, j] = J.s(self.A[i], self.B[j])
        ri, ci = linear_sum_assignment(-S)
        assigned = [(i, j) for i, j in zip(ri, ci) if S[i, j] > 0]
        credit = float(sum(S[i, j] for i, j in assigned))
        prec, rec = credit / len(self.A), credit / len(self.B)
        b2 = rv.BETA ** 2
        r["clause-hung-F05"] = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec > 0 else 0.0
        r["clause-hung-mean"] = credit / len(assigned) if assigned else 0.0
        r["n_cand"], r["n_assigned"] = len(self.cand), len(assigned)
        Eb = [J.get(self.a, bj)[0] for bj in self.B]; Cb = [J.get(self.a, bj)[1] for bj in self.B]
        Ea = [J.get(self.b, ai)[0] for ai in self.A]; Ca = [J.get(self.b, ai)[1] for ai in self.A]
        r["asym-mean"] = clip01(0.5 * float(np.mean(Eb)) + 0.5 * float(np.mean(Ea)) - max(Cb + Ca))
        return r


def score_pairs(J, emb, pairs, log_every=200):
    """pairs: list of (a, b). Returns list of result dicts (configs + raw + calls)."""
    plans, reqs = [], []
    for a, b in pairs:
        p = PairPlan(a, b, emb)
        plans.append(p)
        reqs.extend(p.requests())
    print(f"[jb] {len(pairs)} pairs -> {len(set(reqs))} unique NLI requests ({len(reqs)} total)", flush=True)
    J.flush(reqs)
    return [p.score(J) for p in plans]


# ------------------------------------------------------------------------------------------------------------ 3c: the reward's matching
class RewardAdapter:
    """reward_v2.Scorers stand-in: nli_matrix from the chosen judge (memoised), dedupe via the reward's MiniLM rule."""

    def __init__(self, J):
        from sentence_transformers import SentenceTransformer
        self.J = J
        self.tok = J.tok
        self.sbert = SentenceTransformer(rv.SBERT_MODEL, device=str(J.dev))
        self.last = None

    def nli_matrix(self, roll, gold):
        if not roll or not gold:
            m = np.zeros((len(roll), len(gold)))
        else:
            reqs = [(r, g) for r in roll for g in gold] + [(g, r) for r in roll for g in gold]
            self.J.flush(reqs)
            m = np.array([[self.J.s(r, g) for g in gold] for r in roll])
            over = [i for i, r in enumerate(roll) if len(self.tok.encode(r)) > rv.NLI_MAX_TOKENS]
            if over:
                m[over, :] = 0.0
        self.last = m
        return m

    def dedupe(self, segs):
        return rv.Scorers.dedupe(self, segs)


def reward_match(S, rollout, gold_steps):
    """score_new's match term (mode match, soft gates, no truncation) + the assignment pairs with their credit."""
    b = rv.score_new(rollout, gold_steps, "", S, finish_reason=None, mode="match", gate_mode="soft")
    segs, _ = rv.split_steps(rollout)
    gold = [rv.THINK_RE.sub("", g).strip() for g in gold_steps if not rv.FINAL_RE.match(g.strip())]
    gold = [g for g in gold if g]
    kept, dup = S.dedupe(segs) if segs else ([], [])
    pairs = []
    if segs and gold and S.last is not None and S.last.size:
        m = S.last
        ri, ci = linear_sum_assignment(-m)
        pairs = [(kept[i], int(j), float(m[i, j])) for i, j in zip(ri, ci) if m[i, j] > 0]
    return b, pairs, kept, dup


def edited_index(pos_steps, neg_steps):
    d = [i for i, (x, y) in enumerate(zip(pos_steps, neg_steps)) if x != y]
    return d[0] if len(pos_steps) == len(neg_steps) and len(d) == 1 else None


def step_category(step, edit):
    """bare equation / prose with number / mixed, judged on the edited line (the one whose RHS number changed) and the step."""
    old = edit.split("->")[0]
    lines = [l for l in step.split("\n") if l.strip()]
    edited = next((l for l in lines if re.search(r"=\s*" + re.escape(old) + r"(?![\d.])", l)), lines[0] if lines else step)
    words = re.findall(r"[A-Za-z]{3,}", re.sub(r"\\[a-zA-Z]+", " ", edited))
    line_kind = "equation" if len(words) <= 2 else "prose"
    step_words = re.findall(r"[A-Za-z]{3,}", re.sub(r"\\[a-zA-Z]+", " ", step))
    if line_kind == "equation" and len(step_words) <= 2:
        return "bare equation"
    if line_kind == "prose":
        return "prose with number"
    return "mixed"
