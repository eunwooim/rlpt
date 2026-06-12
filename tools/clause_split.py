"""Deterministic clause splitter for the description-reward bipartite matcher.

Turns a free-text reasoning/solution string into a list of near-atomic claim
chunks WITHOUT any LLM in the loop — only spaCy's frozen syntactic parser, which
decides *boundaries*, not *truth* (truth stays with BERT-vs-GT at reward time).

Two levels:
  sent_split(text)   -> sentence units (baseline, too coarse for ScienceQA)
  clause_split(text) -> clause units (recommended): splits each sentence at
                        clausal boundaries (advcl/ccomp/relcl/parataxis) and
                        coordinated verb phrases that have their own subject, then
                        strips leading connectives (and/but/so/because/...).
                        Relative clauses re-attach their antecedent noun so they
                        don't fragment ("land that would become Maine" -> the
                        chunk "land would later become Maine", not "would become
                        Maine").

The set of clause types to split on is configurable via `clause_deps` (the one
real tuning knob); everything else is grammar-driven, not keyword-driven.

Usage:
    from clause_split import clause_split
    chunks = clause_split("Both move at the same speed, but B has more mass, "
                          "so B is hotter.")
"""

import argparse
import os
import re

os.environ.setdefault("NLTK_DATA", "/scratch/sghos104/rlpt/data/nltk_data")

import spacy

# Default clause-introducing dependency labels whose head is a (sub)clause of its
# own. Deliberately excludes xcomp/acl (non-finite "to see", share the subject ->
# not a standalone fact) to avoid over-splitting. Override per call via clause_deps.
DEFAULT_CLAUSE_DEPS = frozenset({"advcl", "ccomp", "relcl", "parataxis"})
# subject labels: a coordinated verb only starts a new clause if it has one of these
_SUBJ_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass"})
# leading discourse connectives / relative pronouns to strip from a chunk's front
# (cosmetic only — does NOT decide split points; matching is robust to a stray lead)
_LEAD_STRIP = {
    "and", "but", "so", "because", "since", "while", "whereas", "however",
    "therefore", "thus", "then", "or", "yet", "as", "that", "which", "who",
}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        # parser only; no NER/lemmatizer needed -> faster
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


def sent_split(text):
    """Baseline: sentence-level units."""
    doc = _get_nlp()(text.strip())
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def _clause_heads(sent, clause_deps):
    """Verb-like tokens that head their own clause within a sentence."""
    heads = []
    for tok in sent:
        if tok.dep_ == "ROOT":
            heads.append(tok)
        elif tok.dep_ in clause_deps and tok.pos_ in {"VERB", "AUX"}:
            heads.append(tok)
        elif tok.dep_ == "conj" and tok.pos_ in {"VERB", "AUX"} and tok.head.pos_ in {"VERB", "AUX"}:
            # coordinated clause only if it has its OWN subject ("particle B has more
            # mass" splits; bare "attach or stick" does not -> shared subject).
            if any(c.dep_ in _SUBJ_DEPS for c in tok.children):
                heads.append(tok)
    return heads


def _norm_edge(tok):
    return re.sub(r"[^\w]", "", tok.lower())


def _clean(text):
    toks = text.split()
    # strip leading AND trailing connectives / punctuation (cosmetic only)
    while toks and _norm_edge(toks[0]) in _LEAD_STRIP:
        toks = toks[1:]
    while toks and _norm_edge(toks[-1]) in _LEAD_STRIP:
        toks = toks[:-1]
    out = " ".join(toks).strip(" ,;:.")
    return out


def _is_descendant(node, ancestor):
    # NB: spaCy makes a fresh Token on every .head access, so identity (`is`)
    # never matches even at the ROOT -> compare by token index instead.
    cur = node
    while cur.head.i != cur.i:      # not yet at ROOT (ROOT is its own head)
        cur = cur.head
        if cur.i == ancestor.i:
            return True
    return False


def clause_split(text, min_tokens=2, clause_deps=DEFAULT_CLAUSE_DEPS):
    """Recommended: near-atomic clause units.

    clause_deps: which clause-type dependency labels to split on (tuning knob).
    """
    nlp = _get_nlp()
    doc = nlp(text.strip())
    chunks = []
    for sent in doc.sents:
        heads = _clause_heads(sent, clause_deps)
        if len(heads) <= 1:
            c = _clean(sent.text)
            if c:
                chunks.append(c)
            continue
        # each clause = its head's subtree MINUS tokens owned by a descendant clause head
        spans = []
        for h in heads:
            owned = set(t.i for t in h.subtree)
            for other in heads:
                if other.i == h.i:
                    continue
                if other.i in owned and _is_descendant(other, h):
                    owned -= set(t.i for t in other.subtree)
            if owned:
                spans.append((min(owned), sorted(owned), h))
        for _, idxs, h in sorted(spans, key=lambda s: s[0]):
            txt = " ".join(sent.doc[i].text for i in idxs)
            # relative clauses lose the noun they modify -> re-attach the antecedent
            # so the chunk is a standalone fact ("land that would become Maine").
            if h.dep_ == "relcl" and h.head.i not in idxs:
                txt = h.head.text + " " + txt
            c = _clean(txt)
            if c and len(c.split()) >= min_tokens:
                chunks.append(c)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="num ScienceQA solutions to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/scratch/sghos104/rlpt/data/hf_cache")
    from datasets import load_dataset
    sols = load_dataset("derek-thomas/ScienceQA", split=args.split)["solution"]

    import random
    random.seed(args.seed)
    picks = [s for s in sols if s and len(s.split()) >= 12]
    random.shuffle(picks)
    for sol in picks[: args.n]:
        print("=" * 78)
        print("SOLUTION:", sol.replace("\n", " "))
        print("  -- sentence split --")
        for s in sent_split(sol):
            print("   |", s)
        print("  -- clause split (recommended) --")
        for c in clause_split(sol):
            print("   *", c)
        print()


if __name__ == "__main__":
    main()
