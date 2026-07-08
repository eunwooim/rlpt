"""Scorer wrappers for reliability experiments."""

from .base import Scorer, ScorerError, make_device
from .bertscore import BertScoreScorer
from .bge_reranker import BGERerankerScorer
from .cross_nli import CrossNLIScorer
from .nli import NLIScorer
from .sbert import SbertCosineScorer

SCORER_REGISTRY = {
    "sbert": SbertCosineScorer,
    "nli": NLIScorer,
    "bertscore": BertScoreScorer,
    "cross_nli": CrossNLIScorer,
    "bge_reranker": BGERerankerScorer,
}

__all__ = [
    "BGERerankerScorer",
    "BertScoreScorer",
    "CrossNLIScorer",
    "NLIScorer",
    "SCORER_REGISTRY",
    "Scorer",
    "ScorerError",
    "SbertCosineScorer",
    "make_device",
]
