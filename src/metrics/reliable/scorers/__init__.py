"""Scorer wrappers for reliability experiments."""

from .base import Scorer, ScorerError, make_device
from .bertscore import BertScoreScorer
from .nli import NLIScorer
from .sbert import SbertCosineScorer

SCORER_REGISTRY = {
    "sbert": SbertCosineScorer,
    "nli": NLIScorer,
    "bertscore": BertScoreScorer,
}

__all__ = [
    "BertScoreScorer",
    "NLIScorer",
    "SCORER_REGISTRY",
    "Scorer",
    "ScorerError",
    "SbertCosineScorer",
    "make_device",
]

