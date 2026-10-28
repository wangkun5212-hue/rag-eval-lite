"""rag-eval-lite: Pure offline, zero-LLM-cost retrieval evaluation toolkit."""

from .metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    first_relevant_rank,
    mean,
    percentile,
    aggregate_metrics,
)
from .runner import RetrievalEvaluator, EvalQuestion, EvalResult

__version__ = "0.1.0"

__all__ = [
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "first_relevant_rank",
    "mean",
    "percentile",
    "aggregate_metrics",
    "RetrievalEvaluator",
    "EvalQuestion",
    "EvalResult",
]
