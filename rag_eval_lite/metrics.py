"""Pure, deterministic metric computation functions for retrieval evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

MATCH_MODES = ("path", "basename")


def normalize_doc_id(p: str | None) -> str:
    """Normalize file path or document ID: strip whitespace, handle None."""
    if p is None:
        return ""
    return str(p).strip()


def _valid_match_mode(match_mode: str) -> str:
    if match_mode not in MATCH_MODES:
        raise ValueError(f"Invalid match_mode={match_mode!r}. Must be one of {MATCH_MODES}")
    return match_mode


def _expected_keys(expected_files: Iterable[str], match_mode: str) -> set[str]:
    """Convert expected document IDs into matching keys based on match_mode."""
    _valid_match_mode(match_mode)
    keys: set[str] = set()
    for p in expected_files:
        n = normalize_doc_id(p)
        if not n:
            continue
        keys.add(Path(n).name if match_mode == "basename" else n)
    return keys


def _retrieved_key(doc_id: str, match_mode: str) -> str:
    n = normalize_doc_id(doc_id)
    if not n:
        return ""
    return Path(n).name if match_mode == "basename" else n


def first_relevant_rank(
    retrieved_docs: Sequence[str],
    expected_docs: Iterable[str],
    match_mode: str = "path",
) -> int | None:
    """Return the 1-based rank of the first relevant document; None if not found."""
    expected = _expected_keys(expected_docs, match_mode)
    for i, doc in enumerate(retrieved_docs, 1):
        if _retrieved_key(doc, match_mode) in expected:
            return i
    return None


def recall_at_k(
    retrieved_docs: Sequence[str],
    expected_docs: Iterable[str],
    k: int,
    match_mode: str = "path",
) -> int:
    """Binary Recall@k for a single query (1 if any relevant doc is in top-k, else 0)."""
    if k <= 0:
        return 0
    return 1 if first_relevant_rank(retrieved_docs[:k], expected_docs, match_mode) is not None else 0


def reciprocal_rank(
    retrieved_docs: Sequence[str],
    expected_docs: Iterable[str],
    match_mode: str = "path",
) -> float:
    """Reciprocal Rank for a single query: 1 / rank of first relevant doc, or 0.0."""
    rank = first_relevant_rank(retrieved_docs, expected_docs, match_mode)
    return (1.0 / rank) if rank is not None else 0.0


def ndcg_at_k(
    retrieved_docs: Sequence[str],
    expected_docs: Iterable[str],
    k: int,
    match_mode: str = "path",
) -> float:
    """Normalized Discounted Cumulative Gain (nDCG@k) for binary relevance."""
    if k <= 0:
        return 0.0
    expected = _expected_keys(expected_docs, match_mode)
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k], 1):
        if _retrieved_key(doc, match_mode) in expected:
            dcg += 1.0 / math.log2(i + 1)
    
    # Ideal DCG: best possible ordering of relevant docs
    ideal_hits = min(len(expected), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, returning 0.0 for empty sequences."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear interpolation percentile (p in [0, 100])."""
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(xs[0])
    rank = (float(p) / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def aggregate_metrics(details: Sequence[dict]) -> dict:
    """Aggregate individual query results into summary metrics."""
    n = len(details)
    if n == 0:
        return {
            "total_queries": 0,
            "recall_strict": 0.0,
            "recall_basename": 0.0,
            "mrr_strict": 0.0,
            "mrr_basename": 0.0,
            "ndcg_strict": 0.0,
            "ndcg_basename": 0.0,
            "duplicate_replica_noise": 0.0,
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
        }

    recalls_strict = [d.get("recall_strict", 0) for d in details]
    recalls_basename = [d.get("recall_basename", 0) for d in details]
    mrrs_strict = [d.get("mrr_strict", 0.0) for d in details]
    mrrs_basename = [d.get("mrr_basename", 0.0) for d in details]
    ndcgs_strict = [d.get("ndcg_strict", 0.0) for d in details]
    ndcgs_basename = [d.get("ndcg_basename", 0.0) for d in details]
    latencies = [d.get("latency_ms", 0.0) for d in details]

    mean_recall_strict = mean(recalls_strict)
    mean_recall_base = mean(recalls_basename)

    return {
        "total_queries": n,
        "recall_strict": round(mean_recall_strict, 4),
        "recall_basename": round(mean_recall_base, 4),
        "mrr_strict": round(mean(mrrs_strict), 4),
        "mrr_basename": round(mean(mrrs_basename), 4),
        "ndcg_strict": round(mean(ndcgs_strict), 4),
        "ndcg_basename": round(mean(ndcgs_basename), 4),
        "duplicate_replica_noise": round(mean_recall_base - mean_recall_strict, 4),
        "latency_mean_ms": round(mean(latencies), 2),
        "latency_p50_ms": round(percentile(latencies, 50.0), 2),
        "latency_p95_ms": round(percentile(latencies, 95.0), 2),
    }
