"""Runner engine for executing retrieval evaluation with warmup and dual-caliber matching."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Sequence

from .metrics import (
    aggregate_metrics,
    first_relevant_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


@dataclass
class EvalQuestion:
    id: str
    query: str
    expected_docs: List[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    summary: dict
    details: List[dict]


class RetrievalEvaluator:
    """Zero-LLM evaluation orchestrator for any search or retrieval function.

    Features:
    - Pure offline: no model inference or token cost required.
    - Cold-cache warmup: executes initial queries before timing to prevent cache rebuild outliers.
    - Dual-caliber matching: evaluates exact path match vs basename match to identify replica noise.
    - Detailed latency percentiles (mean, p50, p95).
    """

    def __init__(
        self,
        retrieval_fn: Callable[[str, int], Sequence[str]],
        k: int = 5,
        warmup_queries: int = 3,
    ):
        """
        Args:
            retrieval_fn: Callable taking (query_string, top_k) -> List[doc_ids_or_paths].
            k: Top-k cutoff for evaluation (default: 5).
            warmup_queries: Number of queries to pre-run to warm up BM25/index caches (default: 3).
        """
        self.retrieval_fn = retrieval_fn
        self.k = k
        self.warmup_queries = warmup_queries

    def evaluate(
        self,
        questions: Sequence[EvalQuestion],
        verbose: bool = False,
    ) -> EvalResult:
        if not questions:
            return EvalResult(summary=aggregate_metrics([]), details=[])

        # Phase 1: Warmup to eliminate cold-cache latency spikes
        if self.warmup_queries > 0:
            warmup_count = min(self.warmup_queries, len(questions))
            for i in range(warmup_count):
                try:
                    self.retrieval_fn(questions[i].query, self.k)
                except Exception:
                    pass

        # Phase 2: Formal timed evaluation
        details = []
        for q in questions:
            t0 = time.monotonic()
            try:
                retrieved = list(self.retrieval_fn(q.query, self.k))
            except Exception as e:
                retrieved = []
            latency_ms = (time.monotonic() - t0) * 1000.0

            r_strict = recall_at_k(retrieved, q.expected_docs, self.k, match_mode="path")
            r_base = recall_at_k(retrieved, q.expected_docs, self.k, match_mode="basename")
            rr_strict = reciprocal_rank(retrieved, q.expected_docs, match_mode="path")
            rr_base = reciprocal_rank(retrieved, q.expected_docs, match_mode="basename")
            ndcg_strict = ndcg_at_k(retrieved, q.expected_docs, self.k, match_mode="path")
            ndcg_base = ndcg_at_k(retrieved, q.expected_docs, self.k, match_mode="basename")

            item = {
                "id": q.id,
                "query": q.query,
                "expected": q.expected_docs,
                "retrieved": retrieved[: self.k],
                "recall_strict": r_strict,
                "recall_basename": r_base,
                "mrr_strict": rr_strict,
                "mrr_basename": rr_base,
                "ndcg_strict": ndcg_strict,
                "ndcg_basename": ndcg_base,
                "latency_ms": latency_ms,
            }
            details.append(item)

        summary = aggregate_metrics(details)
        return EvalResult(summary=summary, details=details)

    @staticmethod
    def load_jsonl(path: str | Path) -> List[EvalQuestion]:
        """Load evaluation questions from a JSONL file."""
        questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                data = json.loads(line)
                q_id = data.get("id") or str(len(questions) + 1)
                query = data.get("query") or data.get("question", "")
                expected = data.get("expected_docs") or data.get("expected_files", [])
                if isinstance(expected, str):
                    expected = [expected]
                meta = {k: v for k, v in data.items() if k not in ("id", "query", "question", "expected_docs", "expected_files")}
                questions.append(EvalQuestion(id=q_id, query=query, expected_docs=expected, metadata=meta))
        return questions
