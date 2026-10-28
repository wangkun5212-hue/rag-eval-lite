from rag_eval_lite.metrics import (
    aggregate_metrics,
    first_relevant_rank,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval_lite.runner import EvalQuestion, RetrievalEvaluator


def test_metrics_basic():
    retrieved = ["docs/api.md", "docs/guide.md", "docs/faq.md"]
    expected = ["docs/guide.md"]

    assert first_relevant_rank(retrieved, expected, match_mode="path") == 2
    assert recall_at_k(retrieved, expected, k=1, match_mode="path") == 0
    assert recall_at_k(retrieved, expected, k=2, match_mode="path") == 1
    assert reciprocal_rank(retrieved, expected, match_mode="path") == 0.5


def test_dual_caliber_basename():
    # File is in different directory: duplicate replica noise test
    retrieved = ["backup/guide.md", "archive/api.md"]
    expected = ["docs/guide.md"]

    # Strict path fails
    assert recall_at_k(retrieved, expected, k=2, match_mode="path") == 0
    # Basename matches
    assert recall_at_k(retrieved, expected, k=2, match_mode="basename") == 1


def test_ndcg_at_k():
    retrieved = ["a", "b", "c"]
    expected = ["a", "b"]

    score = ndcg_at_k(retrieved, expected, k=3)
    assert score == 1.0  # Perfect order

    # Worse order
    retrieved_worse = ["c", "b", "a"]
    score_worse = ndcg_at_k(retrieved_worse, expected, k=3)
    assert 0.0 < score_worse < 1.0


def test_evaluator_runner():
    mock_db = {
        "python": ["docs/python.md", "docs/intro.md"],
        "rust": ["docs/rust.md"],
        "none": ["docs/other.md"],
    }

    def dummy_search(query: str, top_k: int):
        return mock_db.get(query, [])[:top_k]

    questions = [
        EvalQuestion(id="q1", query="python", expected_docs=["docs/python.md"]),
        EvalQuestion(id="q2", query="rust", expected_docs=["docs/rust.md"]),
        EvalQuestion(id="q3", query="none", expected_docs=["docs/target.md"]),
    ]

    evaluator = RetrievalEvaluator(retrieval_fn=dummy_search, k=2, warmup_queries=1)
    result = evaluator.evaluate(questions)

    summary = result.summary
    assert summary["total_queries"] == 3
    assert round(summary["recall_strict"], 2) == 0.67
    assert summary["recall_basename"] == summary["recall_strict"]
    assert "latency_p50_ms" in summary
