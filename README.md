# rag-eval-lite

[![PyPI](https://img.shields.io/pypi/v/rag-eval-lite.svg)](https://pypi.org/project/rag-eval-lite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Pure offline, zero-LLM-cost retrieval evaluation toolkit for RAG and search pipelines.**

When optimizing RAG systems, tuning chunk sizes, BM25 weights, or vector embeddings usually doesn't require burning thousands of LLM API tokens with heavy frameworks (like Ragas or TruLens). 

`rag-eval-lite` provides pure, deterministic, instant retrieval evaluation with zero external dependencies:
- ⚡ **Zero-LLM cost**: 100% offline, deterministic mathematical evaluation.
- 🎯 **Dual-Caliber Matching**: Compare `path` (strict match) vs `basename` (filename match) to isolate duplicate document replica noise.
- ⏱️ **Cold-Cache Warmup**: Built-in warmup queries to eliminate cold index initialization latency spikes from your p50/p95 benchmarks.
- 📊 **Key IR Metrics**: Recall@K, MRR (Mean Reciprocal Rank), nDCG@K, and Latency percentiles (Mean, P50, P95).

---

## Installation

```bash
pip install rag-eval-lite
```

*(Zero dependencies beyond the Python standard library)*

---

## Quickstart

### 1. Benchmark any Search / RAG Function

```python
from rag_eval_lite import RetrievalEvaluator, EvalQuestion

# 1. Define or wrap your search function (BM25, vector search, or hybrid)
def my_search(query: str, top_k: int) -> list[str]:
    # e.g., returns ["docs/deploy.md", "docs/intro.md"]
    return search_engine.query(query, limit=top_k)

# 2. Define benchmark questions (or load from JSONL)
questions = [
    EvalQuestion(
        id="q1",
        query="How to configure SSL?",
        expected_docs=["docs/ssl_setup.md"]
    ),
    EvalQuestion(
        id="q2",
        query="Deploy to Kubernetes",
        expected_docs=["docs/k8s_deploy.md"]
    ),
]

# 3. Run evaluation with cold-cache warmup
evaluator = RetrievalEvaluator(retrieval_fn=my_search, k=5, warmup_queries=2)
result = evaluator.evaluate(questions)

print(result.summary)
```

**Output:**
```json
{
  "total_queries": 2,
  "recall_strict": 1.0,
  "recall_basename": 1.0,
  "mrr_strict": 1.0,
  "mrr_basename": 1.0,
  "ndcg_strict": 1.0,
  "ndcg_basename": 1.0,
  "duplicate_replica_noise": 0.0,
  "latency_mean_ms": 1.45,
  "latency_p50_ms": 1.42,
  "latency_p95_ms": 1.58
}
```

---

## Why Dual-Caliber Matching?

In enterprise document repositories or large codebases, the same document or markdown file often exists across multiple paths (e.g. `docs/v1/guide.md` and `backup/guide.md`). 

If your search engine retrieves the replica instead of the canonical path:
- **Strict `path` matching** marks it as a MISS.
- **`basename` matching** marks it as a HIT.

The gap (`duplicate_replica_noise = recall_basename - recall_strict`) quantifies whether your retrieval failure is caused by **true retrieval deficiency** or simply **duplicate file dispersion**.

---

## Related Projects

- **[compact-bm25](https://github.com/wangkun5212-hue/compact-bm25)**: Memory-efficient BM25Okapi replacement via CSR/CSC sparse matrices.
- **[mcp-guard](https://github.com/wangkun5212-hue/mcp-guard)**: Production guardrails for MCP servers (auth, per-tool rate limiting, redaction, audit).
- **[fast-agent-loop](https://github.com/wangkun5212-hue/fast-agent-loop)**: Minimal FastAPI Agent loop with streaming tool calling & fallback.

---

## License

MIT License.
