# rag-eval-lite (中文文档)

**纯离线、零 LLM Token 成本的检索评测工具包（RAG / 搜索工程专用）。**

在优化 RAG 检索管线（调整切片大小、BM25/向量权重、检索召回策略）时，常规的 LLM-as-a-judge 方案（如 Ragas、TruLens）不仅消耗昂贵的 API Token，还会带来网络波动和评测不确定性。

`rag-eval-lite` 提取自企业级知识库评测体系，专注于**纯离线、确定性、高精度**的检索质量基准评估：
- ⚡ **零 LLM 成本**：纯数学与统计指标计算，零外部模型 API 依赖。
- 🎯 **双口径匹配 (Dual-Caliber)**：对比 `path`（严格精确路径匹配）与 `basename`（仅文件名匹配），精准量化“同名文档副本噪声”。
- ⏱️ **冷缓存预热 (Warmup)**：计时评测前自动预热检索缓存，剔除首条查询索引重建的高延迟假象，保障 p50 / p95 延迟真实可靠。
- 📊 **核心信息检索指标**：Recall@K、MRR（平均倒数排名）、nDCG@K 及精确延迟百分位数（Mean、P50、P95）。

---

## 核心指标与双口径价值

在大型企业文档库或代码库中，常出现不同目录存在同名或备份文档的情况（例如 `docs/v1/api.md` 与 `archive/api.md`）：
- 严格路径口径（`path`）未命中，而文件名口径（`basename`）命中：说明检索到了等价副本（**副本噪声**）。
- 两者均未命中：说明检索算法存在**真召回缺口**。

`duplicate_replica_noise = recall_basename - recall_strict` 直接将这两种情况拆解得清清楚楚。

---

## 开源家族项目

- **[compact-bm25](https://github.com/wangkun5212-hue/compact-bm25)**: 基于 CSR/CSC 稀疏矩阵加速的极轻量 BM25。
- **[mcp-guard](https://github.com/wangkun5212-hue/mcp-guard)**: 轻量级 MCP 守门层（多租户鉴权、按工具分级限流、敏感脱敏、审计日志）。
- **[fast-agent-loop](https://github.com/wangkun5212-hue/fast-agent-loop)**: 极简高性能 SSE 流式 Agent 网关（支持模型多候选降级）。
