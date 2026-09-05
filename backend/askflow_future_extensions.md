# AskFlow Future Extensions

> 本文档整理 AskFlow 当前版本中已经识别、但暂不在本轮实现的后续工程扩展点。
>
> 当前原则：**先冻结稳定版本、完成正式 Eval 与项目包装，再进行下一阶段优化。**
>
> 这些扩展点不属于当前版本的必要功能，而是后续可以继续深化的工程方向，重点集中在：
>
> - LangGraph 原生编排能力
> - Researcher 输出协议
> - Deep Research 延迟优化
> - 可观测性与执行效率

---

## 0. 当前版本边界

当前主流程已经形成：

```text
START
  ↓
clarify_with_user
  ↓
write_research_brief
  ↓
research_supervisor
  ↓
evidence_verifier
  ↓
route_after_verification
  ├── final_report_generation
  └── plan_targeted_research
          ↓
      targeted_research
          ↓
      evidence_verifier
          ↓
         ...
  ↓
END
```

当前版本优先保证：

1. Research Budget 的硬限制；
2. Evidence Verifier 的覆盖度、可信度、冲突与证据缺口判断；
3. Evidence Gap → narrow targeted task；
4. Adaptive Research 的软停止与硬停止；
5. 多轮 `notes/raw_notes` 正确累积；
6. 无新增 evidence 时停止；
7. Tool retry / fallback；
8. Dynamic Model Router；
9. 正式 Eval 可复现。

后续优化应建立在上述稳定主链路之上，避免在正式评测阶段继续修改系统行为。

---

# 1. Targeted Research 改为 LangGraph `Send` 动态 Fan-out

## 当前实现

当前 `targeted_research()` 在单个 LangGraph node 内部并发调用多个 Researcher：

```python
research_jobs = [
    researcher_subgraph.ainvoke(...)
    for task in tasks
]

results = await asyncio.gather(
    *research_jobs,
    return_exceptions=True,
)
```

执行结构大致为：

```text
plan_targeted_research
        ↓
targeted_research
        │
        ├─ researcher_subgraph(task A)
        ├─ researcher_subgraph(task B)
        └─ researcher_subgraph(task C)
        ↓
evidence_verifier
```

## 当前问题

这种方式功能上可以正常工作，但多个 Researcher 被封装在同一个 `targeted_research` node 内部。

主要问题：

- Studio 图上只能观察到一个 `targeted_research` 节点；
- 无法直接观察每个 targeted worker；
- worker 粒度的 tracing 不够清晰；
- worker 粒度的 checkpoint 不够自然；
- worker state aggregation 被隐藏在 Python 并发逻辑内部；
- 与 LangGraph 原生 orchestrator-worker / map-reduce 模式相比，可观测性较弱。

## 后续目标

考虑使用 LangGraph `Send` 实现动态 fan-out：

```text
plan_targeted_research
        ↓
   dynamic Send
   /     |      \
  ↓      ↓       ↓
worker  worker   worker
  \      |       /
        merge
          ↓
evidence_verifier
```

每个 `TargetedResearchTask` 动态生成一个 worker。

## 需要重点研究

- `Send` 的动态节点派发方式；
- `TargetedResearchTask` 如何映射到 worker state；
- worker state 与主 `AgentState` 的边界；
- `notes` / `raw_notes` reducer 如何聚合；
- 多 worker 完成后的 barrier / fan-in；
- Studio 中每个 targeted task 的可视化；
- LangSmith tracing 是否能展示单 worker 级别执行；
- 是否直接复用 `researcher_subgraph`；
- 是否需要额外增加 targeted worker adapter node。

## 预期收益

```text
当前：
一个 targeted_research node
    ↓
内部隐藏多个 researcher

未来：
LangGraph 显式 fan-out
    ↓
每个 researcher 独立可观察
```

主要价值不是提高研究质量，而是：

- 编排语义更原生；
- 调试更直观；
- tracing 更细；
- checkpoint 更自然；
- 后续更容易实现 worker 级重试与限流。

## 建议优先级

**P2 — 中期优化**

当前实现功能正确，因此不应为了代码形式提前重构。

---

# 2. Researcher 输出协议结构化

## 当前实现

`compress_research()` 压缩失败时可能返回：

```python
{
    "compressed_research":
        "Error synthesizing research report: Maximum retries exceeded",
    "raw_notes": [raw_notes_content],
}
```

调用方可能通过：

```python
compressed_research.startswith(
    "Error synthesizing research report"
)
```

判断 compression 是否失败。

## 当前问题

这是典型的 **stringly-typed error handling**。

存在以下问题：

- 错误文案发生变化后，流程判断可能失效；
- `compressed_research` 同时承担业务结果和错误状态；
- 无法明确区分完整成功、部分成功和完全失败；
- compression 失败并不代表 Researcher 没有获取 evidence；
- `raw_notes` 仍可能具有研究价值；
- 后续 Verifier 难以准确理解上游结果状态。

目前实际上存在三种不同语义：

```text
1. Researcher 整体异常
2. Researcher 已获得 raw evidence，但 compression 失败
3. Researcher 正常完成
```

但当前输出协议没有显式表达三者差异。

## 后续方案

优先考虑为 Researcher 输出增加明确状态字段：

```python
class ResearcherOutputState(BaseModel):
    compressed_research: str = ""
    raw_notes: list[str] = []

    compression_succeeded: bool = True
    compression_error: str | None = None
```

### 正常完成

```python
{
    "compressed_research": "...",
    "raw_notes": [...],
    "compression_succeeded": True,
    "compression_error": None,
}
```

### Compression 失败，但 evidence 可用

```python
{
    "compressed_research": "",
    "raw_notes": [...],
    "compression_succeeded": False,
    "compression_error": "Maximum retries exceeded",
}
```

进一步可考虑：

```python
status: Literal[
    "success",
    "partial",
    "failed",
]
```

不过第一阶段没有必要为了抽象完整性引入过多状态。

## 最终目标

调用方能够通过结构化状态判断：

```text
subgraph 抛出未捕获异常
    ↓
failed

subgraph 正常返回
+ compression_succeeded=True
    ↓
success

subgraph 正常返回
+ compression_succeeded=False
+ raw_notes 非空
    ↓
partial
```

从而彻底移除：

```python
startswith("Error ...")
```

这样的字符串流程控制。

## 预期收益

- 输出协议更明确；
- 错误处理更可靠；
- partial success 可以被保留；
- Verifier 可以利用仍然有效的 raw evidence；
- 单元测试更容易覆盖；
- 后续更容易做 typed state 与 tracing。

## 建议优先级

**P1/P2 — 适合在下一轮内部重构中完成**

这是相对低风险、工程收益明确的优化点。

---

# 3. Deep Research 延迟优化

## 背景

当前 AskFlow 的优化目标首先是：

```text
Research Quality
Evidence Reliability
Failure Recovery
```

而不是最低端到端延迟。

当前链路相比基础 Research 流程新增：

```text
Research
   ↓
Evidence Verifier
   ↓
[Targeted Re-Research]
   ↓
Re-Verification
   ↓
Final Report
```

因此 AskFlow 在质量和可靠性提升的同时，会天然引入额外 latency。

近期 Eval 中还观察到一个更明显的结构性瓶颈：

```text
1 次 Search Tool Call
        ↓
返回多个网页
        ↓
每个网页进入 webpage_summarization
        ↓
产生大量 LLM 调用
```

逻辑搜索次数并不高，但一次搜索可能触发十几甚至几十次网页摘要。

因此：

```text
logical_tool_calls 较低
≠
LLM 调用次数较低
```

## 当前主要延迟来源

### 3.1 大量网页摘要

当前搜索结果可能直接进入：

```text
Search Results
    ↓
Fetch / Raw Content
    ↓
LLM Summarization × N
```

当搜索结果较多时，摘要调用数量迅速放大。

这是当前最值得优先优化的结构性瓶颈。

### 3.2 Summarization Timeout

网页内容较长或模型响应较慢时，可能出现：

```text
Summarization timed out after 60 seconds
```

即使最终 fallback 到原始内容，这 60 秒仍然已经成为端到端延迟的一部分。

### 3.3 多轮 Researcher ReAct

Researcher 会进行：

```text
LLM
↓
Search
↓
LLM
↓
Search
↓
...
```

复杂任务可能持续多轮。

Research Budget 已经限制最大迭代，但合理的多轮探索仍然会增加 latency。

### 3.4 Compression

Researcher 完成后需要：

```text
raw evidence
    ↓
compression
    ↓
compressed_research
```

当输入 evidence 很长时，Compression 是一个明显的长上下文调用。

### 3.5 Evidence Verification

AskFlow 在 Initial Research 后增加：

```text
notes + raw_notes
        ↓
Evidence Verifier
```

Verifier 需要读取大量 evidence，因此也是长上下文节点。

### 3.6 Targeted Re-Research

当初次 evidence 不足时：

```text
Verifier
   ↓
Evidence Gaps
   ↓
Targeted Research
   ↓
Verifier
```

这属于系统有意引入的质量换延迟机制。

不能简单通过删除该流程解决，否则会破坏 AskFlow 的核心设计目标。

---

# 4. 延迟优化的推荐方向

## 4.1 Search Result Ranking Before Summarization

当前可能是：

```text
Search
  ↓
10~20 results
  ↓
LLM summarize all
```

未来建议改为：

```text
Search
  ↓
URL/domain dedup
  ↓
title + snippet relevance scoring
  ↓
Top-K
  ↓
full content
  ↓
LLM summarization
```

例如只精读 Top 3~5 个高价值页面。

### 核心思想

不要让昂贵的 LLM Summarization 承担第一层筛选职责。

先通过廉价信号完成：

- URL 去重；
- domain 去重；
- title 相关性；
- snippet 相关性；
- official domain boost；
- source freshness；
- 与当前 evidence gap 的相关度。

再进入 LLM。

### 预期收益

这是目前最可能同时降低：

- LLM 调用次数；
- Token 消耗；
- 延迟；
- API 成本；

的优化点。

## 4.2 Semantic Deduplication

多个搜索结果可能：

- 内容高度重复；
- 来自同一篇内容的镜像；
- 重复引用同一官方文档；
- 只是不同 URL 参数。

可以在 Summarization 前做：

```text
URL normalization
        ↓
domain/path dedup
        ↓
snippet similarity
        ↓
semantic dedup
```

避免重复摘要。

## 4.3 Webpage Summary Cache

对于同一个 URL：

```text
Researcher A
Researcher B
Targeted Research
```

可能重复访问。

可以建立：

```text
normalized_url
    ↓
content_hash
    ↓
summary cache
```

缓存：

- raw content；
- summary；
- source metadata；
- summarization model/version。

命中缓存后：

```text
Skip LLM Summarization
```

这对 Eval、重复问题和同域研究尤其有效。

## 4.4 Parallel / Batched Summarization

当前网页摘要可以评估进一步并行化。

目标：

```text
Page A ─┐
Page B ─┼→ concurrent summarization
Page C ─┘
```

需要结合：

- Provider concurrency limit；
- Semaphore；
- RPM / TPM；
- timeout；
- retry；
- 成本预算。

如果 Provider 支持适合的 Batch API，也可以评估非实时场景下的批处理。

但 Agent 在线研究链路仍应优先考虑受控并发，而不是为了并行度无限 fan-out。

## 4.5 Adaptive Summarization

不是所有网页都需要相同处理强度。

可以根据：

```text
页面长度
来源类型
相关度
是否官方
是否包含关键 evidence
```

动态决定：

```text
无需摘要
短摘要
标准摘要
深度摘要
```

例如：

```text
短页面
→ 直接保留

官方 API reference
→ evidence extraction

超长博客
→ coarse summary first

低相关页面
→ skip
```

## 4.6 Early Stop for Low-Information Sources

如果某个页面：

- 与 research topic 相关度低；
- 与已有 evidence 重复；
- 不包含目标字段；
- 只是导航页；
- 只引用其他来源；

可以尽早终止处理。

未来可以引入：

```text
expected information gain
```

到 source 级别，而不仅仅用于 Adaptive Research 的下一轮决策。

## 4.7 Latency-aware Model Routing

当前 Dynamic Model Router 已经考虑：

- task type；
- complexity；
- context；
- cost；
- capability。

未来可以再增加：

```python
latency_score
```

模型选择变为：

```text
Capability hard filter
        ↓
Task affinity
        ↓
Quality
Cost
Latency
```

例如：

```text
Webpage Summarization
→ 优先低 latency model

Evidence Verification
→ 保留较强模型

Final Report
→ 在质量和 latency 之间平衡
```

从而实现：

```text
Cheap/Fast models
用于高频简单节点

Strong models
用于低频关键节点
```

## 4.8 Verifier 输入压缩

Evidence Verifier 当前需要读取较多：

```text
notes
raw_notes
```

未来可以考虑先构建：

```text
Evidence Index / Evidence Bundle
```

仅保留：

- claim；
- source；
- source type；
- date；
- supporting excerpt；
- unresolved gap。

然后 Verifier 读取结构化 evidence bundle，而不是完整 raw evidence。

这可能同时降低：

- context 长度；
- Verifier 延迟；
- Token 成本；
- 证据判断噪声。

---

# 5. 不建议的“延迟优化”

以下方式虽然可能让数字变快，但容易直接损害研究质量。

## 5.1 直接大幅减少 Search Budget

例如：

```text
12 tool calls
→ 2 tool calls
```

可能让 Agent 根本没有足够机会找到官方证据。

不应为了 latency 指标单独压低。

## 5.2 删除 Evidence Verifier

Verifier 是 AskFlow 当前架构的重要差异点。

删除：

```text
Research
→ Verifier
```

虽然会降低延迟，但会直接破坏：

- evidence sufficiency；
- credibility checking；
- conflict detection；
- targeted re-research；

不属于合理优化。

## 5.3 禁用 Targeted Research

Targeted Research 是：

```text
发现 evidence gap
→ 定向修复
```

它的额外 latency 是一种有意识的 reliability trade-off。

正确方向是让 targeted task 更窄、更高信息增益，而不是完全删除。

## 5.4 所有节点统一使用最快模型

不同节点的错误成本不同。

例如：

```text
Webpage Summarization
错误一次
→ 影响有限

Evidence Verifier
判断错误
→ 可能错误终止整个 research
```

因此应继续保持 task-aware model routing，而不是全局切换到单一 fast model。

---

# 6. 后续可观测性扩展

延迟优化必须建立在 profiling 上，而不是凭感觉修改。

后续可以增加节点级 metrics：

```text
Task
├── research_brief        1.2s
├── supervisor            4.8s
├── researcher           80.3s
│   ├── search            3.1s
│   └── summarization    65.4s
├── compression          12.7s
├── verifier             15.2s
└── final_report         18.6s
```

重点记录：

- node latency；
- model latency；
- search latency；
- webpage summarization count；
- summarization latency；
- timeout count；
- retry count；
- input/output tokens；
- cache hit；
- logical tool calls；
- actual model calls；
- targeted research rounds。

这样可以进一步形成：

```text
Latency Attribution
Cost Attribution
Quality Attribution
```

三类 profiling 数据。

---

# 7. 扩展点优先级建议

| 优先级 | 扩展点 | 主要收益 | 风险 |
|---|---|---|---|
| P1 | Search Result Ranking + Top-K | 大幅降低摘要调用与延迟 | 中 |
| P1 | Webpage Summary Cache | 降低重复调用、成本和延迟 | 低 |
| P1 | Researcher 输出协议结构化 | 提升状态语义与错误处理可靠性 | 低 |
| P1 | Node-level Latency / Cost Profiling | 为后续优化提供真实依据 | 低 |
| P2 | Semantic Deduplication | 减少重复 evidence processing | 中 |
| P2 | Adaptive Summarization | 减少不必要的长文本处理 | 中 |
| P2 | LangGraph `Send` Fan-out | 提升编排与可观测性 | 中 |
| P2 | Verifier Evidence Bundle | 降低长上下文与判断噪声 | 中 |
| P3 | Latency-aware Model Router | 进一步优化质量/成本/延迟平衡 | 中 |
| P3 | Batch / 更激进并发 | 提升吞吐 | 较高 |

---

# 8. 推荐后续路线

当前 AskFlow 版本完成 Eval 后，可以按以下顺序继续：

```text
Phase 1
Profiling
    ↓
确认真实 latency hotspot

Phase 2
Source Ranking
+ Dedup
+ Cache
    ↓
减少无效 Summarization

Phase 3
Researcher Structured Output
    ↓
完善 typed state

Phase 4
LangGraph Send Fan-out
    ↓
增强 worker 级 tracing / checkpoint

Phase 5
Verifier Evidence Bundle
+ Latency-aware Model Routing
    ↓
继续优化 Quality / Cost / Latency
```

---

# 9. 当前版本的工程取舍

AskFlow 当前选择：

```text
优先 Evidence Quality
优先 Research Reliability
优先 Failure Recovery
        ↓
接受一定 Latency Overhead
```

这是一种明确的工程取舍，而不是单纯的性能缺陷。

当前版本的重点是先证明：

```text
Verifier
Adaptive Re-Research
Research Budget
Tool Recovery
Dynamic Model Routing
```

能够提升 Deep Research 系统的可靠性和可控性。

后续再通过：

```text
Source Ranking
Semantic Dedup
Caching
Adaptive Summarization
Latency-aware Routing
```

逐步降低这一可靠性增强所引入的额外延迟。

因此，后续性能优化的目标不是简单追求：

```text
更少的步骤
```

而是：

```text
减少低价值计算
        +
保留高价值验证
        =
更好的 Quality / Cost / Latency 平衡
```
