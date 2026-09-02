"""Search provider contracts and fallback decisions."""

from dataclasses import dataclass
from typing import Any, Protocol
from open_deep_research.tool_recovery import infer_tool_policy

from open_deep_research.tool_recovery import ToolErrorInfo, ToolKind, ToolPolicy


# 不同服务对外提供相同的输入
@dataclass(frozen=True)
class SearchRequest:
    """Provider-independent search input."""

    query: str
    max_results: int = 5


# 一个搜索结构返回一个SearchSource
@dataclass(frozen=True)
class SearchSource:
    """One source returned by a search provider."""

    title: str
    url: str
    content: str


# 不同服务对外提供相同的输出
@dataclass(frozen=True)
class SearchResponse:
    """Provider-independent search output."""

    provider: str
    sources: tuple[SearchSource, ...]  # 任意长度的tuple，必须全部是SearchSource


# 接口契约，只要某个类长得像如下这种形式，就可以被当作SearchProvider使用
class SearchProvider(Protocol):
    """Contract implemented by each search provider adapter."""

    name: str

    async def search(
        self,
        request: SearchRequest,
        config: Any,
    ) -> SearchResponse: ...


@dataclass(frozen=True)
class FallbackDecision:
    """Whether a failed search may switch providers."""

    should_fallback: bool
    reason: str


# 决定是否可以降级到另一个search provider
class SearchFallbackPolicy:
    """Decide whether a search provider switch is allowed."""

    def decide(
        self,
        *,
        error: ToolErrorInfo,
        tool: ToolPolicy,
        primary_provider: str,
        fallback_provider: str | None,
        retries_exhausted: bool,
        fallback_used: bool,
    ) -> FallbackDecision:

        if tool.kind != ToolKind.SEARCH:
            return FallbackDecision(
                False,
                "not_search_tool",
            )

        if fallback_used:
            return FallbackDecision(
                False,
                "fallback_already_used",
            )

        if not fallback_provider:
            return FallbackDecision(
                False,
                "fallback_not_configured",
            )

        if fallback_provider == primary_provider:
            return FallbackDecision(
                False,
                "same_provider",
            )

        if not error.transient:
            return FallbackDecision(
                False,
                "non_transient_error",
            )

        if not retries_exhausted:
            return FallbackDecision(
                False,
                "primary_retries_remaining",
            )

        return FallbackDecision(
            True,
            "transient_failure_after_retries",
        )

# 从工具列表中找出备选搜索工具
def resolve_search_fallback_tool(
  *,
  primary_tool: Any,
  available_tools: list[Any],
  preferred_fallback_tool_name: str | None = None,
)->Any|None:
  """Find a fallback search tool for a primary search tool."""
  primary_policy = infer_tool_policy(primary_tool)
  
  if primary_policy.kind is not ToolKind.SEARCH:
        return None
  
  # 备选工具列表
  candidates=[]
      
  for candidate in available_tools:
        candidate_policy = infer_tool_policy(candidate)

        if candidate_policy.kind is not ToolKind.SEARCH:
            continue

        if candidate_policy.name == primary_policy.name:
            continue

        candidates.append(candidate)
    
  # 如果能从工具列表找到指定的备选工具就找,找不到就直接返回None
  if preferred_fallback_tool_name is not None:
        for candidate in candidates:
            candidate_policy = infer_tool_policy(candidate)

            if candidate_policy.name == preferred_fallback_tool_name:
                return candidate

        return None
      
  if candidates:return candidates[0]
      
  return None