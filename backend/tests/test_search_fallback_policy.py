import pytest

from open_deep_research.search_fallback import SearchFallbackPolicy
from open_deep_research.tool_recovery import (
    ToolErrorCategory,
    ToolErrorInfo,
    ToolKind,
    ToolPolicy,
)


def make_error(transient=True):
    return ToolErrorInfo(
        category=(
            ToolErrorCategory.TIMEOUT
            if transient
            else ToolErrorCategory.BAD_REQUEST
        ),
        transient=transient,
        status_code=None,
        status_source=None,
        exception_type="FakeError",
        exception_module="tests",
        message="fake error",
    )


def make_tool(kind=ToolKind.SEARCH):
    return ToolPolicy(
        name="fake_tool",
        kind=kind,
        idempotent=(kind == ToolKind.SEARCH),
    )


@pytest.mark.parametrize(
    "overrides, expected, reason",
    [
        (
            {},
            True,
            "transient_failure_after_retries",
        ),
        (
            {"error": make_error(transient=False)},
            False,
            "non_transient_error",
        ),
        (
            {"tool": make_tool(ToolKind.MCP)},
            False,
            "not_search_tool",
        ),
        (
            {"tool": make_tool(ToolKind.UNKNOWN)},
            False,
            "not_search_tool",
        ),
        (
            {"fallback_provider": None},
            False,
            "fallback_not_configured",
        ),
        (
            {"fallback_provider": "tavily"},
            False,
            "same_provider",
        ),
        (
            {"fallback_used": True},
            False,
            "fallback_already_used",
        ),
        (
            {"retries_exhausted": False},
            False,
            "primary_retries_remaining",
        ),
    ],
)
def test_search_fallback_policy(overrides, expected, reason):
    inputs = {
        "error": make_error(),
        "tool": make_tool(),
        "primary_provider": "tavily",
        "fallback_provider": "backup_search",
        "retries_exhausted": True,
        "fallback_used": False,
    }
    inputs.update(overrides)

    decision = SearchFallbackPolicy().decide(**inputs)

    assert decision.should_fallback is expected
    assert decision.reason == reason
    
from open_deep_research.search_fallback import resolve_search_fallback_tool


class FakeTool:
    def __init__(self, name, tool_type):
        self.name = name
        self.metadata = {
            "type": tool_type,
        }


def test_resolves_another_search_tool_as_fallback():
    primary = FakeTool("tavily_search", "search")
    fallback = FakeTool("openai_search", "search")

    result = resolve_search_fallback_tool(
        primary_tool=primary,
        available_tools=[primary, fallback],
    )

    assert result is fallback


def test_does_not_resolve_fallback_for_non_search_primary():
    primary = FakeTool("create_order", "mcp")
    fallback = FakeTool("openai_search", "search")

    result = resolve_search_fallback_tool(
        primary_tool=primary,
        available_tools=[primary, fallback],
    )

    assert result is None


def test_ignores_non_search_candidates():
    primary = FakeTool("tavily_search", "search")
    mcp_tool = FakeTool("create_order", "mcp")

    result = resolve_search_fallback_tool(
        primary_tool=primary,
        available_tools=[primary, mcp_tool],
    )

    assert result is None


def test_does_not_use_same_tool_as_fallback():
    primary = FakeTool("tavily_search", "search")

    result = resolve_search_fallback_tool(
        primary_tool=primary,
        available_tools=[primary],
    )

    assert result is None
    
def test_resolves_preferred_search_fallback_tool():
    primary = FakeTool("tavily_search", "search")
    fallback_a = FakeTool("openai_search", "search")
    fallback_b = FakeTool("anthropic_search", "search")

    result = resolve_search_fallback_tool(
        primary_tool=primary,
        available_tools=[primary, fallback_a, fallback_b],
        preferred_fallback_tool_name="anthropic_search",
    )

    assert result is fallback_b


def test_returns_none_when_preferred_search_fallback_is_missing():
    primary = FakeTool("tavily_search", "search")
    fallback = FakeTool("openai_search", "search")

    result = resolve_search_fallback_tool(
        primary_tool=primary,
        available_tools=[primary, fallback],
        preferred_fallback_tool_name="missing_search",
    )

    assert result is None


def test_does_not_use_primary_as_preferred_fallback():
    primary = FakeTool("tavily_search", "search")
    fallback = FakeTool("openai_search", "search")

    result = resolve_search_fallback_tool(
        primary_tool=primary,
        available_tools=[primary, fallback],
        preferred_fallback_tool_name="tavily_search",
    )

    assert result is None
    
