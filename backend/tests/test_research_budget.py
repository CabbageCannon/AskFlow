import asyncio

from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
import open_deep_research.deep_researcher as dr

# ============================================================
# Test Helpers
# ============================================================


def make_tool_call(name: str, call_id: str):
    """构造一个假的 LLM tool_call。"""
    return {
        "name": name,
        "args": {"value": call_id},
        "id": call_id,
        "type": "tool_call",
    }


def make_state(
    tool_calls,
    *,
    react_iterations: int = 1,
    total_tool_calls: int = 0,
):
    """构造 researcher_tools 所需要的最小 State。"""
    return {
        "researcher_messages": [
            AIMessage(
                content="",
                tool_calls=tool_calls,
            )
        ],
        "react_iterations": react_iterations,
        "total_tool_calls": total_tool_calls,
        "research_topic": "Research budget controller test",
    }


class FakeTool:
    """普通的假 Tool。"""

    def __init__(self, name: str):
        self.name = name

    async def ainvoke(self, args, config):
        return f"result:{self.name}"


def patch_config(monkeypatch, **overrides):
    """
    固定 Configuration，避免测试受到 .env / Studio Runtime Config 影响。
    """
    values = {
        "max_react_iterations": 10,
        "max_tool_calls_per_iteration": 10,
        "max_total_tool_calls": 100,
        "max_concurrent_tool_calls": 10,
        "max_tool_retries": 0,
    }

    values.update(overrides)

    config = dr.Configuration(**values)

    monkeypatch.setattr(
        dr.Configuration,
        "from_runnable_config",
        classmethod(lambda cls, runnable_config=None: config),
    )

    return config


def patch_tools(monkeypatch, tools):
    """让 researcher_tools 获取我们自己提供的假 Tool。"""

    async def fake_get_all_tools(config):
        return tools

    monkeypatch.setattr(
        dr,
        "get_all_tools",
        fake_get_all_tools,
    )


def run_researcher_tools(state):
    """普通 pytest 中运行 async researcher_tools。"""
    return asyncio.run(
        dr.researcher_tools(
            state,
            {},
        )
    )


# ============================================================
# Test 1
# max_tool_calls_per_iteration
# + control tool 不计预算
# + ToolMessage 原始顺序保持
# ============================================================


def test_tool_calls_per_iteration_limit(monkeypatch):
    patch_config(
        monkeypatch,
        max_tool_calls_per_iteration=1,
        max_total_tool_calls=10,
        max_react_iterations=10,
    )

    tools = [
        FakeTool("search_a", "search"),
        FakeTool("think_tool", "control"),
        FakeTool("search_b", "search"),
        FakeTool("search_c", "search"),
    ]

    patch_tools(monkeypatch, tools)

    executed_tools = []

    async def fake_execute_tool_safely(
        tool,
        args,
        config,
        *extra_args,
        **extra_kwargs,
    ):
        executed_tools.append(tool.name)
        return f"result:{tool.name}"

    monkeypatch.setattr(
        dr,
        "execute_tool_safely",
        fake_execute_tool_safely,
    )

    tool_calls = [
        make_tool_call("search_a", "call_a"),
        make_tool_call("think_tool", "call_think"),
        make_tool_call("search_b", "call_b"),
        make_tool_call("search_c", "call_c"),
    ]

    state = make_state(tool_calls)

    command = run_researcher_tools(state)

    # --------------------------------------------------------
    # 1. search_a 消耗预算
    # 2. think_tool 不消耗预算，所以也应该执行
    # 3. search_b / search_c 被拒绝
    # --------------------------------------------------------

    assert executed_tools == [
        "search_a",
        "think_tool",
    ]

    # 本轮只消费 1 个 budgeted tool
    assert command.update["total_tool_calls"] == 1

    # 总预算没耗尽，应该继续 researcher
    assert command.goto == "researcher"

    tool_outputs = command.update["researcher_messages"]

    # ToolMessage 必须严格保持原始 AIMessage.tool_calls 的顺序
    assert [msg.tool_call_id for msg in tool_outputs] == [
        "call_a",
        "call_think",
        "call_b",
        "call_c",
    ]

    assert "result:search_a" in str(tool_outputs[0].content)
    assert "result:think_tool" in str(tool_outputs[1].content)

    assert "Skipped" in str(tool_outputs[2].content)
    assert "Skipped" in str(tool_outputs[3].content)


# ============================================================
# Test 2
# max_total_tool_calls
# + 剩余 budget 不足
# + budget 耗尽进入 compression
# + terminal branch 仍然记最后一笔账
# ============================================================


def test_total_tool_budget_exhaustion_routes_to_compression(monkeypatch):
    patch_config(
        monkeypatch,
        max_tool_calls_per_iteration=5,
        max_total_tool_calls=3,
        max_react_iterations=10,
    )

    tools = [
        FakeTool("search_a", "search"),
        FakeTool("search_b", "search"),
    ]

    patch_tools(monkeypatch, tools)

    executed_tools = []

    async def fake_execute_tool_safely(
        tool,
        args,
        config,
        *extra_args,
        **extra_kwargs,
    ):
        executed_tools.append(tool.name)
        return f"result:{tool.name}"

    monkeypatch.setattr(
        dr,
        "execute_tool_safely",
        fake_execute_tool_safely,
    )

    tool_calls = [
        make_tool_call("search_a", "call_a"),
        make_tool_call("search_b", "call_b"),
    ]

    # 已经用了 2 / 3
    # 所以本轮虽然生成两个 Tool，
    # 实际只能再批准一个。
    state = make_state(
        tool_calls,
        total_tool_calls=2,
    )

    command = run_researcher_tools(state)

    assert executed_tools == ["search_a"]

    # 注意：
    # researcher_tools 返回的是 reducer delta，
    # 不是最终累计值。
    #
    # State:
    # 2 + update(1) = 3
    assert command.update["total_tool_calls"] == 1

    # 3 / 3，必须直接进入 compression
    assert command.goto == "compress_research"

    tool_outputs = command.update["researcher_messages"]

    assert [msg.tool_call_id for msg in tool_outputs] == [
        "call_a",
        "call_b",
    ]

    assert "result:search_a" in str(tool_outputs[0].content)
    assert "Skipped" in str(tool_outputs[1].content)


# ============================================================
# Test 3
# max_react_iterations
# 达到第 N 轮以后完成本轮 Tool，再进入 compression
# ============================================================


def test_react_iteration_limit_routes_to_compression(monkeypatch):
    patch_config(
        monkeypatch,
        max_react_iterations=3,
        max_tool_calls_per_iteration=5,
        max_total_tool_calls=20,
    )

    tools = [
        FakeTool("search_a", "search"),
    ]

    patch_tools(monkeypatch, tools)

    tool_calls = [
        make_tool_call("search_a", "call_a"),
    ]

    # 注意：
    # react_iterations 在 researcher() 中已经 +1。
    #
    # 这里进入 researcher_tools 时已经是第 3 轮。
    state = make_state(
        tool_calls,
        react_iterations=3,
        total_tool_calls=0,
    )

    command = run_researcher_tools(state)

    # 第三轮 Tool 应该仍然执行并计费
    assert command.update["total_tool_calls"] == 1

    # 完整执行第 3 轮后进入 compression
    assert command.goto == "compress_research"


# ============================================================
# Test 4
# max_concurrent_tool_calls
# 真正验证 Semaphore
# ============================================================


def test_max_concurrent_tool_calls(monkeypatch):
    patch_config(
        monkeypatch,
        max_react_iterations=10,
        max_tool_calls_per_iteration=6,
        max_total_tool_calls=20,
        max_concurrent_tool_calls=2,
        max_tool_retries=0,
    )

    tracker = {
        "active": 0,
        "peak": 0,
    }

    class SlowTool:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, args, config):
            tracker["active"] += 1

            tracker["peak"] = max(
                tracker["peak"],
                tracker["active"],
            )

            try:
                # 故意 sleep，让多个 Tool 有机会发生并发重叠
                await asyncio.sleep(0.05)

                return f"result:{self.name}"

            finally:
                tracker["active"] -= 1

    tools = [SlowTool(f"search_{i}") for i in range(6)]

    patch_tools(monkeypatch, tools)

    tool_calls = [
        make_tool_call(
            f"search_{i}",
            f"call_{i}",
        )
        for i in range(6)
    ]

    state = make_state(tool_calls)

    command = run_researcher_tools(state)

    # 一共有 6 个 Tool 被批准
    assert command.update["total_tool_calls"] == 6

    # 关键断言：
    # 任意时刻真正执行 Tool 的数量不能超过 2
    assert tracker["peak"] <= 2

    # 同时确认确实发生过并发，而不是退化成串行
    assert tracker["peak"] == 2

    assert command.goto == "researcher"


# ============================================================
# Test 5
# Retry 不应该重复消耗 logical tool budget
# ============================================================


def test_retry_does_not_consume_extra_tool_budget(monkeypatch):
    patch_config(
        monkeypatch,
        max_react_iterations=10,
        max_tool_calls_per_iteration=5,
        max_total_tool_calls=20,
        max_concurrent_tool_calls=1,
        max_tool_retries=1,
    )

    class RetryOnceTool:
        def __init__(self):
            self.name = "retry_search"
            self.metadata = {
                "type": "search",
            }
            self.attempts = 0

        async def ainvoke(self, args, config):
            self.attempts += 1

            if self.attempts == 1:
                raise TimeoutError("temporary timeout")

            return "success after retry"

    retry_tool = RetryOnceTool()

    patch_tools(
        monkeypatch,
        [retry_tool],
    )

    tool_calls = [
        make_tool_call(
            "retry_search",
            "call_retry",
        )
    ]

    state = make_state(tool_calls)

    command = run_researcher_tools(state)

    # 实际 Tool attempt 有两次
    assert retry_tool.attempts == 2

    # 但是逻辑 Tool Invocation 只有一次
    assert command.update["total_tool_calls"] == 1


# ============================================================
# Test 6
# react_iteration 和 total_tool_calls会累加
# ============================================================


def test_graph_total_budget_one_stops_after_first_iteration(monkeypatch):
    """
    集成测试：
    max_total_tool_calls = 1

    预期：
    researcher -> researcher_tools -> compress_research -> END

    最终 State:
    react_iterations == 1
    total_tool_calls == 1
    """

    patch_config(
        monkeypatch,
        max_react_iterations=5,
        max_tool_calls_per_iteration=3,
        max_total_tool_calls=1,
        max_concurrent_tool_calls=1,
        max_tool_retries=0,
    )

    fake_tool = FakeTool(
        "search",
        "search",
    )

    patch_tools(
        monkeypatch,
        [fake_tool],
    )

    researcher_call_count = 0

    async def fake_researcher(state, config):
        nonlocal researcher_call_count
        researcher_call_count += 1

        response = AIMessage(
            content="",
            tool_calls=[
                make_tool_call(
                    "search",
                    f"call_{researcher_call_count}",
                )
            ],
        )

        return Command(
            goto="researcher_tools",
            update={
                "researcher_messages": [response],
                # 注意：
                # reducer 会执行旧值 + 1
                "react_iterations": 1,
            },
        )

    async def fake_compress_research(state, config):
        return {
            "compressed_research": "compressed",
            "raw_notes": [],
        }

    # ------------------------------
    # 构造一个测试专用 Researcher Graph
    # ------------------------------

    builder = StateGraph(dr.ResearcherState)

    builder.add_node(
        "researcher",
        fake_researcher,
    )

    # 这里故意使用你真正写的 researcher_tools
    builder.add_node(
        "researcher_tools",
        dr.researcher_tools,
    )

    builder.add_node(
        "compress_research",
        fake_compress_research,
    )

    builder.add_edge(
        START,
        "researcher",
    )

    builder.add_edge(
        "compress_research",
        END,
    )

    graph = builder.compile()

    result = asyncio.run(
        graph.ainvoke(
            {
                "researcher_messages": [],
                "research_topic": "Budget integration test",
            },
            {},
        )
    )

    # ------------------------------
    # Assertions
    # ------------------------------

    # 只允许发生一次 researcher iteration
    assert researcher_call_count == 1

    # 验证 react_iterations reducer
    assert result["react_iterations"] == 1

    # 验证 total_tool_calls reducer
    assert result["total_tool_calls"] == 1

    # 确实成功进入 compression
    assert result["compressed_research"] == "compressed"


# ============================================================
# Test 7
# react_iteration 和 total_tool_calls 到达上限会进入压缩节点
# ============================================================


def test_graph_budget_accumulates_across_two_iterations(monkeypatch):
    """
    集成测试：

    max_tool_calls_per_iteration = 1
    max_total_tool_calls = 2

    预期：
    第 1 轮 total: 0 -> 1
    第 2 轮 total: 1 -> 2
    然后进入 compression

    最终：
    react_iterations == 2
    total_tool_calls == 2
    """

    patch_config(
        monkeypatch,
        max_react_iterations=5,
        max_tool_calls_per_iteration=1,
        max_total_tool_calls=2,
        max_concurrent_tool_calls=1,
        max_tool_retries=0,
    )

    fake_tool = FakeTool("search", "search")

    patch_tools(
        monkeypatch,
        [fake_tool],
    )

    researcher_call_count = 0

    async def fake_researcher(state, config):
        nonlocal researcher_call_count
        researcher_call_count += 1

        response = AIMessage(
            content="",
            tool_calls=[
                make_tool_call(
                    "search",
                    f"call_{researcher_call_count}",
                )
            ],
        )

        return Command(
            goto="researcher_tools",
            update={
                "researcher_messages": [response],
                "react_iterations": 1,
            },
        )

    async def fake_compress_research(state, config):
        return {
            "compressed_research": "compressed",
            "raw_notes": [],
        }

    builder = StateGraph(dr.ResearcherState)

    builder.add_node(
        "researcher",
        fake_researcher,
    )

    builder.add_node(
        "researcher_tools",
        dr.researcher_tools,
    )

    builder.add_node(
        "compress_research",
        fake_compress_research,
    )

    builder.add_edge(
        START,
        "researcher",
    )

    builder.add_edge(
        "compress_research",
        END,
    )

    graph = builder.compile()

    result = asyncio.run(
        graph.ainvoke(
            {
                "researcher_messages": [],
                "research_topic": "Two iteration budget test",
            },
            {},
        )
    )

    # 一共真实进入 researcher 两次
    assert researcher_call_count == 2

    # 证明 react reducer：
    #
    # 0 + 1 + 1 = 2
    assert result["react_iterations"] == 2

    # 证明 budget reducer：
    #
    # 0 + 1 + 1 = 2
    assert result["total_tool_calls"] == 2

    assert result["compressed_research"] == "compressed"


class AlwaysTimeoutPrimarySearchTool:
    def __init__(self):
        self.name = "primary_search"
        self.metadata = {
            "type": "search",
        }
        self.call_count = 0

    async def ainvoke(self, args, config):
        self.call_count += 1
        raise TimeoutError("primary timeout")


class SuccessfulFallbackSearchTool:
    def __init__(self):
        self.name = "fallback_search"
        self.metadata = {
            "type": "search",
        }
        self.call_count = 0

    async def ainvoke(self, args, config):
        self.call_count += 1
        return "fallback result"


def test_researcher_tools_uses_search_fallback_after_retry_exhausted(monkeypatch):
    patch_config(
        monkeypatch,
        max_react_iterations=10,
        max_tool_calls_per_iteration=5,
        max_total_tool_calls=20,
        max_concurrent_tool_calls=1,
        max_tool_retries=1,
    )

    primary_tool = AlwaysTimeoutPrimarySearchTool()
    fallback_tool = SuccessfulFallbackSearchTool()

    patch_tools(
        monkeypatch,
        [
            primary_tool,
            fallback_tool,
        ],
    )

    tool_calls = [
        make_tool_call(
            "primary_search",
            "call_primary",
        )
    ]

    state = make_state(tool_calls)

    command = run_researcher_tools(state)

    assert primary_tool.call_count == 2
    assert fallback_tool.call_count == 1
    assert command.update["total_tool_calls"] == 1


def test_researcher_tools_does_not_use_search_fallback_for_non_transient_error(
    monkeypatch,
):
    patch_config(
        monkeypatch,
        max_react_iterations=10,
        max_tool_calls_per_iteration=5,
        max_total_tool_calls=20,
        max_concurrent_tool_calls=1,
        max_tool_retries=1,
    )

    primary_tool = AuthFailurePrimarySearchTool()
    fallback_tool = SuccessfulFallbackSearchTool()

    patch_tools(
        monkeypatch,
        [
            primary_tool,
            fallback_tool,
        ],
    )

    tool_calls = [
        make_tool_call(
            "auth_failure_search",
            "call_auth_failure",
        )
    ]

    state = make_state(tool_calls)

    command = run_researcher_tools(state)

    assert primary_tool.call_count == 1
    assert fallback_tool.call_count == 0
    assert command.update["total_tool_calls"] == 1


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = FakeResponse(status_code)


class AuthFailurePrimarySearchTool:
    def __init__(self):
        self.name = "auth_failure_search"
        self.metadata = {
            "type": "search",
        }
        self.call_count = 0

    async def ainvoke(self, args, config):
        self.call_count += 1
        raise FakeHTTPError(401)


def test_researcher_tools_respects_search_fallback_disabled(monkeypatch):
    patch_config(
        monkeypatch,
        max_react_iterations=10,
        max_tool_calls_per_iteration=5,
        max_total_tool_calls=20,
        max_concurrent_tool_calls=1,
        max_tool_retries=1,
        search_fallback_enabled=False,
    )

    primary_tool = AlwaysTimeoutPrimarySearchTool()
    fallback_tool = SuccessfulFallbackSearchTool()

    patch_tools(
        monkeypatch,
        [
            primary_tool,
            fallback_tool,
        ],
    )

    tool_calls = [
        make_tool_call(
            "primary_search",
            "call_primary",
        )
    ]

    state = make_state(tool_calls)

    command = run_researcher_tools(state)

    assert primary_tool.call_count == 2
    assert fallback_tool.call_count == 0
    assert command.update["total_tool_calls"] == 1

    tool_outputs = command.update["researcher_messages"]

    assert len(tool_outputs) == 1
    assert tool_outputs[0].tool_call_id == "call_primary"
    assert "retry_budget_exhausted" in str(tool_outputs[0].content)


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
            ToolErrorCategory.TIMEOUT if transient else ToolErrorCategory.BAD_REQUEST
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


def test_researcher_tools_uses_configured_search_fallback_tool(monkeypatch):
    patch_config(
        monkeypatch,
        max_react_iterations=10,
        max_tool_calls_per_iteration=5,
        max_total_tool_calls=20,
        max_concurrent_tool_calls=1,
        max_tool_retries=1,
        search_fallback_enabled=True,
        search_fallback_tool_name="fallback_search",
    )

    primary_tool = AlwaysTimeoutPrimarySearchTool()
    fallback_tool = SuccessfulFallbackSearchTool()
    other_fallback_tool = SuccessfulFallbackSearchTool()
    other_fallback_tool.name = "other_fallback_search"

    patch_tools(
        monkeypatch,
        [
            primary_tool,
            other_fallback_tool,
            fallback_tool,
        ],
    )

    tool_calls = [
        make_tool_call(
            "primary_search",
            "call_primary",
        )
    ]

    state = make_state(tool_calls)

    command = run_researcher_tools(state)

    assert primary_tool.call_count == 2
    assert other_fallback_tool.call_count == 0
    assert fallback_tool.call_count == 1
    assert command.update["total_tool_calls"] == 1

    tool_outputs = command.update["researcher_messages"]

    assert "fallback result" in str(tool_outputs[0].content)
