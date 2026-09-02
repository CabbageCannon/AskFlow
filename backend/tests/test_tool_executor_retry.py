import asyncio

import open_deep_research.deep_researcher as dr
from open_deep_research.tool_recovery import ToolKind
from open_deep_research.tool_recovery import ToolErrorCategory


class FakeSearchTool:
    def __init__(self, failures):
        self.name = "fake_search"
        self.metadata = {
            "type": "search",
        }
        self.failures = list(failures)
        self.call_count = 0

    async def ainvoke(self, args, config):
        self.call_count += 1

        if self.failures:
            error = self.failures.pop(0)
            raise error

        return "success"


class FakeMCPTool:
    def __init__(self):
        self.name = "create_order"
        self.metadata = {
            "type": "mcp",
        }
        self.call_count = 0

    async def ainvoke(self, args, config):
        self.call_count += 1
        raise TimeoutError("timeout")
      
def patch_config(monkeypatch, max_tool_retries=3):
    config = dr.Configuration(
        max_tool_retries=max_tool_retries,
    )

    monkeypatch.setattr(
        dr.Configuration,
        "from_runnable_config",
        classmethod(
            lambda cls, runnable_config=None: config
        ),
    )

    return config
  
def patch_sleep(monkeypatch):
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(
        dr.asyncio,
        "sleep",
        fake_sleep,
    )

    return sleep_calls
  
def test_transient_search_failure_retries_then_succeeds(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=3)
    sleep_calls = patch_sleep(monkeypatch)

    tool = FakeSearchTool(
        failures=[
            TimeoutError("timeout"),
        ]
    )

    result = asyncio.run(
        dr.execute_tool_safely(
            tool,
            {},
            {},
            asyncio.Semaphore(1),
        )
    )

    assert result == "success"

    assert tool.call_count == 2

    assert len(sleep_calls) == 1
    
class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = FakeResponse(status_code)
        
def test_500_retries(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=2)
    patch_sleep(monkeypatch)

    tool = FakeSearchTool(
        failures=[
            FakeHTTPError(500),
        ]
    )

    result = asyncio.run(
        dr.execute_tool_safely(
            tool,
            {},
            {},
            asyncio.Semaphore(1),
        )
    )

    assert result == "success"
    assert tool.call_count == 2
    
def test_401_does_not_retry(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=3)
    sleep_calls = patch_sleep(monkeypatch)

    tool = FakeSearchTool(
        failures=[
            FakeHTTPError(401),
        ]
    )

    result = asyncio.run(
        dr.execute_tool_safely(
            tool,
            {},
            {},
            asyncio.Semaphore(1),
        )
    )

    assert tool.call_count == 1
    assert len(sleep_calls) == 0

    assert "authentication" in result
    assert "non_transient_error" in result
    
def test_non_idempotent_mcp_tool_does_not_retry(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=3)
    sleep_calls = patch_sleep(monkeypatch)

    tool = FakeMCPTool()

    result = asyncio.run(
        dr.execute_tool_safely(
            tool,
            {},
            {},
            asyncio.Semaphore(1),
        )
    )

    assert tool.call_count == 1
    assert len(sleep_calls) == 0

    assert "non_idempotent_tool" in result
    
class AlwaysTimeoutSearchTool:
    def __init__(self):
        self.name = "always_timeout_search"
        self.metadata = {
            "type": "search",
        }
        self.call_count = 0

    async def ainvoke(self, args, config):
        self.call_count += 1
        raise TimeoutError("timeout")
    
def test_structured_result_success(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=3)
    patch_sleep(monkeypatch)

    tool = FakeSearchTool(failures=[])

    result = asyncio.run(
        dr.execute_tool_with_recovery_result(
            tool,
            {},
            {},
            asyncio.Semaphore(1),
        )
    )

    assert result.success is True
    assert result.output == "success"
    assert result.error is None
    assert result.exception is None
    assert result.attempts == 1
    assert result.retry_budget_exhausted is False
    assert result.decision is None


def test_structured_result_marks_retry_budget_exhausted(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=2)
    sleep_calls = patch_sleep(monkeypatch)

    tool = AlwaysTimeoutSearchTool()

    result = asyncio.run(
        dr.execute_tool_with_recovery_result(
            tool,
            {},
            {},
            asyncio.Semaphore(1),
        )
    )

    assert result.success is False
    assert result.output is None
    assert result.error.category == ToolErrorCategory.TIMEOUT
    assert result.exception is not None
    assert result.attempts == 3
    assert result.retry_budget_exhausted is True
    assert result.decision.reason == "retry_budget_exhausted"

    assert tool.call_count == 3
    assert len(sleep_calls) == 2


def test_structured_result_non_transient_is_not_retry_exhaustion(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=3)
    sleep_calls = patch_sleep(monkeypatch)

    tool = FakeSearchTool(
        failures=[
            FakeHTTPError(401),
        ]
    )

    result = asyncio.run(
        dr.execute_tool_with_recovery_result(
            tool,
            {},
            {},
            asyncio.Semaphore(1),
        )
    )

    assert result.success is False
    assert result.error.category == ToolErrorCategory.AUTHENTICATION
    assert result.attempts == 1
    assert result.retry_budget_exhausted is False
    assert result.decision.reason == "non_transient_error"

    assert tool.call_count == 1
    assert len(sleep_calls) == 0
    
class SuccessfulFallbackSearchTool:
    def __init__(self):
        self.name = "fallback_search"
        self.metadata = {
            "type": "search",
        }
        self.call_count = 0

    async def ainvoke(self, args, config):
        self.call_count += 1
        return "fallback success"
    
def test_search_fallback_runs_after_primary_retry_budget_exhausted(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=1)
    sleep_calls = patch_sleep(monkeypatch)

    primary_tool = AlwaysTimeoutSearchTool()
    fallback_tool = SuccessfulFallbackSearchTool()

    result = asyncio.run(
        dr.execute_tool_safely(
            primary_tool,
            {},
            {},
            asyncio.Semaphore(1),
            fallback_tool=fallback_tool,
        )
    )

    assert result == "fallback success"

    assert primary_tool.call_count == 2
    assert fallback_tool.call_count == 1
    assert len(sleep_calls) == 1
    
def test_search_fallback_does_not_run_for_non_transient_error(monkeypatch):
    patch_config(monkeypatch, max_tool_retries=3)
    sleep_calls = patch_sleep(monkeypatch)

    primary_tool = FakeSearchTool(
        failures=[
            FakeHTTPError(401),
        ]
    )
    fallback_tool = SuccessfulFallbackSearchTool()

    result = asyncio.run(
        dr.execute_tool_safely(
            primary_tool,
            {},
            {},
            asyncio.Semaphore(1),
            fallback_tool=fallback_tool,
        )
    )

    assert primary_tool.call_count == 1
    assert fallback_tool.call_count == 0
    assert len(sleep_calls) == 0

    assert "authentication" in result
    assert "non_transient_error" in result