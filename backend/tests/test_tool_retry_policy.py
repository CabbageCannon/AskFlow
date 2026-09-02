from open_deep_research.tool_recovery import (
    RetryPolicy,
    ToolErrorCategory,
    ToolErrorInfo,
    ToolKind,
    ToolPolicy,
)


def make_error(*, transient: bool):
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


def test_transient_idempotent_tool_can_retry():
    policy = RetryPolicy()

    decision = policy.should_retry(
        error=make_error(transient=True),
        tool=ToolPolicy(
            name="tavily_search",
            kind=ToolKind.SEARCH,
            idempotent=True,
        ),
        attempt=1,
        max_retries=3,
    )

    assert decision.should_retry is True
    assert decision.reason == "transient_idempotent_failure"


def test_non_transient_error_does_not_retry():
    policy = RetryPolicy()

    decision = policy.should_retry(
        error=make_error(transient=False),
        tool=ToolPolicy(
            name="tavily_search",
            kind=ToolKind.SEARCH,
            idempotent=True,
        ),
        attempt=1,
        max_retries=3,
    )

    assert decision.should_retry is False
    assert decision.reason == "non_transient_error"


def test_non_idempotent_tool_does_not_retry():
    policy = RetryPolicy()

    decision = policy.should_retry(
        error=make_error(transient=True),
        tool=ToolPolicy(
            name="create_order",
            kind=ToolKind.MCP,
            idempotent=False,
        ),
        attempt=1,
        max_retries=3,
    )

    assert decision.should_retry is False
    assert decision.reason == "non_idempotent_tool"


def test_retry_disabled_tool_does_not_retry():
    policy = RetryPolicy()

    decision = policy.should_retry(
        error=make_error(transient=True),
        tool=ToolPolicy(
            name="think_tool",
            kind=ToolKind.CONTROL,
            idempotent=True,
            retry_enabled=False,
        ),
        attempt=1,
        max_retries=3,
    )

    assert decision.should_retry is False
    assert decision.reason == "retry_disabled_for_tool"


def test_retry_budget_exhausted():
    policy = RetryPolicy()

    decision = policy.should_retry(
        error=make_error(transient=True),
        tool=ToolPolicy(
            name="tavily_search",
            kind=ToolKind.SEARCH,
            idempotent=True,
        ),
        attempt=4,
        max_retries=3,
    )

    assert decision.should_retry is False
    assert decision.reason == "retry_budget_exhausted"