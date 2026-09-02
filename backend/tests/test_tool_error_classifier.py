from open_deep_research.tool_recovery import (
    ToolErrorCategory,
    classify_tool_error,
)


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class ResponseStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = FakeResponse(status_code)


class DirectStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_timeout_is_transient():
    info = classify_tool_error(TimeoutError("timeout"))

    assert info.category == ToolErrorCategory.TIMEOUT
    assert info.transient is True
    assert info.status_code is None


def test_429_is_transient_rate_limit():
    info = classify_tool_error(ResponseStatusError(429))

    assert info.category == ToolErrorCategory.RATE_LIMIT
    assert info.transient is True
    assert info.status_code == 429
    assert info.status_source == "exception.response.status_code"


def test_500_is_transient_server_error():
    info = classify_tool_error(ResponseStatusError(500))

    assert info.category == ToolErrorCategory.SERVER_ERROR
    assert info.transient is True


def test_401_is_non_transient_auth_error():
    info = classify_tool_error(ResponseStatusError(401))

    assert info.category == ToolErrorCategory.AUTHENTICATION
    assert info.transient is False


def test_422_is_non_transient_validation_error():
    info = classify_tool_error(ResponseStatusError(422))

    assert info.category == ToolErrorCategory.VALIDATION
    assert info.transient is False


def test_direct_status_code_is_supported():
    info = classify_tool_error(DirectStatusError(408))

    assert info.category == ToolErrorCategory.TIMEOUT
    assert info.transient is True
    assert info.status_source == "exception.status_code"


def test_unknown_error_is_not_transient():
    info = classify_tool_error(ValueError("unexpected"))

    assert info.category == ToolErrorCategory.UNKNOWN
    assert info.transient is False