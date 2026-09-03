"""Tool failure classification and recovery primitives."""

from typing import Any
from dataclasses import dataclass
from enum import StrEnum

# 不同的库对于不同的错误可能有不同的命名,这里归一到当前错误系统当中
TRANSIENT_TIMEOUT_TYPES = {
    "TimeoutError",
    "Timeout",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "TimeoutException",
    "ServerTimeoutError",
    "ConnectionTimeoutError",
    "SocketTimeoutError",
}


TRANSIENT_CONNECTION_TYPES = {
    "ConnectionError",
    "ConnectError",
    "ReadError",
    "WriteError",
    "CloseError",
    "NetworkError",
    "ClientConnectionError",
    "ClientConnectorError",
    "ServerConnectionError",
    "ServerDisconnectedError",
    "RemoteProtocolError",
}


RATE_LIMIT_TYPES = {
    "UsageLimitExceededError",
    "TavilyKeylessLimitError",
}


AUTHENTICATION_TYPES = {
    "InvalidAPIKeyError",
    "MissingAPIKeyError",
}


AUTHORIZATION_TYPES = {
    "ForbiddenError",
}


BAD_REQUEST_TYPES = {
    "BadRequestError",
}

# 工具错误类别(继承StrEnum等价于多继承str,Enum既具有枚举属性又具有字符串属性)
class ToolErrorCategory(StrEnum):
  """Normalized categories for tool execution failures."""
  TIMEOUT = "timeout"
  CONNECTION = "connection"
  RATE_LIMIT = "rate_limit"
  SERVER_ERROR = "server_error"

  BAD_REQUEST = "bad_request"
  AUTHENTICATION = "authentication"
  AUTHORIZATION = "authorization"
  NOT_FOUND = "not_found"
  VALIDATION = "validation"
  CLIENT_ERROR = "client_error"

  TOOL_ERROR = "tool_error"
  UNKNOWN = "unknown"
  
# 自动补充如_init_这样的函数
@dataclass(frozen=True) # frozen取true表示只读
class ToolErrorInfo:
    """Normalized information extracted from an arbitrary tool exception."""

    category: ToolErrorCategory

    # 这里只表示“这个错误本身是否像 transient failure”
    # 不代表最终一定允许 retry。
    # 是否是临时问题
    transient: bool

    status_code: int | None
    status_source: str | None

    exception_type: str
    exception_module: str
    message: str
    
    
# 提取状态码
def _normalize_status_code(value)->int|None:
  """Convert a possible HTTP status value into an integer."""
  
  # bool 是 int 子类,因此如果value是bool类型后面用int来转换成int会成功,这里先排除掉
  if value is None or isinstance(value,bool):
    return None

  try:
    status_code=int(value)
  except:
    return None
  
  if 100<=status_code <=599:
    return status_code
  
  return None

# 从各种不同类型的 HTTP 异常对象里，尽可能统一地提取出 HTTP 状态码
def extract_http_status(exception:Exception)->tuple[int|None,str|None]: # 返回(状态码,状态码来源)
  """Extract HTTP status code from common Python HTTP exception shapes."""
  
  # 1.OpenAI / provider SDK 等部分异常可能直接提供 status_code
  status_code = _normalize_status_code(
    getattr(exception, "status_code", None)
  )
      
  if status_code is not None:
    return status_code, "exception.status_code"
  
  # 2.requests.HTTPError / httpx.HTTPStatusError 常见形式      
  response = getattr(exception, "response", None)
  
  if response is not None:
    status_code = _normalize_status_code(
        getattr(response, "status_code", None)
    )

    if status_code is not None:
        return status_code, "exception.response.status_code"
      
  # 3.aiohttp.ClientResponseError 使用 .status
  status_code = _normalize_status_code(
      getattr(exception, "status", None)
  )

  if status_code is not None:
      return status_code, "exception.status"
    
  # 做一层兼容
  if response is not None:
      status_code = _normalize_status_code(
          getattr(response, "status", None)
      )

      if status_code is not None:
          return status_code, "exception.response.status"
        
  return None,None

# 根据状态码判断能否重试
def _classify_http_status(
    status_code: int,
) -> tuple[ToolErrorCategory, bool]:
    """Classify an HTTP failure by status code."""

    if status_code == 408:
        return ToolErrorCategory.TIMEOUT, True

    if status_code == 429:
        return ToolErrorCategory.RATE_LIMIT, True

    if 500 <= status_code < 600:
        return ToolErrorCategory.SERVER_ERROR, True

    if status_code == 400:
        return ToolErrorCategory.BAD_REQUEST, False

    if status_code == 401:
        return ToolErrorCategory.AUTHENTICATION, False

    if status_code == 403:
        return ToolErrorCategory.AUTHORIZATION, False

    if status_code == 404:
        return ToolErrorCategory.NOT_FOUND, False

    if status_code == 422:
        return ToolErrorCategory.VALIDATION, False

    if 400 <= status_code < 500:
        return ToolErrorCategory.CLIENT_ERROR, False

    return ToolErrorCategory.UNKNOWN, False
  
# 根据异常对象判断错误类型
def classify_tool_error(exception: Exception) -> ToolErrorInfo:
    """Normalize an arbitrary tool exception into a common error model."""

    exception_type = exception.__class__.__name__
    exception_module = getattr(
        exception.__class__,
        "__module__",
        "",
    )

    status_code, status_source = extract_http_status(exception)

    # HTTP 状态是最强证据，优先使用,我们无法从错误名字判断错误是什么,但可以从状态码这一个数字知道
    if status_code is not None:
        category, transient = _classify_http_status(status_code)

        return ToolErrorInfo(
            category=category,
            transient=transient,
            status_code=status_code,
            status_source=status_source,
            exception_type=exception_type,
            exception_module=exception_module,
            message=str(exception),
        )

    # 没有HTTP状态码,再看Python原生异常
    # Python 原生异常
    if isinstance(exception, TimeoutError):
        category = ToolErrorCategory.TIMEOUT
        transient = True

    elif isinstance(exception, ConnectionError):
        category = ToolErrorCategory.CONNECTION
        transient = True

    # 如果也不是Python异常,那只能根据名字判断了
    # Provider / HTTP client exception type
    elif exception_type in TRANSIENT_TIMEOUT_TYPES:
        category = ToolErrorCategory.TIMEOUT
        transient = True

    elif exception_type in TRANSIENT_CONNECTION_TYPES:
        category = ToolErrorCategory.CONNECTION
        transient = True

    elif exception_type in RATE_LIMIT_TYPES:
        category = ToolErrorCategory.RATE_LIMIT
        transient = True

    elif exception_type in AUTHENTICATION_TYPES:
        category = ToolErrorCategory.AUTHENTICATION
        transient = False

    elif exception_type in AUTHORIZATION_TYPES:
        category = ToolErrorCategory.AUTHORIZATION
        transient = False

    elif exception_type in BAD_REQUEST_TYPES:
        category = ToolErrorCategory.BAD_REQUEST
        transient = False

    elif exception_type == "ToolException":
        category = ToolErrorCategory.TOOL_ERROR
        transient = False

    else:
        category = ToolErrorCategory.UNKNOWN
        transient = False

    return ToolErrorInfo(
        category=category,
        transient=transient,
        status_code=status_code,
        status_source=status_source,
        exception_type=exception_type,
        exception_module=exception_module,
        message=str(exception),
    )
    
# 工具的种类,当前项目中只有这三种
class ToolKind(StrEnum):
    SEARCH = "search"
    MCP = "mcp"
    CONTROL = "control"
    UNKNOWN = "unknown"

# 工具类型
@dataclass(frozen=True)
class ToolPolicy:
    """Recovery-related properties of a tool."""

    name: str
    kind: ToolKind

    # 是否可以安全重复调用(幂等性，是否重试后是无副作用的，一般是一些只读类型的tool)
    idempotent: bool

    # 是否允许 infrastructure-level retry(本身是否是值得重试的)
    retry_enabled: bool = True

CONTROL_TOOL_NAMES = {
    "think_tool",
    "ResearchComplete",
}

# 根据工具类型来指定策略
def infer_tool_policy(tool) -> ToolPolicy:
    """Infer a conservative recovery policy from tool metadata."""

    name = getattr(tool, "name", "unknown")

    metadata = getattr(tool, "metadata", None) or {}
    tool_type = metadata.get("type")

    # Search tools are read-only by nature in this project.
    if tool_type == "search":
        return ToolPolicy(
            name=name,
            kind=ToolKind.SEARCH,
            idempotent=True,
            retry_enabled=True,
        )

    if name in CONTROL_TOOL_NAMES:
        return ToolPolicy(
            name=name,
            kind=ToolKind.CONTROL,
            idempotent=True,
            retry_enabled=False,
        )

    # MCP tools默认采用最保守策略。
    if tool_type == "mcp":
        return ToolPolicy(
            name=name,
            kind=ToolKind.MCP,
            idempotent=False, # 全都认为不安全，比如create_order这种
            retry_enabled=True,
        )

    # 不知道是什么工具时，不允许盲目重试。
    return ToolPolicy(
        name=name,
        kind=ToolKind.UNKNOWN,
        idempotent=False,
        retry_enabled=True,
    )
    
# 重试类型
@dataclass(frozen=True)
class RetryDecision:
    """Decision returned by the retry policy."""

    should_retry: bool
    reason: str
    
class RetryPolicy:
    """Decide whether an infrastructure-level tool retry is allowed."""

    def should_retry(
        self,
        *,
        error: ToolErrorInfo,
        tool: ToolPolicy,
        attempt: int,
        max_retries: int,
    ) -> RetryDecision:

        if not tool.retry_enabled:
            return RetryDecision(
                should_retry=False,
                reason="retry_disabled_for_tool",
            )

        if not error.transient:
            return RetryDecision(
                should_retry=False,
                reason="non_transient_error",
            )

        if not tool.idempotent:
            return RetryDecision(
                should_retry=False,
                reason="non_idempotent_tool",
            )

        # attempt=1 表示第一次执行已经失败。
        # max_retries=3，则允许 attempt 1/2/3 后继续重试，
        # attempt=4 时已经没有 retry budget。
        if attempt > max_retries:
            return RetryDecision(
                should_retry=False,
                reason="retry_budget_exhausted",
            )

        return RetryDecision(
            should_retry=True,
            reason="transient_idempotent_failure",
        )
        
# 工具运行结果
@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured result from infrastructure-level tool execution."""

    output: Any | None
    error: ToolErrorInfo | None
    exception: Exception | None
    attempts: int
    retry_budget_exhausted: bool
    decision: RetryDecision | None

    @property
    def success(self) -> bool:
        return self.error is None