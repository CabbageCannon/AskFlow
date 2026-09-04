"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
from typing import Literal, Any
import random

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
    evidence_verification_prompt,
    targeted_research_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
    VerificationResult,
    EvidenceGap,
    TargetedResearchTask,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)
from open_deep_research.tool_recovery import (
    RetryPolicy,
    classify_tool_error,
    infer_tool_policy,
    ToolExecutionResult,
)
from open_deep_research.search_fallback import (
    SearchFallbackPolicy,
    resolve_search_fallback_tool,
)

from open_deep_research.model_utils import (
    build_routed_model_runtime_config,
)

from open_deep_research.model_router import (
    TaskType,
    route_model_for_text,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    configurable_fields=("model", "max_tokens", "api_key", "base_url", "extra_body"),
)


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.

    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences

    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")

    # Step 2: Prepare the model for structured clarification analysis
    messages = state.get("messages",[])

    routing_text = get_buffer_string(messages)

    model_decision = route_model_for_text(
        task_type=TaskType.CLARIFICATION,
        text=routing_text,
        dynamic_enabled=(configurable.model_router_dynamic_enabled),
        prefer_low_cost=(configurable.model_router_prefer_low_cost),
    )

    model_config = build_routed_model_runtime_config(
        model_decision,
        api_key=configurable.bailian_api_key,
        base_url=configurable.bailian_base_url,
    )


    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model.with_structured_output(
            ClarifyWithUser, method="function_calling"
        )  # 强制LLM输出结构
        .with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        )  # 允许LLM重试输出
        .with_config(model_config)  # 一些运行配置
    )

    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        # get_buffer_string用于将messages这个list转换成字符串
        messages=get_buffer_string(messages),
        date=get_today_str(),
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.

    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)

    messages=state.get("messages", [])
    
    routing_text = get_buffer_string(messages)
    
    model_decision = route_model_for_text(
        task_type=TaskType.RESEARCH_BRIEF,
        text=routing_text,
        dynamic_enabled=(configurable.model_router_dynamic_enabled),
        prefer_low_cost=(configurable.model_router_prefer_low_cost),
    )

    research_model_config = build_routed_model_runtime_config(
        model_decision,
        api_key=configurable.bailian_api_key,
        base_url=configurable.bailian_base_url,
    )

    # Configure model for structured research question generation
    research_model = (
        configurable_model.with_structured_output(
            ResearchQuestion,
            method="function_calling",
        )
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief),
                ],
            },
        },
    )


async def supervisor(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.

    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.

    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    routing_text = state.get(
        "research_brief",
        "",
    )
    model_decision = route_model_for_text(
        task_type=TaskType.SUPERVISOR,
        text=routing_text,
        dynamic_enabled=(configurable.model_router_dynamic_enabled),
        prefer_low_cost=(configurable.model_router_prefer_low_cost),
    )
    research_model_config = build_routed_model_runtime_config(
        model_decision,
        api_key=configurable.bailian_api_key,
        base_url=configurable.bailian_base_url,
    )

    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # Configure model with tools, retry logic, and model settings
    # bind_tools强制LLM输出一种tool格式，与with_structured_output强制格式化输出是两种不同的模式
    research_model = (
        configurable_model.bind_tools(
            lead_researcher_tools
        )  # 告知LLM有哪些工具，其内部会是类似{"name": "get_weather", "description": "查询天气", "parameters": { "type": "object", "properties": { "city": { "type": "string" } } } }的结构
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(
        supervisor_messages
    )  # 这里返回的是类似AIMessage(content="", tool_calls=[ { "name": "ConductResearch", "args": { "research_topic": "LangGraph..." }, "id": "call_123" } ] )

    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.

    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase

    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings

    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Define exit criteria for research phase
    exceeded_allowed_iterations = (
        research_iterations > configurable.max_researcher_iterations
    )
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )

    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(
            ToolMessage(
                content=f"Reflection recorded: {reflection_content}",
                name="think_tool",
                tool_call_id=tool_call["id"],
            )
        )

    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[
                : configurable.max_concurrent_research_units
            ]
            overflow_conduct_research_calls = conduct_research_calls[
                configurable.max_concurrent_research_units :
            ]

            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke(
                    {
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"],
                    },
                    config,
                )
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # Create tool messages with research results
            for observation, tool_call in zip(
                tool_results, allowed_conduct_research_calls
            ):
                all_tool_messages.append(
                    ToolMessage(
                        content=observation.get(
                            "compressed_research",
                            "Error synthesizing research report: Maximum retries exceeded",
                        ),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(
                    ToolMessage(
                        content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                        name="ConductResearch",
                        tool_call_id=overflow_call["id"],
                    )
                )

            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join(
                [
                    "\n".join(observation.get("raw_notes", []))
                    for observation in tool_results
                ]
            )

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model):
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", ""),
                    },
                )

    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)  # Main supervisor logic
supervisor_builder.add_node(
    "supervisor_tools", supervisor_tools
)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.

    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.

    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability

    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )

    # Step 2: Configure the researcher model with tools
    routing_text = state.get(
        "research_topic",
        "",
    )
    model_decision = route_model_for_text(
        task_type=TaskType.RESEARCHER,
        text=routing_text,
        dynamic_enabled=(configurable.model_router_dynamic_enabled),
        prefer_low_cost=(configurable.model_router_prefer_low_cost),
    )

    research_model_config = build_routed_model_runtime_config(
        model_decision,
        api_key=configurable.bailian_api_key,
        base_url=configurable.bailian_base_url,
    )


    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", date=get_today_str()
    )

    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model.bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )

    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "react_iterations": 1,
        },
    )


# 检查当前工具调用错误是否需要重试
def is_retryable_tool_error(exception: Exception) -> bool:
    """Determine whether a tool error is likely transient and worth retrying."""

    # 1. Python / network level transient errors
    if isinstance(exception, (TimeoutError, ConnectionError)):
        return True

    # 2. Try to extract an HTTP status code
    status_code = getattr(exception, "status_code", None)

    if status_code is None:
        response = getattr(exception, "response", None)
        status_code = getattr(response, "status_code", None)

    if status_code is not None:
        # Rate limiting
        if status_code == 429:
            return True

        # Request timeout
        if status_code == 408:
            return True

        # Server-side errors
        if 500 <= status_code < 600:
            return True

        # Most other 4xx errors are permanent request errors
        if 400 <= status_code < 500:
            return False

    return False


# 控制工具，不计入预算型工具中
CONTROL_TOOL_NAMES = {"think_tool", "ResearchComplete"}


# 验证某个工具调用是否是预算型工具
def is_budgeted_tool_call(tool_call) -> bool:
    tool_call_name = tool_call.get("name", "")
    if tool_call_name in CONTROL_TOOL_NAMES:
        return False
    return True


async def execute_tool_with_recovery_result(
    tool, args, config, semaphore: asyncio.Semaphore
) -> ToolExecutionResult:
    """Execute a tool and return structured recovery state."""
    configurable = Configuration.from_runnable_config(config)

    # 最大重试次数
    max_retries = configurable.max_tool_retries
    # 工具尝试最大次数
    max_attempts = max_retries + 1

    # 拿到这类工具的策略
    tool_policy = infer_tool_policy(tool)
    retry_policy = RetryPolicy()

    for attempt in range(1, max_attempts + 1):
        try:
            async with semaphore:
                output = await tool.ainvoke(args, config)

            # 工具运行成功
            return ToolExecutionResult(
                output=output,
                error=None,
                exception=None,
                attempts=attempt,
                retry_budget_exhausted=False,
                decision=None,
            )

        except Exception as e:
            error_info = classify_tool_error(e)

            # 拿到重试决定
            decision = retry_policy.should_retry(
                error=error_info,
                tool=tool_policy,
                attempt=attempt,
                max_retries=max_retries,
            )

            print(
                "[TOOL_RECOVERY] "
                f"tool={tool_policy.name!r} | "
                f"kind={tool_policy.kind.value} | "
                f"attempt={attempt}/{max_attempts} | "
                f"category={error_info.category.value} | "
                f"status={error_info.status_code} | "
                f"transient={error_info.transient} | "
                f"idempotent={tool_policy.idempotent} | "
                f"retry={decision.should_retry} | "
                f"reason={decision.reason}"
            )

            if not decision.should_retry:
                return ToolExecutionResult(
                    output=None,
                    error=error_info,
                    exception=e,
                    attempts=attempt,
                    retry_budget_exhausted=(
                        decision.reason == "retry_budget_exhausted"
                    ),
                    decision=decision,
                )

            # 指数退让，基础等待最多8秒
            base_delay = min(2 ** (attempt - 1), 8)

            # 增加一个随机的摆动，避免大量工具同时失败，同时重试
            jitter = random.uniform(0, 1)

            delay = base_delay + jitter

            print(
                "[TOOL_RECOVERY] "
                f"tool={tool_policy.name!r} | "
                f"retry_in={delay:.2f}s"
            )

            await asyncio.sleep(delay)

    # 最终无论什么情况都会成功返回ToolExecuteResult,不会走到这里
    raise RuntimeError("unreachable tool recovery state")


# Tool Execution Helper Function
# fallback_tool为备选搜索源
async def execute_tool_safely(
    tool, args, config, semaphore: asyncio.Semaphore, fallback_tool=None
):
    """Execute a tool with infrastructure-level failure recovery."""
    result = await execute_tool_with_recovery_result(tool, args, config, semaphore)

    # 运行成功直接返回
    if result.output:
        return result.output

    # 理论上没有运行成功的话,这些一定不是None
    assert result.error is not None
    assert result.exception is not None
    assert result.decision is not None

    # 使用fallback_tool重试
    if fallback_tool is not None:
        primary_policy = infer_tool_policy(tool)
        fallback_policy = infer_tool_policy(fallback_tool)

        fallback_decision = SearchFallbackPolicy().decide(
            error=result.error,
            tool=primary_policy,
            primary_provider=primary_policy.name,
            fallback_provider=fallback_policy.name,
            retries_exhausted=result.retry_budget_exhausted,
            fallback_used=False,
        )

        print(
            "[SEARCH_FALLBACK] "
            f"primary={primary_policy.name!r} | "
            f"fallback={fallback_policy.name!r} | "
            f"allowed={fallback_decision.should_fallback} | "
            f"reason={fallback_decision.reason}"
        )

        if fallback_decision.should_fallback:
            fallback_result = await execute_tool_with_recovery_result(
                fallback_tool, args, config, semaphore
            )

            if fallback_result.success:
                return fallback_result.output

            assert fallback_result.error is not None
            assert fallback_result.exception is not None
            assert fallback_result.decision is not None

            return (
                f"Error executing fallback tool: {fallback_result.exception} "
                f"(category={fallback_result.error.category.value}, "
                f"reason={fallback_result.decision.reason})"
            )

    return (
        f"Error executing tool: {result.exception} "
        f"(category={result.error.category.value}, "
        f"reason={result.decision.reason})"
    )


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.

    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task

    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings

    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = openai_websearch_called(
        most_recent_message
    ) or anthropic_websearch_called(most_recent_message)

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls

    # 当前researcher总共运行了预算型tool的次数
    current_total_tool_calls = state.get("total_tool_calls", 0)

    # 本次研究剩余的可用预算型工具调用
    # 这里和0取max是因为有可能运行过程中用户更改配置导致算出来的是负数，这里为了后续运算合理，因此最少都要取0
    remaining_total_budget = max(
        configurable.max_total_tool_calls - current_total_tool_calls, 0
    )

    # 其中的预算型工具list
    budgeted_tool_calls = [
        tool_call for tool_call in tool_calls if is_budgeted_tool_call(tool_call)
    ]

    # 本轮实际最多批准的预算型工具调用
    allowed_budgeted_count = min(
        len(budgeted_tool_calls),  # 模型希望调用的预算型工具数
        configurable.max_tool_calls_per_iteration,  # 配置中允许的每次调用工具最大数
        remaining_total_budget,  # 配置中限制的每个researcher最大允许调用工具最大数
    )

    # 允许调用的工具列表
    allowed_tool_calls = []
    # 拒绝调用的工具列表
    rejected_tool_calls = []
    # 对应预算型工具被拒绝调用的原因
    rejected_reason_by_id = {}

    # 允许调用的预算型工具数目
    admitted_budgeted_count = 0

    # 选出可以调用的工具list
    for tool_call in tool_calls:
        if not is_budgeted_tool_call(tool_call):
            allowed_tool_calls.append(tool_call)
            continue

        if admitted_budgeted_count < allowed_budgeted_count:
            allowed_tool_calls.append(tool_call)
            admitted_budgeted_count += 1
            continue

        rejected_tool_calls.append(tool_call)

        # 这里先判断total budget是因为当两个限制同时触发时,优先告知总的到达上限,表达下一步就进入compression了,没必要再请求一次tool list
        if admitted_budgeted_count >= remaining_total_budget:
            rejected_reason_by_id[tool_call["id"]] = (
                "Skipped because the total research tool budget was exhausted."  # researcher总的tool调用次数到达上限
            )
        else:
            rejected_reason_by_id[tool_call["id"]] = (
                "Skipped because the per-iteration tool call limit was reached."  # 这一轮单次迭代到达上限
            )

    print(
        "[RESEARCH_BUDGET] "
        f"topic={state.get('research_topic', '')[:50]!r} | "
        f"react={state.get('react_iterations', 0)}/{configurable.max_react_iterations} | "
        f"total_before={current_total_tool_calls}/{configurable.max_total_tool_calls} | "
        f"generated={len(tool_calls)} | "
        f"budgeted={len(budgeted_tool_calls)} | "
        f"admitted={admitted_budgeted_count} | "
        f"rejected={len(rejected_tool_calls)} | "
        f"remaining_before={remaining_total_budget}"
    )

    # 创建并发信号量,限制并发数
    tool_semaphore = asyncio.Semaphore(configurable.max_concurrent_tool_calls)

    tool_execution_tasks = []

    for tool_call in allowed_tool_calls:
        tool = tools_by_name[tool_call["name"]]

        fallback_tool = None

        if configurable.search_fallback_enabled:
            fallback_tool = resolve_search_fallback_tool(
                primary_tool=tool,
                available_tools=list(tools_by_name.values()),
                preferred_fallback_tool_name=configurable.search_fallback_tool_name,
            )

        tool_execution_tasks.append(
            execute_tool_safely(
                tool,
                tool_call["args"],
                config,
                semaphore=tool_semaphore,
                fallback_tool=fallback_tool,
            )
        )

    observations = await asyncio.gather(*tool_execution_tasks)

    # Create tool messages from execution results
    # 创建ToolMessage
    tool_outputs_by_id = {}

    # 放入允许调用工具的结果
    for tool_call, observation in zip(allowed_tool_calls, observations):
        tool_outputs_by_id[tool_call["id"]] = ToolMessage(
            content=observation, name=tool_call["name"], tool_call_id=tool_call["id"]
        )

    # 放入被拒绝调用工具的结果(被拒绝原因，此处分为两类)
    for tool_call in rejected_tool_calls:
        tool_outputs_by_id[tool_call["id"]] = ToolMessage(
            content=rejected_reason_by_id[tool_call["id"]],
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )

    # 按照原本传进来的顺序创建outputs
    tool_outputs = [tool_outputs_by_id[tool_call["id"]] for tool_call in tool_calls]

    # Step 3: Check late exit conditions (after processing tools)
    reached_max_iterations = (
        state.get("react_iterations", 0) >= configurable.max_react_iterations
    )
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    total_budget_exhausted = (
        current_total_tool_calls + admitted_budgeted_count
        >= configurable.max_total_tool_calls
    )

    next_total_tool_calls = current_total_tool_calls + admitted_budgeted_count

    print(
        "[RESEARCH_BUDGET] "
        f"total_after={next_total_tool_calls}/{configurable.max_total_tool_calls} | "
        f"react_limit={reached_max_iterations} | "
        f"total_limit={total_budget_exhausted} | "
        f"research_complete={research_complete_called}"
    )

    # 到达researcher最大迭代次数 | 工具列表中有complete | 总的工具调用次数上限
    if reached_max_iterations or research_complete_called or total_budget_exhausted:
        print("[RESEARCH_BUDGET] route -> compress_research")
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "total_tool_calls": admitted_budgeted_count,
            },
        )

    print("[RESEARCH_BUDGET] route -> researcher")
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
            "total_tool_calls": admitted_budgeted_count,
        },
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.

    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.

    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings

    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = list(state.get("researcher_messages", []))
    routing_text = get_buffer_string(
        researcher_messages
    )
    model_decision=route_model_for_text(
        task_type=TaskType.COMPRESSION,
        text=routing_text,
        dynamic_enabled=(configurable.model_router_dynamic_enabled),
        prefer_low_cost=(configurable.model_router_prefer_low_cost),
    )

    compression_model_config = build_routed_model_runtime_config(
        model_decision,
        api_key=configurable.bailian_api_key,
        base_url=configurable.bailian_base_url,
    )

    synthesizer_model = configurable_model.with_config(compression_model_config)

    # Step 2: Prepare messages for compression
    

    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(
        HumanMessage(content=compress_research_simple_human_message)
    )

    # Step 3: Attempt compression with retry logic for token limit issues
    # 这里最多只允许裁剪3次，如果超过3次了则不压缩，返回错误信息
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(
                date=get_today_str()
            )
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            # Execute compression
            response = await synthesizer_model.ainvoke(messages)

            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join(
                [
                    str(message.content)
                    for message in filter_messages(
                        researcher_messages, include_types=["tool", "ai"]
                    )  # 筛选出工具信息和ai信息
                ]
            )

            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
            }

        except Exception as e:
            synthesis_attempts += 1

            # 上下文过长时，牺牲一次完整的研究过程，因为正常一般的研究过程就是一次AIMessage跟着多个ToolMessage
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, model_decision.model.model_name):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # For other errors, continue retrying
            continue

    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join(
        [
            str(message.content)
            for message in filter_messages(
                researcher_messages, include_types=["tool", "ai"]
            )
        ]
    )

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
    }


# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, output=ResearcherOutputState, config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)  # Main researcher logic
researcher_builder.add_node(
    "researcher_tools", researcher_tools
)  # Tool execution handler
researcher_builder.add_node(
    "compress_research", compress_research
)  # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")  # Entry point to researcher
researcher_builder.add_edge("compress_research", END)  # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


# 过滤Evidence_gap列表中预期信息收益低的那些
def get_actionable_evidence_gaps(
    state: AgentState,
    config: RunnableConfig,
) -> list[EvidenceGap]:
    """Return evidence gaps worth another targeted research round."""

    configurable = Configuration.from_runnable_config(config)

    verification_result = state.get("verification_result")

    if not verification_result:
        return []

    actionable_gaps = [
        gap
        for gap in verification_result.evidence_gaps
        if (gap.expected_information_gain >= configurable.min_expected_information_gain)
    ]

    actionable_gaps.sort(
        key=lambda gap: (gap.importance * gap.expected_information_gain),
        reverse=True,
    )

    # 最大的researcher并发数和每轮targeted_research允许最大的researcher数
    max_actionable_gaps = min(
        configurable.max_concurrent_research_units,
        configurable.max_targeted_research_tasks_per_round,
    )

    return actionable_gaps[:max_actionable_gaps]


# 检查搜索资料节点(总图)
async def evidence_verifier(state: AgentState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    researcher_brief=state.get("research_brief","")
    notes=state.get("notes",[])
    raw_notes=state.get("raw_notes",[])
    raw_notes_text = "\n".join(
        str(note)
        for note in raw_notes
    )
    notes_text="\n".join(
        str(note) for note in notes
    )
    routing_text = (
        researcher_brief
        + "\n"
        + notes_text
        + "\n"
        + raw_notes_text
    )
    model_decision = route_model_for_text(
        task_type=TaskType.EVIDENCE_VERIFICATION,
        text=routing_text,
        dynamic_enabled=(configurable.model_router_dynamic_enabled),
        prefer_low_cost=(configurable.model_router_prefer_low_cost),
    )

    verifier_model_config = build_routed_model_runtime_config(
        model_decision,
        api_key=configurable.bailian_api_key,
        base_url=configurable.bailian_base_url,
    )

    verification_model = (
        configurable_model.with_structured_output(
            VerificationResult, method="function_calling"
        )
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(verifier_model_config)
    )

    # 准备verfier输入
    researcher_brief = state.get("research_brief", "")
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])

    notes_text = "\n\n".join(notes)
    raw_notes_text = "\n\n".join(raw_notes)

    # 准备完整prompt
    prompt_content = evidence_verification_prompt.format(
        date=get_today_str(),
        research_brief=researcher_brief,  # 总的研究主题
        notes=notes_text,  # 最终final_report节点拿到的资料
        raw_notes=raw_notes_text,  # 原始材料
    )

    verification_result = await verification_model.ainvoke(
        [HumanMessage(content=prompt_content)]
    )

    print(
        "[EVIDENCE_VERIFIER] "
        f"result={verification_result!r} | "
        f"type={type(verification_result)}"
    )

    # 未来路由固定是到final_report_generation的，因此这里无需使用command
    return {
        "verification_result": verification_result,
        "verification_iterations": 1,  # 审查迭代
    }


# 审查后的路由节点
async def route_after_verification(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["plan_targeted_research", "final_report_generation"]]:
    """Decide whether evidence quality justifies another research round."""

    configurable = Configuration.from_runnable_config(config)

    verification_result = state.get("verification_result")

    verification_iterations = state.get(
        "verification_iterations",
        0,
    )

    # 1.Hard Stop(是否到达审查上限)->final
    if verification_iterations >= configurable.max_verification_iterations:
        print(
            "[ADAPTIVE_RESEARCH] "
            "route -> final_report_generation | "
            "reason=max_verification_iterations"
        )
        return Command(goto="final_report_generation")

    # 2.Soft Stop:Verifier认为资料已足够->final
    if verification_result and verification_result.evidence_sufficient:
        print(
            "[ADAPTIVE_RESEARCH] "
            "route -> final_report_generation | "
            "reason=evidence_sufficient"
        )

        return Command(goto="final_report_generation")

    # 3.没有审查结果时,Fail Safe->final 这里肯定是流程哪里出错了,因为当前节点只能verifier进入,不会没有verification_result
    if not verification_result:
        print(
            "[ADAPTIVE_RESEARCH] "
            "route -> final_report_generation | "
            "reason=missing_verification_result"
        )

        return Command(goto="final_report_generation")
    
    # 4.不期望继续搜索了
    if (
        not verification_result
        .further_research_likely_to_help
    ):
        print(
            "[ADAPTIVE_RESEARCH] "
            "route -> final_report_generation | "
            "reason=further_research_unlikely_to_help"
        )

        return Command(
            goto="final_report_generation"
        )

    # 拿到有价值的gap
    actionable_gaps = get_actionable_evidence_gaps(
        state,
        config,
    )

    # 4.没有值得补的gap时->final
    # 即使审查节点标识当前不满意,但是所需的gap都没有搜索的价值,那也直接final
    if not actionable_gaps:
        print(
            "[ADAPTIVE_RESEARCH] "
            "route -> final_report_generation | "
            "reason=no_actionable_evidence_gaps"
        )

        return Command(goto="final_report_generation")

    # 除以上4种情况，一律进入plan_targeted_research节点
    print(
        "[ADAPTIVE_RESEARCH] "
        "route -> plan_targeted_research | "
        f"actionable_gaps={len(actionable_gaps)} | "
        f"verification_iterations={verification_iterations}/"
        f"{configurable.max_verification_iterations}"
    )

    return Command(goto="plan_targeted_research")


# 根据gap不同的type设置不同策略并返回prompt
def build_targeted_research_instruction(gap: EvidenceGap, research_brief: str) -> str:
    """Convert one evidence gap into a narrowly scoped research task."""
    if gap.gap_type == "coverage":
        strategy = (
            "Find evidence specifically for this missing topic. "
            "Prioritize authoritative and primary sources when available."
        )

    elif gap.gap_type == "credibility":
        strategy = (
            "Find stronger evidence for this topic. "
            "Prioritize first-party documentation, primary sources, "
            "official data, or other authoritative evidence over weak "
            "secondary sources."
        )

    else:
        strategy = (
            "Investigate the conflicting evidence specifically. "
            "Determine whether the disagreement can be explained by "
            "date, version, scope, methodology, geography, or another "
            "material difference. Prefer authoritative sources that can "
            "help resolve the conflict."
        )

    return targeted_research_prompt.format(
        research_brief=research_brief,
        gap_topic=gap.topic,
        gap_reason=gap.reason,
        strategy=strategy,
    ).strip()


# 基于审查出来的缺口生成给researcher的新任务
async def plan_targeted_research(state: AgentState, config: RunnableConfig):
    """Convert verifier evidence gaps into targeted follow-up research tasks."""
    verification_result = state.get("verification_result")

    if not verification_result:
        return {"targeted_research_tasks": []}

    evidence_gaps = get_actionable_evidence_gaps(state, config)

    if not evidence_gaps:
        return {"targeted_research_tasks": []}

    research_brief = state.get("research_brief", "")

    tasks = []

    for gap in evidence_gaps:
        research_topic = build_targeted_research_instruction(
            gap,
            research_brief,
        )

        priority = gap.importance * gap.expected_information_gain

        tasks.append(
            TargetedResearchTask(
                gap_type=gap.gap_type,
                gap_topic=gap.topic,
                research_topic=research_topic,
                priority=priority,
            )
        )

    tasks.sort(key=lambda task: task.priority, reverse=True)

    print(
        "[ADAPTIVE_RESEARCH] "
        "planned targeted research | "
        f"tasks={len(tasks)} | "
        f"topics={[task.gap_topic for task in tasks]}"
    )

    return {"targeted_research_tasks": tasks}


# 标准化资料文本
def normalize_evidence_text(text: str) -> str:
    """Normalize evidence text for lightweight duplicate detection."""
    return " ".join(str(text).split())  # 按任意空格 \n 来切分成数组


# 过滤新资料中与就资料重复的资料
def filter_new_evidence(
    existing_items: list[str],
    candidate_items: list[str],
) -> list[str]:
    """Remove empty and exact/whitespace-equivalent duplicate evidence."""
    seen = {
        normalize_evidence_text(item)
        for item in existing_items
        if normalize_evidence_text(item)
    }

    new_items = []

    for item in candidate_items:
        text = str(item).strip()

        if not text:
            continue

        normalized = normalize_evidence_text(text)

        if normalized in seen:
            continue

        seen.add(normalized)
        new_items.append(text)

    return new_items


# 进行重新搜索
async def targeted_research(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["evidence_verifier", "final_report_generation"]]:
    """Execute targeted follow-up research without re-entering the Supervisor."""

    tasks = state.get("targeted_research_tasks", [])

    print(
        "[ADAPTIVE_RESEARCH] "
        "targeted research start | "
        f"tasks={len(tasks)} | "
        f"gaps={[task.gap_topic for task in tasks]}"
    )

    # 理论上Router 和 Planner已保证这里有任务，再做一次防御
    # Defensive fallback.
    if not tasks:
        print(
            "[ADAPTIVE_RESEARCH] "
            "route -> final_report_generation | "
            "reason=no_targeted_research_tasks"
        )

        return Command(
            goto="final_report_generation",
            update={
                "targeted_research_tasks": [],
            },
        )

    research_jobs = [
        researcher_subgraph.ainvoke(
            {
                "researcher_messages": [HumanMessage(content=task.research_topic)],
                "research_topic": (task.research_topic),
            },
            config,
        )
        for task in tasks
    ]

    results = await asyncio.gather(
        *research_jobs, return_exceptions=True  # 其中某个任务出错不会进入整个exception
    )

    # 收集新的evidence
    candidate_notes = []
    candidate_raw_notes = []

    # 记录正常返回结果的researcher子图,但不保证researcher内部节点运行正常,比如压缩节点出错,这里里面会自己处理错误最终返回字符串
    completed_tasks = 0

    for task, result in zip(
        tasks,
        results,
    ):
        # 如果失败
        if isinstance(result, Exception):
            print(
                "[ADAPTIVE_RESEARCH] "
                "targeted task failed | "
                f"gap={task.gap_topic!r} | "
                f"error={result}"
            )
            continue

        # 如果成功
        compressed_research = str(
            result.get(
                "compressed_research",
                "",
            )
        ).strip()

        raw_notes = result.get(
            "raw_notes",
            [],
        )

        # 判断压缩结果并保留压缩资料
        if compressed_research and not compressed_research.startswith(
            "Error synthesizing research report"
        ):
            candidate_notes.append(compressed_research)

        # 保留原始资料
        candidate_raw_notes.extend(str(note) for note in raw_notes if str(note).strip())

        completed_tasks += 1

    # 和已有资料去重
    existing_notes = state.get(
        "notes",
        [],
    )

    existing_raw_notes = state.get(
        "raw_notes",
        [],
    )

    new_notes = filter_new_evidence(
        existing_notes,
        candidate_notes,
    )

    new_raw_notes = filter_new_evidence(
        existing_raw_notes,
        candidate_raw_notes,
    )

    # 1.如果没有搜出任何有价值的资料，直接->final
    if not new_notes and not new_raw_notes:
        print(
            "[ADAPTIVE_RESEARCH] "
            "route -> final_report_generation | "
            "reason=no_new_evidence | "
            f"completed_tasks={completed_tasks}"
        )

        return Command(
            goto="final_report_generation",
            update={
                "targeted_research_tasks": [],
            },
        )

    # 2.放入notes和raw_notes中，并清空任务列表，为下一轮补充搜索做准备
    print(
        "[ADAPTIVE_RESEARCH] "
        "route -> evidence_verifier | "
        f"tasks={len(tasks)} | "
        f"completed={completed_tasks} | "
        f"new_notes={len(new_notes)} | "
        f"new_raw_notes={len(new_raw_notes)}"
    )

    return Command(
        goto="evidence_verifier",
        update={
            "notes": new_notes,
            "raw_notes": new_raw_notes,
            # 当前这一批任务已经执行完，
            # 必须清空，不能下一轮重复执行。
            "targeted_research_tasks": [],
        },
    )


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.

    This function takes all collected research findings and synthesizes them into a
    well-structured, comprehensive final report using the configured report generation model.

    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys

    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    research_brief=state.get("research_brief","")
    cleared_state = {
        "notes": {"type": "override", "value": []}
    }  # 用于更新状态时把notes清空
    findings = "\n".join(notes)
    verification_result = state.get("verification_result", "")
    # model_dump_json将baseModel转换成json格式字符串
    verification_result_text = (
        verification_result.model_dump_json(indent=2) if verification_result else "{}"
    )

    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    routing_text=(research_brief+"\n"+findings+"\n"+verification_result_text)
    model_decision = route_model_for_text(
        task_type=TaskType.FINAL_REPORT,
        text=routing_text,
        dynamic_enabled=(configurable.model_router_dynamic_enabled),
        prefer_low_cost=(configurable.model_router_prefer_low_cost),
    )

    writer_model_config = build_routed_model_runtime_config(
        model_decision,
        api_key=configurable.bailian_api_key,
        base_url=configurable.bailian_base_url,
        no_stream=False,
    )


    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                verification_result=verification_result_text,
                date=get_today_str(),
            )

            # Generate the final report
            final_report = await configurable_model.with_config(
                writer_model_config
            ).ainvoke([HumanMessage(content=final_report_prompt)])

            # Return successful report generation
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state,
            }

        except Exception as e:
            token_limit_exceeded = is_token_limit_exceeded(
                e,
                model_decision.model.model_name,
            )

            print(
                "[FINAL_REPORT_ERROR] "
                f"model={model_decision.model.model_name!r} | "
                f"attempt={current_retry + 1}/{max_retries + 1} | "
                f"type={type(e).__name__} | "
                f"token_limit={token_limit_exceeded} | "
                f"error={e}"
            )
            # Handle token limit exceeded errors with progressive truncation
            if token_limit_exceeded:
                current_retry += 1

                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = model_decision.model.context_window
                    # 理论上ModelSpec一定会有这个字段
                    # if not model_token_limit:
                    #     return {
                    #         "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                    #         "messages": [
                    #             AIMessage(
                    #                 content="Report generation failed due to token limits"
                    #             )
                    #         ],
                    #         **cleared_state,
                    #     }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)

                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [
                        AIMessage(content="Report generation failed due to an error")
                    ],
                    **cleared_state,
                }

    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [
            AIMessage(content="Report generation failed after maximum retries")
        ],
        **cleared_state,
    }


# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState,
    # agent的输入只需要AgentInputState
    input=AgentInputState,
    config_schema=Configuration,
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node(
    "clarify_with_user", clarify_with_user
)  # User clarification phase
deep_researcher_builder.add_node(
    "write_research_brief", write_research_brief
)  # Research planning phase
deep_researcher_builder.add_node(
    "research_supervisor", supervisor_subgraph
)  # Research execution phase
deep_researcher_builder.add_node(
    "evidence_verifier", evidence_verifier
)  # 搜索资料检查阶段
deep_researcher_builder.add_node(
    "route_after_verification", route_after_verification
)  # 审查后路由
deep_researcher_builder.add_node(
    "plan_targeted_research",
    plan_targeted_research,
)  # 根据审查结果撰写补充research任务
deep_researcher_builder.add_node(
    "targeted_research",
    targeted_research,
)  # 进行补充research
deep_researcher_builder.add_node(
    "final_report_generation", final_report_generation
)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")  # Entry point
deep_researcher_builder.add_edge(
    "research_supervisor", "evidence_verifier"
)  # Research to verifier
deep_researcher_builder.add_edge(
    "evidence_verifier", "route_after_verification"
)  # verifier->router
deep_researcher_builder.add_edge(
    "plan_targeted_research", "targeted_research"
)  # plan->targeted_research
deep_researcher_builder.add_edge("final_report_generation", END)  # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
