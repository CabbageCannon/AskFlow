"""Task-aware model routing primitives."""

from enum import StrEnum
from dataclasses import dataclass
from typing import Any

# 任务类型
class TaskType(StrEnum):
    """LLM-backed task types in the research workflow."""

    CLARIFICATION = "clarification"
    RESEARCH_BRIEF = "research_brief"
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    WEBPAGE_SUMMARIZATION = "webpage_summarization"
    COMPRESSION = "compression"
    EVIDENCE_VERIFICATION = "evidence_verification"
    FINAL_REPORT = "final_report"

# 硬需求
@dataclass(frozen=True)
class TaskRequirements:
    requires_tools: bool = False
    requires_structured_output: bool = False

# 任务对硬需求映射
TASK_REQUIREMENTS = {
    TaskType.CLARIFICATION: TaskRequirements(
        requires_structured_output=True,
    ),

    TaskType.RESEARCH_BRIEF: TaskRequirements(
        requires_structured_output=True,
    ),

    TaskType.SUPERVISOR: TaskRequirements(
        requires_tools=True,
    ),

    TaskType.RESEARCHER: TaskRequirements(
        requires_tools=True,
    ),

    TaskType.WEBPAGE_SUMMARIZATION: TaskRequirements(
        requires_structured_output=True,
    ),

    TaskType.COMPRESSION: TaskRequirements(),

    TaskType.EVIDENCE_VERIFICATION: TaskRequirements(
        requires_structured_output=True,
    ),

    TaskType.FINAL_REPORT: TaskRequirements(),
}

# 任务复杂度
class TaskComplexity(StrEnum):
    SIMPLE="simple"
    MEDIUM="medium"
    COMPLEX="complex"

# 任务复杂度评价
@dataclass(frozen=True)
class ComplexityAssessment:
    complexity: TaskComplexity
    score: int
    reasons: tuple[str, ...]

# 具体任务分析
@dataclass(frozen=True)
class RoutingContext:
    complexity: TaskComplexity = TaskComplexity.MEDIUM

    estimated_input_tokens: int = 0
    reserved_output_tokens: int = 0

    prefer_low_cost: bool = False
    context_safety_margin_tokens: int = 8_192

# 任务与输出空间token的映射
TASK_RESERVED_OUTPUT_TOKENS = {
    TaskType.CLARIFICATION: 1_024,
    TaskType.RESEARCH_BRIEF: 2_048,

    TaskType.SUPERVISOR: 4_096,
    TaskType.RESEARCHER: 8_192,

    TaskType.WEBPAGE_SUMMARIZATION: 4_096,

    TaskType.COMPRESSION: 16_384,

    TaskType.EVIDENCE_VERIFICATION: 8_192,

    TaskType.FINAL_REPORT: 16_384,
}

TOOL_REQUIRED_TASKS = {
    TaskType.SUPERVISOR,
    TaskType.RESEARCHER,
}
STRUCTURED_OUTPUT_TASKS = {
    TaskType.CLARIFICATION,
    TaskType.RESEARCH_BRIEF,
    TaskType.WEBPAGE_SUMMARIZATION,
    TaskType.EVIDENCE_VERIFICATION,
}

# 任务需求
class ModelProfile(StrEnum):
    """Provider-independent model capability profiles."""

    FAST = "fast"
    REASONING = "reasoning"
    SYNTHESIS="synthesis"
    WRITER = "writer"

# 在router中注册模型配置结构
@dataclass(frozen=True)
class ModelSpec:
    model_name: str

    # 上下文/输出限制
    max_tokens: int
    context_window: int

    # 硬能力要求
    supports_tools: bool
    supports_structured_output: bool

    # 价格成本
    input_cost_per_million: float
    output_cost_per_million: float

    # reasoning_strength: int
    # writing_strength: int
    # synthesis_strength: int
    # speed_score: int

    # 模型额外的运行时参数
    extra_body: dict[str, Any] | None = None

# 模型分类
MODEL_CATALOG:tuple[ModelSpec,...]=(
    ModelSpec(
        model_name="openai:deepseek-v4-flash",
        max_tokens=8192,
        context_window=1_000_000,
        supports_tools=True,
        supports_structured_output=True,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
        extra_body={
            "enable_thinking": False,
        },
    ),
    ModelSpec(
        model_name="openai:glm-5.2",
        max_tokens=16_384,
        context_window=1_048_576,
        supports_tools=True,
        supports_structured_output=True,
        input_cost_per_million=8.0,
        output_cost_per_million=28.0,
        extra_body={
            "enable_thinking": False,
            "tool_stream": True,
        },
    ),
    ModelSpec(
        model_name="openai:qwen3.8-27b",
        max_tokens=16_384,
        context_window=1_000_000,
        supports_tools=True,
        supports_structured_output=True,
        input_cost_per_million=3.0,
        output_cost_per_million=12.0,
        extra_body={
            "enable_thinking": False,
        },
    ),
    ModelSpec(
        model_name="openai:MiniMax/MiniMax-M3",
        max_tokens=16_384,
        context_window=1_048_576,
        supports_tools=True,
        supports_structured_output=False,
        input_cost_per_million=4.2,
        output_cost_per_million=16.8,
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
    ),
)

DEFAULT_TASK_PROFILES = {
    TaskType.CLARIFICATION: ModelProfile.FAST,
    TaskType.RESEARCH_BRIEF: ModelProfile.FAST,

    TaskType.SUPERVISOR: ModelProfile.REASONING,
    TaskType.RESEARCHER: ModelProfile.REASONING,

    TaskType.WEBPAGE_SUMMARIZATION: ModelProfile.FAST,

    TaskType.COMPRESSION: ModelProfile.SYNTHESIS,

    TaskType.EVIDENCE_VERIFICATION: ModelProfile.REASONING,

    TaskType.FINAL_REPORT: ModelProfile.WRITER,
}


def get_default_profile_for_task(
    task_type: TaskType,
) -> ModelProfile:
    """Return the baseline model profile for a task."""

    return DEFAULT_TASK_PROFILES[task_type]

MODEL_BY_NAME={
    model.model_name:model
    for model in MODEL_CATALOG
}

# Fast使用deepseek-v4-flash
# Reasoning使用glm-5.2
# Long_context使用qwen3.8
# Writing使用minmax
PROFILE_MODELS = {
    ModelProfile.FAST: MODEL_BY_NAME[
        "openai:deepseek-v4-flash"
    ],

    ModelProfile.REASONING: MODEL_BY_NAME[
        "openai:glm-5.2"
    ],

    ModelProfile.SYNTHESIS: MODEL_BY_NAME[
         "openai:qwen3.8-27b"
    ],
    ModelProfile.WRITER: MODEL_BY_NAME[
        "openai:MiniMax/MiniMax-M3"
    ],
}


@dataclass(frozen=True)
class ModelDecision:
    """Result produced by the model router."""

    task_type: TaskType
    profile: ModelProfile
    model: ModelSpec
    reason: str
    
    candidates: tuple[str, ...] = ()
    score: float | None = None
    estimated_cost: float | None = None

# 日志函数
def log_model_decision(
    decision: ModelDecision,
) -> None:

    print(
        "[MODEL_ROUTER] "
        f"task={decision.task_type.value} | "
        f"profile={decision.profile.value} | "
        f"model={decision.model.model_name!r} | "
        f"score={decision.score} | "
        f"estimated_cost="
        f"{decision.estimated_cost} | "
        f"candidates="
        f"{decision.candidates} | "
        f"reason={decision.reason}"
    )
      
# 筛选模型时如果没有满足硬能力需求的,就抛出当前错误
class NoCompatibleModelError(RuntimeError):
    """Raised when no configured model satisfies hard requirements."""
           
def filter_compatible_models(
    task_type:TaskType,
    context: RoutingContext,
    models: tuple[ModelSpec, ...] = MODEL_CATALOG,
) -> list[ModelSpec]:
    """Filter models using hard task requirements."""

    # 硬需求
    # 1.支持工具调用和结构化输出
    requirements = TASK_REQUIREMENTS[task_type]
    
    # 2.上下文需求
    required_context=(
        context.estimated_input_tokens
        +context.reserved_output_tokens
        +context.context_safety_margin_tokens
    )

    # 候选模型列表
    candidates:list[ModelSpec] = []

    for model in models:

        # 从硬能力(是否支持tool_calling,structed output和上下文限制进行筛选)
        # 1. Tool Calling hard constraint
        if (
            requirements.requires_tools
            and not model.supports_tools
        ):
            continue

        # 2. Structured Output hard constraint
        if (
            requirements.requires_structured_output
            and not model.supports_structured_output
        ):
            continue

        # 3. Context capacity hard constraint
        if (
            required_context > 0
            and required_context
            > model.context_window
        ):
            continue

        candidates.append(model)

    return candidates

# 近似将2个中文字定为1个token,后续可优化
def estimate_text_tokens(
    text: str,
) -> int:
    """Conservative provider-neutral token estimate."""

    if not text:
        return 0

    # Conservative for mixed English/Chinese text.
    return max(1, len(text) // 2)

# 不借助LLM的前提下评估任务的复杂度
def estimate_task_complexity(
    text: str,
    task_type: TaskType,
) -> ComplexityAssessment:
    """Estimate task complexity without another LLM call."""

    normalized = text.lower().strip()

    score = 0
    reasons: list[str] = []

    # -----------------------------
    # 1. Prompt length
    # -----------------------------

    text_length = len(text)

    if text_length >= 4_000:
        score += 2
        reasons.append("long_input")

    elif text_length >= 1_500:
        score += 1
        reasons.append("medium_input")

    # -----------------------------
    # 2. Research / comparison
    # -----------------------------

    comparison_keywords = (
        "compare",
        "comparison",
        "versus",
        " vs ",
        "对比",
        "比较",
        "区别",
        "差异",
    )

    if any(
        keyword in normalized
        for keyword in comparison_keywords
    ):
        score += 1
        reasons.append("comparison")

    # -----------------------------
    # 3. Evidence requirements
    # -----------------------------

    evidence_keywords = (
        "evidence",
        "source",
        "citation",
        "verify",
        "verification",
        "official",
        "primary source",
        "证据",
        "来源",
        "引用",
        "核实",
        "验证",
        "官方",
        "一手资料",
    )

    if any(
        keyword in normalized
        for keyword in evidence_keywords
    ):
        score += 1
        reasons.append("evidence_requirement")

    # -----------------------------
    # 4. Current / time-sensitive
    # -----------------------------

    current_keywords = (
        "current",
        "latest",
        "today",
        "recent",
        "最新",
        "当前",
        "截至",
        "现在",
    )

    if any(
        keyword in normalized
        for keyword in current_keywords
    ):
        score += 1
        reasons.append("time_sensitive")

    # -----------------------------
    # 5. Multi-dimensional task
    # -----------------------------

    dimension_markers = (
        "architecture",
        "performance",
        "deployment",
        "pricing",
        "security",
        "cost",
        "生态",
        "架构",
        "性能",
        "部署",
        "价格",
        "成本",
        "安全",
    )

    dimension_count = sum(
        1
        for marker in dimension_markers
        if marker in normalized
    )

    if dimension_count >= 3:
        score += 2
        reasons.append(
            f"multiple_dimensions={dimension_count}"
        )

    elif dimension_count >= 2:
        score += 1
        reasons.append(
            f"multiple_dimensions={dimension_count}"
        )

    # -----------------------------
    # 6. Task type itself
    # -----------------------------

    if task_type in {
        TaskType.SUPERVISOR,
        TaskType.EVIDENCE_VERIFICATION,
    }:
        score += 1
        reasons.append(
            f"task_type={task_type.value}"
        )

    # -----------------------------
    # Final classification
    # -----------------------------

    if score <= 1:
        complexity = TaskComplexity.SIMPLE

    elif score <= 4:
        complexity = TaskComplexity.MEDIUM

    else:
        complexity = TaskComplexity.COMPLEX

    return ComplexityAssessment(
        complexity=complexity,
        score=score,
        reasons=tuple(reasons),
    )
    
# 构建路由context
def build_routing_context(
    *,
    task_type: TaskType,
    text: str,
    prefer_low_cost: bool = True,
) -> RoutingContext:
    """Build runtime routing signals for one task."""

    complexity = estimate_task_complexity(
        text,
        task_type,
    )

    return RoutingContext(
        complexity=complexity.complexity,
        estimated_input_tokens=(
            estimate_text_tokens(text)
        ),
        reserved_output_tokens=(
            TASK_RESERVED_OUTPUT_TOKENS[
                task_type
            ]
        ),
        prefer_low_cost=prefer_low_cost,
    )
    
# 模型分数
@dataclass(frozen=True)
class ModelScore:
    model: ModelSpec

    total_score: float

    task_affinity: float
    task_weight: float

    cost_score: float
    cost_weight: float

    estimated_cost: float
    
# 任务类型对于各模型偏好
PROFILE_MODEL_AFFINITY = {
    ModelProfile.FAST: {
        "openai:deepseek-v4-flash": 30,
        "openai:qwen3.8-27b": 15,
        "openai:glm-5.2": 8,
        "openai:MiniMax/MiniMax-M3": 5,
    },

    ModelProfile.REASONING: {
        "openai:glm-5.2": 30,
        "openai:qwen3.8-27b": 20,
        "openai:deepseek-v4-flash": 12,
        "openai:MiniMax/MiniMax-M3": 8,
    },

    ModelProfile.SYNTHESIS: {
        "openai:qwen3.8-27b": 30,
        "openai:glm-5.2": 22,
        "openai:MiniMax/MiniMax-M3": 18,
        "openai:deepseek-v4-flash": 12,
    },

    ModelProfile.WRITER: {
        "openai:MiniMax/MiniMax-M3": 30,
        "openai:qwen3.8-27b": 22,
        "openai:glm-5.2": 18,
        "openai:deepseek-v4-flash": 10,
    },
}

# 计算预估调用价格
def estimate_model_cost(
    model: ModelSpec,
    context: RoutingContext,
) -> float:
    """Estimate request cost in RMB."""

    input_cost = (
        context.estimated_input_tokens
        / 1_000_000
        * model.input_cost_per_million
    )

    output_cost = (
        context.reserved_output_tokens
        / 1_000_000
        * model.output_cost_per_million
    )

    return input_cost + output_cost

# 计算价格分数
def calculate_cost_scores(
    candidates: list[ModelSpec],
    context: RoutingContext,
) -> dict[str, float]:

    estimated_costs = {
        model.model_name:
            estimate_model_cost(
                model,
                context,
            )
        for model in candidates
    }

    values = list(
        estimated_costs.values()
    )

    min_cost = min(values)
    max_cost = max(values)

    if max_cost == min_cost:
        return {
            name: 10.0
            for name in estimated_costs
        }

    return {
        name: (
            (max_cost - cost)
            / (max_cost - min_cost)
            * 20.0
        )
        for name, cost
        in estimated_costs.items()
    }
    
# 根据任务复杂度决定"任务偏好"和"成本"权重
def get_scoring_weights(
    complexity: TaskComplexity,
    prefer_low_cost: bool,
) -> tuple[float, float]:
    """Return task-affinity and cost weights."""

    if complexity == TaskComplexity.SIMPLE:
        task_weight = 0.45
        cost_weight = 1.0

    elif complexity == TaskComplexity.MEDIUM:
        task_weight = 1.0
        cost_weight = 0.45

    else:
        task_weight = 1.35
        cost_weight = 0.15

    if not prefer_low_cost:
        cost_weight = 0.0

    return (
        task_weight,
        cost_weight,
    )

# 总的评分系统
def score_candidate_models(
    *,
    task_type: TaskType,
    context: RoutingContext,
    candidates: list[ModelSpec],
) -> list[ModelScore]:

    default_profile = (
        get_default_profile_for_task(
            task_type
        )
    )

    affinity = PROFILE_MODEL_AFFINITY[
        default_profile
    ]

    cost_scores = calculate_cost_scores(
        candidates,
        context,
    )

    task_weight, cost_weight = (
        get_scoring_weights(
            context.complexity,
            context.prefer_low_cost,
        )
    )

    scored_models: list[ModelScore] = []

    for model in candidates:

        task_affinity = float(
            affinity.get(
                model.model_name,
                0,
            )
        )

        cost_score = cost_scores[
            model.model_name
        ]

        total_score = (
            task_affinity * task_weight
            + cost_score * cost_weight
        )

        scored_models.append(
            ModelScore(
                model=model,
                total_score=total_score,

                task_affinity=task_affinity,
                task_weight=task_weight,

                cost_score=cost_score,
                cost_weight=cost_weight,

                estimated_cost=estimate_model_cost(
                    model,
                    context,
                ),
            )
        )

    return sorted(
        scored_models,
        key=lambda item: item.total_score,
        reverse=True,
    )
    
# 模型路由v1
# # 根据类型分配ModelDecision
# def route_model(
#     task_type: TaskType,
#     context:RoutingContext | None=None
#     ) -> ModelDecision:
#     """Route one workflow task to a concrete model."""

#     if context==None:
#         # 从任务类型映射到模型类型
#         profile = get_default_profile_for_task(task_type)
        
#         reason=(
#             f"default rule: task={task_type.value} "
#             f"maps to profile={profile.value}"
#         )
#     else:
#         profile = get_default_profile_for_task(
#             task_type
#         )

#         reasons = [
#             f"default_profile={profile.value}"
#         ]
        
#         # 1.final report优先保持writer
#         if task_type==TaskType.FINAL_REPORT:
#             profile=ModelProfile.WRITER
            
#             reasons.append(
#                 "final_report requires writer profile"
#             )
            
#         # 2.超长上下文优先Long Context
#         elif context.context_tokens>=200_000:
#             profile=ModelProfile.LONG_CONTEXT
            
#             reasons.append(
#                 "context_tokens>=200000"
#             )
            
#         # 3.简单Researcher可以降级
#         elif(
#             task_type==TaskType.RESEARCHER
#             and context.complexity==TaskComplexity.SIMPLE
#             and context.prefer_low_cost
#         ):
#             profile=ModelProfile.FAST
            
#             reasons.append(
#                 "simple researcher with low-cost preference"
#             )
#         # 4.简单Supervisor也可降级
#         elif(
#             task_type == TaskType.SUPERVISOR
#             and context.complexity
#             == TaskComplexity.SIMPLE
#             and context.prefer_low_cost
#         ):
#             profile = ModelProfile.FAST

#             reasons.append(
#                 "simple supervisor with low-cost preference"
#             )
            
#         # 5.原本 FAST 的任务如果变复杂可以升级到 reasoning
#         elif(
#             task_type
#             in {
#                 TaskType.CLARIFICATION,
#                 TaskType.RESEARCH_BRIEF,
#                 TaskType.WEBPAGE_SUMMARIZATION,
#             }
#             and context.complexity
#             == TaskComplexity.COMPLEX
#         ):
#             profile = ModelProfile.REASONING

#             reasons.append(
#                 "complex fast-path task upgraded to reasoning"
#             )
            
#         reason=";".join(reasons)
        
#     model = PROFILE_MODELS[profile]
        
#     # 能力的配置
#     # 优先看任务难度中是否要求,没有的话看选出的model本身的能力
#     requires_tools = (
#         context.requires_tools
#         if context is not None
#         else task_type
#         in {
#             TaskType.SUPERVISOR,
#             TaskType.RESEARCHER,
#         }
#     )
#     requires_structured_output = (
#         context.requires_structured_output
#         if context is not None
#         else task_type
#         in {
#             TaskType.CLARIFICATION,
#             TaskType.RESEARCH_BRIEF,
#             TaskType.WEBPAGE_SUMMARIZATION,
#             TaskType.EVIDENCE_VERIFICATION,
#         }
#     )
    
#     # 需要工具但模型不支持工具
#     if (
#         requires_tools
#         and not model.supports_tools
#     ):
#         raise ValueError(
#             f"Model {model.model_name!r} does not "
#             f"support tools required by "
#             f"task {task_type.value!r}."
#         )
        
#     # 需要结构化输出但是模型不支持
#     if (
#         requires_structured_output
#         and not model.supports_structured_output
#     ):
#         raise ValueError(
#             f"Model {model.model_name!r} does not "
#             f"support structured output required by "
#             f"task {task_type.value!r}."
#         )
        
#     return ModelDecision(
#         task_type=task_type,
#         profile=profile,
#         model=model,
#         reason=reason,
#     )
 
# 模型路由v2
def route_model(
    task_type: TaskType,
    context: RoutingContext | None = None,
) -> ModelDecision:
    """Route one task to the best compatible model."""

    default_profile = (
        get_default_profile_for_task(
            task_type
        )
    )

    # -----------------------------------
    # V1 compatibility
    # -----------------------------------

    # 没有routeContext则直接按照任务对应节点分配默认model
    if context is None:

        model = PROFILE_MODELS[
            default_profile
        ]

        return ModelDecision(
            task_type=task_type,
            profile=default_profile,
            model=model,
            reason=(
                "default rule: "
                f"task={task_type.value} "
                f"maps to "
                f"profile={default_profile.value}"
            ),
        )

    # -----------------------------------
    # V2: hard constraints
    # -----------------------------------

    candidates = filter_compatible_models(
        task_type,
        context,
    )

    if not candidates:
        raise NoCompatibleModelError(
            "No configured model satisfies "
            f"task={task_type.value!r}, "
            f"input_tokens="
            f"{context.estimated_input_tokens}, "
            f"reserved_output_tokens="
            f"{context.reserved_output_tokens}."
        )

    # -----------------------------------
    # V2: optimization
    # -----------------------------------

    scored = score_candidate_models(
        task_type=task_type,
        context=context,
        candidates=candidates,
    )

    winner = scored[0]

    return ModelDecision(
        task_type=task_type,
        profile=default_profile,
        model=winner.model,
        candidates=tuple(
            model.model_name
            for model in candidates
        ),
        score=winner.total_score,
        estimated_cost=winner.estimated_cost,
        reason=(
            "dynamic rule: "
            f"complexity={context.complexity.value}; "
            f"task_affinity="
            f"{winner.task_affinity:.2f}; "
            f"task_weight="
            f"{winner.task_weight:.2f}; "
            f"cost_score="
            f"{winner.cost_score:.2f}; "
            f"cost_weight="
            f"{winner.cost_weight:.2f}"
        ),
    )
    
def route_model_for_text(
    *,
    task_type: TaskType,
    text: str,
    dynamic_enabled: bool = True,
    prefer_low_cost: bool = True,
) -> ModelDecision:

    if not dynamic_enabled:
        decision = route_model(
            task_type
        )

        log_model_decision(
            decision
        )

        return decision

    context = build_routing_context(
        task_type=task_type,
        text=text,
        prefer_low_cost=prefer_low_cost,
    )

    decision = route_model(
        task_type,
        context,
    )

    log_model_decision(
        decision
    )

    return decision