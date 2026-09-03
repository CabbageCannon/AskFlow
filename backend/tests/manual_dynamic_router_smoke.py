from open_deep_research.model_router import (
    TaskType,
    route_model_for_text,
)
from open_deep_research.model_router import (
    RoutingContext,
    TaskComplexity,
    route_model,
)


def show_case(
    name: str,
    task_type: TaskType,
    text: str,
):
    print("\n" + "=" * 80)
    print(f"CASE: {name}")
    print("=" * 80)

    decision = route_model_for_text(
        task_type=task_type,
        text=text,
        dynamic_enabled=True,
        prefer_low_cost=True,
    )

    print(f"task={decision.task_type.value}")
    print(f"profile={decision.profile.value}")
    print(f"model={decision.model.model_name}")
    print(f"score={decision.score}")
    print(f"estimated_cost={decision.estimated_cost}")
    print(f"candidates={decision.candidates}")
    print(f"reason={decision.reason}")


show_case(
    "simple researcher",
    TaskType.RESEARCHER,
    "查询 LangGraph 当前最新稳定版本。",
)


show_case(
    "medium researcher",
    TaskType.RESEARCHER,
    """
请研究 LangGraph 当前的主要功能，
包括状态管理、Tool Calling 和持久化机制，
优先参考官方文档。
""",
)


show_case(
    "complex researcher",
    TaskType.RESEARCHER,
    """
请对 LangGraph、AutoGen、CrewAI 和 OpenAI Agents SDK
做一次生产级技术调研。

需要比较它们的架构、状态管理、Tool Calling、
并发能力、部署方式、失败恢复、成本和生态，
优先使用官方资料，并对冲突信息进行验证。
""",
)


show_case(
    "simple final report",
    TaskType.FINAL_REPORT,
    "根据已有结果写一份简短总结。",
)


show_case(
    "complex final report",
    TaskType.FINAL_REPORT,
    """
根据完整研究证据生成正式生产接入评估报告。
需要覆盖架构、性能、成本、部署、风险和证据冲突，
清晰区分可靠证据、弱证据和未解决证据缺口。
""",
)


show_case(
    "structured verifier",
    TaskType.EVIDENCE_VERIFICATION,
    """
检查现有研究证据是否足以支持结论，
识别来源可信度问题、冲突和证据缺口。
""",
)

print("\n" + "=" * 80)
print("CASE: context hard constraint")
print("=" * 80)

decision = route_model(
    TaskType.EVIDENCE_VERIFICATION,
    RoutingContext(
        complexity=TaskComplexity.COMPLEX,
        estimated_input_tokens=1_020_000,
        reserved_output_tokens=10_000,
        prefer_low_cost=True,
    ),
)

print(f"model={decision.model.model_name}")
print(f"candidates={decision.candidates}")
print(f"reason={decision.reason}")