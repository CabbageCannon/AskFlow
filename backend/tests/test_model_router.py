from open_deep_research.model_router import (
    ModelDecision,
    ModelProfile,
    TaskType,
    get_default_profile_for_task,
    route_model,
)


def test_fast_tasks_use_fast_profile():
    assert (
        get_default_profile_for_task(TaskType.CLARIFICATION)
        == ModelProfile.FAST
    )

    assert (
        get_default_profile_for_task(TaskType.RESEARCH_BRIEF)
        == ModelProfile.FAST
    )

    assert (
        get_default_profile_for_task(TaskType.WEBPAGE_SUMMARIZATION)
        == ModelProfile.FAST
    )


def test_reasoning_tasks_use_reasoning_profile():
    assert (
        get_default_profile_for_task(TaskType.SUPERVISOR)
        == ModelProfile.REASONING
    )

    assert (
        get_default_profile_for_task(TaskType.RESEARCHER)
        == ModelProfile.REASONING
    )

    assert (
        get_default_profile_for_task(TaskType.EVIDENCE_VERIFICATION)
        == ModelProfile.REASONING
    )


def test_compression_uses_synthesis_profile():
    assert (
        get_default_profile_for_task(TaskType.COMPRESSION)
        == ModelProfile.SYNTHESIS
    )


def test_final_report_uses_writer_profile():
    assert (
        get_default_profile_for_task(TaskType.FINAL_REPORT)
        == ModelProfile.WRITER
    )
    
def test_reasoning_task_routes_to_glm():
    decision = route_model(
        TaskType.EVIDENCE_VERIFICATION
    )

    assert isinstance(decision, ModelDecision)

    assert decision.profile == ModelProfile.REASONING
    assert decision.model.model_name == "openai:glm-5.2"

    assert decision.model.supports_structured_output is True


def test_synthesis_task_routes_to_qwen():
    decision = route_model(
        TaskType.COMPRESSION
    )

    assert decision.profile == ModelProfile.SYNTHESIS
    assert (
        decision.model.model_name
        == "openai:qwen3.8-27b"
    )


def test_writer_routes_to_minimax():
    decision = route_model(
        TaskType.FINAL_REPORT
    )

    assert decision.profile == ModelProfile.WRITER

    assert (
        decision.model.model_name
        == "openai:MiniMax/MiniMax-M3"
    )

    assert decision.model.supports_structured_output is False


def test_fast_task_routes_to_deepseek():
    decision = route_model(
        TaskType.CLARIFICATION
    )

    assert decision.profile == ModelProfile.FAST

    assert (
        decision.model.model_name
        == "openai:deepseek-v4-flash"
    )


def test_route_decision_contains_reason():
    decision = route_model(
        TaskType.RESEARCHER
    )

    assert decision.reason
    assert "researcher" in decision.reason
    assert "reasoning" in decision.reason
    
def test_fast_task_routes_to_deepseek():
    decision = route_model(
        TaskType.CLARIFICATION
    )

    assert decision.profile == ModelProfile.FAST

    assert (
        decision.model.model_name
        == "openai:deepseek-v4-flash"
    )

    assert (
        decision.model.supports_structured_output
        is True
    )