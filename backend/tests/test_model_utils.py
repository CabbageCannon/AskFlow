from open_deep_research.model_router import (
    TaskType,
    route_model,
)

import pytest

from open_deep_research.configuration import Configuration

from open_deep_research.model_utils import (
    build_model_runtime_config,
    build_routed_model_runtime_config,
    get_provider_model_kwargs,
)

def test_deepseek_provider_config_disables_thinking():
    provider_config = get_provider_model_kwargs(
        "deepseek:deepseek-v4-flash"
    )

    assert provider_config == {
        "extra_body": {
            "thinking": {
                "type": "disabled",
            }
        }
    }


def test_non_deepseek_model_has_no_deepseek_config():
    provider_config = get_provider_model_kwargs(
        "openai:gpt-4.1-mini"
    )

    assert provider_config == {}


def test_build_model_runtime_config_for_deepseek():
    config = build_model_runtime_config(
        model_name="deepseek:deepseek-v4-flash",
        max_tokens=10000,
        api_key="test-key",
    )

    assert config["model"] == "deepseek:deepseek-v4-flash"
    assert config["max_tokens"] == 10000
    assert config["api_key"] == "test-key"
    assert config["tags"] == ["langsmith:nostream"]
    assert config["extra_body"] == {
        "thinking": {
            "type": "disabled",
        }
    }


def test_build_model_runtime_config_for_openai():
    config = build_model_runtime_config(
        model_name="openai:gpt-4.1-mini",
        max_tokens=4000,
        api_key="test-key",
    )

    assert config["model"] == "openai:gpt-4.1-mini"
    assert config["max_tokens"] == 4000
    assert config["api_key"] == "test-key"

    # Provider-specific DeepSeek arguments must not leak.
    assert "extra_body" not in config


def test_final_writer_can_enable_streaming():
    config = build_model_runtime_config(
        model_name="deepseek:deepseek-v4-flash",
        max_tokens=10000,
        api_key=None,
        no_stream=False,
    )

    assert "tags" not in config
    assert "api_key" not in config
    
def test_build_routed_reasoning_model_config():
    decision = route_model(
        TaskType.EVIDENCE_VERIFICATION
    )

    config = build_routed_model_runtime_config(
        decision,
        api_key="test-bailian-key",
        base_url=(
            "https://dashscope.aliyuncs.com/"
            "compatible-mode/v1"
        ),
    )

    assert config["model"] == "openai:glm-5.2"
    assert config["max_tokens"] == 16_384

    assert config["api_key"] == "test-bailian-key"

    assert config["base_url"] == (
        "https://dashscope.aliyuncs.com/"
        "compatible-mode/v1"
    )

    assert config["extra_body"] == {
        "enable_thinking": False,
        "tool_stream": True,
    }

    assert config["tags"] == [
        "langsmith:nostream"
    ]
    
def test_minimax_uses_its_own_thinking_parameter():
    decision = route_model(
        TaskType.FINAL_REPORT
    )

    config = build_routed_model_runtime_config(
        decision,
        api_key="test-bailian-key",
        base_url="https://example.com/v1",
        no_stream=False,
    )

    assert config["model"] == (
        "openai:MiniMax/MiniMax-M3"
    )

    assert config["extra_body"] == {
        "thinking": {
            "type": "disabled",
        }
    }

    assert "enable_thinking" not in config["extra_body"]

    # Final Writer 要允许 streaming。
    assert "tags" not in config
    
def test_configuration_reads_bailian_gateway_from_env(
    monkeypatch,
):
    monkeypatch.setenv(
        "BAILIAN_API_KEY",
        "test-bailian-key",
    )

    monkeypatch.setenv(
        "BAILIAN_BASE_URL",
        "https://example.com/v1",
    )

    configurable = Configuration.from_runnable_config(
        {}
    )

    assert (
        configurable.bailian_api_key
        == "test-bailian-key"
    )

    assert (
        configurable.bailian_base_url
        == "https://example.com/v1"
    )
    
def test_routed_model_requires_bailian_api_key():
    decision = route_model(
        TaskType.CLARIFICATION
    )

    with pytest.raises(
        ValueError,
        match="BAILIAN_API_KEY",
    ):
        build_routed_model_runtime_config(
            decision,
            api_key=None,
            base_url="https://example.com/v1",
        )