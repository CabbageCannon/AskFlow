from langchain.chat_models import init_chat_model
from open_deep_research.model_utils import build_model_runtime_config


def test_flat_with_config_switches_model():
    model = init_chat_model(
        model="deepseek:deepseek-v4-flash",
        configurable_fields=("model", "max_tokens", "api_key"),
        extra_body={"thinking": {"type": "disabled"}},
    )

    configured_model = model.with_config(
        {
            "model": "openai:gpt-4.1-mini",
            "max_tokens": 1234,
        }
    )

    assert configured_model._default_config["model"] == "openai:gpt-4.1-mini"
    assert configured_model._default_config["max_tokens"] == 1234


def test_nested_configurable_switches_model():
    model = init_chat_model(
        model="deepseek:deepseek-v4-flash",
        configurable_fields=("model", "max_tokens", "api_key"),
        extra_body={"thinking": {"type": "disabled"}},
    )

    configured_model = model.with_config(
        {
            "configurable": {
                "model": "openai:gpt-4.1-mini",
                "max_tokens": 1234,
            }
        }
    )

    assert configured_model._default_config["model"] == "openai:gpt-4.1-mini"
    assert configured_model._default_config["max_tokens"] == 1234
    
def test_provider_specific_config_does_not_leak_between_models():
    model = init_chat_model(
        model="deepseek:deepseek-v4-flash",
        configurable_fields=(
            "model",
            "max_tokens",
            "api_key",
            "extra_body",
        ),
    )

    deepseek_config = build_model_runtime_config(
        model_name="deepseek:deepseek-v4-flash",
        max_tokens=4000,
        api_key=None,
    )

    deepseek_model = model.with_config(deepseek_config)

    assert deepseek_model._default_config["model"] == (
        "deepseek:deepseek-v4-flash"
    )

    assert deepseek_model._default_config["extra_body"] == {
        "thinking": {
            "type": "disabled",
        }
    }

    openai_config = build_model_runtime_config(
        model_name="openai:gpt-4.1-mini",
        max_tokens=4000,
        api_key=None,
    )

    openai_model = model.with_config(openai_config)

    assert openai_model._default_config["model"] == "openai:gpt-4.1-mini"

    # DeepSeek-only arguments must not leak into another provider.
    assert "extra_body" not in openai_model._default_config