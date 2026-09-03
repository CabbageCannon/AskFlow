import asyncio

from langchain_core.messages import HumanMessage

import open_deep_research.deep_researcher as dr


class FakeConfigurableModel:
    """Minimal fake model for router integration testing."""

    def __init__(self, response):
        self.response = response
        self.runtime_config = None
        self.structured_schema = None
        self.retry_kwargs = None

    def with_structured_output(
        self,
        schema,
        **kwargs,
    ):
        self.structured_schema = schema
        return self

    def with_retry(self, **kwargs):
        self.retry_kwargs = kwargs
        return self

    def with_config(self, config):
        self.runtime_config = config
        return self

    async def ainvoke(self, messages):
        return self.response


def test_clarification_uses_fast_routed_model(
    monkeypatch,
    capsys,
):
    configurable = dr.Configuration(
        allow_clarification=True,
        bailian_api_key="test-bailian-key",
        bailian_base_url="https://example.com/v1",
    )

    monkeypatch.setattr(
        dr.Configuration,
        "from_runnable_config",
        classmethod(
            lambda cls, runnable_config=None: configurable
        ),
    )

    fake_model = FakeConfigurableModel(
        dr.ClarifyWithUser(
        need_clarification=False,
        question="",
        verification="The request is clear.",
    )
    )

    monkeypatch.setattr(
        dr,
        "configurable_model",
        fake_model,
    )

    command = asyncio.run(
        dr.clarify_with_user(
            {
                "messages": [
                    HumanMessage(
                        content="Research the latest DeepSeek API."
                    )
                ]
            },
            {},
        )
    )

    assert command.goto == "write_research_brief"

    assert (
        fake_model.structured_schema
        is dr.ClarifyWithUser
    )

    assert fake_model.runtime_config["model"] == (
        "openai:deepseek-v4-flash"
    )

    assert (
        fake_model.runtime_config["api_key"]
        == "test-bailian-key"
    )

    assert (
        fake_model.runtime_config["base_url"]
        == "https://example.com/v1"
    )

    assert fake_model.runtime_config["extra_body"] == {
        "enable_thinking": False,
    }

    output = capsys.readouterr().out

    assert "[MODEL_ROUTER]" in output
    assert "task=clarification" in output
    assert "profile=fast" in output
    assert "deepseek-v4-flash" in output
    
def test_research_brief_uses_fast_routed_model(
    monkeypatch,
    capsys,
):
    configurable = dr.Configuration(
        bailian_api_key="test-bailian-key",
        bailian_base_url="https://example.com/v1",
    )

    monkeypatch.setattr(
        dr.Configuration,
        "from_runnable_config",
        classmethod(
            lambda cls, runnable_config=None: configurable
        ),
    )

    fake_model = FakeConfigurableModel(
        dr.ResearchQuestion(
            research_brief=(
                "Research the current DeepSeek API models, "
                "context windows, and tool calling support."
            )
        )
    )

    monkeypatch.setattr(
        dr,
        "configurable_model",
        fake_model,
    )

    command = asyncio.run(
        dr.write_research_brief(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Research the current DeepSeek API."
                        )
                    )
                ]
            },
            {},
        )
    )

    assert command.goto == "research_supervisor"

    assert (
        fake_model.structured_schema
        is dr.ResearchQuestion
    )

    assert fake_model.runtime_config["model"] == (
        "openai:deepseek-v4-flash"
    )

    assert (
        fake_model.runtime_config["api_key"]
        == "test-bailian-key"
    )

    assert (
        fake_model.runtime_config["base_url"]
        == "https://example.com/v1"
    )

    output = capsys.readouterr().out

    assert "[MODEL_ROUTER]" in output
    assert "task=research_brief" in output
    assert "profile=fast" in output