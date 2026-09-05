import pytest

import open_deep_research.utils as utils


class FakeSummarizationModel:
    def with_structured_output(self, *args, **kwargs):
        return self

    def with_retry(self, *args, **kwargs):
        return self


@pytest.mark.asyncio
async def test_tavily_search_reuses_one_summarization_model_per_batch(
    monkeypatch,
):
    """One Tavily search batch should route/init once and reuse the model."""

    async def fake_tavily_search_async(
        search_queries,
        max_results=5,
        topic="general",
        include_raw_content=True,
        config=None,
    ):
        return [
            {
                "query": "test query",
                "results": [
                    {
                        "url": "https://example.com/1",
                        "title": "Page 1",
                        "content": "snippet 1",
                        "raw_content": "raw page 1",
                    },
                    {
                        "url": "https://example.com/2",
                        "title": "Page 2",
                        "content": "snippet 2",
                        "raw_content": "raw page 2",
                    },
                    {
                        "url": "https://example.com/3",
                        "title": "Page 3",
                        "content": "snippet 3",
                        "raw_content": "raw page 3",
                    },
                ],
            }
        ]

    route_call_count = 0

    def fake_route_model_for_text(**kwargs):
        nonlocal route_call_count
        route_call_count += 1
        return object()

    def fake_build_runtime_config(*args, **kwargs):
        return {}

    init_call_count = 0
    created_model = FakeSummarizationModel()

    def fake_init_chat_model(**kwargs):
        nonlocal init_call_count
        init_call_count += 1
        return created_model

    models_seen = []

    async def fake_summarize_webpage(model, webpage_content):
        models_seen.append(model)
        return f"summary:{webpage_content}"

    monkeypatch.setattr(
        utils,
        "tavily_search_async",
        fake_tavily_search_async,
    )
    monkeypatch.setattr(
        utils,
        "route_model_for_text",
        fake_route_model_for_text,
    )
    monkeypatch.setattr(
        utils,
        "build_routed_model_runtime_config",
        fake_build_runtime_config,
    )
    monkeypatch.setattr(
        utils,
        "init_chat_model",
        fake_init_chat_model,
    )
    monkeypatch.setattr(
        utils,
        "summarize_webpage",
        fake_summarize_webpage,
    )

    config = {
        "configurable": {
            "max_content_length": 50_000,
            "model_router_dynamic_enabled": True,
            "model_router_prefer_low_cost": True,
            "bailian_api_key": "test-key",
            "bailian_base_url": "https://example.com/v1",
        }
    }

    result = await utils.tavily_search.coroutine(
        queries=["test query"],
        max_results=5,
        topic="general",
        config=config,
    )

    # 核心回归断言：
    # 一个 search batch 只能 route 一次。
    assert route_call_count == 1

    # 一个 search batch 只能 init model 一次。
    assert init_call_count == 1

    # 三个网页仍然各自被总结。
    assert len(models_seen) == 3

    # 但三个网页共享完全同一个 model instance。
    assert all(
        model is created_model
        for model in models_seen
    )

    assert "summary:raw page 1" in result
    assert "summary:raw page 2" in result
    assert "summary:raw page 3" in result