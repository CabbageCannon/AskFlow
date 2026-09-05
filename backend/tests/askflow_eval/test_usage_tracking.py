from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

try:
    from .pricing import PricingSnapshot
    from .usage_tracking import (
        EvalUsageCallbackHandler,
        EvalUsageTracker,
        reset_current_usage_tracker,
        set_current_usage_tracker,
        wrap_async_external_search,
    )
except ImportError:
    from pricing import PricingSnapshot
    from usage_tracking import (
        EvalUsageCallbackHandler,
        EvalUsageTracker,
        reset_current_usage_tracker,
        set_current_usage_tracker,
        wrap_async_external_search,
    )


@pytest.fixture
def pricing() -> PricingSnapshot:
    return PricingSnapshot.from_dict(
        {
            "effective_date": "2026-09-05",
            "currency": "CNY",
            "pricing_basis": "unit_test",
            "models": {
                "model-a": {
                    "aliases": ["openai:model-a"],
                    "input_per_million": 2.0,
                    "cached_input_per_million": 0.5,
                    "output_per_million": 4.0,
                },
                "model-b": {
                    "aliases": ["openai:model-b"],
                    "input_per_million": 10.0,
                    "output_per_million": 20.0,
                },
                "model-tiered": {
                    "aliases": ["openai:model-tiered"],
                    "tiers": [
                        {
                            "max_input_tokens": 100,
                            "input_per_million": 1.0,
                            "cached_input_per_million": 0.1,
                            "output_per_million": 2.0,
                        },
                        {
                            "max_input_tokens": 200,
                            "input_per_million": 3.0,
                            "cached_input_per_million": 0.3,
                            "output_per_million": 6.0,
                        },
                    ],
                },
            },
            "search": {
                "tavily": {
                    "cost_per_request": 0.05,
                }
            },
        }
    )


def fake_result(
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_tokens: int = 0,
):
    usage = None
    if input_tokens is not None and output_tokens is not None:
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {
                "cache_read": cached_tokens,
            },
        }

    message = SimpleNamespace(
        usage_metadata=usage,
        response_metadata={"model_name": model},
    )
    generation = SimpleNamespace(message=message)
    return SimpleNamespace(
        generations=[[generation]],
        llm_output=None,
    )


def record_call(
    handler: EvalUsageCallbackHandler,
    *,
    model: str,
    stage: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_tokens: int = 0,
) -> None:
    run_id = uuid4()
    handler.on_chat_model_start(
        {},
        [[]],
        run_id=run_id,
        metadata={
            "langgraph_node": stage,
            "ls_model_name": model,
        },
    )
    handler.on_llm_end(
        fake_result(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        ),
        run_id=run_id,
    )


def test_one_llm_call_accumulates(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    record_call(
        handler,
        model="model-a",
        stage="researcher",
        input_tokens=1000,
        output_tokens=200,
    )

    snapshot = tracker.snapshot(pricing)
    assert snapshot["llm_calls"] == 1
    assert snapshot["input_tokens"] == 1000
    assert snapshot["output_tokens"] == 200
    assert snapshot["usage_by_stage"]["researcher"]["calls"] == 1


def test_two_models_are_accounted_separately(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    record_call(
        handler,
        model="model-a",
        stage="researcher",
        input_tokens=100,
        output_tokens=10,
    )
    record_call(
        handler,
        model="model-b",
        stage="evidence_verifier",
        input_tokens=200,
        output_tokens=20,
    )

    snapshot = tracker.snapshot(pricing)
    assert snapshot["usage_by_model"]["model-a"]["calls"] == 1
    assert snapshot["usage_by_model"]["model-b"]["calls"] == 1
    assert snapshot["llm_calls"] == 2


def test_retry_is_counted_as_two_real_llm_calls(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    for _ in range(2):
        record_call(
            handler,
            model="model-a",
            stage="researcher",
            input_tokens=100,
            output_tokens=10,
        )

    assert tracker.snapshot(pricing)["llm_calls"] == 2


@pytest.mark.asyncio
async def test_three_tavily_queries_count_three_external_requests(
    pricing: PricingSnapshot,
):
    tracker = EvalUsageTracker()
    token = set_current_usage_tracker(tracker)

    async def fake_search(_client, query: str):
        await asyncio.sleep(0)
        return {"query": query, "results": []}

    wrapped = wrap_async_external_search(
        fake_search,
        provider="tavily",
    )

    try:
        queries = ["a", "b", "c"]
        await asyncio.gather(
            *(wrapped(object(), query) for query in queries)
        )
    finally:
        reset_current_usage_tracker(token)

    snapshot = tracker.snapshot(pricing)
    assert snapshot["external_search_requests"] == 3
    assert snapshot["external_search_requests_by_provider"]["tavily"] == 3


@pytest.mark.asyncio
async def test_concurrent_tasks_do_not_pollute_trackers(
    pricing: PricingSnapshot,
):
    async def worker(model: str, count: int):
        tracker = EvalUsageTracker()
        handler = EvalUsageCallbackHandler(tracker)
        token = set_current_usage_tracker(tracker)
        try:
            for _ in range(count):
                await asyncio.sleep(0)
                record_call(
                    handler,
                    model=model,
                    stage="researcher",
                    input_tokens=10,
                    output_tokens=1,
                )
            return tracker.snapshot(pricing)
        finally:
            reset_current_usage_tracker(token)

    first, second = await asyncio.gather(
        worker("model-a", 2),
        worker("model-b", 3),
    )

    assert first["llm_calls"] == 2
    assert second["llm_calls"] == 3
    assert set(first["usage_by_model"]) == {"model-a"}
    assert set(second["usage_by_model"]) == {"model-b"}


def test_missing_usage_is_explicit(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    record_call(
        handler,
        model="model-a",
        stage="researcher",
        input_tokens=None,
        output_tokens=None,
    )

    snapshot = tracker.snapshot(pricing)
    assert snapshot["missing_usage_count"] == 1
    assert snapshot["llm_cost"] is None
    assert snapshot["cost_complete"] is False


def test_cost_formula_uses_cached_input_rate(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    record_call(
        handler,
        model="model-a",
        stage="researcher",
        input_tokens=1_000_000,
        output_tokens=500_000,
        cached_tokens=250_000,
    )

    snapshot = tracker.snapshot(pricing)
    expected_llm = 0.75 * 2.0 + 0.25 * 0.5 + 0.5 * 4.0
    assert snapshot["llm_cost"] == pytest.approx(expected_llm)


def test_eval_record_is_jsonl_serializable(pricing: PricingSnapshot, tmp_path: Path):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)
    record_call(
        handler,
        model="model-a",
        stage="researcher",
        input_tokens=50,
        output_tokens=5,
    )

    record = {
        "task_id": "unit-01",
        **tracker.snapshot(pricing),
    }
    path = tmp_path / "result.jsonl"
    path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    loaded = json.loads(path.read_text(encoding="utf-8").strip())
    assert loaded["task_id"] == "unit-01"
    assert loaded["llm_calls"] == 1


def test_stage_cost_sums_multiple_models(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    record_call(
        handler,
        model="model-a",
        stage="researcher",
        input_tokens=1_000_000,
        output_tokens=0,
    )
    record_call(
        handler,
        model="model-b",
        stage="researcher",
        input_tokens=1_000_000,
        output_tokens=0,
    )

    snapshot = tracker.snapshot(pricing)
    assert snapshot["usage_by_stage"]["researcher"]["cost"] == pytest.approx(12.0)


def test_openai_compatible_response_metadata_is_normalized(
    pricing: PricingSnapshot,
):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)
    run_id = uuid4()

    handler.on_chat_model_start(
        {},
        [[]],
        run_id=run_id,
        metadata={
            "langgraph_node": "researcher",
            "ls_model_name": "model-a",
        },
    )

    message = SimpleNamespace(
        usage_metadata=None,
        response_metadata={
            "model_name": "model-a",
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "prompt_tokens_details": {
                    "cached_tokens": 400,
                },
            },
        },
    )
    response = SimpleNamespace(
        generations=[[SimpleNamespace(message=message)]],
        llm_output=None,
    )

    handler.on_llm_end(response, run_id=run_id)

    snapshot = tracker.snapshot(pricing)
    assert snapshot["input_tokens"] == 1000
    assert snapshot["output_tokens"] == 200
    assert snapshot["cached_input_tokens"] == 400


def test_tiered_pricing_uses_each_request_input_size(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    # If pricing were incorrectly selected from the 150-token aggregate, both
    # calls would use tier 2. Correct behavior prices each real request.
    record_call(
        handler,
        model="model-tiered",
        stage="researcher",
        input_tokens=50,
        output_tokens=10,
    )
    record_call(
        handler,
        model="model-tiered",
        stage="researcher",
        input_tokens=100,
        output_tokens=20,
    )

    snapshot = tracker.snapshot(pricing)
    expected = (
        (50 / 1_000_000 * 1.0 + 10 / 1_000_000 * 2.0)
        + (100 / 1_000_000 * 1.0 + 20 / 1_000_000 * 2.0)
    )
    assert snapshot["llm_cost"] == pytest.approx(expected)


def test_tiered_pricing_moves_to_higher_tier_per_call(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    record_call(
        handler,
        model="model-tiered",
        stage="researcher",
        input_tokens=150,
        output_tokens=20,
    )

    snapshot = tracker.snapshot(pricing)
    expected = 150 / 1_000_000 * 3.0 + 20 / 1_000_000 * 6.0
    assert snapshot["llm_cost"] == pytest.approx(expected)


def test_missing_usage_nulls_model_and_stage_cost(pricing: PricingSnapshot):
    tracker = EvalUsageTracker()
    handler = EvalUsageCallbackHandler(tracker)

    record_call(
        handler,
        model="model-a",
        stage="researcher",
        input_tokens=None,
        output_tokens=None,
    )

    snapshot = tracker.snapshot(pricing)
    assert snapshot["usage_by_model"]["model-a"]["cost"] is None
    assert snapshot["usage_by_stage"]["researcher"]["cost"] is None
