from __future__ import annotations

try:
    from .cost_summary import summarize_serving_cost
except ImportError:
    from cost_summary import summarize_serving_cost


def judged_row(task_id: str, *, agent: str, success: bool):
    return {
        "task_id": task_id,
        "category": "unit",
        agent: {
            "quality": {
                "research_success": success,
            }
        },
    }


def test_cost_summary_metrics():
    raw = {
        "a": {
            "task_id": "a",
            "total_api_cost": 2.0,
            "known_api_cost": 2.0,
            "llm_calls": 10,
            "input_tokens": 100,
            "output_tokens": 20,
            "external_search_requests": 3,
            "missing_usage_count": 0,
            "pricing_currency": "CNY",
            "pricing_snapshot": "2026-09-05",
        },
        "b": {
            "task_id": "b",
            "total_api_cost": 4.0,
            "known_api_cost": 4.0,
            "llm_calls": 20,
            "input_tokens": 300,
            "output_tokens": 40,
            "external_search_requests": 5,
            "missing_usage_count": 0,
            "pricing_currency": "CNY",
            "pricing_snapshot": "2026-09-05",
        },
    }
    judged = [
        judged_row("a", agent="askflow", success=True),
        judged_row("b", agent="askflow", success=False),
    ]

    summary = summarize_serving_cost(
        raw_records=raw,
        judged_rows=judged,
        agent="askflow",
    )

    assert summary["mean_api_cost_per_task"] == 3.0
    assert summary["median_api_cost_per_task"] == 3.0
    assert summary["total_benchmark_api_cost"] == 6.0
    assert summary["mean_llm_calls"] == 15.0
    assert summary["mean_input_tokens"] == 200.0
    assert summary["mean_output_tokens"] == 30.0
    assert summary["mean_external_search_requests"] == 4.0
    assert summary["cost_per_research_success"] == 6.0


def test_zero_research_success_is_safe():
    raw = {
        "a": {
            "task_id": "a",
            "total_api_cost": 2.0,
            "known_api_cost": 2.0,
            "llm_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "external_search_requests": 1,
        }
    }
    judged = [
        judged_row("a", agent="baseline", success=False),
    ]

    summary = summarize_serving_cost(
        raw_records=raw,
        judged_rows=judged,
        agent="baseline",
    )

    assert summary["research_success_count"] == 0
    assert summary["cost_per_research_success"] is None
