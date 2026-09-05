"""Offline API-cost aggregation for AskFlow controlled evaluation.

This module intentionally reads already-produced agent serving records and judge
results. It never invokes a model or search provider, so judge/evaluation overhead
cannot leak into agent serving cost.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Literal


AgentName = Literal["askflow", "baseline"]


def load_jsonl_files(
    paths: list[Path],
) -> dict[str, dict[str, Any]]:
    """Load JSONL records keyed by task_id; later files override earlier files."""

    records: dict[str, dict[str, Any]] = {}

    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in {path} at line {line_number}: {exc}"
                    ) from exc

                task_id = record.get("task_id")
                if not task_id:
                    raise ValueError(
                        f"Missing task_id in {path} at line {line_number}"
                    )

                records[str(task_id)] = record

    return records


def load_judged_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid judged JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), 8)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.median(values), 8)


def _research_success_count(
    judged_rows: list[dict[str, Any]],
    agent: AgentName,
) -> int:
    return sum(
        1
        for row in judged_rows
        if (
            row.get(agent)
            and row[agent].get("quality")
            and row[agent]["quality"].get("research_success") is True
        )
    )


def _selected_task_ids(
    judged_rows: list[dict[str, Any]],
    agent: AgentName,
) -> list[str]:
    return [
        str(row["task_id"])
        for row in judged_rows
        if row.get(agent) is not None
    ]


def summarize_serving_cost(
    *,
    raw_records: dict[str, dict[str, Any]],
    judged_rows: list[dict[str, Any]],
    agent: AgentName,
) -> dict[str, Any]:
    """Compute serving-cost metrics for one agent over the judged task set."""

    selected_ids = _selected_task_ids(judged_rows, agent)
    records = [
        raw_records[task_id]
        for task_id in selected_ids
        if task_id in raw_records
    ]

    missing_raw_task_ids = [
        task_id
        for task_id in selected_ids
        if task_id not in raw_records
    ]

    complete_costs = [
        float(record["total_api_cost"])
        for record in records
        if record.get("total_api_cost") is not None
    ]

    known_costs = [
        float(record.get("known_api_cost", 0.0) or 0.0)
        for record in records
    ]

    cost_complete_task_count = len(complete_costs)
    all_costs_complete = (
        not missing_raw_task_ids
        and len(records) == len(selected_ids)
        and cost_complete_task_count == len(records)
    )

    total_benchmark_api_cost = (
        round(sum(complete_costs), 8)
        if all_costs_complete
        else None
    )

    known_total_benchmark_api_cost = round(
        sum(known_costs),
        8,
    )

    llm_calls = [
        float(record.get("llm_calls", 0) or 0)
        for record in records
    ]
    input_tokens = [
        float(record.get("input_tokens", 0) or 0)
        for record in records
    ]
    output_tokens = [
        float(record.get("output_tokens", 0) or 0)
        for record in records
    ]
    external_search_requests = [
        float(record.get("external_search_requests", 0) or 0)
        for record in records
    ]

    research_success_count = _research_success_count(
        judged_rows,
        agent,
    )

    cost_per_research_success = None
    if (
        total_benchmark_api_cost is not None
        and research_success_count > 0
    ):
        cost_per_research_success = round(
            total_benchmark_api_cost / research_success_count,
            8,
        )

    currencies = sorted(
        {
            str(record.get("pricing_currency"))
            for record in records
            if record.get("pricing_currency")
        }
    )
    pricing_snapshots = sorted(
        {
            str(record.get("pricing_snapshot"))
            for record in records
            if record.get("pricing_snapshot")
        }
    )

    return {
        "serving_cost_task_count": len(records),
        "cost_complete_task_count": cost_complete_task_count,
        "cost_incomplete_task_count": len(records) - cost_complete_task_count,
        "missing_raw_task_ids": missing_raw_task_ids,
        "all_costs_complete": all_costs_complete,
        "pricing_currencies": currencies,
        "pricing_snapshots": pricing_snapshots,
        "mean_api_cost_per_task": (
            _mean(complete_costs)
            if all_costs_complete
            else None
        ),
        "median_api_cost_per_task": (
            _median(complete_costs)
            if all_costs_complete
            else None
        ),
        "total_benchmark_api_cost": total_benchmark_api_cost,
        "known_total_benchmark_api_cost": known_total_benchmark_api_cost,
        "mean_llm_calls": _mean(llm_calls),
        "mean_input_tokens": _mean(input_tokens),
        "mean_output_tokens": _mean(output_tokens),
        "mean_external_search_requests": _mean(external_search_requests),
        "research_success_count": research_success_count,
        "cost_per_research_success": cost_per_research_success,
        "total_missing_usage_count": sum(
            int(record.get("missing_usage_count", 0) or 0)
            for record in records
        ),
    }


def summarize_categories(
    *,
    raw_records: dict[str, dict[str, Any]],
    judged_rows: list[dict[str, Any]],
    agent: AgentName,
) -> dict[str, Any]:
    categories = sorted(
        {
            str(row.get("category", "unknown"))
            for row in judged_rows
            if row.get(agent) is not None
        }
    )

    output: dict[str, Any] = {}
    for category in categories:
        subset = [
            row
            for row in judged_rows
            if (
                row.get(agent) is not None
                and str(row.get("category", "unknown")) == category
            )
        ]
        output[category] = summarize_serving_cost(
            raw_records=raw_records,
            judged_rows=subset,
            agent=agent,
        )

    return output


def augment_summary(
    *,
    summary: dict[str, Any],
    judged_rows: list[dict[str, Any]],
    askflow_records: dict[str, dict[str, Any]],
    baseline_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return a copy of judge summary augmented with serving-cost metrics."""

    output = dict(summary)
    output["serving_cost"] = {
        "scope": "agent_execution_only_excludes_judge_overhead",
        "askflow": summarize_serving_cost(
            raw_records=askflow_records,
            judged_rows=judged_rows,
            agent="askflow",
        ),
        "baseline": summarize_serving_cost(
            raw_records=baseline_records,
            judged_rows=judged_rows,
            agent="baseline",
        ),
        "categories": {
            "askflow": summarize_categories(
                raw_records=askflow_records,
                judged_rows=judged_rows,
                agent="askflow",
            ),
            "baseline": summarize_categories(
                raw_records=baseline_records,
                judged_rows=judged_rows,
                agent="baseline",
            ),
        },
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge agent-serving API cost metrics into a judge summary without "
            "counting judge-model usage."
        )
    )
    parser.add_argument("--askflow-results", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-results", nargs="+", type=Path, required=True)
    parser.add_argument("--judged-results", type=Path, required=True)
    parser.add_argument("--judge-summary", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    askflow_records = load_jsonl_files(
        [path.resolve() for path in args.askflow_results]
    )
    baseline_records = load_jsonl_files(
        [path.resolve() for path in args.baseline_results]
    )
    judged_rows = load_judged_rows(
        args.judged_results.resolve()
    )

    with args.judge_summary.resolve().open("r", encoding="utf-8") as file:
        summary = json.load(file)

    output = augment_summary(
        summary=summary,
        judged_rows=judged_rows,
        askflow_records=askflow_records,
        baseline_records=baseline_records,
    )

    output_path = args.output_summary.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(
        "[COST_SUMMARY] "
        f"wrote={output_path} | "
        "scope=agent_execution_only_excludes_judge_overhead"
    )


if __name__ == "__main__":
    main()
