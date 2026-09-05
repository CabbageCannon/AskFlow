param(
    [string]$RepoRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne "feat/node-latency-observability") {
    throw "Expected branch feat/node-latency-observability, current branch is '$currentBranch'."
}

$evalDir = Join-Path $RepoRoot "backend\tests\askflow_eval"
Copy-Item (Join-Path $PSScriptRoot "node_timing.py") (Join-Path $evalDir "node_timing.py") -Force
Copy-Item (Join-Path $PSScriptRoot "test_node_timing.py") (Join-Path $evalDir "test_node_timing.py") -Force
Copy-Item (Join-Path $PSScriptRoot "LATENCY_OBSERVABILITY.md") (Join-Path $evalDir "LATENCY_OBSERVABILITY.md") -Force

$env:ASKFLOW_REPO_ROOT = $RepoRoot
@'
import os
from pathlib import Path

repo = Path(os.environ["ASKFLOW_REPO_ROOT"])
eval_dir = repo / "backend" / "tests" / "askflow_eval"

runner = eval_dir / "run_agent_eval.py"
text = runner.read_text(encoding="utf-8")

replacements = [
(
'''    from .pricing import load_pricing_snapshot
    from .usage_tracking import (
''',
'''    from .pricing import load_pricing_snapshot
    from .node_timing import (
        NodeTimingCallbackHandler,
        NodeTimingCollector,
        reset_current_node_timing_collector,
        set_current_node_timing_collector,
    )
    from .usage_tracking import (
'''
),
(
'''    from pricing import load_pricing_snapshot
    from usage_tracking import (
''',
'''    from pricing import load_pricing_snapshot
    from node_timing import (
        NodeTimingCallbackHandler,
        NodeTimingCollector,
        reset_current_node_timing_collector,
        set_current_node_timing_collector,
    )
    from usage_tracking import (
'''
),
(
'''    usage_tracker = EvalUsageTracker()
    usage_handler = EvalUsageCallbackHandler(usage_tracker)
    usage_token = set_current_usage_tracker(usage_tracker)

    graph = dr.deep_researcher_builder.compile(
''',
'''    usage_tracker = EvalUsageTracker()
    usage_handler = EvalUsageCallbackHandler(usage_tracker)
    usage_token = set_current_usage_tracker(usage_tracker)

    node_timing_collector = NodeTimingCollector()
    node_timing_handler = NodeTimingCallbackHandler(
        node_timing_collector
    )
    node_timing_token = set_current_node_timing_collector(
        node_timing_collector
    )

    graph = dr.deep_researcher_builder.compile(
'''
),
(
'''    config["callbacks"] = [usage_handler]

    start = time.perf_counter()
''',
'''    config["callbacks"] = [
        usage_handler,
        node_timing_handler,
    ]

    start = time.perf_counter()
'''
),
(
'''        _CURRENT_METRICS.reset(metrics_token)
        reset_current_usage_tracker(usage_token)

    api_usage = usage_tracker.snapshot(
        PRICING_SNAPSHOT
    )
''',
'''        _CURRENT_METRICS.reset(metrics_token)
        reset_current_usage_tracker(usage_token)
        reset_current_node_timing_collector(
            node_timing_token
        )

    api_usage = usage_tracker.snapshot(
        PRICING_SNAPSHOT
    )
    node_timing = node_timing_collector.snapshot()
'''
),
(
'''        "latency_seconds": round(
            latency_seconds,
            3,
        ),
        "logical_tool_calls": (
''',
'''        "latency_seconds": round(
            latency_seconds,
            3,
        ),
        # Inclusive graph-node work. This may exceed E2E wall-clock
        # latency because nodes can overlap and subgraph nodes contain
        # child-node execution.
        **node_timing,
        "logical_tool_calls": (
'''
),
(
'''            f"latency={result['latency_seconds']}s | "
            f"tool_calls={result['logical_tool_calls']} | "
''',
'''            f"latency={result['latency_seconds']}s | "
            f"aggregate_node_work="
            f"{result['aggregate_recorded_node_seconds']}s | "
            f"tool_calls={result['logical_tool_calls']} | "
'''
),
]

for old, new in replacements:
    if old not in text:
        raise RuntimeError("run_agent_eval.py patch anchor not found:\n" + old[:160])
    text = text.replace(old, new, 1)

runner.write_text(text, encoding="utf-8")

judge = eval_dir / "judge_eval.py"
text = judge.read_text(encoding="utf-8")

old = '''def deterministic_rng(
'''
new = '''def safe_percentile(
    values: list[float],
    percentile: float,
) -> float | None:
    """Return a deterministic linear-interpolated percentile."""

    if not values:
        return None
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0.0 and 1.0")

    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)

    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    value = ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction
    return round(value, 4)


def summarize_node_latency(
    agent_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate raw node timing without claiming wall-clock attribution."""

    node_names = sorted({
        str(node)
        for record in agent_rows
        for node in (record.get("node_latency", {}) or {})
    })
    output: dict[str, Any] = {}

    for node in node_names:
        per_task_totals = [
            float(
                (record.get("node_latency", {}) or {})
                .get(node, {})
                .get("total_seconds", 0.0)
                or 0.0
            )
            for record in agent_rows
        ]
        invocation_values = [
            float(event.get("duration_seconds", 0.0) or 0.0)
            for record in agent_rows
            for event in (record.get("node_timings", []) or [])
            if event.get("node") == node
        ]
        total_calls = sum(
            int(
                (record.get("node_latency", {}) or {})
                .get(node, {})
                .get("calls", 0)
                or 0
            )
            for record in agent_rows
        )
        output[node] = {
            "mean_total_seconds_per_task": safe_mean(per_task_totals),
            "median_total_seconds_per_task": safe_median(per_task_totals),
            "mean_invocation_seconds": safe_mean(invocation_values),
            "p50_invocation_seconds": safe_percentile(invocation_values, 0.50),
            "p95_invocation_seconds": safe_percentile(invocation_values, 0.95),
            "total_calls": total_calls,
        }

    return output


def deterministic_rng(
'''
if old not in text:
    raise RuntimeError("judge_eval.py helper insertion anchor not found")
text = text.replace(old, new, 1)

old = '''    tool_call_values = [
'''
new = '''    aggregate_node_work_values = [
        float(record.get("aggregate_recorded_node_seconds", 0.0) or 0.0)
        for record in agent_rows
        if record.get("aggregate_recorded_node_seconds") is not None
    ]

    node_latency_summary = summarize_node_latency(agent_rows)

    tool_call_values = [
'''
if old not in text:
    raise RuntimeError("judge_eval.py summarize insertion anchor not found")
text = text.replace(old, new, 1)

old = '''        "mean_latency_seconds": safe_mean(
            latency_values
        ),
        "median_latency_seconds": safe_median(
            latency_values
        ),
        "mean_logical_tool_calls": safe_mean(
'''
new = '''        "mean_latency_seconds": safe_mean(
            latency_values
        ),
        "median_latency_seconds": safe_median(
            latency_values
        ),
        "mean_end_to_end_latency": safe_mean(
            latency_values
        ),
        "median_end_to_end_latency": safe_median(
            latency_values
        ),
        "mean_aggregate_recorded_node_seconds": safe_mean(
            aggregate_node_work_values
        ),
        "median_aggregate_recorded_node_seconds": safe_median(
            aggregate_node_work_values
        ),
        "node_latency_summary": node_latency_summary,
        "mean_logical_tool_calls": safe_mean(
'''
if old not in text:
    raise RuntimeError("judge_eval.py summarize return anchor not found")
text = text.replace(old, new, 1)

old = '''        "logical_tool_calls": raw_record.get(
            "logical_tool_calls"
        ),
'''
new = '''        "aggregate_recorded_node_seconds": raw_record.get(
            "aggregate_recorded_node_seconds"
        ),
        "node_latency": raw_record.get(
            "node_latency",
            {},
        ),
        "node_timings": raw_record.get(
            "node_timings",
            [],
        ),
        "logical_tool_calls": raw_record.get(
            "logical_tool_calls"
        ),
'''
if old not in text:
    raise RuntimeError("judge_eval.py build_agent_record anchor not found")
text = text.replace(old, new, 1)

judge.write_text(text, encoding="utf-8")
print("Patched run_agent_eval.py and judge_eval.py")
'@ | python -
Remove-Item Env:ASKFLOW_REPO_ROOT

Write-Host "Applied node latency observability changes." -ForegroundColor Green
Write-Host "Run tests:" -ForegroundColor Cyan
Write-Host '  cd <AskFlow-repo>\backend'
Write-Host '  Remove-Item -Recurse -Force .\.pytest_tmp -ErrorAction SilentlyContinue'
Write-Host '  New-Item -ItemType Directory .\.pytest_tmp | Out-Null'
Write-Host '  uv run python -m pytest tests/askflow_eval/test_node_timing.py tests/askflow_eval/test_usage_tracking.py tests/askflow_eval/test_cost_summary.py -q -s --basetemp=.pytest_tmp'
Write-Host "Review:" -ForegroundColor Cyan
Write-Host '  git diff -- backend/tests/askflow_eval'
