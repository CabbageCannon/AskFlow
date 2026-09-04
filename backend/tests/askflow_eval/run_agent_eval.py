"""Behavior-level evaluation runner for AskFlow and its pre-enhancement baseline.

Run this same file inside each worktree so imports resolve to that worktree's
open_deep_research package.

Examples:
    uv run python tests/askflow_eval/run_agent_eval.py \
        --agent askflow \
        --task-ids api-01 conflict-01 efficiency-03 \
        --output tests/askflow_eval/results/askflow_smoke.jsonl

    uv run python tests/askflow_eval/run_agent_eval.py \
        --agent baseline \
        --task-ids api-01 conflict-01 efficiency-03 \
        --output tests/askflow_eval/results/baseline_smoke.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
EVAL_DIR = THIS_FILE.parent
BACKEND_ROOT = THIS_FILE.parents[2]

load_dotenv(BACKEND_ROOT / ".env")


# Import only after loading the target worktree's .env.
# The script should be executed from that worktree's backend directory.
import open_deep_research.deep_researcher as dr  # noqa: E402


CONTROL_TOOL_NAMES = {
    "think_tool",
    "ResearchComplete",
}


@dataclass
class RunMetrics:
    """Metrics collected during one graph run."""

    logical_tool_calls: int = 0
    logical_tool_names: Counter[str] = field(default_factory=Counter)

    verification_history: list[dict[str, Any]] = field(default_factory=list)

    targeted_research_rounds: int = 0
    targeted_research_tasks: int = 0


_CURRENT_METRICS: ContextVar[RunMetrics | None] = ContextVar(
    "askflow_eval_metrics",
    default=None,
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    """Convert common LangChain/Pydantic objects into JSON-safe values."""

    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def get_tool_name(tool: Any) -> str:
    """Extract a tool name conservatively."""

    name = getattr(tool, "name", None)

    if name:
        return str(name)

    if isinstance(tool, dict):
        return str(tool.get("name", "unknown"))

    return getattr(tool, "__name__", tool.__class__.__name__)


# ---------------------------------------------------------------------------
# Tool instrumentation
# ---------------------------------------------------------------------------

_ORIGINAL_EXECUTE_TOOL_SAFELY = dr.execute_tool_safely


async def instrumented_execute_tool_safely(*args, **kwargs):
    """Count one logical research-tool invocation.

    A retry performed *inside* execute_tool_safely is still the same logical
    invocation and therefore does not increment this counter again.
    """

    if args:
        tool = args[0]
    else:
        tool = kwargs.get("tool")

    tool_name = get_tool_name(tool)

    metrics = _CURRENT_METRICS.get()

    if (
        metrics is not None
        and tool_name not in CONTROL_TOOL_NAMES
    ):
        metrics.logical_tool_calls += 1
        metrics.logical_tool_names[tool_name] += 1

    return await _ORIGINAL_EXECUTE_TOOL_SAFELY(*args, **kwargs)


# Patch the module-global function used by researcher_tools.
dr.execute_tool_safely = instrumented_execute_tool_safely


# ---------------------------------------------------------------------------
# Stream update instrumentation
# ---------------------------------------------------------------------------

def iter_dicts(value: Any):
    """Recursively yield dictionaries from a nested update payload."""

    if isinstance(value, dict):
        yield value

        for item in value.values():
            yield from iter_dicts(item)

    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_dicts(item)


def collect_update_metrics(
    update: Any,
    metrics: RunMetrics,
) -> None:
    """Collect AskFlow-only metrics without breaking baseline compatibility."""

    if not isinstance(update, dict):
        return

    # A completed targeted_research node is one adaptive re-research round.
    if "targeted_research" in update:
        metrics.targeted_research_rounds += 1

    for item in iter_dicts(update):
        verification_result = item.get("verification_result")

        if verification_result is not None:
            serialized = to_jsonable(verification_result)

            # Avoid accidental duplicate capture from nested references.
            if (
                not metrics.verification_history
                or metrics.verification_history[-1] != serialized
            ):
                metrics.verification_history.append(serialized)

        tasks = item.get("targeted_research_tasks")

        if isinstance(tasks, list) and tasks:
            metrics.targeted_research_tasks += len(tasks)


# ---------------------------------------------------------------------------
# Eval task loading
# ---------------------------------------------------------------------------

def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("tasks.json must contain a JSON array")

    return data


def select_tasks(
    tasks: list[dict[str, Any]],
    task_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if task_ids:
        wanted = set(task_ids)
        selected = [
            task
            for task in tasks
            if task.get("id") in wanted
        ]

        missing = wanted - {
            str(task.get("id"))
            for task in selected
        }

        if missing:
            raise ValueError(
                f"Unknown task ids: {sorted(missing)}"
            )
    else:
        selected = list(tasks)

    if limit is not None:
        selected = selected[:limit]

    return selected


# ---------------------------------------------------------------------------
# Shared runtime config
# ---------------------------------------------------------------------------

def build_runtime_config(agent: str) -> dict[str, Any]:
    """Build a reproducible shared behavior profile.

    Unknown fields are ignored by the old baseline Configuration, while the
    current AskFlow Configuration consumes the new budget/verifier fields.
    """

    configurable: dict[str, Any] = {
        "thread_id": str(uuid.uuid4()),

        # Shared supervisor behavior
        "allow_clarification": False,
        "search_api": "tavily",
        "max_structured_output_retries": 3,
        "max_concurrent_research_units": 3,
        "max_researcher_iterations": 3,

        # Old baseline researcher round limit.
        "max_react_tool_calls": 6,

        # AskFlow researcher resource governance.
        "max_react_iterations": 6,
        "max_tool_calls_per_iteration": 4,
        "max_total_tool_calls": 12,
        "max_concurrent_tool_calls": 3,
        "max_tool_retries": 2,

        # AskFlow adaptive verification.
        "max_verification_iterations": 3,
        "max_targeted_research_tasks_per_round": 3,
        "min_expected_information_gain": 0.30,

        # AskFlow model routing.
        "model_router_dynamic_enabled": True,
        "model_router_prefer_low_cost": True,
    }

    return {
        "configurable": configurable,
        "metadata": {
            "eval_agent": agent,
            "eval_suite": "askflow-v1",
        },
    }


# ---------------------------------------------------------------------------
# One graph run
# ---------------------------------------------------------------------------

async def run_one_task(
    task: dict[str, Any],
    *,
    agent: str,
) -> dict[str, Any]:
    metrics = RunMetrics()
    token = _CURRENT_METRICS.set(metrics)

    graph = dr.deep_researcher_builder.compile(
        checkpointer=MemorySaver()
    )

    config = build_runtime_config(agent)

    start = time.perf_counter()
    error: str | None = None

    try:
        async for update in graph.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": task["prompt"],
                    }
                ]
            },
            config,
            stream_mode="updates",
        ):
            collect_update_metrics(
                update,
                metrics,
            )

        snapshot = await graph.aget_state(config)
        final_state = dict(snapshot.values)

    except Exception as exc:
        final_state = {}
        error = (
            f"{exc.__class__.__name__}: {exc}"
        )

    finally:
        latency_seconds = (
            time.perf_counter() - start
        )
        _CURRENT_METRICS.reset(token)

    final_report = str(
        final_state.get("final_report", "")
        or ""
    )

    execution_success = bool(
        final_report.strip()
    ) and not final_report.startswith(
        "Error generating final report"
    )

    verification_history = (
        metrics.verification_history
    )

    initial_gap_count = None
    final_gap_count = None
    final_evidence_sufficient = None

    if verification_history:
        first = verification_history[0]
        last = verification_history[-1]

        initial_gap_count = len(
            first.get("evidence_gaps", [])
        )
        final_gap_count = len(
            last.get("evidence_gaps", [])
        )
        final_evidence_sufficient = last.get(
            "evidence_sufficient"
        )

    return {
        "task_id": task["id"],
        "category": task.get("category"),
        "prompt": task["prompt"],
        "required_aspects": task.get(
            "required_aspects",
            [],
        ),

        "agent": agent,

        # Runtime / execution metrics
        "execution_success": execution_success,
        "latency_seconds": round(
            latency_seconds,
            3,
        ),
        "logical_tool_calls": (
            metrics.logical_tool_calls
        ),
        "logical_tool_names": dict(
            metrics.logical_tool_names
        ),

        # AskFlow-specific adaptive research metrics
        "verification_rounds": len(
            verification_history
        ),
        "verification_history": (
            verification_history
        ),
        "initial_evidence_gap_count": (
            initial_gap_count
        ),
        "final_evidence_gap_count": (
            final_gap_count
        ),
        "final_evidence_sufficient": (
            final_evidence_sufficient
        ),
        "targeted_research_rounds": (
            metrics.targeted_research_rounds
        ),
        "targeted_research_tasks": (
            metrics.targeted_research_tasks
        ),

        # Needed later by the judge phase
        "research_brief": to_jsonable(
            final_state.get("research_brief")
        ),
        "final_report": final_report,

        "error": error,
    }


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

async def run_suite(
    tasks: list[dict[str, Any]],
    *,
    agent: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[dict[str, Any]] = []

    for index, task in enumerate(
        tasks,
        start=1,
    ):
        print(
            f"\n[EVAL] "
            f"{index}/{len(tasks)} | "
            f"agent={agent} | "
            f"task={task['id']} | "
            f"category={task.get('category')}"
        )

        result = await run_one_task(
            task,
            agent=agent,
        )

        records.append(result)

        print(
            "[EVAL_RESULT] "
            f"task={result['task_id']} | "
            f"ok={result['execution_success']} | "
            f"latency={result['latency_seconds']}s | "
            f"tool_calls={result['logical_tool_calls']} | "
            f"verification_rounds="
            f"{result['verification_rounds']} | "
            f"targeted_rounds="
            f"{result['targeted_research_rounds']} | "
            f"error={result['error']!r}"
        )

        # Incremental write so a long experiment is not lost
        # if a later task crashes or the process is interrupted.
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(
        f"\n[EVAL] wrote "
        f"{len(records)} records -> "
        f"{output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        choices=("askflow", "baseline"),
        required=True,
    )

    parser.add_argument(
        "--tasks",
        type=Path,
        default=EVAL_DIR / "tasks.json",
    )

    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    tasks = load_tasks(
        args.tasks.resolve()
    )

    selected = select_tasks(
        tasks,
        args.task_ids,
        args.limit,
    )

    print(
        "[EVAL_CONFIG] "
        f"agent={args.agent} | "
        f"tasks={len(selected)} | "
        f"task_file={args.tasks.resolve()} | "
        f"output={args.output.resolve()}"
    )

    await run_suite(
        selected,
        agent=args.agent,
        output_path=args.output.resolve(),
    )


if __name__ == "__main__":
    asyncio.run(main())
