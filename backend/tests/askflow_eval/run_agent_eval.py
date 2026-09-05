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
import os
import subprocess
import time
import uuid
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from tavily import AsyncTavilyClient

try:
    from .experiment_groups import ExperimentGroup, resolve_experiment_group
    from .pricing import load_pricing_snapshot
    from .usage_tracking import (
        EvalUsageCallbackHandler,
        EvalUsageTracker,
        reset_current_usage_tracker,
        run_in_stage,
        set_current_usage_tracker,
        wrap_async_external_search,
    )
except ImportError:
    from experiment_groups import ExperimentGroup, resolve_experiment_group
    from pricing import load_pricing_snapshot
    from usage_tracking import (
        EvalUsageCallbackHandler,
        EvalUsageTracker,
        reset_current_usage_tracker,
        run_in_stage,
        set_current_usage_tracker,
        wrap_async_external_search,
    )


# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
EVAL_DIR = THIS_FILE.parent
BACKEND_ROOT = THIS_FILE.parents[2]

load_dotenv(BACKEND_ROOT / ".env")


def _configure_bailian_openai_compat_env() -> None:
    """Expose Bailian credentials through OpenAI-compatible env names.

    The pre-router baseline only knows the generic OpenAI provider path.
    Mapping these values inside the Eval process lets Groups A and B use the
    exact same static Bailian model without changing production source or the
    user's .env file.
    """

    bailian_key = os.environ.get("BAILIAN_API_KEY")
    bailian_base_url = os.environ.get(
        "BAILIAN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if bailian_key:
        os.environ["OPENAI_API_KEY"] = bailian_key
        os.environ["OPENAI_BASE_URL"] = bailian_base_url


_configure_bailian_openai_compat_env()


# Import only after loading the target worktree's .env.
# The script should be executed from that worktree's backend directory.
import open_deep_research.deep_researcher as dr  # noqa: E402
import open_deep_research.utils as research_utils  # noqa: E402


PRICING_SNAPSHOT = load_pricing_snapshot(
    EVAL_DIR / "pricing.json"
)

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
# Tool / external API instrumentation
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


# Count actual Tavily SDK search requests rather than logical tavily_search calls.
# A tool-level retry re-enters AsyncTavilyClient.search and is therefore counted
# again, matching external request / billing semantics.
_ORIGINAL_TAVILY_SEARCH = AsyncTavilyClient.search
AsyncTavilyClient.search = wrap_async_external_search(
    _ORIGINAL_TAVILY_SEARCH,
    provider="tavily",
)


# Webpage summarization happens inside the Tavily tool rather than as a named
# LangGraph node. This wrapper only adds a ContextVar stage label and delegates
# to the original function unchanged.
_ORIGINAL_SUMMARIZE_WEBPAGE = research_utils.summarize_webpage


async def instrumented_summarize_webpage(*args, **kwargs):
    return await run_in_stage(
        "webpage_summarization",
        lambda: _ORIGINAL_SUMMARIZE_WEBPAGE(*args, **kwargs),
    )


research_utils.summarize_webpage = instrumented_summarize_webpage


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
# Controlled experiment configuration
# ---------------------------------------------------------------------------

CONTROLLED_ENV_VALUES = {
    "ALLOW_CLARIFICATION": "false",
    "SEARCH_API": "tavily",
    "MAX_STRUCTURED_OUTPUT_RETRIES": "3",
    "MAX_CONCURRENT_RESEARCH_UNITS": "3",
    "MAX_RESEARCHER_ITERATIONS": "3",
    "MAX_REACT_TOOL_CALLS": "6",
    "MAX_REACT_ITERATIONS": "6",
    "MAX_TOOL_CALLS_PER_ITERATION": "4",
    "MAX_TOTAL_TOOL_CALLS": "12",
    "MAX_CONCURRENT_TOOL_CALLS": "3",
    "MAX_TOOL_RETRIES": "2",
    "MAX_VERIFICATION_ITERATIONS": "3",
    "MAX_TARGETED_RESEARCH_TASKS_PER_ROUND": "3",
    "MIN_EXPECTED_INFORMATION_GAIN": "0.30",
}


def get_code_revision() -> str:
    """Best-effort git revision for reproducible result records."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BACKEND_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except Exception:
        return "unknown"


def apply_controlled_eval_environment(group: ExperimentGroup) -> None:
    """Lock controlled Eval fields despite Configuration's env-first policy."""

    for name, value in CONTROLLED_ENV_VALUES.items():
        os.environ[name] = value

    if group.code == "B":
        os.environ["MODEL_ROUTER_DYNAMIC_ENABLED"] = "false"
        os.environ["MODEL_ROUTER_PREFER_LOW_COST"] = "true"
    elif group.code == "C":
        os.environ["MODEL_ROUTER_DYNAMIC_ENABLED"] = "true"
        os.environ["MODEL_ROUTER_PREFER_LOW_COST"] = "true"

    if group.static_model:
        # Baseline source reads these legacy per-stage model fields. Setting all
        # four to one value guarantees Group A is truly single-model. Keeping
        # the same values for B provides a second guard against accidental
        # profile-model leakage.
        os.environ["RESEARCH_MODEL"] = group.static_model
        os.environ["SUMMARIZATION_MODEL"] = group.static_model
        os.environ["COMPRESSION_MODEL"] = group.static_model
        os.environ["FINAL_REPORT_MODEL"] = group.static_model


def validate_group_worktree(group: ExperimentGroup) -> None:
    """Fail early on the most dangerous A/B/C worktree mix-ups."""

    has_askflow_verifier = hasattr(dr, "evidence_verifier")

    if group.code == "A" and has_askflow_verifier:
        raise RuntimeError(
            "Group A must be run from the pre-enhancement baseline worktree; "
            "this worktree exposes AskFlow evidence_verifier."
        )

    if group.code in {"B", "C"} and not has_askflow_verifier:
        raise RuntimeError(
            f"Group {group.code} must be run from the AskFlow worktree; "
            "this worktree does not expose evidence_verifier."
        )


def install_askflow_static_router(static_model: str) -> None:
    """Eval-only override that forces every AskFlow LLM stage to one model."""

    import open_deep_research.model_router as model_router

    candidates = {
        spec.model_name: spec
        for spec in model_router.MODEL_CATALOG
    }

    model = candidates.get(static_model)

    if model is None:
        available = ", ".join(sorted(candidates))
        raise ValueError(
            f"Static model {static_model!r} is not in MODEL_CATALOG. "
            f"Available: {available}"
        )

    def static_route_model_for_text(
        *,
        task_type,
        text,
        dynamic_enabled=True,
        prefer_low_cost=True,
    ):
        del text, dynamic_enabled, prefer_low_cost

        profile = model_router.get_default_profile_for_task(task_type)
        decision = model_router.ModelDecision(
            task_type=task_type,
            profile=profile,
            model=model,
            candidates=(model.model_name,),
            score=None,
            estimated_cost=None,
            reason=(
                "eval static override: "
                f"group=B; model={model.model_name}"
            ),
        )
        model_router.log_model_decision(decision)
        return decision

    # deep_researcher.py and utils.py import route_model_for_text directly, so
    # patch their bound globals as well as the defining module.
    model_router.route_model_for_text = static_route_model_for_text
    if hasattr(dr, "route_model_for_text"):
        dr.route_model_for_text = static_route_model_for_text
    if hasattr(research_utils, "route_model_for_text"):
        research_utils.route_model_for_text = static_route_model_for_text


# ---------------------------------------------------------------------------
# Shared runtime config
# ---------------------------------------------------------------------------

def build_runtime_config(group: ExperimentGroup) -> dict[str, Any]:
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
        "model_router_dynamic_enabled": (group.code == "C"),
        "model_router_prefer_low_cost": True,
    }

    return {
        "configurable": configurable,
        "metadata": {
            "eval_agent": group.agent,
            "eval_group": group.code,
            "eval_router_mode": group.router_mode,
            "eval_suite": "askflow-v2",
        },
    }


# ---------------------------------------------------------------------------
# One graph run
# ---------------------------------------------------------------------------

async def run_one_task(
    task: dict[str, Any],
    *,
    group: ExperimentGroup,
    eval_run_id: str,
    code_revision: str,
) -> dict[str, Any]:
    metrics = RunMetrics()
    metrics_token = _CURRENT_METRICS.set(metrics)

    usage_tracker = EvalUsageTracker()
    usage_handler = EvalUsageCallbackHandler(usage_tracker)
    usage_token = set_current_usage_tracker(usage_tracker)

    graph = dr.deep_researcher_builder.compile(
        checkpointer=MemorySaver()
    )

    config = build_runtime_config(group)
    task_run_id = str(config["configurable"]["thread_id"])
    config["callbacks"] = [usage_handler]

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
        _CURRENT_METRICS.reset(metrics_token)
        reset_current_usage_tracker(usage_token)

    api_usage = usage_tracker.snapshot(
        PRICING_SNAPSHOT
    )

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

        "agent": group.agent,
        "experiment_group": group.code,
        "router_mode": group.router_mode,
        "static_model": group.static_model,
        "eval_run_id": eval_run_id,
        "task_run_id": task_run_id,
        "code_revision": code_revision,

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

        # API serving cost / usage observability.
        **api_usage,

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
    group: ExperimentGroup,
    output_path: Path,
    eval_run_id: str,
    code_revision: str,
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
            f"group={group.code} | "
            f"agent={group.agent} | "
            f"task={task['id']} | "
            f"category={task.get('category')}"
        )

        result = await run_one_task(
            task,
            group=group,
            eval_run_id=eval_run_id,
            code_revision=code_revision,
        )

        records.append(result)

        print(
            "[EVAL_RESULT] "
            f"task={result['task_id']} | "
            f"ok={result['execution_success']} | "
            f"latency={result['latency_seconds']}s | "
            f"tool_calls={result['logical_tool_calls']} | "
            f"llm_calls={result['llm_calls']} | "
            f"search_requests={result['external_search_requests']} | "
            f"api_cost={result['total_api_cost']} "
            f"{result['pricing_currency']} | "
            f"cost_complete={result['cost_complete']} | "
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
        "--group",
        choices=("A", "B", "C", "a", "b", "c"),
        default=None,
        help="Formal controlled experiment arm. Prefer this over --agent.",
    )

    parser.add_argument(
        "--static-model",
        default=None,
        help=(
            "Optional static model override for Groups A/B. "
            "Defaults to openai:qwen3.5-plus."
        ),
    )

    parser.add_argument(
        "--agent",
        choices=("askflow", "baseline"),
        default=None,
        help="Legacy smoke mode; formal benchmark should use --group.",
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

    if args.group is None:
        if args.agent is None:
            raise SystemExit("Provide --group A/B/C (preferred) or legacy --agent")

        # Preserve the old smoke interface without pretending it is a formal
        # controlled group.
        group = ExperimentGroup(
            code="C" if args.agent == "askflow" else "A",
            agent=args.agent,
            router_mode="dynamic" if args.agent == "askflow" else "unavailable",
            static_model=None,
            description="legacy --agent compatibility mode",
        )
        formal_group = False
    else:
        group = resolve_experiment_group(
            args.group,
            static_model=args.static_model,
        )
        formal_group = True

    if formal_group:
        apply_controlled_eval_environment(group)
        validate_group_worktree(group)

        if group.code == "B":
            assert group.static_model is not None
            install_askflow_static_router(group.static_model)

    tasks = load_tasks(
        args.tasks.resolve()
    )

    selected = select_tasks(
        tasks,
        args.task_ids,
        args.limit,
    )

    eval_run_id = str(uuid.uuid4())
    code_revision = get_code_revision()

    print(
        "[EVAL_CONFIG] "
        f"group={group.code if formal_group else 'legacy'} | "
        f"agent={group.agent} | "
        f"router_mode={group.router_mode} | "
        f"static_model={group.static_model} | "
        f"tasks={len(selected)} | "
        f"revision={code_revision} | "
        f"task_file={args.tasks.resolve()} | "
        f"output={args.output.resolve()} | "
        f"pricing_snapshot={PRICING_SNAPSHOT.effective_date} | "
        f"pricing_currency={PRICING_SNAPSHOT.currency}"
    )

    await run_suite(
        selected,
        group=group,
        output_path=args.output.resolve(),
        eval_run_id=eval_run_id,
        code_revision=code_revision,
    )


if __name__ == "__main__":
    asyncio.run(main())
