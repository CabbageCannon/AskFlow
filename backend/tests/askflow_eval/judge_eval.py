"""
AskFlow evaluation judge.

This script evaluates already-generated research reports.

It does NOT run AskFlow or the baseline agent itself.

Main metrics:
1. Aspect Coverage
2. Research Success Rate
3. Blind Pairwise Win Rate
4. Evidence Gap Repair Rate

It also aggregates:
- execution success
- logical tool calls
- latency
- verification rounds
- targeted research trigger rate

Example:

uv run python tests/askflow_eval/judge_eval.py \
    --askflow-results tests/askflow_eval/results/askflow_smoke.jsonl \
                       tests/askflow_eval/results/askflow_conflict_retry.jsonl \
    --baseline-results ..\\..\\AskFlow-baseline\\backend\\tests\\askflow_eval\\results\\baseline_test.jsonl \
                        ..\\..\\AskFlow-baseline\\backend\\tests\\askflow_eval\\results\\baseline_smoke_remaining.jsonl \
    --task-ids api-01 conflict-01 efficiency-03 \
    --output tests/askflow_eval/results/judged_smoke.jsonl \
    --summary tests/askflow_eval/results/judged_smoke_summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# =========================================================
# Paths
# =========================================================

SCRIPT_PATH = Path(__file__).resolve()

# backend/tests/askflow_eval/judge_eval.py
# parents[0] -> askflow_eval
# parents[1] -> tests
# parents[2] -> backend
BACKEND_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_TASK_FILE = (
    BACKEND_ROOT
    / "tests"
    / "askflow_eval"
    / "tasks.json"
)


# =========================================================
# Judge structured output schemas
# =========================================================


class AspectEvaluation(BaseModel):
    """Evaluation of one required aspect."""

    aspect: str = Field(
        description=(
            "The exact required aspect identifier supplied "
            "in the evaluation request."
        )
    )

    covered: bool = Field(
        description=(
            "Whether the report materially satisfies this "
            "required aspect."
        )
    )

    reasoning: str = Field(
        description=(
            "A concise explanation grounded in the report."
        )
    )


class ResearchJudgment(BaseModel):
    """Single-report research quality judgment."""

    aspect_evaluations: list[AspectEvaluation]

    critical_factual_error: bool = Field(
        description=(
            "True only when the report contains a clear, "
            "material factual, evidential, temporal, or "
            "instruction-following failure that undermines "
            "the answer."
        )
    )

    critical_error_reason: str | None = Field(
        default=None,
        description=(
            "Explain the critical error if one exists."
        ),
    )

    overall_reasoning: str = Field(
        description=(
            "Concise overall assessment of the report."
        )
    )


class PairwiseVerdict(BaseModel):
    """Blind comparison between Report A and Report B."""

    winner: Literal["A", "B", "tie"]

    reasoning: str = Field(
        description=(
            "Explain why one report is better, or why "
            "they are materially equivalent."
        )
    )


class GapRepairJudgment(BaseModel):
    """Whether one original evidence gap was repaired."""

    repaired: bool = Field(
        description=(
            "Whether the original evidence gap was "
            "substantively resolved by later research."
        )
    )

    reasoning: str = Field(
        description=(
            "Explain whether the same unresolved issue "
            "still remains after follow-up research."
        )
    )


# =========================================================
# Prompts
# =========================================================


ASPECT_JUDGE_SYSTEM_PROMPT = """
You are an evaluation judge for deep-research systems.

You are NOT conducting new research.
You must evaluate only the supplied user task and research report.

Do not use web search.
Do not invent evidence.
Do not reward a report merely for being long.

Your job is to determine whether the report materially satisfies
each required aspect.

Important evaluation rules:

1. Required aspects

Each required_aspect is an evaluation target derived from the
original user request.

Mark covered=true only when the report materially addresses that
aspect.

A passing mention is not enough.


2. Evidence requirements

When the user explicitly requires:
- official sources
- primary sources
- dates
- direct evidence
- explicit evidence gaps
- conflict resolution
- scope control

those requirements are part of coverage.

For example:

If the required aspect is "official_only", but the report relies on
third-party claims for its core conclusion, that aspect should not
be considered fully covered.

However, the presence of some secondary sources does NOT by itself
constitute a critical failure if the report's core conclusions are
supported by appropriate evidence and secondary sources are used only
as supplements.


3. Evidence gaps

If the requested information cannot be established from the
available evidence, a responsible report may still satisfy the
requirement by explicitly identifying the evidence gap instead of
inventing a value.

Do not penalize uncertainty when uncertainty is the correct answer.


4. Temporal grounding

The evaluation runtime date is 2026-09-04.

The explicit cutoff date or temporal scope stated in the supplied
user task is authoritative.

For example, if the task asks for information "as of 2026-09-01",
you MUST evaluate the report relative to 2026-09-01.

Do NOT:
- treat 2026-09-01 as a future date
- use your pretrained knowledge cutoff to override the task date
- use an internal or remembered current date instead of the supplied
  evaluation date
- declare a model, release, API feature, or event impossible merely
  because it postdates your pretrained knowledge

For time-sensitive tasks, evaluate whether the report distinguishes
old information from current information and explains version or
date conflicts when relevant.


5. Source-status caution

Do NOT classify a source URL, domain, or documentation path as
fabricated, unofficial, or invalid merely because it is unfamiliar
to you.

Do NOT infer that a documentation domain must be unofficial simply
because you remember a different historical domain.

Only penalize source provenance when:
- the supplied report itself clearly identifies the source as
  third-party, or
- the non-official status is unambiguous from the supplied material.

If you are uncertain whether a domain is an official documentation
domain, treat its provenance as uncertain rather than as a critical
factual error.

Do not independently fact-check URLs from memory.


6. Factual evaluation

Do NOT independently fact-check claims primarily from your pretrained
world knowledge.

Judge factual reliability primarily from:
- the evidence and citations presented in the report
- consistency between claims and supplied evidence
- internal consistency of the report
- the explicit requirements of the user task
- appropriate handling of uncertainty and evidence gaps

Your own remembered knowledge may be incomplete or outdated and
must not override the temporal scope of the task.


7. Critical factual error

Set critical_factual_error=true ONLY for a clear and MATERIAL failure
that substantially undermines the usefulness or reliability of the
answer.

Examples include:
- a major internal contradiction
- presenting an unsupported CORE claim as confirmed fact
- mixing incompatible versions or dates in a way that materially
  changes the answer
- directly violating a CENTRAL evidence constraint in a way that
  undermines the report
- answering a materially different question

Do NOT mark critical_factual_error=true merely because:
- some secondary sources are present
- source quality is imperfect
- one non-core source is uncertain
- one required aspect is incomplete
- citation formatting is imperfect
- a URL or domain is unfamiliar to you
- a claim is newer than your pretrained knowledge

A weakness in one aspect should normally be reflected by
covered=false for that aspect.

Only escalate it to critical_factual_error=true when the problem is
severe enough to materially invalidate the report as a whole.


8. Exact output coverage

Return exactly one AspectEvaluation for every required aspect.

Use the exact aspect identifier supplied to you.

Before returning, verify that:
- every required aspect appears exactly once
- covered=false is used for materially missing or unsupported aspects
- critical_factual_error is reserved for genuinely report-invalidating
  failures
"""

PAIRWISE_JUDGE_SYSTEM_PROMPT = """
You are a blind evaluator comparing two deep-research reports.

You are NOT conducting new research.
You must compare only the reports and the supplied task.

You do not know which system produced either report.

Do not prefer Report A or Report B because of its position.
Do not reward verbosity by itself.

Evaluate which report better satisfies the user's actual research
request.

Consider:

- required-aspect coverage
- correctness and internal consistency
- quality and appropriateness of evidence
- compliance with official/primary-source requirements
- handling of missing evidence
- handling of conflicting evidence
- temporal and version awareness
- scope control
- unsupported certainty or hallucinated claims
- usefulness of the final answer

A report can be better even if it is shorter.

Use "tie" when neither report has a meaningful overall advantage.

Temporal and source-grounding rules:

- The evaluation runtime date is 2026-09-04.
- Treat the explicit cutoff date in the supplied task as authoritative.
- Do not use your pretrained knowledge cutoff or an internal remembered
  date to override the task's temporal scope.
- Do not reject newer models, releases, or API features merely because
  they postdate your pretrained knowledge.
- Do not classify an unfamiliar documentation URL or domain as
  fabricated or unofficial merely from memory.
- Judge source quality from the supplied reports and evidence rather
  than unsupported assumptions about domains.

Do not use outside web search or invent facts that are absent from
the reports.
"""


GAP_REPAIR_SYSTEM_PROMPT = """
You are evaluating whether targeted follow-up research repaired an
earlier evidence gap.

You are NOT conducting new research.

You will receive:

1. The original evidence gap.
2. One or more later Evidence Verifier results.

Determine whether the SAME substantive unresolved issue has been
resolved.

Set repaired=true only when later verification provides strong
indication that the original problem has been materially addressed.

Important:

- A gap disappearing from a list does NOT automatically mean it was repaired.
- Rewording the gap does NOT mean it was repaired.
- If the same problem remains under a different name or category,
  repaired=false.
- If later verification explicitly considers the relevant evidence
  sufficient and no longer identifies the substantive issue,
  repaired may be true.
- If evidence remains missing, weak, or conflicting,
  repaired=false.
"""


# =========================================================
# Utility
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge AskFlow and baseline deep-research "
            "evaluation results."
        )
    )

    parser.add_argument(
        "--askflow-results",
        nargs="+",
        required=True,
        help=(
            "One or more AskFlow JSONL result files. "
            "If duplicate task IDs exist, the LAST file wins."
        ),
    )

    parser.add_argument(
        "--baseline-results",
        nargs="+",
        required=True,
        help=(
            "One or more baseline JSONL result files. "
            "If duplicate task IDs exist, the LAST file wins."
        ),
    )

    parser.add_argument(
        "--tasks",
        default=str(DEFAULT_TASK_FILE),
        help="Canonical task dataset JSON file.",
    )

    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help=(
            "Optional subset of task IDs. "
            "Default: evaluate every matched task."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output per-task judged JSONL file.",
    )

    parser.add_argument(
        "--summary",
        required=True,
        help="Output aggregate summary JSON file.",
    )

    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.80,
        help=(
            "Minimum aspect coverage for Research Success. "
            "Default: 0.80"
        ),
    )

    parser.add_argument(
        "--pairwise-seed",
        type=int,
        default=42,
        help=(
            "Deterministic randomization seed for A/B order."
        ),
    )

    parser.add_argument(
        "--gap-min-information-gain",
        type=float,
        default=0.30,
        help=(
            "Minimum expected_information_gain used to "
            "reconstruct actionable evidence gaps."
        ),
    )

    parser.add_argument(
        "--gap-max-tasks-per-round",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--gap-max-concurrent",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing judged JSONL file and "
            "skip task IDs already present."
        ),
    )

    parser.add_argument(
        "--skip-pairwise",
        action="store_true",
    )

    parser.add_argument(
        "--skip-gap-repair",
        action="store_true",
    )

    return parser.parse_args()


def load_json_file(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_jsonl_files(
    paths: list[str],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    """
    Load one or more result JSONL files.

    Later files override earlier files for duplicate task IDs.
    """

    records: dict[str, dict[str, Any]] = {}

    for raw_path in paths:
        path = Path(raw_path).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"{label} result file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_number, line in enumerate(
                f,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in {path} "
                        f"at line {line_number}: {exc}"
                    ) from exc

                task_id = record.get(
                    "task_id"
                )

                if not task_id:
                    raise ValueError(
                        f"Missing task_id in {path} "
                        f"line {line_number}"
                    )

                if task_id in records:
                    print(
                        "[JUDGE_LOAD] "
                        f"{label} duplicate task={task_id} | "
                        "later record overrides earlier record"
                    )

                records[task_id] = record

    return records


def normalize_aspect_name(
    value: str,
) -> str:
    return str(value).strip().lower()


def safe_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return round(
        statistics.mean(values),
        4,
    )


def safe_median(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return round(
        statistics.median(values),
        4,
    )


def safe_percentile(
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
    task_id: str,
    seed: int,
) -> random.Random:
    text = f"{seed}:{task_id}"

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    derived_seed = int(
        digest[:16],
        16,
    )

    return random.Random(
        derived_seed
    )


def dump_jsonl_record(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def truncate_for_console(
    text: str | None,
    length: int = 100,
) -> str:
    if not text:
        return ""

    text = " ".join(
        str(text).split()
    )

    if len(text) <= length:
        return text

    return text[:length] + "..."


# =========================================================
# Judge model
# =========================================================


def build_judge_model() -> tuple[ChatOpenAI, str, str]:
    """
    Build an OpenAI-compatible judge.

    Required:
        EVAL_JUDGE_MODEL

    Credentials:
        EVAL_JUDGE_API_KEY
        or
        BAILIAN_API_KEY

    Base URL:
        EVAL_JUDGE_BASE_URL
        or
        BAILIAN_BASE_URL

    Optional:
        EVAL_JUDGE_STRUCTURED_METHOD
        EVAL_JUDGE_TEMPERATURE
        EVAL_JUDGE_TIMEOUT_SECONDS
        EVAL_JUDGE_MAX_TOKENS
        EVAL_JUDGE_EXTRA_BODY_JSON
    """

    model_name = os.getenv(
        "EVAL_JUDGE_MODEL"
    )

    api_key = (
        os.getenv(
            "EVAL_JUDGE_API_KEY"
        )
        or os.getenv(
            "BAILIAN_API_KEY"
        )
    )

    base_url = (
        os.getenv(
            "EVAL_JUDGE_BASE_URL"
        )
        or os.getenv(
            "BAILIAN_BASE_URL"
        )
    )

    if not model_name:
        raise RuntimeError(
            "Missing EVAL_JUDGE_MODEL in environment."
        )

    if not api_key:
        raise RuntimeError(
            "Missing EVAL_JUDGE_API_KEY "
            "or BAILIAN_API_KEY."
        )

    if not base_url:
        raise RuntimeError(
            "Missing EVAL_JUDGE_BASE_URL "
            "or BAILIAN_BASE_URL."
        )

    temperature = float(
        os.getenv(
            "EVAL_JUDGE_TEMPERATURE",
            "0",
        )
    )

    timeout = float(
        os.getenv(
            "EVAL_JUDGE_TIMEOUT_SECONDS",
            "180",
        )
    )

    structured_method = os.getenv(
        "EVAL_JUDGE_STRUCTURED_METHOD",
        "function_calling",
    )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "timeout": timeout,
    }

    max_tokens = os.getenv(
        "EVAL_JUDGE_MAX_TOKENS"
    )

    if max_tokens:
        kwargs["max_tokens"] = int(
            max_tokens
        )

    extra_body_raw = os.getenv(
        "EVAL_JUDGE_EXTRA_BODY_JSON"
    )

    if extra_body_raw:
        try:
            kwargs["extra_body"] = json.loads(
                extra_body_raw
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "EVAL_JUDGE_EXTRA_BODY_JSON "
                "must be valid JSON."
            ) from exc

    return (
        ChatOpenAI(**kwargs),
        model_name,
        structured_method,
    )


def structured_judge(
    base_model: ChatOpenAI,
    schema: type[BaseModel],
    *,
    structured_method: str,
):
    retry_count = int(
        os.getenv(
            "EVAL_JUDGE_RETRIES",
            "3",
        )
    )

    return (
        base_model
        .with_structured_output(
            schema,
            method=structured_method,
        )
        .with_retry(
            stop_after_attempt=retry_count
        )
    )


# =========================================================
# Single report judgment
# =========================================================


async def judge_single_report(
    *,
    judge_model: ChatOpenAI,
    structured_method: str,
    task_id: str,
    prompt: str,
    required_aspects: list[str],
    report: str,
    execution_success: bool,
    coverage_threshold: float,
) -> dict[str, Any]:
    """
    Judge one final report.

    Research Success:
        execution_success
        AND aspect coverage >= threshold
        AND no critical factual error
    """

    if (
        not execution_success
        or not str(report).strip()
    ):
        return {
            "judged": True,
            "aspect_coverage": 0.0,
            "covered_aspects": [],
            "missing_aspects": list(
                required_aspects
            ),
            "aspect_evaluations": [
                {
                    "aspect": aspect,
                    "covered": False,
                    "reasoning": (
                        "No successful final report was produced."
                    ),
                }
                for aspect in required_aspects
            ],
            "critical_factual_error": False,
            "critical_error_reason": None,
            "research_success": False,
            "overall_reasoning": (
                "Agent execution did not produce "
                "a usable final report."
            ),
            "judge_error": None,
        }

    human_prompt = f"""
<Task ID>
{task_id}
</Task ID>

<User Task>
{prompt}
</User Task>

<Required Aspects>
{json.dumps(required_aspects, ensure_ascii=False)}
</Required Aspects>

<Research Report>
{report}
</Research Report>

Evaluate the report against every required aspect.

Return exactly one evaluation for each required aspect using the
exact identifier from Required Aspects.
"""

    model = structured_judge(
        judge_model,
        ResearchJudgment,
        structured_method=structured_method,
    )

    try:
        judgment = await model.ainvoke(
            [
                SystemMessage(
                    content=ASPECT_JUDGE_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=human_prompt
                ),
            ]
        )

    except Exception as exc:
        return {
            "judged": False,
            "aspect_coverage": None,
            "covered_aspects": [],
            "missing_aspects": [],
            "aspect_evaluations": [],
            "critical_factual_error": None,
            "critical_error_reason": None,
            "research_success": None,
            "overall_reasoning": None,
            "judge_error": str(exc),
        }

    evaluations_by_name: dict[
        str,
        AspectEvaluation,
    ] = {}

    for evaluation in judgment.aspect_evaluations:
        normalized = normalize_aspect_name(
            evaluation.aspect
        )

        if normalized not in evaluations_by_name:
            evaluations_by_name[
                normalized
            ] = evaluation

    reconciled: list[
        dict[str, Any]
    ] = []

    covered_count = 0

    for required_aspect in required_aspects:
        normalized = normalize_aspect_name(
            required_aspect
        )

        evaluation = evaluations_by_name.get(
            normalized
        )

        if evaluation is None:
            covered = False
            reasoning = (
                "Judge omitted this required aspect "
                "from the structured response."
            )
        else:
            covered = bool(
                evaluation.covered
            )
            reasoning = (
                evaluation.reasoning
            )

        if covered:
            covered_count += 1

        reconciled.append(
            {
                "aspect": required_aspect,
                "covered": covered,
                "reasoning": reasoning,
            }
        )

    if required_aspects:
        coverage = (
            covered_count
            / len(required_aspects)
        )
    else:
        coverage = 1.0

    coverage = round(
        coverage,
        4,
    )

    covered_aspects = [
        item["aspect"]
        for item in reconciled
        if item["covered"]
    ]

    missing_aspects = [
        item["aspect"]
        for item in reconciled
        if not item["covered"]
    ]

    research_success = (
        execution_success
        and coverage >= coverage_threshold
        and not judgment.critical_factual_error
    )

    return {
        "judged": True,
        "aspect_coverage": coverage,
        "covered_aspects": covered_aspects,
        "missing_aspects": missing_aspects,
        "aspect_evaluations": reconciled,
        "critical_factual_error": (
            judgment.critical_factual_error
        ),
        "critical_error_reason": (
            judgment.critical_error_reason
        ),
        "research_success": (
            research_success
        ),
        "overall_reasoning": (
            judgment.overall_reasoning
        ),
        "judge_error": None,
    }


# =========================================================
# Pairwise judgment
# =========================================================


async def judge_pairwise(
    *,
    judge_model: ChatOpenAI,
    structured_method: str,
    task_id: str,
    prompt: str,
    required_aspects: list[str],
    baseline_report: str,
    askflow_report: str,
    baseline_execution_success: bool,
    askflow_execution_success: bool,
    seed: int,
) -> dict[str, Any]:
    """
    Blind randomized comparison.

    Judge never sees "AskFlow" or "baseline".
    """

    if (
        not baseline_execution_success
        or not askflow_execution_success
        or not baseline_report.strip()
        or not askflow_report.strip()
    ):
        return {
            "judged": False,
            "winner_agent": None,
            "winner_label": None,
            "reasoning": None,
            "A_agent": None,
            "B_agent": None,
            "judge_error": (
                "Pairwise comparison requires "
                "two successful reports."
            ),
        }

    rng = deterministic_rng(
        task_id,
        seed,
    )

    if rng.random() < 0.5:
        report_a = baseline_report
        report_b = askflow_report

        a_agent = "baseline"
        b_agent = "askflow"

    else:
        report_a = askflow_report
        report_b = baseline_report

        a_agent = "askflow"
        b_agent = "baseline"

    human_prompt = f"""
<Task ID>
{task_id}
</Task ID>

<User Task>
{prompt}
</User Task>

<Required Aspects>
{json.dumps(required_aspects, ensure_ascii=False)}
</Required Aspects>

<Report A>
{report_a}
</Report A>

<Report B>
{report_b}
</Report B>

Compare Report A and Report B blindly.

Choose:
- A
- B
- tie
"""

    model = structured_judge(
        judge_model,
        PairwiseVerdict,
        structured_method=structured_method,
    )

    try:
        verdict = await model.ainvoke(
            [
                SystemMessage(
                    content=PAIRWISE_JUDGE_SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=human_prompt
                ),
            ]
        )

    except Exception as exc:
        return {
            "judged": False,
            "winner_agent": None,
            "winner_label": None,
            "reasoning": None,
            "A_agent": a_agent,
            "B_agent": b_agent,
            "judge_error": str(exc),
        }

    if verdict.winner == "tie":
        winner_agent = "tie"

    elif verdict.winner == "A":
        winner_agent = a_agent

    else:
        winner_agent = b_agent

    return {
        "judged": True,
        "winner_agent": winner_agent,
        "winner_label": verdict.winner,
        "reasoning": verdict.reasoning,
        "A_agent": a_agent,
        "B_agent": b_agent,
        "judge_error": None,
    }


# =========================================================
# Evidence Gap Repair
# =========================================================


def select_initial_triggered_gaps(
    record: dict[str, Any],
    *,
    min_information_gain: float,
    max_tasks_per_round: int,
    max_concurrent: int,
) -> list[dict[str, Any]]:
    """
    Reconstruct the first targeted-research round's
    actionable gaps.

    This mirrors AskFlow's routing policy:
    - expected_information_gain threshold
    - sort by importance * information gain
    - max tasks bounded by concurrent and per-round limits
    """

    if (
        record.get(
            "targeted_research_rounds",
            0,
        )
        <= 0
    ):
        return []

    history = record.get(
        "verification_history",
        [],
    )

    if not history:
        return []

    initial_verification = history[0]

    gaps = initial_verification.get(
        "evidence_gaps",
        [],
    ) or []

    actionable = []

    for gap in gaps:
        gain = float(
            gap.get(
                "expected_information_gain",
                0.0,
            )
        )

        if gain < min_information_gain:
            continue

        importance = float(
            gap.get(
                "importance",
                0.0,
            )
        )

        item = dict(
            gap
        )

        item["_priority"] = (
            importance * gain
        )

        actionable.append(
            item
        )

    actionable.sort(
        key=lambda item: item["_priority"],
        reverse=True,
    )

    limit = min(
        max_tasks_per_round,
        max_concurrent,
    )

    selected = actionable[:limit]

    for item in selected:
        item.pop(
            "_priority",
            None,
        )

    return selected


async def judge_gap_repair(
    *,
    judge_model: ChatOpenAI,
    structured_method: str,
    record: dict[str, Any],
    min_information_gain: float,
    max_tasks_per_round: int,
    max_concurrent: int,
) -> dict[str, Any]:
    """
    Evaluate repair of first-round actionable gaps.
    """

    targeted_rounds = int(
        record.get(
            "targeted_research_rounds",
            0,
        )
        or 0
    )

    if targeted_rounds <= 0:
        return {
            "applicable": False,
            "triggered_gap_count": 0,
            "repaired_gap_count": 0,
            "repair_rate": None,
            "gap_judgments": [],
            "judge_errors": [],
        }

    history = record.get(
        "verification_history",
        [],
    ) or []

    initial_gaps = select_initial_triggered_gaps(
        record,
        min_information_gain=min_information_gain,
        max_tasks_per_round=max_tasks_per_round,
        max_concurrent=max_concurrent,
    )

    if not initial_gaps:
        return {
            "applicable": True,
            "triggered_gap_count": 0,
            "repaired_gap_count": 0,
            "repair_rate": None,
            "gap_judgments": [],
            "judge_errors": [
                (
                    "Targeted research was recorded, "
                    "but no initial actionable gaps could "
                    "be reconstructed from verification_history."
                )
            ],
        }

    later_history = history[1:]

    results: list[
        dict[str, Any]
    ] = []

    judge_errors: list[
        str
    ] = []

    # If targeted research happened but there was no
    # later verification, conservatively count the gap
    # as not repaired.
    if not later_history:
        for gap in initial_gaps:
            results.append(
                {
                    "initial_gap": gap,
                    "repaired": False,
                    "reasoning": (
                        "Targeted research occurred, but no "
                        "post-targeted Evidence Verifier result "
                        "was available. Conservatively counted "
                        "as not repaired."
                    ),
                    "judge_error": None,
                }
            )

        return {
            "applicable": True,
            "triggered_gap_count": len(
                initial_gaps
            ),
            "repaired_gap_count": 0,
            "repair_rate": 0.0,
            "gap_judgments": results,
            "judge_errors": [],
        }

    model = structured_judge(
        judge_model,
        GapRepairJudgment,
        structured_method=structured_method,
    )

    later_history_json = json.dumps(
        later_history,
        ensure_ascii=False,
        indent=2,
    )

    for gap in initial_gaps:
        human_prompt = f"""
<Original Evidence Gap>
{json.dumps(gap, ensure_ascii=False, indent=2)}
</Original Evidence Gap>

<Later Verification Results>
{later_history_json}
</Later Verification Results>

Determine whether the ORIGINAL substantive evidence gap was
actually repaired by the follow-up research.
"""

        try:
            verdict = await model.ainvoke(
                [
                    SystemMessage(
                        content=GAP_REPAIR_SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=human_prompt
                    ),
                ]
            )

            results.append(
                {
                    "initial_gap": gap,
                    "repaired": verdict.repaired,
                    "reasoning": verdict.reasoning,
                    "judge_error": None,
                }
            )

        except Exception as exc:
            error_text = str(
                exc
            )

            judge_errors.append(
                error_text
            )

            # Conservative treatment:
            # a triggered gap that cannot be verified as
            # repaired is counted as unrepaired.
            results.append(
                {
                    "initial_gap": gap,
                    "repaired": False,
                    "reasoning": (
                        "Gap repair judge failed. "
                        "Conservatively counted as not repaired."
                    ),
                    "judge_error": error_text,
                }
            )

    repaired_count = sum(
        1
        for item in results
        if item["repaired"]
    )

    triggered_count = len(
        results
    )

    repair_rate = (
        repaired_count
        / triggered_count
        if triggered_count
        else None
    )

    if repair_rate is not None:
        repair_rate = round(
            repair_rate,
            4,
        )

    return {
        "applicable": True,
        "triggered_gap_count": triggered_count,
        "repaired_gap_count": repaired_count,
        "repair_rate": repair_rate,
        "gap_judgments": results,
        "judge_errors": judge_errors,
    }


# =========================================================
# Aggregation
# =========================================================


def summarize_agent(
    rows: list[dict[str, Any]],
    agent_name: Literal[
        "askflow",
        "baseline",
    ],
) -> dict[str, Any]:
    agent_rows = [
        row.get(
            agent_name
        )
        for row in rows
        if row.get(
            agent_name
        )
    ]

    total = len(
        agent_rows
    )

    execution_success_count = sum(
        1
        for record in agent_rows
        if record.get(
            "execution_success"
        )
    )

    judged_rows = [
        record
        for record in agent_rows
        if (
            record.get(
                "quality"
            )
            and record[
                "quality"
            ].get(
                "judged"
            )
        )
    ]

    judged_count = len(
        judged_rows
    )

    research_success_count = sum(
        1
        for record in judged_rows
        if record[
            "quality"
        ].get(
            "research_success"
        )
    )

    coverage_values = [
        float(
            record[
                "quality"
            ][
                "aspect_coverage"
            ]
        )
        for record in judged_rows
        if record[
            "quality"
        ].get(
            "aspect_coverage"
        )
        is not None
    ]

    latency_values = [
        float(
            record.get(
                "latency_seconds",
                0,
            )
        )
        for record in agent_rows
        if record.get(
            "latency_seconds"
        )
        is not None
    ]

    aggregate_node_work_values = [
        float(record.get("aggregate_recorded_node_seconds", 0.0) or 0.0)
        for record in agent_rows
        if record.get("aggregate_recorded_node_seconds") is not None
    ]

    node_latency_summary = summarize_node_latency(agent_rows)

    tool_call_values = [
        float(
            record.get(
                "logical_tool_calls",
                0,
            )
        )
        for record in agent_rows
        if record.get(
            "logical_tool_calls"
        )
        is not None
    ]

    successful_tool_calls = [
        float(
            record.get(
                "logical_tool_calls",
                0,
            )
        )
        for record in judged_rows
        if record[
            "quality"
        ].get(
            "research_success"
        )
    ]

    execution_success_rate = (
        execution_success_count / total
        if total
        else None
    )

    research_success_rate_on_judged = (
        research_success_count
        / judged_count
        if judged_count
        else None
    )

    # Only call it the final Research Success Rate
    # when every record was successfully judged.
    research_success_rate = (
        research_success_count / total
        if total and judged_count == total
        else None
    )

    return {
        "total_tasks": total,
        "execution_success_count": (
            execution_success_count
        ),
        "execution_success_rate": (
            round(
                execution_success_rate,
                4,
            )
            if execution_success_rate is not None
            else None
        ),
        "judged_count": judged_count,
        "unjudged_count": (
            total - judged_count
        ),
        "research_success_count": (
            research_success_count
        ),
        "research_success_rate": (
            round(
                research_success_rate,
                4,
            )
            if research_success_rate is not None
            else None
        ),
        "research_success_rate_on_judged": (
            round(
                research_success_rate_on_judged,
                4,
            )
            if research_success_rate_on_judged
            is not None
            else None
        ),
        "mean_aspect_coverage": safe_mean(
            coverage_values
        ),
        "median_aspect_coverage": safe_median(
            coverage_values
        ),
        "mean_latency_seconds": safe_mean(
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
            tool_call_values
        ),
        "median_logical_tool_calls": safe_median(
            tool_call_values
        ),
        "mean_logical_tool_calls_on_research_success": (
            safe_mean(
                successful_tool_calls
            )
        ),
    }


def summarize_pairwise(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    judgments = [
        row.get(
            "pairwise"
        )
        for row in rows
        if (
            row.get(
                "pairwise"
            )
            and row[
                "pairwise"
            ].get(
                "judged"
            )
        )
    ]

    askflow_wins = sum(
        1
        for item in judgments
        if item.get(
            "winner_agent"
        )
        == "askflow"
    )

    baseline_wins = sum(
        1
        for item in judgments
        if item.get(
            "winner_agent"
        )
        == "baseline"
    )

    ties = sum(
        1
        for item in judgments
        if item.get(
            "winner_agent"
        )
        == "tie"
    )

    total = len(
        judgments
    )

    decisive = (
        askflow_wins
        + baseline_wins
    )

    return {
        "judged_pairs": total,
        "askflow_wins": askflow_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "askflow_win_rate": (
            round(
                askflow_wins / total,
                4,
            )
            if total
            else None
        ),
        "baseline_win_rate": (
            round(
                baseline_wins / total,
                4,
            )
            if total
            else None
        ),
        "tie_rate": (
            round(
                ties / total,
                4,
            )
            if total
            else None
        ),
        "askflow_decisive_win_rate": (
            round(
                askflow_wins / decisive,
                4,
            )
            if decisive
            else None
        ),
    }


def summarize_askflow_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    records = [
        row.get(
            "askflow"
        )
        for row in rows
        if row.get(
            "askflow"
        )
    ]

    if not records:
        return {}

    verification_rounds = [
        float(
            record.get(
                "verification_rounds",
                0,
            )
        )
        for record in records
    ]

    targeted_tasks = [
        record
        for record in records
        if (
            int(
                record.get(
                    "targeted_research_rounds",
                    0,
                )
                or 0
            )
            > 0
        )
    ]

    triggered_gap_count = 0
    repaired_gap_count = 0

    applicable_gap_tasks = 0

    for row in rows:
        repair = row.get(
            "gap_repair"
        )

        if not repair:
            continue

        if not repair.get(
            "applicable"
        ):
            continue

        applicable_gap_tasks += 1

        triggered_gap_count += int(
            repair.get(
                "triggered_gap_count",
                0,
            )
            or 0
        )

        repaired_gap_count += int(
            repair.get(
                "repaired_gap_count",
                0,
            )
            or 0
        )

    repair_rate = (
        repaired_gap_count
        / triggered_gap_count
        if triggered_gap_count
        else None
    )

    return {
        "mean_verification_rounds": safe_mean(
            verification_rounds
        ),
        "median_verification_rounds": safe_median(
            verification_rounds
        ),
        "targeted_research_trigger_count": len(
            targeted_tasks
        ),
        "targeted_research_trigger_rate": round(
            len(targeted_tasks)
            / len(records),
            4,
        ),
        "gap_repair_applicable_tasks": (
            applicable_gap_tasks
        ),
        "triggered_initial_actionable_gaps": (
            triggered_gap_count
        ),
        "repaired_initial_actionable_gaps": (
            repaired_gap_count
        ),
        "evidence_gap_repair_rate": (
            round(
                repair_rate,
                4,
            )
            if repair_rate is not None
            else None
        ),
    }


def summarize_categories(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    categories = sorted(
        {
            row.get(
                "category",
                "unknown",
            )
            for row in rows
        }
    )

    output: dict[
        str,
        Any,
    ] = {}

    for category in categories:
        subset = [
            row
            for row in rows
            if row.get(
                "category",
                "unknown",
            )
            == category
        ]

        output[category] = {
            "askflow": summarize_agent(
                subset,
                "askflow",
            ),
            "baseline": summarize_agent(
                subset,
                "baseline",
            ),
            "pairwise": summarize_pairwise(
                subset
            ),
        }

    return output


# =========================================================
# Main task evaluation
# =========================================================


def build_agent_record(
    raw_record: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    """
    Keep evaluation-relevant runtime metrics without
    duplicating the huge final report into judged JSONL.
    """

    return {
        "execution_success": raw_record.get(
            "execution_success"
        ),
        "latency_seconds": raw_record.get(
            "latency_seconds"
        ),
        "aggregate_recorded_node_seconds": raw_record.get(
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
        "logical_tool_names": raw_record.get(
            "logical_tool_names"
        ),
        "verification_rounds": raw_record.get(
            "verification_rounds",
            0,
        ),
        "targeted_research_rounds": raw_record.get(
            "targeted_research_rounds",
            0,
        ),
        "targeted_research_tasks": raw_record.get(
            "targeted_research_tasks",
            0,
        ),
        "final_evidence_sufficient": raw_record.get(
            "final_evidence_sufficient"
        ),
        "runtime_error": raw_record.get(
            "error"
        ),
        "quality": quality,
    }


async def evaluate_task(
    *,
    task: dict[str, Any],
    askflow_record: dict[str, Any] | None,
    baseline_record: dict[str, Any] | None,
    judge_model: ChatOpenAI,
    structured_method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = task["id"]

    prompt = task.get(
        "prompt",
        "",
    )

    required_aspects = list(
        task.get(
            "required_aspects",
            [],
        )
    )

    category = task.get(
        "category",
        "unknown",
    )

    output: dict[
        str,
        Any,
    ] = {
        "task_id": task_id,
        "category": category,
        "prompt": prompt,
        "required_aspects": required_aspects,
    }

    # -----------------------------------------------------
    # AskFlow individual judgment
    # -----------------------------------------------------

    if askflow_record is not None:
        askflow_quality = await judge_single_report(
            judge_model=judge_model,
            structured_method=structured_method,
            task_id=task_id,
            prompt=prompt,
            required_aspects=required_aspects,
            report=str(
                askflow_record.get(
                    "final_report",
                    "",
                )
            ),
            execution_success=bool(
                askflow_record.get(
                    "execution_success",
                    False,
                )
            ),
            coverage_threshold=args.coverage_threshold,
        )

        output["askflow"] = build_agent_record(
            askflow_record,
            askflow_quality,
        )

    # -----------------------------------------------------
    # Baseline individual judgment
    # -----------------------------------------------------

    if baseline_record is not None:
        baseline_quality = await judge_single_report(
            judge_model=judge_model,
            structured_method=structured_method,
            task_id=task_id,
            prompt=prompt,
            required_aspects=required_aspects,
            report=str(
                baseline_record.get(
                    "final_report",
                    "",
                )
            ),
            execution_success=bool(
                baseline_record.get(
                    "execution_success",
                    False,
                )
            ),
            coverage_threshold=args.coverage_threshold,
        )

        output["baseline"] = build_agent_record(
            baseline_record,
            baseline_quality,
        )

    # -----------------------------------------------------
    # Pairwise
    # -----------------------------------------------------

    if (
        not args.skip_pairwise
        and askflow_record is not None
        and baseline_record is not None
    ):
        output["pairwise"] = await judge_pairwise(
            judge_model=judge_model,
            structured_method=structured_method,
            task_id=task_id,
            prompt=prompt,
            required_aspects=required_aspects,
            baseline_report=str(
                baseline_record.get(
                    "final_report",
                    "",
                )
            ),
            askflow_report=str(
                askflow_record.get(
                    "final_report",
                    "",
                )
            ),
            baseline_execution_success=bool(
                baseline_record.get(
                    "execution_success",
                    False,
                )
            ),
            askflow_execution_success=bool(
                askflow_record.get(
                    "execution_success",
                    False,
                )
            ),
            seed=args.pairwise_seed,
        )

    # -----------------------------------------------------
    # AskFlow Gap Repair
    # -----------------------------------------------------

    if (
        not args.skip_gap_repair
        and askflow_record is not None
    ):
        output[
            "gap_repair"
        ] = await judge_gap_repair(
            judge_model=judge_model,
            structured_method=structured_method,
            record=askflow_record,
            min_information_gain=(
                args.gap_min_information_gain
            ),
            max_tasks_per_round=(
                args.gap_max_tasks_per_round
            ),
            max_concurrent=(
                args.gap_max_concurrent
            ),
        )

    return output


# =========================================================
# Main
# =========================================================


async def async_main() -> None:
    args = parse_args()

    load_dotenv(
        BACKEND_ROOT / ".env"
    )

    judge_model, model_name, structured_method = (
        build_judge_model()
    )

    print(
        "[JUDGE_CONFIG] "
        f"model={model_name!r} | "
        f"structured_method={structured_method!r} | "
        f"coverage_threshold={args.coverage_threshold} | "
        f"pairwise_seed={args.pairwise_seed}"
    )

    task_file = Path(
        args.tasks
    ).resolve()

    tasks_raw = load_json_file(
        task_file
    )

    if not isinstance(
        tasks_raw,
        list,
    ):
        raise ValueError(
            "tasks.json must contain a JSON array."
        )

    tasks_by_id = {
        task["id"]: task
        for task in tasks_raw
    }

    askflow_results = load_jsonl_files(
        args.askflow_results,
        label="askflow",
    )

    baseline_results = load_jsonl_files(
        args.baseline_results,
        label="baseline",
    )

    if args.task_ids:
        selected_ids = list(
            args.task_ids
        )

    else:
        selected_ids = [
            task["id"]
            for task in tasks_raw
            if (
                task["id"]
                in askflow_results
                or task["id"]
                in baseline_results
            )
        ]

    for task_id in selected_ids:
        if task_id not in tasks_by_id:
            raise KeyError(
                f"Task {task_id!r} not found in {task_file}"
            )

    output_path = Path(
        args.output
    ).resolve()

    summary_path = Path(
        args.summary
    ).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_rows: list[
        dict[str, Any]
    ] = []

    completed_ids: set[
        str
    ] = set()

    if args.resume and output_path.exists():
        with output_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                row = json.loads(
                    line
                )

                completed_rows.append(
                    row
                )

                completed_ids.add(
                    row["task_id"]
                )

        print(
            "[JUDGE_RESUME] "
            f"loaded={len(completed_rows)}"
        )

    else:
        output_path.write_text(
            "",
            encoding="utf-8",
        )

    tasks_to_run = [
        task_id
        for task_id in selected_ids
        if task_id not in completed_ids
    ]

    print(
        "[JUDGE] "
        f"tasks={len(tasks_to_run)} | "
        f"task_file={task_file} | "
        f"output={output_path}"
    )

    for index, task_id in enumerate(
        tasks_to_run,
        start=1,
    ):
        task = tasks_by_id[
            task_id
        ]

        askflow_record = askflow_results.get(
            task_id
        )

        baseline_record = baseline_results.get(
            task_id
        )

        print()
        print(
            f"[JUDGE] {index}/{len(tasks_to_run)} | "
            f"task={task_id} | "
            f"category={task.get('category')}"
        )

        row = await evaluate_task(
            task=task,
            askflow_record=askflow_record,
            baseline_record=baseline_record,
            judge_model=judge_model,
            structured_method=structured_method,
            args=args,
        )

        dump_jsonl_record(
            output_path,
            row,
        )

        completed_rows.append(
            row
        )

        askflow_quality = (
            row.get(
                "askflow",
                {},
            )
            .get(
                "quality",
                {},
            )
        )

        baseline_quality = (
            row.get(
                "baseline",
                {},
            )
            .get(
                "quality",
                {},
            )
        )

        pairwise = row.get(
            "pairwise",
            {},
        )

        gap_repair = row.get(
            "gap_repair",
            {},
        )

        print(
            "[JUDGE_RESULT] "
            f"task={task_id} | "
            f"askflow_coverage="
            f"{askflow_quality.get('aspect_coverage')} | "
            f"askflow_success="
            f"{askflow_quality.get('research_success')} | "
            f"baseline_coverage="
            f"{baseline_quality.get('aspect_coverage')} | "
            f"baseline_success="
            f"{baseline_quality.get('research_success')} | "
            f"pairwise="
            f"{pairwise.get('winner_agent')} | "
            f"gap_repair="
            f"{gap_repair.get('repair_rate')}"
        )

        if askflow_quality.get(
            "judge_error"
        ):
            print(
                "[JUDGE_WARNING] "
                "AskFlow quality judge failed: "
                + truncate_for_console(
                    askflow_quality.get(
                        "judge_error"
                    )
                )
            )

        if baseline_quality.get(
            "judge_error"
        ):
            print(
                "[JUDGE_WARNING] "
                "Baseline quality judge failed: "
                + truncate_for_console(
                    baseline_quality.get(
                        "judge_error"
                    )
                )
            )

        if pairwise.get(
            "judge_error"
        ):
            print(
                "[JUDGE_WARNING] "
                "Pairwise judge failed: "
                + truncate_for_console(
                    pairwise.get(
                        "judge_error"
                    )
                )
            )

    # =====================================================
    # Summary
    # =====================================================

    summary = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "protocol": {
            "judge_model": model_name,
            "structured_method": structured_method,
            "coverage_threshold": (
                args.coverage_threshold
            ),
            "pairwise_seed": (
                args.pairwise_seed
            ),
            "gap_min_information_gain": (
                args.gap_min_information_gain
            ),
            "gap_max_tasks_per_round": (
                args.gap_max_tasks_per_round
            ),
            "gap_max_concurrent": (
                args.gap_max_concurrent
            ),
            "task_file": str(
                task_file
            ),
            "askflow_result_files": [
                str(
                    Path(path).resolve()
                )
                for path in args.askflow_results
            ],
            "baseline_result_files": [
                str(
                    Path(path).resolve()
                )
                for path in args.baseline_results
            ],
        },
        "total_tasks": len(
            completed_rows
        ),
        "askflow": summarize_agent(
            completed_rows,
            "askflow",
        ),
        "baseline": summarize_agent(
            completed_rows,
            "baseline",
        ),
        "pairwise": summarize_pairwise(
            completed_rows
        ),
        "askflow_diagnostics": (
            summarize_askflow_diagnostics(
                completed_rows
            )
        ),
        "categories": summarize_categories(
            completed_rows
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "[JUDGE_DONE] "
        f"records={len(completed_rows)} | "
        f"output={output_path} | "
        f"summary={summary_path}"
    )

    print()
    print(
        json.dumps(
            {
                "askflow": (
                    summary[
                        "askflow"
                    ]
                ),
                "baseline": (
                    summary[
                        "baseline"
                    ]
                ),
                "pairwise": (
                    summary[
                        "pairwise"
                    ]
                ),
                "askflow_diagnostics": (
                    summary[
                        "askflow_diagnostics"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(
        async_main()
    )