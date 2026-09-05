"""Controlled experiment group definitions for AskFlow evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


DEFAULT_STATIC_MODEL = "openai:qwen3.5-plus"


@dataclass(frozen=True)
class ExperimentGroup:
    """One controlled benchmark arm.

    A and B deliberately share the exact same static model so differences
    between them isolate AskFlow's workflow-level enhancements. C keeps the
    AskFlow workflow but enables dynamic model routing.
    """

    code: Literal["A", "B", "C"]
    agent: Literal["baseline", "askflow"]
    router_mode: Literal["unavailable", "static_override", "dynamic"]
    static_model: str | None
    description: str


GROUPS: dict[str, ExperimentGroup] = {
    "A": ExperimentGroup(
        code="A",
        agent="baseline",
        router_mode="unavailable",
        static_model=DEFAULT_STATIC_MODEL,
        description="Baseline workflow + one static model",
    ),
    "B": ExperimentGroup(
        code="B",
        agent="askflow",
        router_mode="static_override",
        static_model=DEFAULT_STATIC_MODEL,
        description="AskFlow workflow + the same static model as Group A",
    ),
    "C": ExperimentGroup(
        code="C",
        agent="askflow",
        router_mode="dynamic",
        static_model=None,
        description="AskFlow workflow + dynamic model router",
    ),
}


def resolve_experiment_group(
    code: str,
    *,
    static_model: str | None = None,
) -> ExperimentGroup:
    normalized = code.upper()

    if normalized not in GROUPS:
        raise ValueError(
            f"Unknown experiment group {code!r}; expected one of A, B, C"
        )

    group = GROUPS[normalized]

    if group.code in {"A", "B"} and static_model:
        return replace(group, static_model=static_model)

    return group
