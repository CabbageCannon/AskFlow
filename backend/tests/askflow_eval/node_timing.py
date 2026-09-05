"""Low-intrusion LangGraph node latency instrumentation for AskFlow Eval."""

from __future__ import annotations

import time
from collections import defaultdict
from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Literal
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

TimingStatus = Literal["success", "error"]
Clock = Callable[[], float]


@dataclass(frozen=True)
class NodeTimingEvent:
    """One inclusive graph-node invocation timing."""

    node: str
    duration_seconds: float
    status: TimingStatus
    run_id: str
    parent_run_id: str | None
    start_offset_seconds: float
    end_offset_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "duration_seconds": round(self.duration_seconds, 6),
            "status": self.status,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "start_offset_seconds": round(self.start_offset_seconds, 6),
            "end_offset_seconds": round(self.end_offset_seconds, 6),
        }


@dataclass(frozen=True)
class _ActiveNodeTiming:
    node: str
    started_at: float
    parent_run_id: str | None


_CURRENT_NODE_TIMING_COLLECTOR: ContextVar["NodeTimingCollector | None"] = ContextVar(
    "askflow_eval_node_timing_collector",
    default=None,
)


def set_current_node_timing_collector(collector: "NodeTimingCollector") -> Token:
    return _CURRENT_NODE_TIMING_COLLECTOR.set(collector)


def reset_current_node_timing_collector(token: Token) -> None:
    _CURRENT_NODE_TIMING_COLLECTOR.reset(token)


def current_node_timing_collector() -> "NodeTimingCollector | None":
    return _CURRENT_NODE_TIMING_COLLECTOR.get()


class NodeTimingCollector:
    """Thread-safe per-Eval-task accumulator for inclusive node timings."""

    def __init__(self, *, clock: Clock = time.perf_counter) -> None:
        self._clock = clock
        self._origin = clock()
        self._lock = Lock()
        self._active: dict[str, _ActiveNodeTiming] = {}
        self._events: list[NodeTimingEvent] = []

    def start_node(
        self,
        *,
        run_id: UUID | str,
        node: str,
        parent_run_id: UUID | str | None = None,
    ) -> bool:
        run_key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else None
        started_at = self._clock()

        with self._lock:
            if run_key in self._active:
                return False
            self._active[run_key] = _ActiveNodeTiming(
                node=str(node),
                started_at=started_at,
                parent_run_id=parent_key,
            )
        return True

    def finish_node(
        self,
        *,
        run_id: UUID | str,
        status: TimingStatus,
    ) -> NodeTimingEvent | None:
        run_key = str(run_id)
        ended_at = self._clock()

        with self._lock:
            active = self._active.pop(run_key, None)
            if active is None:
                return None
            event = NodeTimingEvent(
                node=active.node,
                duration_seconds=max(0.0, ended_at - active.started_at),
                status=status,
                run_id=run_key,
                parent_run_id=active.parent_run_id,
                start_offset_seconds=max(0.0, active.started_at - self._origin),
                end_offset_seconds=max(0.0, ended_at - self._origin),
            )
            self._events.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)

        events.sort(
            key=lambda event: (
                event.start_offset_seconds,
                event.end_offset_seconds,
                event.node,
                event.run_id,
            )
        )

        durations_by_node: dict[str, list[float]] = defaultdict(list)
        errors_by_node: dict[str, int] = defaultdict(int)

        for event in events:
            durations_by_node[event.node].append(event.duration_seconds)
            if event.status == "error":
                errors_by_node[event.node] += 1

        node_latency: dict[str, dict[str, Any]] = {}
        for node in sorted(durations_by_node):
            durations = durations_by_node[node]
            total = sum(durations)
            calls = len(durations)
            node_latency[node] = {
                "calls": calls,
                "total_seconds": round(total, 6),
                "mean_seconds": round(total / calls, 6),
                "max_seconds": round(max(durations), 6),
                "error_calls": errors_by_node[node],
            }

        return {
            "aggregate_recorded_node_seconds": round(
                sum(event.duration_seconds for event in events),
                6,
            ),
            "node_latency": node_latency,
            "node_timings": [event.to_dict() for event in events],
        }


def _resolve_graph_node(
    *,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    callback_name: Any,
) -> str | None:
    metadata = metadata or {}
    node = metadata.get("langgraph_node")
    if not node:
        return None

    if not any(
        isinstance(tag, str) and tag.startswith("graph:step:")
        for tag in (tags or [])
    ):
        return None

    if callback_name is not None and str(callback_name) != str(node):
        return None

    return str(node)


class NodeTimingCallbackHandler(BaseCallbackHandler):
    """Observe real LangGraph node runs without changing graph behavior."""

    raise_error = False
    run_inline = True

    def __init__(self, collector: NodeTimingCollector) -> None:
        super().__init__()
        self._collector = collector

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del serialized, inputs
        try:
            node = _resolve_graph_node(
                tags=tags,
                metadata=metadata,
                callback_name=kwargs.get("name"),
            )
            if node is None:
                return
            self._collector.start_node(
                run_id=run_id,
                node=node,
                parent_run_id=parent_run_id,
            )
        except Exception:
            return

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del outputs, parent_run_id, tags, kwargs
        try:
            self._collector.finish_node(run_id=run_id, status="success")
        except Exception:
            return

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del error, parent_run_id, tags, kwargs
        try:
            self._collector.finish_node(run_id=run_id, status="error")
        except Exception:
            return
