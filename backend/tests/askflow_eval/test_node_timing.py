from __future__ import annotations

import asyncio
import json
from typing import TypedDict
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph

try:
    from .node_timing import (
        NodeTimingCallbackHandler,
        NodeTimingCollector,
        current_node_timing_collector,
        reset_current_node_timing_collector,
        set_current_node_timing_collector,
    )
except ImportError:
    from node_timing import (
        NodeTimingCallbackHandler,
        NodeTimingCollector,
        current_node_timing_collector,
        reset_current_node_timing_collector,
        set_current_node_timing_collector,
    )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def start_callback(handler, *, node: str, run_id=None, parent_run_id=None):
    run_id = run_id or uuid4()
    handler.on_chain_start(
        {},
        {},
        run_id=run_id,
        parent_run_id=parent_run_id,
        tags=["graph:step:1"],
        metadata={"langgraph_node": node},
        name=node,
    )
    return run_id


def test_async_node_records_latency_and_preserves_return_value():
    clock = FakeClock()
    collector = NodeTimingCollector(clock=clock)
    handler = NodeTimingCallbackHandler(collector)

    async def node():
        clock.advance(1.25)
        return {"answer": 42}

    async def run():
        run_id = start_callback(handler, node="researcher")
        result = await node()
        handler.on_chain_end(result, run_id=run_id)
        return result

    result = asyncio.run(run())
    snapshot = collector.snapshot()

    assert result == {"answer": 42}
    assert snapshot["node_latency"]["researcher"]["calls"] == 1
    assert snapshot["node_latency"]["researcher"]["total_seconds"] == pytest.approx(1.25)


def test_same_node_twice_aggregates_calls_total_mean_and_max():
    clock = FakeClock()
    collector = NodeTimingCollector(clock=clock)
    handler = NodeTimingCallbackHandler(collector)

    first = start_callback(handler, node="researcher")
    clock.advance(1.0)
    handler.on_chain_end({}, run_id=first)

    second = start_callback(handler, node="researcher")
    clock.advance(3.0)
    handler.on_chain_end({}, run_id=second)

    item = collector.snapshot()["node_latency"]["researcher"]
    assert item["calls"] == 2
    assert item["total_seconds"] == pytest.approx(4.0)
    assert item["mean_seconds"] == pytest.approx(2.0)
    assert item["max_seconds"] == pytest.approx(3.0)


def test_different_nodes_are_aggregated_separately():
    clock = FakeClock()
    collector = NodeTimingCollector(clock=clock)
    handler = NodeTimingCallbackHandler(collector)

    first = start_callback(handler, node="supervisor")
    clock.advance(2.0)
    handler.on_chain_end({}, run_id=first)

    second = start_callback(handler, node="evidence_verifier")
    clock.advance(5.0)
    handler.on_chain_end({}, run_id=second)

    snapshot = collector.snapshot()
    assert set(snapshot["node_latency"]) == {"supervisor", "evidence_verifier"}
    assert snapshot["node_latency"]["supervisor"]["total_seconds"] == pytest.approx(2.0)
    assert snapshot["node_latency"]["evidence_verifier"]["total_seconds"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_concurrent_same_node_invocations_do_not_overwrite_each_other():
    collector = NodeTimingCollector()
    handler = NodeTimingCallbackHandler(collector)
    started = asyncio.Event()
    release = asyncio.Event()
    entered = 0
    entered_lock = asyncio.Lock()

    async def worker():
        nonlocal entered
        run_id = start_callback(handler, node="researcher")
        async with entered_lock:
            entered += 1
            if entered == 2:
                started.set()
        await release.wait()
        handler.on_chain_end({}, run_id=run_id)

    first = asyncio.create_task(worker())
    second = asyncio.create_task(worker())
    await started.wait()
    release.set()
    await asyncio.gather(first, second)

    snapshot = collector.snapshot()
    assert snapshot["node_latency"]["researcher"]["calls"] == 2
    assert len(snapshot["node_timings"]) == 2
    assert len({item["run_id"] for item in snapshot["node_timings"]}) == 2


@pytest.mark.asyncio
async def test_different_eval_tasks_do_not_share_contextvar_collectors():
    async def worker(label: str):
        collector = NodeTimingCollector()
        token = set_current_node_timing_collector(collector)
        try:
            await asyncio.sleep(0)
            assert current_node_timing_collector() is collector
            collector.start_node(run_id=label, node=label)
            collector.finish_node(run_id=label, status="success")
            return collector.snapshot()
        finally:
            reset_current_node_timing_collector(token)

    first, second = await asyncio.gather(worker("task-a"), worker("task-b"))
    assert set(first["node_latency"]) == {"task-a"}
    assert set(second["node_latency"]) == {"task-b"}
    assert current_node_timing_collector() is None


@pytest.mark.asyncio
async def test_exception_records_duration_and_original_exception_is_reraised():
    clock = FakeClock()
    collector = NodeTimingCollector(clock=clock)
    handler = NodeTimingCallbackHandler(collector)

    class ExpectedError(RuntimeError):
        pass

    async def failing_node():
        clock.advance(2.5)
        raise ExpectedError("boom")

    run_id = start_callback(handler, node="researcher")

    with pytest.raises(ExpectedError, match="boom"):
        try:
            await failing_node()
        except Exception as exc:
            handler.on_chain_error(exc, run_id=run_id)
            raise

    snapshot = collector.snapshot()
    item = snapshot["node_latency"]["researcher"]
    assert item["calls"] == 1
    assert item["error_calls"] == 1
    assert item["total_seconds"] == pytest.approx(2.5)
    assert snapshot["node_timings"][0]["status"] == "error"


def test_snapshot_is_json_serializable():
    clock = FakeClock()
    collector = NodeTimingCollector(clock=clock)
    collector.start_node(run_id="run-1", node="compress_research")
    clock.advance(0.75)
    collector.finish_node(run_id="run-1", status="success")

    decoded = json.loads(json.dumps(collector.snapshot(), ensure_ascii=False))
    assert decoded["node_latency"]["compress_research"]["calls"] == 1
    assert decoded["node_timings"][0]["duration_seconds"] == pytest.approx(0.75)


def test_child_chain_with_inherited_metadata_is_not_double_counted():
    clock = FakeClock()
    collector = NodeTimingCollector(clock=clock)
    handler = NodeTimingCallbackHandler(collector)

    real_run = uuid4()
    handler.on_chain_start(
        {}, {}, run_id=real_run,
        tags=["graph:step:2"],
        metadata={"langgraph_node": "researcher"},
        name="researcher",
    )

    child_run = uuid4()
    handler.on_chain_start(
        {}, {}, run_id=child_run, parent_run_id=real_run,
        tags=["graph:step:2"],
        metadata={"langgraph_node": "researcher"},
        name="RunnableSequence",
    )

    clock.advance(1.0)
    handler.on_chain_end({}, run_id=child_run)
    clock.advance(1.0)
    handler.on_chain_end({}, run_id=real_run)

    snapshot = collector.snapshot()
    assert snapshot["node_latency"]["researcher"]["calls"] == 1
    assert snapshot["node_latency"]["researcher"]["total_seconds"] == pytest.approx(2.0)


class MiniState(TypedDict):
    value: int


@pytest.mark.asyncio
async def test_real_stategraph_records_each_graph_node_once():
    async def node_a(state: MiniState):
        return {"value": state["value"] + 1}

    async def node_b(state: MiniState):
        return {"value": state["value"] + 1}

    builder = StateGraph(MiniState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)

    collector = NodeTimingCollector()
    handler = NodeTimingCallbackHandler(collector)
    graph = builder.compile()
    result = await graph.ainvoke({"value": 0}, {"callbacks": [handler]})
    snapshot = collector.snapshot()

    assert result["value"] == 2
    assert snapshot["node_latency"]["node_a"]["calls"] == 1
    assert snapshot["node_latency"]["node_b"]["calls"] == 1
