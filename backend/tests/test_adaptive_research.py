import asyncio

import open_deep_research.deep_researcher as dr

from langgraph.graph import START, END, StateGraph

from open_deep_research.state import (
    AgentState,
    EvidenceGap,
    TargetedResearchTask,
    VerificationResult,
)
import pytest
from pydantic import ValidationError

def patch_adaptive_config(monkeypatch, **overrides):
    config_values = {
        "research_model": "fake:model",
        "max_verification_iterations": 3,
        "min_expected_information_gain": 0.30,
        "max_targeted_research_tasks_per_round": 3,
        "max_concurrent_research_units": 3,
    }

    config_values.update(overrides)

    config = dr.Configuration(
        **config_values
    )

    monkeypatch.setattr(
        dr.Configuration,
        "from_runnable_config",
        classmethod(
            lambda cls, runnable_config=None: config
        ),
    )

    return config


def make_verification_result(
    *,
    sufficient=False,
    gaps=None,
    further_research_likely_to_help=None,
):
    gaps = gaps or []

    if further_research_likely_to_help is None:
        further_research_likely_to_help = (
            not sufficient
            and bool(gaps)
        )

    return VerificationResult(
        coverage_score=0.7,
        credibility_score=0.8,
        credibility_issues=[],
        conflicts=[],
        missing_evidence=[],
        evidence_gaps=gaps,
        evidence_sufficient=sufficient,
        further_research_likely_to_help=(
            further_research_likely_to_help
        ),
        summary="Fake verification result.",
    )


def make_gap(
    topic,
    importance,
    gain,
    gap_type="coverage",
):
    return EvidenceGap(
        gap_type=gap_type,
        topic=topic,
        reason=f"Evidence for {topic} is insufficient.",
        importance=importance,
        expected_information_gain=gain,
    )


# ---------------------------------------------------------
# 1. actionable gap：过滤 + 排序 + 数量硬限制
# ---------------------------------------------------------

def test_actionable_gaps_are_filtered_sorted_and_limited(
    monkeypatch,
):
    patch_adaptive_config(
        monkeypatch,
        min_expected_information_gain=0.30,
        max_targeted_research_tasks_per_round=3,
        max_concurrent_research_units=2,
    )

    gaps = [
        make_gap(
            "Low-value gap",
            importance=0.9,
            gain=0.2,
        ),
        make_gap(
            "Pricing",
            importance=0.95,
            gain=0.90,
        ),
        make_gap(
            "Rate limits",
            importance=0.80,
            gain=0.80,
        ),
        make_gap(
            "Benchmark",
            importance=0.90,
            gain=0.50,
        ),
    ]

    state = {
        "verification_result":
            make_verification_result(
                gaps=gaps
            )
    }

    result = dr.get_actionable_evidence_gaps(
        state,
        {},
    )

    # Low-value gap 因 gain < 0.3 被过滤
    assert all(
        gap.topic != "Low-value gap"
        for gap in result
    )

    # min(3 targeted limit, 2 concurrent limit) = 2
    assert len(result) == 2

    # priority = importance * gain
    assert result[0].topic == "Pricing"
    assert result[1].topic == "Rate limits"


# ---------------------------------------------------------
# 2. Router：达到最大 verification 次数必须停止
# ---------------------------------------------------------

def test_router_stops_at_verification_hard_limit(
    monkeypatch,
):
    patch_adaptive_config(
        monkeypatch,
        max_verification_iterations=3,
    )

    state = {
        "verification_iterations": 3,
        "verification_result":
            make_verification_result(
                sufficient=False,
                gaps=[
                    make_gap(
                        "Pricing",
                        1.0,
                        1.0,
                    )
                ],
            ),
    }

    result = asyncio.run(
        dr.route_after_verification(
            state,
            {},
        )
    )

    assert (
        result.goto
        == "final_report_generation"
    )


# ---------------------------------------------------------
# 3. Router：证据充分 -> Final
# ---------------------------------------------------------

def test_router_stops_when_evidence_is_sufficient(
    monkeypatch,
):
    patch_adaptive_config(monkeypatch)

    state = {
        "verification_iterations": 1,
        "verification_result":
            make_verification_result(
                sufficient=True,
            ),
    }

    result = asyncio.run(
        dr.route_after_verification(
            state,
            {},
        )
    )

    assert (
        result.goto
        == "final_report_generation"
    )


# ---------------------------------------------------------
# 4. Router：存在 gap，但 expected gain 太低 -> Final
# ---------------------------------------------------------

def test_router_stops_when_no_gap_has_enough_information_gain(
    monkeypatch,
):
    patch_adaptive_config(
        monkeypatch,
        min_expected_information_gain=0.30,
    )

    state = {
        "verification_iterations": 1,
        "verification_result":
            make_verification_result(
                sufficient=False,
                gaps=[
                    make_gap(
                        "Hard-to-find private data",
                        importance=0.95,
                        gain=0.10,
                    )
                ],
            ),
    }

    result = asyncio.run(
        dr.route_after_verification(
            state,
            {},
        )
    )

    assert (
        result.goto
        == "final_report_generation"
    )


# ---------------------------------------------------------
# 5. Router：有 actionable gap -> Planner
# ---------------------------------------------------------

def test_router_continues_when_actionable_gap_exists(
    monkeypatch,
):
    patch_adaptive_config(monkeypatch)

    state = {
        "verification_iterations": 1,
        "verification_result":
            make_verification_result(
                sufficient=False,
                gaps=[
                    make_gap(
                        "DeepSeek API pricing",
                        importance=0.95,
                        gain=0.90,
                    )
                ],
            ),
    }

    result = asyncio.run(
        dr.route_after_verification(
            state,
            {},
        )
    )

    assert (
        result.goto
        == "plan_targeted_research"
    )


# ---------------------------------------------------------
# 6. Planner：EvidenceGap -> TargetedResearchTask
# ---------------------------------------------------------

def test_planner_generates_narrow_targeted_task(
    monkeypatch,
):
    patch_adaptive_config(monkeypatch)

    gap = make_gap(
        "Current DeepSeek API pricing",
        importance=0.95,
        gain=0.90,
    )

    state = {
        "research_brief": (
            "Compare DeepSeek API capabilities, "
            "context window, rate limits and pricing."
        ),
        "verification_result":
            make_verification_result(
                sufficient=False,
                gaps=[gap],
            ),
    }

    result = asyncio.run(
        dr.plan_targeted_research(
            state,
            {},
        )
    )

    tasks = result[
        "targeted_research_tasks"
    ]

    assert len(tasks) == 1

    task = tasks[0]

    assert (
        task.gap_topic
        == "Current DeepSeek API pricing"
    )

    assert task.gap_type == "coverage"

    assert (
        task.priority
        == 0.95 * 0.90
    )

    # Planner 应明确限制 scope
    assert (
        "Research ONLY"
        in task.research_topic
    )

    assert (
        "Current DeepSeek API pricing"
        in task.research_topic
    )

    assert (
        "Do not restart research"
        in task.research_topic
    )


# ---------------------------------------------------------
# 7. duplicate filter
# ---------------------------------------------------------

def test_filter_new_evidence_removes_exact_and_whitespace_duplicates():
    existing = [
        "DeepSeek API pricing is documented.",
    ]

    candidates = [
        "DeepSeek   API   pricing is documented.",
        "A new official pricing source was found.",
        "",
        "   ",
    ]

    result = dr.filter_new_evidence(
        existing,
        candidates,
    )

    assert result == [
        "A new official pricing source was found."
    ]


# ---------------------------------------------------------
# Fake Researcher Subgraph
# ---------------------------------------------------------

class FakeResearcherSubgraph:
    def __init__(self, results):
        self.results = results
        self.inputs = []
        self.call_index = 0

    async def ainvoke(
        self,
        state,
        config,
    ):
        self.inputs.append(state)

        result = self.results[
            self.call_index
        ]

        self.call_index += 1

        if isinstance(result, Exception):
            raise result

        return result


def make_task(
    topic="DeepSeek API pricing",
):
    return TargetedResearchTask(
        gap_type="coverage",
        gap_topic=topic,
        research_topic=(
            f"Research ONLY {topic}. "
            "Do not restart the full research."
        ),
        priority=0.9,
    )


# ---------------------------------------------------------
# 8. Targeted Research：拿到新 evidence -> 再次 Verifier
# ---------------------------------------------------------

def test_targeted_research_returns_new_evidence_to_verifier(
    monkeypatch,
):
    fake_subgraph = FakeResearcherSubgraph(
        [
            {
                "compressed_research":
                    "New pricing evidence.",
                "raw_notes": [
                    "Official pricing raw evidence."
                ],
            }
        ]
    )

    monkeypatch.setattr(
        dr,
        "researcher_subgraph",
        fake_subgraph,
    )

    state = {
        "notes": [
            "Existing capability evidence."
        ],
        "raw_notes": [
            "Existing raw evidence."
        ],
        "targeted_research_tasks": [
            make_task()
        ],
    }

    result = asyncio.run(
        dr.targeted_research(
            state,
            {},
        )
    )

    assert (
        result.goto
        == "evidence_verifier"
    )

    # 注意：这里只应该返回 NEW evidence
    assert result.update["notes"] == [
        "New pricing evidence."
    ]

    assert result.update["raw_notes"] == [
        "Official pricing raw evidence."
    ]

    assert (
        result.update[
            "targeted_research_tasks"
        ]
        == []
    )

    # 确认 Researcher 收到的是 targeted task
    researcher_input = (
        fake_subgraph.inputs[0]
    )

    assert (
        "Research ONLY"
        in researcher_input[
            "research_topic"
        ]
    )


# ---------------------------------------------------------
# 9. Targeted Research：没有任何新 evidence -> Final
# ---------------------------------------------------------

def test_targeted_research_stops_when_no_new_evidence(
    monkeypatch,
):
    fake_subgraph = FakeResearcherSubgraph(
        [
            {
                "compressed_research":
                    "Existing evidence.",
                "raw_notes": [
                    "Existing raw evidence."
                ],
            }
        ]
    )

    monkeypatch.setattr(
        dr,
        "researcher_subgraph",
        fake_subgraph,
    )

    state = {
        "notes": [
            "Existing evidence."
        ],
        "raw_notes": [
            "Existing raw evidence."
        ],
        "targeted_research_tasks": [
            make_task()
        ],
    }

    result = asyncio.run(
        dr.targeted_research(
            state,
            {},
        )
    )

    assert (
        result.goto
        == "final_report_generation"
    )

    assert (
        result.update[
            "targeted_research_tasks"
        ]
        == []
    )


# ---------------------------------------------------------
# 10. 一个 Targeted Researcher 失败，
#     另一个成功时不能丢掉成功结果
# ---------------------------------------------------------

def test_targeted_research_keeps_partial_success(
    monkeypatch,
):
    fake_subgraph = FakeResearcherSubgraph(
        [
            RuntimeError(
                "fake researcher failure"
            ),
            {
                "compressed_research":
                    "Useful rate-limit evidence.",
                "raw_notes": [
                    "Official rate-limit source."
                ],
            },
        ]
    )

    monkeypatch.setattr(
        dr,
        "researcher_subgraph",
        fake_subgraph,
    )

    state = {
        "notes": [],
        "raw_notes": [],
        "targeted_research_tasks": [
            make_task(
                "DeepSeek pricing"
            ),
            make_task(
                "DeepSeek rate limits"
            ),
        ],
    }

    result = asyncio.run(
        dr.targeted_research(
            state,
            {},
        )
    )

    assert (
        result.goto
        == "evidence_verifier"
    )

    assert result.update["notes"] == [
        "Useful rate-limit evidence."
    ]

    assert result.update["raw_notes"] == [
        "Official rate-limit source."
    ]


# ---------------------------------------------------------
# 11. 验证 notes reducer：Targeted Research 返回 new notes
#     后，主 AgentState 应自动 old + new
# ---------------------------------------------------------

def test_agent_state_appends_new_research_evidence():
    def add_new_evidence(state):
        return {
            "notes": [
                "New pricing evidence."
            ],
            "raw_notes": [
                "New pricing raw source."
            ],
        }

    builder = StateGraph(
        AgentState
    )

    builder.add_node(
        "add_new_evidence",
        add_new_evidence,
    )

    builder.add_edge(
        START,
        "add_new_evidence",
    )

    builder.add_edge(
        "add_new_evidence",
        END,
    )

    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [],
            "notes": [
                "Existing architecture evidence."
            ],
            "raw_notes": [
                "Existing architecture source."
            ],
        }
    )

    assert result["notes"] == [
        "Existing architecture evidence.",
        "New pricing evidence.",
    ]

    assert result["raw_notes"] == [
        "Existing architecture source.",
        "New pricing raw source.",
    ]
    
def test_full_adaptive_research_loop(
    monkeypatch,
):
    patch_adaptive_config(
        monkeypatch,
        max_verification_iterations=3,
        min_expected_information_gain=0.30,
    )

    execution_order = []

    # -----------------------------------------
    # 1. 初始 Research
    # -----------------------------------------

    async def fake_initial_research(
        state,
        config,
    ):
        execution_order.append(
            "research"
        )

        return {
            "research_brief": (
                "Compare DeepSeek API capabilities, "
                "context window, rate limits and pricing."
            ),
            "notes": [
                (
                    "DeepSeek API supports the requested "
                    "model capabilities."
                )
            ],
            "raw_notes": [
                (
                    "Official documentation describing "
                    "DeepSeek API capabilities."
                )
            ],
        }

    # -----------------------------------------
    # 2. Fake Verifier
    #
    # 第一次：
    # pricing 缺失 -> insufficient
    #
    # 第二次：
    # pricing 已经加入 -> sufficient
    # -----------------------------------------

    async def fake_verifier(
        state,
        config,
    ):
        verification_number = (
            state.get(
                "verification_iterations",
                0,
            )
            + 1
        )

        execution_order.append(
            f"verifier_{verification_number}"
        )

        notes = state.get(
            "notes",
            [],
        )

        has_pricing = any(
            "pricing" in note.lower()
            for note in notes
        )

        if not has_pricing:
            result = VerificationResult(
                coverage_score=0.70,
                credibility_score=0.90,
                credibility_issues=[],
                conflicts=[],
                missing_evidence=[
                    "Current API pricing is missing."
                ],
                evidence_gaps=[
                    EvidenceGap(
                        gap_type="coverage",
                        topic="Current DeepSeek API pricing",
                        reason=(
                            "Pricing is required by the "
                            "research brief but is not "
                            "covered by current evidence."
                        ),
                        importance=0.95,
                        expected_information_gain=0.90,
                    )
                ],
                evidence_sufficient=False,

                # 新增
                further_research_likely_to_help=True,

                summary=(
                    "Evidence is strong but pricing "
                    "is still missing."
                ),
            )

        else:
            result = VerificationResult(
                coverage_score=0.95,
                credibility_score=0.92,
                credibility_issues=[],
                conflicts=[],
                missing_evidence=[],
                evidence_gaps=[],
                evidence_sufficient=True,

                # 新增
                further_research_likely_to_help=False,

                summary=(
                    "The evidence now sufficiently "
                    "covers the research brief."
                ),
            )

        return {
            "verification_result": result,
            "verification_iterations": 1,
        }

    # -----------------------------------------
    # 3. 给 Planner 加执行记录
    # -----------------------------------------

    async def tracked_planner(
        state,
        config,
    ):
        execution_order.append(
            "planner"
        )

        return await (
            dr.plan_targeted_research(
                state,
                config,
            )
        )

    # -----------------------------------------
    # 4. Fake Targeted Research
    #
    # 模拟只补 pricing
    # -----------------------------------------

    async def fake_targeted_research(
        state,
        config,
    ):
        execution_order.append(
            "targeted_research"
        )

        tasks = state.get(
            "targeted_research_tasks",
            [],
        )

        assert len(tasks) == 1

        assert (
            tasks[0].gap_topic
            == "Current DeepSeek API pricing"
        )

        assert (
            "Research ONLY"
            in tasks[0].research_topic
        )

        return dr.Command(
            goto="evidence_verifier",
            update={
                "notes": [
                    (
                        "Current DeepSeek API pricing "
                        "is documented by the official "
                        "pricing source."
                    )
                ],
                "raw_notes": [
                    (
                        "Official DeepSeek pricing "
                        "documentation."
                    )
                ],
                "targeted_research_tasks": [],
            },
        )

    # -----------------------------------------
    # 5. Fake Writer
    # -----------------------------------------

    async def fake_writer(
        state,
        config,
    ):
        execution_order.append(
            "writer"
        )

        assert (
            state[
                "verification_result"
            ].evidence_sufficient
            is True
        )

        # 必须同时存在初始 evidence + 新 pricing evidence
        notes = state.get(
            "notes",
            [],
        )

        assert any(
            "capabilities" in note.lower()
            for note in notes
        )

        assert any(
            "pricing" in note.lower()
            for note in notes
        )

        return {
            "final_report":
                "fake adaptive final report"
        }

    # -----------------------------------------
    # 6. 构建一个最小 Adaptive Graph
    # -----------------------------------------

    builder = StateGraph(
        AgentState
    )

    builder.add_node(
        "research",
        fake_initial_research,
    )

    builder.add_node(
        "evidence_verifier",
        fake_verifier,
    )

    builder.add_node(
        "route_after_verification",
        dr.route_after_verification,
    )

    builder.add_node(
        "plan_targeted_research",
        tracked_planner,
    )

    builder.add_node(
        "targeted_research",
        fake_targeted_research,
    )

    builder.add_node(
        "final_report_generation",
        fake_writer,
    )

    # -----------------------------------------
    # 7. Edges
    # -----------------------------------------

    builder.add_edge(
        START,
        "research",
    )

    builder.add_edge(
        "research",
        "evidence_verifier",
    )

    builder.add_edge(
        "evidence_verifier",
        "route_after_verification",
    )

    builder.add_edge(
        "plan_targeted_research",
        "targeted_research",
    )

    builder.add_edge(
        "final_report_generation",
        END,
    )

    graph = builder.compile()

    # -----------------------------------------
    # 8. Run
    # -----------------------------------------

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [],
            },
            {},
        )
    )

    # -----------------------------------------
    # 9. 最核心断言
    # -----------------------------------------

    assert execution_order == [
        "research",
        "verifier_1",
        "planner",
        "targeted_research",
        "verifier_2",
        "writer",
    ]

    assert (
        result["verification_iterations"]
        == 2
    )

    assert (
        result["verification_result"]
        .evidence_sufficient
        is True
    )

    assert (
        result["final_report"]
        == "fake adaptive final report"
    )
    
def test_verification_rejects_missing_gap_when_followup_is_useful():
    with pytest.raises(ValidationError):
        VerificationResult(
            coverage_score=0.15,
            credibility_score=0.10,
            credibility_issues=[],
            conflicts=[],
            missing_evidence=[
                "No official source evidence was found."
            ],
            evidence_gaps=[],
            evidence_sufficient=False,
            further_research_likely_to_help=True,
            summary="Evidence is insufficient.",
        )
        
def test_router_stops_when_further_research_is_unlikely(
    monkeypatch,
):
    patch_adaptive_config(monkeypatch)

    state = {
        "verification_iterations": 1,
        "verification_result":
            make_verification_result(
                sufficient=False,
                gaps=[],
                further_research_likely_to_help=False,
            ),
    }

    result = asyncio.run(
        dr.route_after_verification(
            state,
            {},
        )
    )

    assert (
        result.goto
        == "final_report_generation"
    )