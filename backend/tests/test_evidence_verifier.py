import asyncio
import open_deep_research.deep_researcher as dr
from open_deep_research.state import (
    CredibilityIssue,
    EvidenceConflict,
    VerificationResult,
    EvidenceGap
)
from langchain_core.messages import AIMessage
from langgraph.graph import START, END, StateGraph

class FakeVerificationModel:
    """
    模拟 configurable_model 的 structured-output 调用链。

    evidence_verifier 中实际是：
    configurable_model
        .with_structured_output(...)
        .with_retry(...)
        .with_config(...)
        .ainvoke(...)
    """

    def __init__(self, result):
        self.result = result
        self.structured_schema = None
        self.received_messages = None

    def with_structured_output(
        self,
        schema,
        **kwargs,
    ):
        self.structured_schema = schema
        return self

    def with_retry(self, **kwargs):
        return self

    def with_config(self, config):
        return self

    async def ainvoke(self, messages):
        self.received_messages = messages
        return self.result

class FakeWriterModel:
    def __init__(self):
        self.received_messages = None
        self.received_config = None

    def with_config(self, config):
        self.received_config = config
        return self

    async def ainvoke(self, messages):
        self.received_messages = messages

        return AIMessage(
            content="fake final report"
        )

def patch_config(monkeypatch):
    config = dr.Configuration(
        research_model="fake:model",
        research_model_max_tokens=1000,
        max_structured_output_retries=1,

        # Dynamic Model Router gateway config
        bailian_api_key="fake-bailian-api-key",
        bailian_base_url="https://example.com/v1",
    )

    monkeypatch.setattr(
        dr.Configuration,
        "from_runnable_config",
        classmethod(
            lambda cls, runnable_config=None: config
        ),
    )

    return config


# Test 1 :测试架构正确性
def test_evidence_verifier_sufficient_evidence(monkeypatch):
    expected_verification = dr.VerificationResult(
        coverage_score=0.95,
        credibility_score=0.90,
        credibility_issues=[],
        conflicts=[],
        missing_evidence=[],
        evidence_sufficient=True,
        further_research_likely_to_help=False,
        summary=("The research is well covered and supported " "by credible sources."),
    )

    fake_model = FakeVerificationModel(expected_verification)

    monkeypatch.setattr(
        dr,
        "configurable_model",
        fake_model,
    )

    patch_config(monkeypatch)

    state = {
        "research_brief": (
            "Compare Framework A and Framework B in terms of "
            "architecture and deployment."
        ),
        "notes": [
            (
                "Framework A uses a graph-based architecture "
                "according to its official documentation [1]."
            ),
            (
                "Framework B uses a conversation-oriented architecture "
                "according to its official documentation [2]."
            ),
            ("Both frameworks provide documented deployment " "instructions [1][2]."),
        ],
        "raw_notes": [
            (
                "--- SOURCE 1: Framework A Official Documentation ---\n"
                "URL: https://framework-a.example.com/docs\n"
                "Architecture and deployment documentation."
            ),
            (
                "--- SOURCE 2: Framework B Official Documentation ---\n"
                "URL: https://framework-b.example.com/docs\n"
                "Architecture and deployment documentation."
            ),
        ],
    }

    result = asyncio.run(
        dr.evidence_verifier(
            state,
            {},
        )
    )

    assert set(result.keys()) == {"verification_result","verification_iterations"}

    assert result["verification_result"] == expected_verification
    
    assert result["verification_iterations"] == 1

    assert fake_model.structured_schema is dr.VerificationResult

    prompt = fake_model.received_messages[0].content

    assert state["research_brief"] in prompt

    for note in state["notes"]:
        assert note in prompt

    for raw_note in state["raw_notes"]:
        assert raw_note in prompt

    assert "ONLY the evidence" in prompt
    assert "Missing Evidence" in prompt


# Test 2 :测试证据不足的情况
def test_evidence_verifier_insufficient_evidence(monkeypatch):
    expected_verification = VerificationResult(
        coverage_score=0.35,
        credibility_score=0.40,
        credibility_issues=[
            CredibilityIssue(
                source="https://example-blog.com/framework-a",
                concern=(
                    "The claim is supported only by a secondary personal blog "
                    "rather than primary documentation or direct evidence."
                ),
            )
        ],
        conflicts=[],
        missing_evidence=[
            "No evidence was provided for Framework B architecture.",
            "No evidence was provided for deployment comparison.",
            "No benchmark or primary evidence supports the performance claim.",
        ],
        evidence_gaps=[
            EvidenceGap(
                gap_type="coverage",
                topic="Framework B architecture and deployment evidence",
                reason=(
                    "Important parts of the research brief are missing "
                    "and could realistically be resolved with additional research."
                ),
                importance=0.9,
                expected_information_gain=0.8,
            )
        ],
        evidence_sufficient=False,
        further_research_likely_to_help=True,
        summary=(
            "The research covers only part of the brief and several important "
            "claims lack sufficient supporting evidence."
        ),
    )

    fake_model = FakeVerificationModel(expected_verification)

    monkeypatch.setattr(
        dr,
        "configurable_model",
        fake_model,
    )

    patch_config(monkeypatch)

    state = {
        "research_brief": (
            "Compare Framework A and Framework B in terms of "
            "architecture, deployment, and performance."
        ),
        "notes": [
            (
                "Framework A uses a graph-based architecture. "
                "Framework A also appears to perform better than Framework B."
            ),
        ],
        "raw_notes": [
            (
                "--- SOURCE 1: Personal Framework Review ---\n"
                "URL: https://example-blog.com/framework-a\n"
                "The author says Framework A feels faster in their projects."
            ),
        ],
    }

    result = asyncio.run(
        dr.evidence_verifier(
            state,
            {},
        )
    )

    verification = result["verification_result"]

    assert verification == expected_verification

    assert verification.evidence_sufficient is False

    assert verification.coverage_score < 0.5
    assert verification.credibility_score < 0.5

    assert len(verification.missing_evidence) >= 1

    assert len(verification.credibility_issues) >= 1

    assert verification.conflicts == []

    prompt = fake_model.received_messages[0].content

    assert state["research_brief"] in prompt

    for note in state["notes"]:
        assert note in prompt

    for raw_note in state["raw_notes"]:
        assert raw_note in prompt

# Test 3 :测试多来源矛盾
def test_evidence_verifier_detects_source_conflict(monkeypatch):
    expected_verification = VerificationResult(
        coverage_score=0.90,
        credibility_score=0.88,
        credibility_issues=[],
        conflicts=[
            EvidenceConflict(
                topic="Framework A throughput",
                description=(
                    "Two credible sources report materially different throughput "
                    "results for Framework A under apparently comparable conditions."
                ),
                sources=[
                    "https://benchmark-a.example.com/report",
                    "https://benchmark-b.example.com/report",
                ],
            )
        ],
        missing_evidence=[],
        evidence_gaps=[
            EvidenceGap(
                gap_type="conflict",
                topic="Framework A throughput discrepancy",
                reason=(
                    "Two credible sources report materially different "
                    "throughput values and additional research could "
                    "identify differences in version, methodology, or conditions."
                ),
                importance=0.95,
                expected_information_gain=0.8,
            )
        ],
        evidence_sufficient=False,
        further_research_likely_to_help=True,
        summary=(
            "The research covers the requested topic and uses credible sources, "
            "but a material performance conflict remains unresolved."
        ),
    )

    fake_model = FakeVerificationModel(
        expected_verification
    )

    monkeypatch.setattr(
        dr,
        "configurable_model",
        fake_model,
    )

    patch_config(monkeypatch)

    state = {
        "research_brief": (
            "Evaluate Framework A performance and determine its typical throughput."
        ),
        "notes": [
            (
                "Benchmark Report A states that Framework A reaches "
                "approximately 100 requests per second [1]."
            ),
            (
                "Benchmark Report B states that Framework A reaches "
                "approximately 60 requests per second [2]."
            ),
        ],
        "raw_notes": [
            (
                "--- SOURCE 1: Independent Benchmark Report A ---\n"
                "URL: https://benchmark-a.example.com/report\n"
                "Framework A achieved 100 requests per second."
            ),
            (
                "--- SOURCE 2: Independent Benchmark Report B ---\n"
                "URL: https://benchmark-b.example.com/report\n"
                "Framework A achieved 60 requests per second."
            ),
        ],
    }

    result = asyncio.run(
        dr.evidence_verifier(
            state,
            {},
        )
    )

    verification = result["verification_result"]

    assert verification == expected_verification

    # 覆盖度本身没有明显问题
    assert verification.coverage_score >= 0.8

    # 来源本身也可以是可信的
    assert verification.credibility_score >= 0.8
    assert verification.credibility_issues == []

    # 但仍然存在冲突
    assert len(verification.conflicts) == 1

    conflict = verification.conflicts[0]

    assert conflict.topic == "Framework A throughput"

    assert (
        "https://benchmark-a.example.com/report"
        in conflict.sources
    )

    assert (
        "https://benchmark-b.example.com/report"
        in conflict.sources
    )

    # 关键冲突未解决，因此整体证据仍可判定为不足
    assert verification.evidence_sufficient is False
    
# Test 4 :测试Verifier 的结果有没有真的进入 Writer 的上下文
def test_final_report_receives_verification_result(monkeypatch):
    fake_writer = FakeWriterModel()

    monkeypatch.setattr(
        dr,
        "configurable_model",
        fake_writer,
    )

    patch_config(monkeypatch)

    verification_result = VerificationResult(
        coverage_score=0.9,
        credibility_score=0.7,
        credibility_issues=[],
        conflicts=[
            EvidenceConflict(
                topic="Framework throughput",
                description=(
                    "Source A reports 100 requests per second, "
                    "while Source B reports 60 requests per second."
                ),
                sources=[
                    "https://source-a.example.com",
                    "https://source-b.example.com",
                ],
            )
        ],
        missing_evidence=[
            "The conflict has not been independently resolved."
        ],
        evidence_sufficient=True,
        further_research_likely_to_help=False,
        summary=(
            "The evidence is sufficient for a cautious final answer, "
            "but the throughput conflict must be disclosed."
        ),
    )

    state = {
        "messages": [],
        "research_brief": (
            "Determine the typical throughput of Framework A."
        ),
        "notes": [
            (
                "Source A reports Framework A reaches "
                "100 requests per second."
            ),
            (
                "Source B reports Framework A reaches "
                "60 requests per second."
            ),
        ],
        "verification_result": verification_result,
    }

    result = asyncio.run(
        dr.final_report_generation(
            state,
            {},
        )
    )

    assert result["final_report"] == "fake final report"

    prompt = fake_writer.received_messages[0].content

    # Writer 必须收到原始研究结果
    for note in state["notes"]:
        assert note in prompt

    # Writer 必须收到 Verifier 的质量审查结果
    assert "Framework throughput" in prompt
    assert "100 requests per second" in prompt
    assert "60 requests per second" in prompt

    assert (
        "The conflict has not been independently resolved."
        in prompt
    )

    assert (
        "The evidence is sufficient for a cautious final answer"
        in prompt
    )

    # Writer Prompt 中必须存在 Verification 使用规则
    assert "Evidence Verification" in prompt
    assert "weak or credibility concerns" in prompt
    assert "materially conflict" in prompt
    assert "do not invent missing evidence" in prompt.lower()
    
# Test 5 :测试完整Graph链路
def test_graph_routes_research_through_verifier_before_writer(
    monkeypatch,
):
    expected_verification = VerificationResult(
        coverage_score=0.9,
        credibility_score=0.9,
        credibility_issues=[],
        conflicts=[],
        missing_evidence=[],
        evidence_sufficient=True,
        further_research_likely_to_help=False,
        summary="Evidence is sufficient.",
    )

    fake_model = FakeVerificationModel(
        expected_verification
    )

    monkeypatch.setattr(
        dr,
        "configurable_model",
        fake_model,
    )

    patch_config(monkeypatch)

    execution_order = []

    async def fake_research(state, config):
        execution_order.append("research")

        return {
            "research_brief": (
                "Explain Framework A architecture."
            ),
            "notes": [
                (
                    "Framework A uses a graph architecture "
                    "according to official documentation."
                )
            ],
            "raw_notes": [
                (
                    "--- SOURCE 1: Official Documentation ---\n"
                    "URL: https://framework-a.example.com/docs\n"
                    "Framework A uses explicit graph-based workflows."
                )
            ],
        }

    async def tracked_verifier(state, config):
        execution_order.append("verifier")

        return await dr.evidence_verifier(
            state,
            config,
        )

    async def fake_writer(state, config):
        execution_order.append("writer")

        assert (
            state["verification_result"]
            == expected_verification
        )

        return {
            "final_report": "fake report",
        }

    builder = StateGraph(
        dr.AgentState
    )

    builder.add_node(
        "research",
        fake_research,
    )

    builder.add_node(
        "evidence_verifier",
        tracked_verifier,
    )

    builder.add_node(
        "writer",
        fake_writer,
    )

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
        "writer",
    )

    builder.add_edge(
        "writer",
        END,
    )

    graph = builder.compile()

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [],
            },
            {},
        )
    )

    assert execution_order == [
        "research",
        "verifier",
        "writer",
    ]

    assert (
        result["verification_result"]
        == expected_verification
    )

    assert (
        result["final_report"]
        == "fake report"
    )