"""Graph state definitions and data structures for the Deep Research agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

# 解析用户问题时希望LLM返回的结构
class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    # 是否需要追问用户
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    # 追问用户的问题
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    # 当模型判断不需要继续追问后，给用户返回的一段确认信息
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

# 一个researcher研究的问题
class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

# 研究可靠性相关输出结构

# 冲突资料的详细解释
class EvidenceConflict(BaseModel):
    topic:str
    description:str
    sources:list[str]=Field(default_factory=list)
    
# 资料可靠度详细解释(一般认为官方文档更可靠、个人博客相对没这么可靠)
class CredibilityIssue(BaseModel):
    source:str
    concern:str
    
class VerificationResult(BaseModel):
    # 是否覆盖了用户问题中的全部维度
    coverage_score:float=Field(ge=0.0,le=1.0)
    # 资料可靠度
    credibility_score:float=Field(ge=0.0,le=1.0)
    
    credibility_issues:list[CredibilityIssue]=Field(default_factory=list)
    
    conflicts:list[EvidenceConflict]=Field(default_factory=list)
    # 某个结论有没有足够的证据支撑
    missing_evidence:list[str]=Field(default_factory=list)
    # 搜索资料是否满意
    evidence_sufficient:bool
    # 总结
    summary:str

###################
# State Definitions
###################

# 处理状态更新
def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
# 总图输入
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

# 总图状态
class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    # MessagesState继承来的messages存储的是用户和系统之间的对话
    
    # Supervisor Agent 自己使用的消息历史，比如工具调用、LLM生成内容等等
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    # 具体研究任务
    research_brief: Optional[str]
    # 各个研究员直接搜回来的原始资料
    raw_notes: Annotated[list[str], override_reducer] = []
    # 处理后的研究材料
    notes: Annotated[list[str], override_reducer] = []
    # 搜索资料检查结果
    verification_result:Optional[VerificationResult]
    # 最终给用户的研究报告
    final_report: str

# supervisor子图状态
class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

# researcher子图状态
class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    # 当前researcher迭代的次数.researcher调用LLM生成一组工具调用list就算做一次iteration
    react_iterations:Annotated[int,operator.add]=0
    # 预算型tool执行了的次数，即便执行失败了也算，并且这里如果execute_tool_safely中执行失败了重试，也不会额外增加，后续这里应该优化
    total_tool_calls:Annotated[int,operator.add]=0
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0 #后续删除当前字段
    # 当前research负责研究的具体主题
    research_topic: str
    # 压缩后的原始材料（这部分会交给supervisor）
    compressed_research: str
    # 搜索得到的原始材料
    raw_notes: Annotated[list[str], override_reducer] = []

# researcher子图输出
class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []