
from datetime import datetime
from typing import Any
from sqlalchemy import String, Integer, CLOB, JSON, DateTime, Numeric, func, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import BaseEntity, VectorField
from agent.common import SkillExecutionContext


class UserProfileEntity(BaseEntity):
    """
    Stores long-term user characteristics and technical preferences across sessions.
    """
    __tablename__ = "kbot_md_user_profile"

    user_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment="Unique identifier for the user")
    global_preferences: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Persistent technical stack preferences")
    frequent_entities: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Frequently occurring entities (e.g., project names, IPs)")
    entity_relations: Mapped[list[dict]] = mapped_column(JSON, nullable=True, comment="Frequently occurring entity relations (e.g., project -> IP)")
    correction_history: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="User correction history")
    profile_summary: Mapped[str | None] = mapped_column(CLOB, comment="LLM-generated user behavior summary")
    last_update_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConversationContextEntity(BaseEntity):
    """
    Manages active session states and rolling conversation summaries.
    """
    __tablename__ = "kbot_md_conv_context"

    session_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment="Unique session UUID")
    user_id: Mapped[str] = mapped_column(String(256), nullable=False)
    session_title: Mapped[str | None] = mapped_column(String(256), comment="Session title")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), comment="Associated application ID")
    agent_id: Mapped[int] = mapped_column(Numeric(38, 0), comment="Associated AI Agent identifier")

    session_state: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Current active state machine parameters")
    current_plan: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Current active plan")
    step_outputs: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Skill step outputs")

    context_summary: Mapped[str | None] = mapped_column(CLOB, comment="Short-to-medium term rolling summary")
    last_relevance_score: Mapped[float | None] = mapped_column(Numeric(2, 1), comment="Last relevance score computed by the model")
    active_topic: Mapped[str | None] = mapped_column(String(512), comment="Current active topic")

    interaction_count: Mapped[int] = mapped_column(Integer, default=0, comment="Total turns in this session")
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, comment="Whether the session is deleted")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MemoryEntryEntity(BaseEntity):
    """
    Records atomic Q&A interactions with semantic vectors and state snapshots.
    """
    __tablename__ = "kbot_md_memory_entry"

    entry_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment="Unique identifier for the entry")
    session_id: Mapped[str] = mapped_column(String(256), comment="Associated session UUID", nullable=False)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False)
    # Core RAG Fields
    standalone_query: Mapped[str | None] = mapped_column(CLOB, comment="Context-enriched rewritten question")
    search_keywords: Mapped[str | None] = mapped_column(String(1000), comment="Includes extracted keywords and expanded synonyms in hybrid search")
    memory_vector: Mapped[list[float]] = mapped_column(VectorField(), comment="Oracle 23ai native vector for semantic search")
    
    thought: Mapped[str | None] = mapped_column(CLOB, comment="LLM thought process for this turn")
    current_plan: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Current active plan")
    reasoning_path: Mapped[list[SkillExecutionContext] | list[dict[str, Any]]] = mapped_column(JSON, nullable=True, comment="Reasoning path for this turn")
    memory_summary: Mapped[str | None] = mapped_column(CLOB, comment="LLM reflected knowledge snapshot for long-term memory")
    turn_type: Mapped[str | None] = mapped_column(String(64), comment="Intent category for this turn, e.g., FOLLOW-UP, NEW TOPIC, CORRECTION")
    turn_entities: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Entity snapshot for this specific turn")

    # Metadata & Content
    
    raw_question: Mapped[str] = mapped_column(CLOB, comment="Original user input")
    answer: Mapped[str | None] = mapped_column(CLOB, comment="AI generated response")
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=True, comment="Streaming blocks used for this answer")
    feedback: Mapped[int] = mapped_column(Numeric(1, 0), default=0, comment="User feedback for this turn, -1: bad, 0: neutral, 1: good")
    request_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    response_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))



import uuid
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime
from agent.common import SkillExecutionContext

class UserProfileModel(BaseModel):
    """长期用户画像"""
    user_id: str = Field(..., description="用户唯一 ID")
    # --- 权重偏好系统 ---
    # 分为显式设定(user_defined)和隐式统计(inferred)
    global_preferences: dict[str, dict[str, Any]] = Field(default_factory=lambda: {"confirmed": {}, "inferred": {}}, description="技术栈、语言、输出风格等偏好")
    # --- 核心改动：实体认知图谱 ---
    frequent_entities: dict[str, int] = Field(default_factory=dict, description="实体出现频率计数")
    entity_relations: list[dict] = Field(default_factory=list, description="轻量级实体关联，如产线-负责人")
    # --- 核心改动：纠错记忆 (防止污染的关键) ---
    correction_history: list[str] = Field(default_factory=list, description="用户订正过的错误事实或偏好")

    profile_summary: str | None = None
    last_update_time: datetime = Field(default_factory=datetime.now)

class ConversationContextModel(BaseModel):
    """会话上下文与滚动摘要"""
    session_id: str
    user_id: str
    agent_id: str
    session_title: str | None = None

    # --- 状态机与执行流 ---
    session_state: dict[str, Any] = Field(default_factory=dict, description="全局实体参数池")
    current_plan: dict[str, Any] | None = Field(None, description="TaskPlanner 生成的当前待执行步骤")
    step_outputs: dict[str, Any] = Field(default_factory=dict, description="各 Skill 步骤的中间产物，用于下一步输入")

    # --- 抗污染机制 ---
    context_summary: str | None = None
    last_relevance_score: float = Field(1.0, description="最后一轮对话与上下文的相关性，低分触发隔离")
    active_topic: str | None = Field(None, description="当前活跃话题标签")

    interaction_count: int = 0
    is_deleted: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    last_active_at: datetime = Field(default_factory=datetime.now)

class MemoryEntryModel(BaseModel):
    """原子交互记忆"""
    entry_id: str # 问答交互唯一 ID (UUID)
    session_id: str
    user_id: str  # 冗余 user_id，方便跨 Session 检索该用户的历史知识

    # --- 检索增强核心 ---
    standalone_query: str | None = Field(None, description="改写后的问题")
    search_keywords: str | None = None   # 关键词和同义词
    memory_vector: list[float] | None = None  # 向量字段

    # --- 推理链路与知识提取 ---
    thought: str | None = Field(None, description="LLM 在改写阶段的思考过程")
    current_plan: dict[str, Any] | None = Field(None, description="TaskPlanner 生成的当前待执行步骤")
    reasoning_path: list[SkillExecutionContext] | list[dict[str, Any]] = Field(default_factory=list, description="执行的技能链条，如 ['sql_rag', 'compute']")
    memory_summary: str | None = Field(None, description="本轮问答提炼出的核心事实知识")
    turn_type: str | None = Field(None, description="问题意图分类，如 FOLLOW-UP, NEW TOPIC, CORRECTION")
    turn_entities: dict[str, Any] = Field(default_factory=dict, description="本轮提取的实体快照")
    
    # --- 溯源与质量 ---
    raw_question: str
    answer: str | None = None
    blocks: list[dict[str, Any]] = Field(default_factory=list, description="流式输出的块列表，用于前端渲染")
    feedback: int = 0  # -1, 0, 1
    request_time: datetime
    response_time: datetime