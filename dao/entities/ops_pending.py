# dao/entities/ops_pending.py
"""AIOps HITL 挂起请求实体"""

from datetime import datetime, timezone
from sqlalchemy import String, Integer, CLOB, DateTime, text
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, OracleJSON


class OpsPendingRequestEntity(BaseEntity):
    """
    AIOps HITL 挂起请求快照表映射。
    存储 Agent 等待用户输入时的完整执行状态，用于恢复。
    """

    __tablename__ = "kbot_ops_pending_request"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, comment="挂起请求唯一标识 (UUID v7)")
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="关联的会话 ID")
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, comment="发起诊断的用户 ID")
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False, comment="执行诊断的 Agent ID (int)")
    instance_id: Mapped[str] = mapped_column(String(36), nullable=False, comment="目标数据库实例 ID")
    entry_id: Mapped[str] = mapped_column(String(36), nullable=False, comment="原始 memory entry ID")

    # 挂起上下文 (给用户看的)
    suspend_reason: Mapped[str] = mapped_column(CLOB, nullable=False, comment="LLM 给出的挂起原因")
    user_prompt: Mapped[str] = mapped_column(CLOB, nullable=False, comment="展示给用户的操作指引")
    sql_to_run: Mapped[str | None] = mapped_column(CLOB, comment="LLM 生成的用户需执行的 SQL")
    expected_fields: Mapped[dict | None] = mapped_column(OracleJSON, comment="期望用户返回的字段列表 JSON")

    # 执行状态快照 (用于恢复)
    suspended_by_skill: Mapped[str] = mapped_column(String(128), nullable=False, comment="触发挂起的 Skill 名称")
    current_step_index: Mapped[int] = mapped_column(Integer, default=0, comment="挂起时的步骤索引")
    completed_steps: Mapped[dict | None] = mapped_column(OracleJSON, comment="execution_history JSON")
    accumulated_results: Mapped[dict | None] = mapped_column(OracleJSON, comment="metric/monitor/doc_results JSON")
    pending_variables: Mapped[dict | None] = mapped_column(OracleJSON, comment="ctx['variables'] 快照 JSON")
    hitl_history: Mapped[dict | None] = mapped_column(OracleJSON, comment="HITL 多轮 Timeline JSON")
    runtime_plan: Mapped[dict | None] = mapped_column(OracleJSON, comment="ctx['runtime_plan'] 快照 JSON")

    # 生命周期
    status: Mapped[str] = mapped_column(String(16), default="pending", comment="pending / answered / timeout / cancelled")
    requested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc))
    responded_at: Mapped[datetime | None] = mapped_column(DateTime, comment="用户回填时间")
    timeout_at: Mapped[datetime | None] = mapped_column(DateTime, comment="超时时间（默认 +30 分钟）")
    reminder_count: Mapped[int] = mapped_column(Integer, default=0, comment="催办次数")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    __table_args__ = (
        {"comment": "AIOps HITL 挂起请求快照表"}
    )

    def __repr__(self) -> str:
        return (
            f"<OpsPendingRequest(request_id='{self.request_id}', "
            f"session='{self.session_id}', status='{self.status}', "
            f"skill='{self.suspended_by_skill}')>"
        )
