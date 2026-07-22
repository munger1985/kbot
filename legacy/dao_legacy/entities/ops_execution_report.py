"""AIOps 执行报告实体 (Oracle 23ai)"""

from datetime import datetime, timezone
from sqlalchemy import String, Float, DateTime, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, OracleJSON


class OpsExecutionReportEntity(BaseEntity):
    """AIOps 自愈执行报告表 — 存储每次自愈操作的完整审计报告。

    Oracle 23ai 表，JSON 类型支持原生 JSON 查询和搜索索引。
    """

    __tablename__ = "kbot_ops_execution_report"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, server_default=text("SYS_GUID()"),
        comment="报告唯一标识",
    )
    entry_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment="关联的 kbot_md_memory_entry.entry_id",
    )
    session_id: Mapped[str] = mapped_column(
        String(64), nullable=False,
        comment="关联的会话 ID",
    )
    user_id: Mapped[str | None] = mapped_column(
        String(64), comment="操作用户 ID",
    )
    agent_id: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="执行诊断的 Agent ID",
    )

    # 目标实例信息
    instance_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment="目标数据库实例 ID",
    )
    instance_name: Mapped[str] = mapped_column(
        String(256), nullable=False, default="",
        comment="实例名称 (展示用)",
    )
    db_type: Mapped[str] = mapped_column(
        String(32), nullable=False,
        comment="Oracle / PostgreSQL / MySQL",
    )
    environment: Mapped[str] = mapped_column(
        String(32), nullable=False, default="prod",
        comment="prod / staging / dev",
    )

    # 触发信息
    trigger_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="manual",
        comment="manual / webhook / cron",
    )
    original_question: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
        comment="用户原始问题或告警摘要",
    )
    diagnosis_summary: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
        comment="LLM 诊断结论摘要",
    )

    # 执行动作 (Oracle 23ai JSON)
    actions_executed: Mapped[list | None] = mapped_column(
        OracleJSON, comment="[{sql, impact, risk_level, context}]",
    )

    # 验证快照 (JSON)
    pre_snapshot: Mapped[dict | None] = mapped_column(
        OracleJSON, comment="修复前监控快照",
    )
    post_snapshot: Mapped[dict | None] = mapped_column(
        OracleJSON, comment="修复后监控快照",
    )
    health_check_result: Mapped[dict | None] = mapped_column(
        OracleJSON, comment="DB 健康检查结果",
    )

    # 回滚信息 (JSON)
    rollback_info: Mapped[dict | None] = mapped_column(
        OracleJSON, comment="{rollback_sql, executed: bool, result}",
    )

    # 判定
    verification_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="skipped",
        comment="verified / degraded / failed / skipped",
    )

    # 报告内容 (CLOB)
    report_content: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
        comment="Markdown 格式完整报告",
    )
    recommendations: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
        comment="LLM 后续优化建议",
    )

    # 元数据
    total_duration_seconds: Mapped[float] = mapped_column(
        Float, nullable=False, default=0,
        comment="从诊断到验证的总耗时 (秒)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=text("SYSTIMESTAMP"),
        comment="报告创建时间",
    )

    def __repr__(self) -> str:
        return (
            f"<OpsExecutionReport(id='{self.id}', instance='{self.instance_name}', "
            f"status='{self.verification_status}', db_type='{self.db_type}')>"
        )
