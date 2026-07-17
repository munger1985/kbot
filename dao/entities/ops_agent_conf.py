# dao/entities/ops_agent_conf.py

from datetime import datetime, timezone
from sqlalchemy import String, Integer, Boolean, DateTime, ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity


class OpsAgentConfEntity(BaseEntity):
    """
    智能运维线 - 智能体与运维物理资产多对多动态绑定配置表
    【核心价值】: 专为 AIOps 机器人管理数据库实例群设计。
    【⚠️ 注意】: agent_id 为 int 类型 (对齐 kbot3 的自增主键体系)，
               instance_id 为 str 类型 (UUID v7, 外部 CMDB 标识)。
    """
    __tablename__ = "kbot_ops_agent_conf"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="关系配置自增 ID, 主键")
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False, comment="关联的智能体 Agent ID (int, 对应 kbot_md_agent.agent_id)")
    instance_id: Mapped[int] = mapped_column(Integer, nullable=False, comment="关联的运维实例 ID (对应 ops_db_instance.instance_id)")
    is_mutation_allowed: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否允许执行变更动作(如 Kill/DDL),False 则将此 Agent 锁死在只读听诊器状态")
    require_approval: Mapped[bool] = mapped_column(Boolean, default=True, comment="对此实例执行高危自愈动作时,是否必须触发人工审批门禁")
    max_daily_execution: Mapped[int] = mapped_column(Integer, default=10, comment="单日高危变更自愈动作上限频次,防止大模型陷入死循环")
    created_by: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc))
    updated_by: Mapped[str | None] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    __table_args__ = (
        {"comment": "智能运维线 - 智能体与运维物理资产多对多动态绑定配置表"}
    )

    def __repr__(self) -> str:
        return f"<OpsAgentConf(id={self.id}, agent={self.agent_id}, instance='{self.instance_id}')>"
