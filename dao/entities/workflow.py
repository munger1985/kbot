# dao/entities/workflow.py — SOP 编排流程实体（Oracle 适配版）

from typing import Any
from datetime import datetime
from sqlalchemy import String, DateTime, Numeric, func, text
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseEntity, OracleJSON, VectorField

# SOP 强制执行模式
# strict: 严格遵循 / guided: 引导增强(默认) / suggested: 建议参考

class WorkflowEntity(BaseEntity):
    """SOP 编排流程实体：存储用户自定义的任务执行蓝图（Oracle 23ai）"""

    __tablename__ = "KBOT_MD_AGENT_WORKFLOW"

    # --- 核心标识与关联 ---
    id: Mapped[str] = mapped_column(String(36), primary_key=True, comment="UUID, 主键")
    agent_id: Mapped[int] = mapped_column(Numeric, nullable=False, comment="所属 Agent ID")
    name: Mapped[str] = mapped_column(String(255), nullable=False, comment="流程名称")
    description: Mapped[str] = mapped_column(String(1024), nullable=False, comment="流程语义描述")
    embedding: Mapped[list[float] | None] = mapped_column(VectorField(), comment="Oracle 23ai 语义向量")

    # --- 编排数据 (JSON) ---
    nodes: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON, comment="节点信息")
    edges: Mapped[list[dict[str, Any]] | None] = mapped_column(OracleJSON, comment="连接信息")
    config: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON, comment="全局配置")

    # --- 强制执行模式 ---
    exec_mode: Mapped[str] = mapped_column(
        String(20), nullable=False, default="guided", server_default=text("'guided'"),
        comment="SOP 强制执行模式: strict(严格遵循) / guided(引导增强,默认) / suggested(建议参考)"
    )

    # --- 状态控制 ---
    is_active: Mapped[str] = mapped_column(
        String(1), default="1", server_default=text("'1'"),
        comment="是否启用: 1=启用, 0=禁用"
    )

    # --- 审计日志 ---
    created_by: Mapped[str | None] = mapped_column(String(50), comment="创建人")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp(), comment="创建时间"
    )
    updated_by: Mapped[str | None] = mapped_column(String(50), comment="最后修改人")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp(),
        comment="更新时间"
    )

    @property
    def is_active_bool(self) -> bool:
        return self.is_active == "1"

    def to_dict(self) -> dict[str, Any]:
        """将实体转换为字典"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "nodes": self.nodes,
            "edges": self.edges,
            "config": self.config,
            "exec_mode": self.exec_mode,
            "is_active": self.is_active_bool,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at
        }
