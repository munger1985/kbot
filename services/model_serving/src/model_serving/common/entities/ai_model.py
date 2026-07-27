from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UUIDv7Type,
)

class AIModelEntity(BaseEntity):
    """模型托管服务拥有的可调用模型定义。"""

    __tablename__ = "KBOT_AI_MODEL"
    __table_args__ = (
        UniqueConstraint("served_model_name", name="UK_AI_MODEL_SERVED_NAME"),
    )

    model_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
        comment="模型定义 UUIDv7",
    )
    app_id: Mapped[int] = mapped_column(
        Numeric(38, 0), nullable=False, comment="APEX 应用过滤字段",
    )
    served_model_name: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="推理 API 对外暴露的稳定模型名称",
    )
    display_name: Mapped[str] = mapped_column(
        String(256), nullable=False, comment="可修改的界面显示名称",
    )
    provider_model_name: Mapped[str] = mapped_column(
        String(256), nullable=False, comment="传递给上游厂商或本地引擎的真实模型名称",
    )
    category: Mapped[int] = mapped_column(
        Numeric(2, 0), nullable=False, comment="模型类别",
    )
    provider: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="模型提供方",
    )
    api_endpoint: Mapped[str | None] = mapped_column(
        String(1024), comment="上游推理端点",
    )
    api_key: Mapped[str | None] = mapped_column(
        Text, comment="待迁移至 Secret Store 的上游凭据",
    )
    status: Mapped[int] = mapped_column(
        Numeric(1, 0), nullable=False, default=0, comment="0 禁用、1 启用、2 归档",
    )
    embedding_dimension: Mapped[int | None] = mapped_column(
        Numeric(10, 0),
        comment="输出向量维度；文本 Embedding 模型必填且必须匹配平台配置",
    )
    model_params: Mapped[dict | None] = mapped_column(
        OracleNativeJSON(),
        comment="模型默认参数，使用 Oracle 原生 JSON 存储",
    )
    descs: Mapped[str | None] = mapped_column(String(512), comment="模型说明")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="创建者")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间",
    )
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="更新者")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        comment="更新时间",
    )
