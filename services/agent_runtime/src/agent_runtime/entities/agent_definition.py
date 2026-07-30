"""用户可选择的 Root Agent 定义。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class AgentDefinitionEntity(BaseEntity):
    """Agent Runtime 拥有的执行配置，不表示 Specialist 进程实例。"""

    __tablename__ = "KBOT_AGENT_DEFINITION"

    agent_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    agent_key: Mapped[str] = mapped_column(String(128), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="DRAFT"
    )
    enabled_capabilities_json: Mapped[list[str]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    models_json: Mapped[dict[str, str]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    do_rerank: Mapped[int] = mapped_column(
        Numeric(1, 0), nullable=False, default=0
    )
    data_profile_name: Mapped[str | None] = mapped_column(String(256))
    instruction: Mapped[str | None] = mapped_column(Text)
    config_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    __mapper_args__ = {"version_id_col": row_version}
