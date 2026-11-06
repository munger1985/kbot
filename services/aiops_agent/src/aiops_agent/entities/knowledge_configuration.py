"""AIOps 案例处理的 App 私有配置；不创建共享知识集合。"""

from datetime import datetime
from uuid import UUID
from sqlalchemy import Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class OperationsKnowledgeConfigurationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_CONFIG"

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, nullable=False)
    models_json: Mapped[dict] = mapped_column(OracleNativeJSON, nullable=False, default=dict)
    profile_ref: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
