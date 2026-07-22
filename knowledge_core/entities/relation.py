"""Evidence-backed, revision-scoped Knowledge Core relations."""
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity, OracleJSON


class KcRelationEntity(BaseEntity):
    __tablename__ = "KBOT_KC_RELATION"

    relation_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    collection_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    bundle_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    bundle_revision_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    predicate: Mapped[str] = mapped_column(String(48), nullable=False)
    object_type: Mapped[str] = mapped_column(String(32), nullable=False)
    object_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    directionality: Mapped[str] = mapped_column(String(16), nullable=False)
    support_json: Mapped[dict[str, Any]] = mapped_column(OracleJSON, nullable=False)
    derivation_type: Mapped[str] = mapped_column(String(16), nullable=False)
    derivation_key: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(8, 6))
    attributes_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    relation_status: Mapped[str] = mapped_column(String(16), nullable=False, default="STAGED")
    created_by: Mapped[str | None] = mapped_column(String(256))
    updated_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
