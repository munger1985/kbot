"""Discovery projections for Bundle/Document-level retrieval."""
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity, OracleJSON, VectorField


class KcDiscoveryObjectEntity(BaseEntity):
    __tablename__ = "KBOT_KC_DISCOVERY_OBJECT"

    discovery_object_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    collection_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    bundle_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    bundle_revision_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    bundle_revision_document_id: Mapped[int | None] = mapped_column(Numeric(38, 0))
    document_id: Mapped[int | None] = mapped_column(Numeric(38, 0))
    document_version_id: Mapped[int | None] = mapped_column(Numeric(38, 0))
    object_type: Mapped[str] = mapped_column(String(16), nullable=False)
    profile_key: Mapped[str] = mapped_column(String(256), nullable=False)
    display_title: Mapped[str] = mapped_column(String(512), nullable=False)
    profile_text: Mapped[str] = mapped_column(Text, nullable=False)
    facet_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    coverage_json: Mapped[dict[str, Any]] = mapped_column(OracleJSON, nullable=False)
    profile_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(VectorField())
    embedding_model_id: Mapped[int | None] = mapped_column(Numeric(38, 0))
    embedding_model_key: Mapped[str | None] = mapped_column(String(256))
    embedding_config_fingerprint: Mapped[str | None] = mapped_column(String(64))
    embedding_input_hash: Mapped[str | None] = mapped_column(String(64))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    security_level: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Numeric(8, 6))
    discovery_status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(256))
    updated_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
