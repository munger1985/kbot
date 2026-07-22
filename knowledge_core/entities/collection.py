"""Root Knowledge Core entity mappings."""
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity, OracleJSON


class KcCollectionEntity(BaseEntity):
    __tablename__ = "KBOT_KC_COLLECTION"

    collection_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    collection_key: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))
    embedding_model_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="ACTIVE")
    default_security_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str | None] = mapped_column(String(256))
    updated_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KcCollectionBindingEntity(BaseEntity):
    __tablename__ = "KBOT_KC_COLLECTION_BINDING"

    binding_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    collection_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    consumer_type: Mapped[str] = mapped_column(String(32), nullable=False, default="AGENT")
    consumer_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    note: Mapped[str | None] = mapped_column(String(1000))
    created_by: Mapped[str | None] = mapped_column(String(256))
    updated_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KcIngestionReceiptEntity(BaseEntity):
    __tablename__ = "KBOT_KC_INGESTION_RECEIPT"

    ingestion_receipt_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    collection_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    receipt_status: Mapped[str] = mapped_column(String(24), nullable=False)
    bundle_id: Mapped[int | None] = mapped_column(Numeric(38, 0))
    bundle_revision_id: Mapped[int | None] = mapped_column(Numeric(38, 0))
    staging_manifest_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    failure_code: Mapped[str | None] = mapped_column(String(128))
    failure_message: Mapped[str | None] = mapped_column(String(1000))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str | None] = mapped_column(String(256))
    updated_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
