"""Knowledge Core 不可变入库聚合映射。"""
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleJSON,
    UUIDv7Type,
    VectorField,
)


class _AuditEntity(BaseEntity):
    __abstract__ = True
    created_by: Mapped[str | None] = mapped_column(String(256))
    updated_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KcBundleEntity(_AuditEntity):
    __tablename__ = "KBOT_KC_BUNDLE"
    bundle_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
    )
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    source_system: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(256), nullable=False)
    current_revision_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    availability_status: Mapped[str] = mapped_column(String(16), nullable=False, default="EMPTY")
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)


class KcBundleRevisionEntity(_AuditEntity):
    __tablename__ = "KBOT_KC_BUNDLE_REVISION"
    bundle_revision_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
    )
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    revision_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    source_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    snapshot_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_json: Mapped[dict[str, Any]] = mapped_column(OracleJSON, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    canonical_url: Mapped[str | None] = mapped_column(String(2048))
    security_level: Mapped[int] = mapped_column(Integer, nullable=False)
    facet_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_code: Mapped[str | None] = mapped_column(String(128))
    failure_message: Mapped[str | None] = mapped_column(String(1000))


class KcDocumentEntity(_AuditEntity):
    __tablename__ = "KBOT_KC_DOCUMENT"
    document_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
    )
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    external_document_id: Mapped[str] = mapped_column(String(256), nullable=False)
    document_status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")


class KcDocumentVersionEntity(_AuditEntity):
    __tablename__ = "KBOT_KC_DOCUMENT_VERSION"
    document_version_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
    )
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    document_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    version_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_uri: Mapped[str] = mapped_column(String(2048), nullable=False)
    storage_state: Mapped[str] = mapped_column(String(16), nullable=False)
    byte_size: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    detected_mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    security_level: Mapped[int] = mapped_column(Integer, nullable=False)
    content_metadata_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class KcBundleRevisionDocumentEntity(_AuditEntity):
    __tablename__ = "KBOT_KC_BUNDLE_REVISION_DOCUMENT"
    bundle_revision_document_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
    )
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_revision_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    document_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    document_version_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    document_role: Mapped[str] = mapped_column(String(24), nullable=False)
    ordinal: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    required_flag: Mapped[int] = mapped_column(Integer, nullable=False)
    external_document_id: Mapped[str] = mapped_column(String(256), nullable=False)
    declared_name: Mapped[str | None] = mapped_column(String(512))
    declared_mime_type: Mapped[str | None] = mapped_column(String(255))
    source_url: Mapped[str | None] = mapped_column(String(2048))
    member_status: Mapped[str] = mapped_column(String(24), nullable=False)
    failure_stage: Mapped[str | None] = mapped_column(String(32))
    failure_code: Mapped[str | None] = mapped_column(String(128))
    failure_message: Mapped[str | None] = mapped_column(String(1000))
    received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)


class KcIngestionJobEntity(_AuditEntity):
    __tablename__ = "KBOT_KC_INGESTION_JOB"
    ingestion_job_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
    )
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_revision_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    document_version_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    parse_view_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    job_type: Mapped[str] = mapped_column(String(24), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    job_status: Mapped[str] = mapped_column(String(16), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    failure_class: Mapped[str | None] = mapped_column(String(16))
    failure_code: Mapped[str | None] = mapped_column(String(128))
    failure_message: Mapped[str | None] = mapped_column(String(1000))
    trace_id: Mapped[str | None] = mapped_column(String(128))
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)


class KcParseViewEntity(_AuditEntity):
    __tablename__ = "KBOT_KC_PARSE_VIEW"
    parse_view_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
    )
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    view_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    parser_name: Mapped[str] = mapped_column(String(128), nullable=False)
    parser_version: Mapped[str | None] = mapped_column(String(128))
    parse_config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    parse_config_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    view_status: Mapped[str] = mapped_column(String(16), nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Numeric(8, 5))
    quality_report_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    artifact_manifest_json: Mapped[dict[str, Any] | None] = mapped_column(OracleJSON)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KcEvidenceEntity(_AuditEntity):
    __tablename__ = "KBOT_KC_EVIDENCE"
    evidence_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7,
    )
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_revision_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_revision_document_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    document_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    parse_view_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    evidence_key: Mapped[str] = mapped_column(String(256), nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ordinal: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    fragment_index: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=0)
    parent_evidence_key: Mapped[str | None] = mapped_column(String(256))
    source_item_ref: Mapped[str | None] = mapped_column(String(512))
    source_spans_json: Mapped[list[dict[str, Any]]] = mapped_column(OracleJSON, nullable=False)
    heading_path_json: Mapped[list[Any] | None] = mapped_column(OracleJSON)
    section_key: Mapped[str | None] = mapped_column(String(256))
    hierarchy_depth: Mapped[int | None] = mapped_column(Integer)
    heading_level: Mapped[int | None] = mapped_column(Integer)
    locator_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    locator_json: Mapped[dict[str, Any]] = mapped_column(OracleJSON, nullable=False)
    payload_uri: Mapped[str | None] = mapped_column(String(2048))
    provenance_json: Mapped[dict[str, Any]] = mapped_column(OracleJSON, nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    page_start: Mapped[int | None] = mapped_column(Numeric(19, 0))
    page_end: Mapped[int | None] = mapped_column(Numeric(19, 0))
    language_code: Mapped[str | None] = mapped_column(String(32))
    token_count: Mapped[int | None] = mapped_column(Numeric(19, 0))
    quality_score: Mapped[float | None] = mapped_column(Numeric(8, 5))
    # Exactly one text vector is produced by the KC INDEX job.  Parser output
    # must leave this field null; the model identity is persisted beside the
    # vector so equal dimensions can never be mistaken for interchangeable
    # embedding spaces.
    embedding: Mapped[list[float] | None] = mapped_column(VectorField())
    embedding_model_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    embedding_served_model_name: Mapped[str | None] = mapped_column(String(128))
    embedding_config_fingerprint: Mapped[str | None] = mapped_column(String(64))
    embedding_input_hash: Mapped[str | None] = mapped_column(String(64))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
