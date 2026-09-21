"""知识检索应用 X Search 运行时实体。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UUIDv7Type


class KnowledgeRetrievalResearchRunEntity(BaseEntity):
    __tablename__ = "KBOT_KR_RESEARCH_RUN"
    __table_args__ = (
        UniqueConstraint(
            "domain_id", "actor_id", "idempotency_key",
            name="UK_KR_RESEARCH_IDEMP",
        ),
        UniqueConstraint("run_id", "domain_id", name="UK_KR_RESEARCH_SCOPE"),
        Index("IX_KR_RESEARCH_DOMAIN_TIME", "domain_id", "created_at"),
        Index("IX_KR_RESEARCH_CLAIM", "status", "lease_until"),
    )

    run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACCEPTED")
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    model_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    model_snapshot_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON(), nullable=False)
    request_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON(), nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON())
    usage_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON())
    provider_request_id: Mapped[str | None] = mapped_column(String(256))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_token: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class KnowledgeRetrievalResearchEventEntity(BaseEntity):
    __tablename__ = "KBOT_KR_RESEARCH_EVENT"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "sequence_no", name="UK_KR_RESEARCH_EVENT_SEQ"
        ),
    )

    event_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str | None] = mapped_column(String(1000))
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

class KnowledgeRetrievalXSourceEntity(BaseEntity):
    __tablename__ = "KBOT_KR_X_SOURCE"
    __table_args__ = (
        UniqueConstraint("run_id", "citation_label", name="UK_KR_X_SOURCE_LABEL"),
        Index("IX_KR_X_SOURCE_RUN", "run_id", "display_order"),
    )

    source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    citation_label: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_citation_id: Mapped[str] = mapped_column(String(256), nullable=False)
    canonical_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512))
    excerpt: Mapped[str | None] = mapped_column(String(4000))
    author_handle: Mapped[str | None] = mapped_column(String(128))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
