"""数据库托管凭据实体。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Index, LargeBinary, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, UUIDv7Type


class ManagedCredentialEntity(BaseEntity):
    """只保存密文和加密元数据，不保存可逆业务标识。"""

    __tablename__ = "KBOT_MANAGED_CREDENTIAL"
    __table_args__ = (
        UniqueConstraint(
            "domain_id",
            "namespace",
            "credential_kind",
            "external_key",
            name="UK_MANAGED_CREDENTIAL_KEY",
        ),
        UniqueConstraint(
            "credential_id",
            "domain_id",
            name="UK_MANAGED_CREDENTIAL_SCOPE",
        ),
        Index(
            "IX_MANAGED_CREDENTIAL_SCOPE",
            "domain_id",
            "namespace",
            "credential_kind",
            "status",
        ),
    )

    credential_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    namespace: Mapped[str] = mapped_column(String(64), nullable=False)
    credential_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    external_key: Mapped[str] = mapped_column(String(256), nullable=False)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    key_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


__all__ = ["ManagedCredentialEntity"]
