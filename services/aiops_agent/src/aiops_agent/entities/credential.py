"""AIOps 加密数据库凭据映射。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import LargeBinary, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity, UniversalTimestamp, UUIDv7Type


class CredentialEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_CREDENTIAL"

    credential_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    credential_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    username_ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    username_nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    password_ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    password_nonce: Mapped[bytes] = mapped_column(LargeBinary(12), nullable=False)
    key_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
