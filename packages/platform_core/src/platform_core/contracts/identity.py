"""跨服务传播的身份上下文契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PrincipalKind(StrEnum):
    """内部身份来源。"""

    PORTAL = "PORTAL"
    APP_API_CLIENT = "APP_API_CLIENT"
    API_CLIENT = "API_CLIENT"
    SERVICE = "SERVICE"


class IdentityEntryKind(StrEnum):
    """公开身份进入平台或业务 App 的上下文类型。"""

    PLATFORM = "PLATFORM"
    BUSINESS = "BUSINESS"


class AuthContext(BaseModel):
    """由可信边界创建、供下游服务消费的最小身份上下文。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    principal_kind: PrincipalKind
    client_id: str = Field(min_length=1, max_length=128)
    calling_service: str | None = Field(default=None, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    api_key_id: str | None = Field(default=None, max_length=64)
    entry_kind: IdentityEntryKind = IdentityEntryKind.BUSINESS
    app_id: str | None = Field(default=None, max_length=64)
    domain_id: str | None = Field(default=None, max_length=128)
    tenant_id: str | None = Field(default=None, max_length=128)
    asserted_user_id: str | None = Field(default=None, max_length=256)
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    authorized_agent_ids: tuple[UUID, ...] = ()
    delegated_by: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_principal(self) -> "AuthContext":
        if self.principal_kind == PrincipalKind.PORTAL:
            if not self.api_key_id:
                raise ValueError("门户身份必须包含 api_key_id")
            if not self.asserted_user_id:
                raise ValueError("门户身份必须包含 asserted_user_id")
            if self.entry_kind == IdentityEntryKind.PLATFORM:
                if self.domain_id or self.app_id:
                    raise ValueError("平台入口不能包含 App 或 Domain 上下文")
            elif not self.domain_id:
                raise ValueError("业务入口必须包含 domain_id")
        if self.principal_kind == PrincipalKind.API_CLIENT and not self.api_key_id:
            raise ValueError("API Client 身份必须包含 api_key_id")
        if self.principal_kind == PrincipalKind.APP_API_CLIENT:
            if not self.api_key_id:
                raise ValueError("App API Client 身份必须包含 api_key_id")
            if not self.app_id or not self.domain_id or not self.asserted_user_id:
                raise ValueError("App API Client 身份上下文不完整")
            if self.entry_kind != IdentityEntryKind.BUSINESS:
                raise ValueError("App API Client 只能使用业务入口")
        return self


class ServiceIdentity(BaseModel):
    """内部调用方的短期、限定 audience 的服务身份。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issuer: str = Field(min_length=1, max_length=128)
    subject: str = Field(min_length=1, max_length=128)
    audience: str = Field(min_length=1, max_length=128)
    scopes: tuple[str, ...] = Field(min_length=1)
    issued_at: datetime
    expires_at: datetime
    token_id: UUID

    @model_validator(mode="after")
    def validate_identity(self) -> "ServiceIdentity":
        if self.expires_at <= self.issued_at:
            raise ValueError("Service Identity 过期时间必须晚于签发时间")
        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError("Service Identity scopes 不能重复")
        if any(not scope or len(scope) > 128 for scope in self.scopes):
            raise ValueError("Service Identity scope 格式无效")
        return self
