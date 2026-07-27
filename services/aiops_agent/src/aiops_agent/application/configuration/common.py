"""配置用例共享的并发、幂等、游标与消息工具。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import InboxEntity, OutboxEntity
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.contracts import AuthContext
from platform_core.identity import uuid7


@dataclass(frozen=True)
class ConfigurationScope:
    """从可信 AuthContext 派生的配置读写范围。"""

    app_id: int
    domain_id: int
    principal_id: str
    actor_id: str
    request_id: str
    trace_id: str

    @classmethod
    def from_auth(
        cls,
        *,
        app_id: int,
        auth_context: AuthContext,
    ) -> "ConfigurationScope":
        if not auth_context.domain_id or not auth_context.asserted_user_id:
            raise AIOpsApplicationError(
                code="OPS_IDENTITY_CONTEXT_REQUIRED",
                message="配置操作必须包含可信 Domain 和操作人",
                status_code=401,
            )
        try:
            domain_id = int(auth_context.domain_id)
        except ValueError as exc:
            raise AIOpsApplicationError(
                code="OPS_IDENTITY_CONTEXT_INVALID",
                message="Domain 标识格式无效",
                status_code=401,
            ) from exc
        return cls(
            app_id=app_id,
            domain_id=domain_id,
            principal_id=f"{auth_context.principal_kind}:{auth_context.client_id}",
            actor_id=auth_context.asserted_user_id,
            request_id=auth_context.request_id,
            trace_id=auth_context.trace_id,
        )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_etag(value: str | None) -> int:
    """解析强 ETag；缺失与格式错误使用不同稳定错误码。"""
    if value is None:
        raise AIOpsApplicationError(
            code="PRECONDITION_REQUIRED",
            message="该操作必须提供 If-Match",
            status_code=428,
        )
    normalized = value.strip()
    if (
        len(normalized) < 6
        or not normalized.startswith('"rv-')
        or not normalized.endswith('"')
    ):
        raise AIOpsApplicationError(
            code="OPS_ETAG_INVALID",
            message='If-Match 必须使用 "rv-N" 格式',
            status_code=400,
        )
    try:
        version = int(normalized[4:-1])
    except ValueError as exc:
        raise AIOpsApplicationError(
            code="OPS_ETAG_INVALID",
            message='If-Match 必须使用 "rv-N" 格式',
            status_code=400,
        ) from exc
    if version < 1:
        raise AIOpsApplicationError(
            code="OPS_ETAG_INVALID",
            message="If-Match 版本必须大于零",
            status_code=400,
        )
    return version


def format_etag(row_version: int) -> str:
    return f'"rv-{int(row_version)}"'


class SignedCursorCodec:
    """绑定 Scope、调用方、过滤器和过期时间的不透明游标。"""

    def __init__(self, *, secret: str, ttl_seconds: int):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("AIOps Cursor 密钥至少需要 32 字节")
        self._secret = secret.encode("utf-8")
        self._ttl_seconds = ttl_seconds

    def encode(
        self,
        *,
        scope: ConfigurationScope,
        updated_at: datetime,
        resource_id: UUID,
        filters: dict[str, Any],
    ) -> str:
        payload = {
            "v": 1,
            "scope": sha256_json(
                {
                    "app_id": scope.app_id,
                    "domain_id": scope.domain_id,
                    "principal": scope.principal_id,
                }
            ),
            "filters": sha256_json(filters),
            "updated_at": updated_at.astimezone(UTC).isoformat(),
            "resource_id": str(resource_id),
            "exp": int(
                (datetime.now(UTC) + timedelta(seconds=self._ttl_seconds))
                .timestamp()
            ),
        }
        raw = canonical_json(payload).encode("utf-8")
        signature = hmac.new(self._secret, raw, hashlib.sha256).digest()
        return (
            base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
            + "."
            + base64.urlsafe_b64encode(signature)
            .rstrip(b"=")
            .decode("ascii")
        )

    def decode(
        self,
        *,
        token: str,
        scope: ConfigurationScope,
        filters: dict[str, Any],
    ) -> tuple[datetime, UUID]:
        try:
            encoded_payload, encoded_signature = token.split(".", 1)
            raw = base64.urlsafe_b64decode(
                encoded_payload + "=" * (-len(encoded_payload) % 4)
            )
            signature = base64.urlsafe_b64decode(
                encoded_signature + "=" * (-len(encoded_signature) % 4)
            )
            expected = hmac.new(self._secret, raw, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("签名不匹配")
            payload = json.loads(raw)
            expected_scope = sha256_json(
                {
                    "app_id": scope.app_id,
                    "domain_id": scope.domain_id,
                    "principal": scope.principal_id,
                }
            )
            if (
                payload.get("v") != 1
                or payload.get("scope") != expected_scope
                or payload.get("filters") != sha256_json(filters)
                or int(payload.get("exp", 0)) < int(datetime.now(UTC).timestamp())
            ):
                raise ValueError("游标上下文不匹配或已过期")
            updated_at = datetime.fromisoformat(payload["updated_at"])
            if updated_at.tzinfo is None:
                raise ValueError("游标时间缺少时区")
            return updated_at.astimezone(UTC), UUID(payload["resource_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AIOpsApplicationError(
                code="OPS_CURSOR_INVALID",
                message="分页游标无效、已过期或不属于当前查询",
                status_code=400,
            ) from exc


class IdempotencyGuard:
    """把配置命令的请求指纹和响应快照写入 Inbox。"""

    SOURCE_SYSTEM = "AIOPS_CONFIG_API"

    @staticmethod
    def message_key(
        *,
        scope: ConfigurationScope,
        operation: str,
        parent_resource: str,
        idempotency_key: str,
    ) -> str:
        return sha256_json(
            {
                "principal": scope.principal_id,
                "domain": scope.domain_id,
                "operation": operation,
                "parent": parent_resource,
                "key": idempotency_key,
            }
        )

    async def replay(
        self,
        *,
        uow: AIOpsUnitOfWork,
        message_key: str,
        payload_hash: str,
    ) -> dict[str, Any] | None:
        assert uow.inbox is not None
        existing = await uow.inbox.get_by_message(
            source_system=self.SOURCE_SYSTEM,
            message_key=message_key,
            lock=True,
        )
        if existing is None:
            return None
        if existing.payload_hash != payload_hash:
            raise AIOpsApplicationError(
                code="OPS_IDEMPOTENCY_CONFLICT",
                message="Idempotency-Key 已用于不同请求",
                status_code=409,
            )
        payload = existing.payload_json or {}
        result = payload.get("result")
        if existing.status != "PROCESSED" or not isinstance(result, dict):
            raise AIOpsApplicationError(
                code="OPS_IDEMPOTENCY_IN_PROGRESS",
                message="相同请求仍在处理中，请稍后重试",
                status_code=409,
                retryable=True,
            )
        return result

    async def record(
        self,
        *,
        uow: AIOpsUnitOfWork,
        message_key: str,
        operation: str,
        payload_hash: str,
        result: dict[str, Any],
        now: datetime,
    ) -> None:
        assert uow.inbox is not None
        await uow.inbox.add(
            InboxEntity(
                inbox_id=uuid7(),
                source_system=self.SOURCE_SYSTEM,
                message_key=message_key,
                message_type=operation,
                payload_json={"result": result},
                payload_hash=payload_hash,
                status="PROCESSED",
                processed_at=now,
            )
        )


async def add_configuration_event(
    *,
    uow: AIOpsUnitOfWork,
    scope: ConfigurationScope,
    aggregate_type: str,
    aggregate_id: UUID,
    event_type: str,
    row_version: int,
    details: dict[str, Any] | None = None,
) -> None:
    """Outbox 同时承载下游通知和不可丢失的配置操作轨迹。"""
    assert uow.outbox is not None
    payload = {
        "schema_version": "aiops.config.event.v1",
        "app_id": scope.app_id,
        "domain_id": scope.domain_id,
        "aggregate_type": aggregate_type,
        "aggregate_id": str(aggregate_id),
        "event_type": event_type,
        "row_version": row_version,
        "actor_id": scope.actor_id,
        "request_id": scope.request_id,
        "trace_id": scope.trace_id,
        "details": details or {},
    }
    await uow.outbox.add(
        OutboxEntity(
            outbox_id=uuid7(),
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            event_type=event_type,
            idempotency_key=sha256_json(payload),
            payload_json=payload,
            payload_hash=sha256_json(payload),
            status="PENDING",
            trace_id=scope.trace_id,
        )
    )
