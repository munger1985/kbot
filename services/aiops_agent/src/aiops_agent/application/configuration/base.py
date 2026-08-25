"""配置服务共享的事务、幂等和并发基类。"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from urllib.parse import urlparse
from uuid import UUID

from loguru import logger
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    IdempotencyGuard,
    SignedCursorCodec,
    add_configuration_event,
    canonical_json,
    sha256_json,
)
from aiops_agent.application.managed_credentials import (
    AIOpsManagedCredentialService,
)
from aiops_agent.application.configuration.schedule import (
    InspectionTemplateRegistry,
    next_cron_run,
)
from aiops_agent.application.errors import (
    AIOpsApplicationError,
    resource_not_found,
    row_version_changed,
    state_conflict,
    validation_failed,
)
from aiops_agent.config import AIOpsManagementConfig
from aiops_agent.entities import (
    InspectionPlanEntity,
    InspectionTargetEntity,
    DiagnosticSourceEntity,
    PolicyEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetSourceBindingEntity,
)
from aiops_agent.persistence import AIOpsUnitOfWork
from aiops_agent.ports.agent_catalog import AgentCatalogPort
from aiops_agent.ports.diagnostic_source import (
    DiagnosticSourceAdapterCatalogPort,
)
from aiops_agent.ports.secret_store import SecretStorePort
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    AgentBindingCreate,
    AgentBindingPatch,
    AgentBindingView,
    HealthCheckReceipt,
    InspectionPlanCreate,
    InspectionPlanDetail,
    InspectionPlanPage,
    InspectionPlanPatch,
    InspectionPlanSummary,
    InspectionTargetCreate,
    InspectionTargetPatch,
    InspectionTargetView,
    SourceBindingCreate,
    SourceBindingPatch,
    SourceBindingView,
    DiagnosticSourceCreate,
    DiagnosticSourceDetail,
    DiagnosticSourcePage,
    DiagnosticSourcePatch,
    DiagnosticSourceSummary,
    PolicyCreate,
    PolicyDetail,
    PolicyPage,
    PolicySummary,
    SecretRefStatus,
    TargetCreate,
    TargetDetail,
    TargetPage,
    TargetPatch,
    TargetSummary,
    WebhookKeyRotation,
)
from platform_core.identity import uuid7
from platform_core.managed_credentials import ManagedCredentialCipher


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
UowFactory = Callable[[], AIOpsUnitOfWork]
CommandHandler = Callable[
    [AIOpsUnitOfWork, datetime], Awaitable[ResponseModel]
]
ResultTransform = Callable[[dict[str, Any]], dict[str, Any]]


class ConfigurationServiceBase:
    def __init__(
        self,
        *,
        uow_factory: UowFactory,
        cursor_codec: SignedCursorCodec,
        secret_store: SecretStorePort,
        agent_catalog: AgentCatalogPort,
        template_registry: InspectionTemplateRegistry,
        management: AIOpsManagementConfig,
        max_inspection_targets: int,
        credential_cipher: ManagedCredentialCipher,
        managed_credential_service: AIOpsManagedCredentialService,
        diagnostic_source_catalog: DiagnosticSourceAdapterCatalogPort | None = None,
    ):
        self._uow_factory = uow_factory
        self._cursor_codec = cursor_codec
        self._secret_store = secret_store
        self._agent_catalog = agent_catalog
        self._template_registry = template_registry
        self._management = management
        self._max_inspection_targets = max_inspection_targets
        self._idempotency = IdempotencyGuard()
        self._credential_cipher = credential_cipher
        self._managed_credentials = managed_credential_service
        self._diagnostic_source_catalog = diagnostic_source_catalog

    async def _validate_secret_refs(self, *references: str | None) -> None:
        for reference in references:
            if reference is not None:
                await self._secret_store.validate_ref(reference)

    async def _idempotent(
        self,
        *,
        scope: ConfigurationScope,
        operation: str,
        parent_resource: str,
        idempotency_key: str,
        payload: Any,
        response_type: type[ResponseModel],
        handler: CommandHandler[ResponseModel],
        store_transform: ResultTransform | None = None,
        replay_transform: ResultTransform | None = None,
    ) -> ResponseModel:
        if not idempotency_key or len(idempotency_key) > 256:
            raise AIOpsApplicationError(
                code="OPS_IDEMPOTENCY_KEY_REQUIRED",
                message="POST 操作必须提供有效 Idempotency-Key",
                status_code=400,
            )
        message_key = self._idempotency.message_key(
            scope=scope,
            operation=operation,
            parent_resource=parent_resource,
            idempotency_key=idempotency_key,
        )
        # 口令请求不能留下可离线猜测的原始摘要。
        payload_hash = hmac.new(
            self._credential_cipher.fingerprint_key,
            canonical_json(payload).encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        try:
            async with self._uow_factory() as uow:
                replay = await self._idempotency.replay(
                    uow=uow,
                    message_key=message_key,
                    payload_hash=payload_hash,
                )
                if replay is not None:
                    if replay_transform is not None:
                        replay = replay_transform(replay)
                    return response_type.model_validate(replay)
                now = datetime.now(UTC)
                response = await handler(uow, now)
                result = response.model_dump(mode="json")
                stored_result = (
                    store_transform(result)
                    if store_transform is not None
                    else result
                )
                await self._idempotency.record(
                    uow=uow,
                    message_key=message_key,
                    operation=operation,
                    payload_hash=payload_hash,
                    result=stored_result,
                    now=now,
                )
                await uow.commit()
                return response
        except IntegrityError as exc:
            database_error = str(getattr(exc, "orig", exc))
            is_unique_conflict = (
                "ORA-00001" in database_error
                or "unique constraint" in database_error.lower()
            )
            if is_unique_conflict:
                async with self._uow_factory() as retry_uow:
                    replay = await self._idempotency.replay(
                        uow=retry_uow,
                        message_key=message_key,
                        payload_hash=payload_hash,
                    )
                    if replay is not None:
                        if replay_transform is not None:
                            replay = replay_transform(replay)
                        return response_type.model_validate(replay)
                raise state_conflict("配置自然键已存在或并发创建冲突") from exc
            logger.error(
                "AIOps 配置持久化约束失败：operation={} database_error={}",
                operation,
                database_error,
            )
            raise AIOpsApplicationError(
                code="OPS_PERSISTENCE_CONSTRAINT_FAILED",
                message="配置保存未满足数据库约束，请检查 AIOps Schema 与 Domain 配置",
                status_code=500,
            ) from exc

    @staticmethod
    def _check_version(actual: int, expected: int) -> None:
        if int(actual) != expected:
            raise row_version_changed()
