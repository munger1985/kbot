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
    MonitorSourceEntity,
    PolicyEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetMonitorEntity,
)
from aiops_agent.persistence import AIOpsUnitOfWork
from aiops_agent.ports.agent_runtime import AgentRuntimePort
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
    MonitorBindingCreate,
    MonitorBindingPatch,
    MonitorBindingView,
    MonitorSourceCreate,
    MonitorSourceDetail,
    MonitorSourcePage,
    MonitorSourcePatch,
    MonitorSourceSummary,
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
        agent_runtime: AgentRuntimePort,
        template_registry: InspectionTemplateRegistry,
        management: AIOpsManagementConfig,
        max_inspection_targets: int,
    ):
        self._uow_factory = uow_factory
        self._cursor_codec = cursor_codec
        self._secret_store = secret_store
        self._agent_runtime = agent_runtime
        self._template_registry = template_registry
        self._management = management
        self._max_inspection_targets = max_inspection_targets
        self._idempotency = IdempotencyGuard()

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
        payload_hash = sha256_json(payload)
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

    @staticmethod
    def _check_version(actual: int, expected: int) -> None:
        if int(actual) != expected:
            raise row_version_changed()
