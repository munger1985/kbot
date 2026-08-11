"""Monitor Source 与 Target Monitor Binding 配置用例。"""

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

from .projections import (
    _target_detail,
    _target_summary,
    _agent_binding_view,
    _monitor_detail,
    _monitor_summary,
    _monitor_binding_view,
    _policy_detail,
    _policy_summary,
    _inspection_detail,
    _inspection_summary,
    _inspection_target_view,
)


class MonitorConfigurationMixin:
    async def create_monitor_source(
        self,
        *,
        scope: ConfigurationScope,
        request: MonitorSourceCreate,
        idempotency_key: str,
    ) -> MonitorSourceDetail:
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> MonitorSourceDetail:
            assert uow.monitor_sources is not None
            assert uow.managed_credentials is not None
            source_id = uuid7()
            secret_ref = webhook_secret_ref = None
            for kind, values in (
                ("monitor_source", request.credentials),
                ("monitor_webhook", request.webhook_credentials),
            ):
                if values is None:
                    continue
                credential = await self._managed_credentials.put(
                    uow=uow,
                    domain_id=scope.domain_id,
                    external_key=source_id,
                    credential_kind=kind,
                    values=values,
                    actor_id=scope.actor_id,
                )
                reference = self._managed_credentials.reference(
                    domain_id=scope.domain_id,
                    external_key=source_id,
                    credential_kind=kind,
                    credential_id=credential.credential_id,
                )
                if kind == "monitor_source":
                    secret_ref = reference
                else:
                    webhook_secret_ref = reference
            entity = MonitorSourceEntity(
                monitor_source_id=source_id,
                domain_id=scope.domain_id,
                display_name=request.display_name,
                source_type=request.source_type,
                endpoint=str(request.endpoint),
                secret_ref=secret_ref,
                webhook_secret_ref=webhook_secret_ref,
                tls_profile_ref=None,
                capabilities_json=request.capabilities,
                status="DISABLED",
                health_status="UNKNOWN",
                row_version=1,
                health_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.monitor_sources.add(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="MONITOR_SOURCE",
                aggregate_id=entity.monitor_source_id,
                event_type="MONITOR_SOURCE_CREATED",
                row_version=1,
            )
            return _monitor_detail(entity)

        return await self._idempotent(
            scope=scope,
            operation="MONITOR_SOURCE_CREATE",
            parent_resource="monitor-sources",
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=MonitorSourceDetail,
            handler=handler,
        )

    async def get_monitor_source(
        self, *, scope: ConfigurationScope, source_id: UUID
    ) -> MonitorSourceDetail:
        async with self._uow_factory() as uow:
            assert uow.monitor_sources is not None
            entity = await uow.monitor_sources.get_scoped(
                monitor_source_id=source_id,
                domain_id=scope.domain_id,
            )
            if entity is None:
                raise resource_not_found("Monitor Source")
            return _monitor_detail(entity)

    async def list_monitor_sources(
        self,
        *,
        scope: ConfigurationScope,
        status: str | None,
        cursor: str | None,
        limit: int,
    ) -> MonitorSourcePage:
        if status is not None and status not in {"ACTIVE", "DISABLED"}:
            raise validation_failed("Monitor Source status 过滤条件无效")
        filters = {"status": status}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._cursor_codec.decode(
                token=cursor, scope=scope, filters=filters
            )
        async with self._uow_factory() as uow:
            assert uow.monitor_sources is not None
            entities = await uow.monitor_sources.page_scoped(
                domain_id=scope.domain_id,
                statuses=(status,) if status else None,
                before_updated_at=before_at,
                before_id=before_id,
                limit=limit + 1,
            )
            page_entities = entities[:limit]
            next_cursor = None
            if len(entities) > limit and page_entities:
                last = page_entities[-1]
                next_cursor = self._cursor_codec.encode(
                    scope=scope,
                    updated_at=last.updated_at,
                    resource_id=last.monitor_source_id,
                    filters=filters,
                )
            return MonitorSourcePage(
                items=tuple(
                    _monitor_summary(item) for item in page_entities
                ),
                next_cursor=next_cursor,
                has_more=len(entities) > limit,
            )

    async def patch_monitor_source(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        request: MonitorSourcePatch,
        expected_version: int,
    ) -> MonitorSourceDetail:
        fields = request.model_dump(exclude_unset=True, mode="json")
        fields.pop("schema_version", None)
        if not fields:
            raise validation_failed("PATCH 至少需要一个可修改字段")
        credential_updates = {
            "monitor_source": fields.pop("credentials", ...),
            "monitor_webhook": fields.pop("webhook_credentials", ...),
        }
        if "endpoint" in fields:
            if request.endpoint is None:
                raise validation_failed("Monitor Endpoint 不能为空")
            fields["endpoint"] = str(request.endpoint)
        connectivity_changed = bool(
            {
                "endpoint",
                "credentials",
                "webhook_credentials",
            }
            & request.model_fields_set
        )
        if "capabilities" in fields:
            fields["capabilities_json"] = fields.pop("capabilities")
        async with self._uow_factory() as uow:
            assert uow.monitor_sources is not None
            assert uow.managed_credentials is not None
            entity = await uow.monitor_sources.get_scoped(
                monitor_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Monitor Source")
            self._check_version(entity.row_version, expected_version)
            for kind, values in credential_updates.items():
                if values is ...:
                    continue
                field = (
                    "secret_ref"
                    if kind == "monitor_source"
                    else "webhook_secret_ref"
                )
                current_ref = getattr(entity, field)
                if values is None:
                    if current_ref:
                        _, _, _, credential_id = (
                            self._managed_credentials.parse_reference(
                                current_ref
                            )
                        )
                        await self._managed_credentials.revoke(
                            uow=uow,
                            domain_id=scope.domain_id,
                            credential_id=credential_id,
                            credential_kind=kind,
                            actor_id=scope.actor_id,
                        )
                    setattr(entity, field, None)
                    continue
                credential = await self._managed_credentials.put(
                    uow=uow,
                    domain_id=scope.domain_id,
                    external_key=source_id,
                    credential_kind=kind,
                    values=values,
                    actor_id=scope.actor_id,
                )
                setattr(
                    entity,
                    field,
                    self._managed_credentials.reference(
                        domain_id=scope.domain_id,
                        external_key=source_id,
                        credential_kind=kind,
                        credential_id=credential.credential_id,
                    ),
                )
            for name, value in fields.items():
                setattr(entity, name, value)
            if connectivity_changed:
                entity.status = "DISABLED"
                entity.health_status = "UNKNOWN"
                entity.health_version = int(entity.health_version) + 1
                entity.health_check_request_id = None
                entity.health_check_requested_at = None
                entity.last_health_check_at = None
                entity.last_error_code = None
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now(UTC)
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="MONITOR_SOURCE",
                aggregate_id=source_id,
                event_type="MONITOR_SOURCE_UPDATED",
                row_version=int(entity.row_version),
            )
            response = _monitor_detail(entity)
            await uow.commit()
            return response

    async def command_monitor_source(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        command: str,
        expected_version: int,
        idempotency_key: str,
    ) -> MonitorSourceDetail:
        destination = {"enable": "ACTIVE", "disable": "DISABLED"}.get(command)
        if destination is None:
            raise validation_failed("未知 Monitor Source 命令")

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> MonitorSourceDetail:
            assert uow.monitor_sources is not None
            entity = await uow.monitor_sources.get_scoped(
                monitor_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Monitor Source")
            self._check_version(entity.row_version, expected_version)
            if entity.status == destination:
                raise state_conflict(f"Monitor Source 已处于 {destination}")
            if destination == "ACTIVE" and entity.health_status != "HEALTHY":
                raise validation_failed("启用前必须通过至少一次健康检查")
            entity.status = destination
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="MONITOR_SOURCE",
                aggregate_id=source_id,
                event_type=f"MONITOR_SOURCE_{destination}",
                row_version=int(entity.row_version),
            )
            return _monitor_detail(entity)

        return await self._idempotent(
            scope=scope,
            operation=f"MONITOR_SOURCE_{command.upper()}",
            parent_resource=str(source_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=MonitorSourceDetail,
            handler=handler,
        )

    async def delete_monitor_source(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        expected_version: int,
        idempotency_key: str,
    ) -> MonitorSourceDetail:
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> MonitorSourceDetail:
            assert uow.monitor_sources is not None
            assert uow.managed_credentials is not None
            entity = await uow.monitor_sources.get_scoped(
                monitor_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Monitor Source")
            self._check_version(entity.row_version, expected_version)
            if entity.status != "DISABLED":
                raise state_conflict("仅允许删除已停用的 Monitor Source")

            result = _monitor_detail(entity)
            for reference, kind in (
                (entity.secret_ref, "monitor_source"),
                (entity.webhook_secret_ref, "monitor_webhook"),
            ):
                if reference:
                    _, _, _, credential_id = (
                        self._managed_credentials.parse_reference(reference)
                    )
                    await self._managed_credentials.revoke(
                        uow=uow,
                        domain_id=scope.domain_id,
                        credential_id=credential_id,
                        credential_kind=kind,
                        actor_id=scope.actor_id,
                    )
            try:
                await uow.monitor_sources.delete_source(entity)
            except IntegrityError as exc:
                raise state_conflict(
                    "Monitor Source 仍有关联的 Target、Agent 或运行历史，不能删除"
                ) from exc
            return result

        return await self._idempotent(
            scope=scope,
            operation="MONITOR_SOURCE_DELETE",
            parent_resource=str(source_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=MonitorSourceDetail,
            handler=handler,
        )

    async def request_monitor_health_check(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        expected_version: int,
        idempotency_key: str,
    ) -> HealthCheckReceipt:
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> HealthCheckReceipt:
            assert uow.monitor_sources is not None
            entity = await uow.monitor_sources.get_scoped(
                monitor_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Monitor Source")
            self._check_version(entity.row_version, expected_version)
            request_id = uuid7()
            entity.health_check_request_id = request_id
            entity.health_check_requested_at = now
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="MONITOR_SOURCE",
                aggregate_id=source_id,
                event_type="MONITOR_HEALTH_CHECK_REQUESTED",
                row_version=int(entity.row_version),
                details={
                    "health_check_request_id": str(request_id),
                    "health_version": int(entity.health_version),
                },
            )
            return HealthCheckReceipt(
                source_id=source_id,
                request_id=request_id,
                accepted_at=now,
                config_row_version=int(entity.row_version),
                health_version=int(entity.health_version),
            )

        return await self._idempotent(
            scope=scope,
            operation="MONITOR_HEALTH_CHECK",
            parent_resource=str(source_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=HealthCheckReceipt,
            handler=handler,
        )

    async def rotate_webhook_key(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        expected_version: int,
        idempotency_key: str,
    ) -> WebhookKeyRotation:
        key_secret = os.getenv(self._management.webhook_key_secret_env, "")
        if len(key_secret.encode("utf-8")) < 32:
            raise AIOpsApplicationError(
                code="OPS_WEBHOOK_KEY_SECRET_UNAVAILABLE",
                message="Webhook Key 派生密钥未配置",
                status_code=503,
                retryable=True,
            )
        seed = canonical_json(
            {
                "source_id": str(source_id),
                "domain_id": scope.domain_id,
                "principal": scope.principal_id,
                "idempotency_key": idempotency_key,
            }
        )
        material = hmac.new(
            key_secret.encode("utf-8"),
            seed.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        webhook_key = f"whk-{material}"

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> WebhookKeyRotation:
            assert uow.monitor_sources is not None
            entity = await uow.monitor_sources.get_scoped(
                monitor_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Monitor Source")
            self._check_version(entity.row_version, expected_version)
            expires_at = (
                now
                + timedelta(
                    seconds=self._management.webhook_key_overlap_seconds
                )
                if entity.webhook_key_hash
                and self._management.webhook_key_overlap_seconds > 0
                else None
            )
            entity.previous_webhook_key_hash = entity.webhook_key_hash
            entity.previous_webhook_key_expires_at = expires_at
            entity.webhook_key_hash = hashlib.sha256(
                webhook_key.encode("utf-8")
            ).hexdigest()
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="MONITOR_SOURCE",
                aggregate_id=source_id,
                event_type="MONITOR_WEBHOOK_KEY_ROTATED",
                row_version=int(entity.row_version),
                details={
                    "fingerprint": entity.webhook_key_hash[:16],
                    "previous_key_expires_at": (
                        expires_at.isoformat() if expires_at else None
                    ),
                },
            )
            logger.info("监控源 Webhook Key 已轮换：source_id={}", source_id)
            return WebhookKeyRotation(
                source_id=source_id,
                webhook_key=webhook_key,
                previous_key_expires_at=expires_at,
                created_at=now,
            )

        return await self._idempotent(
            scope=scope,
            operation="MONITOR_WEBHOOK_KEY_ROTATE",
            parent_resource=str(source_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=WebhookKeyRotation,
            handler=handler,
            store_transform=lambda result: {
                name: value
                for name, value in result.items()
                if name != "webhook_key"
            },
            replay_transform=lambda result: {
                **result,
                "webhook_key": webhook_key,
            },
        )

    async def create_monitor_binding(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        request: MonitorBindingCreate,
        idempotency_key: str,
    ) -> MonitorBindingView:
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> MonitorBindingView:
            assert uow.targets is not None
            assert uow.monitor_sources is not None
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if target is None:
                raise resource_not_found("Target")
            source = await uow.monitor_sources.get_scoped(
                monitor_source_id=request.source_id,
                domain_id=scope.domain_id,
            )
            if source is None:
                raise resource_not_found("Monitor Source")
            entity = TargetMonitorEntity(
                target_monitor_id=uuid7(),
                target_id=target_id,
                monitor_source_id=request.source_id,
                external_target_key=request.external_target_key,
                role=request.role,
                priority=request.priority,
                metric_scope_json=request.metric_scope,
                mapping_overrides_json=request.mapping_overrides,
                status="ACTIVE",
                health_status="UNKNOWN",
                row_version=1,
                health_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.targets.add_monitor(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET_MONITOR",
                aggregate_id=entity.target_monitor_id,
                event_type="TARGET_MONITOR_CREATED",
                row_version=1,
                details={"target_id": str(target_id)},
            )
            return _monitor_binding_view(entity)

        return await self._idempotent(
            scope=scope,
            operation="TARGET_MONITOR_CREATE",
            parent_resource=str(target_id),
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=MonitorBindingView,
            handler=handler,
        )

    async def list_monitor_bindings(
        self, *, scope: ConfigurationScope, target_id: UUID
    ) -> tuple[MonitorBindingView, ...]:
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
            )
            if target is None:
                raise resource_not_found("Target")
            entities = await uow.targets.list_monitors(
                target_id=target_id,
                domain_id=scope.domain_id,
                active_only=False,
            )
            return tuple(_monitor_binding_view(item) for item in entities)

    async def patch_monitor_binding(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        binding_id: UUID,
        request: MonitorBindingPatch,
        expected_version: int,
    ) -> MonitorBindingView:
        fields = request.model_dump(exclude_unset=True, mode="json")
        fields.pop("schema_version", None)
        if not fields:
            raise validation_failed("PATCH 至少需要一个可修改字段")
        if "metric_scope" in fields:
            fields["metric_scope_json"] = fields.pop("metric_scope")
        if "mapping_overrides" in fields:
            fields["mapping_overrides_json"] = fields.pop("mapping_overrides")
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            entity = await uow.targets.get_monitor_scoped(
                target_monitor_id=binding_id,
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Monitor Binding")
            self._check_version(entity.row_version, expected_version)
            for name, value in fields.items():
                setattr(entity, name, value)
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now(UTC)
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET_MONITOR",
                aggregate_id=binding_id,
                event_type="TARGET_MONITOR_UPDATED",
                row_version=int(entity.row_version),
            )
            response = _monitor_binding_view(entity)
            await uow.commit()
            return response

    async def command_monitor_binding(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        binding_id: UUID,
        command: str,
        expected_version: int,
        idempotency_key: str,
    ) -> MonitorBindingView:
        destination = {"enable": "ACTIVE", "disable": "DISABLED"}.get(command)
        if destination is None:
            raise validation_failed("未知 Monitor Binding 命令")

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> MonitorBindingView:
            assert uow.targets is not None
            entity = await uow.targets.get_monitor_scoped(
                target_monitor_id=binding_id,
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Monitor Binding")
            self._check_version(entity.row_version, expected_version)
            if entity.status == destination:
                raise state_conflict(f"Monitor Binding 已处于 {destination}")
            entity.status = destination
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET_MONITOR",
                aggregate_id=binding_id,
                event_type=f"TARGET_MONITOR_{destination}",
                row_version=int(entity.row_version),
            )
            return _monitor_binding_view(entity)

        return await self._idempotent(
            scope=scope,
            operation=f"TARGET_MONITOR_{command.upper()}",
            parent_resource=str(binding_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=MonitorBindingView,
            handler=handler,
        )
