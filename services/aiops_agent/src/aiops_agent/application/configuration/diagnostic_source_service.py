"""Diagnostic Source 与 Target Source Binding 配置用例。"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

from loguru import logger
from sqlalchemy.exc import IntegrityError

from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    add_configuration_event,
    canonical_json,
)
from aiops_agent.application.errors import (
    AIOpsApplicationError,
    resource_not_found,
    state_conflict,
    validation_failed,
)
from aiops_agent.application.configuration.source_connection_test import (
    run_diagnostic_source_health_check,
)
from aiops_agent.adapters.diagnostic_sources.base import (
    DiagnosticSourceAdapterError,
)
from aiops_agent.entities import (
    DiagnosticSourceEntity,
    TargetSourceBindingEntity,
)
from aiops_agent.domain.evidence import validate_event_class_map
from aiops_agent.persistence import AIOpsUnitOfWork
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_HEALTH_CHECK,
    CAPABILITY_INSTANCE_DISCOVERY,
    CAPABILITY_LOG_QUERY,
    DiagnosticSourceContext,
    InstanceDiscoveryRequest as ProviderInstanceDiscoveryRequest,
    LogSourceLocator,
)
from platform_core.contracts.aiops import (
    SourceBindingCreate,
    SourceBindingPatch,
    SourceBindingView,
    DiagnosticSourceCreate,
    DiagnosticSourceDetail,
    DiagnosticSourcePage,
    DiagnosticSourcePatch,
    DiagnosticSourceSummary,
    InstanceDiscoveryCandidate,
    InstanceDiscoveryPage,
    InstanceDiscoveryRequest,
    InstanceMappingRequest,
    InstanceMappingResult,
    InstanceMappingView,
    WebhookKeyRotation,
)
from platform_core.identity import uuid7

from .projections import (
    _diagnostic_source_detail,
    _diagnostic_source_summary,
    _source_binding_view,
)


class DiagnosticSourceConfigurationMixin:
    @staticmethod
    def _locator_hint(locator_key: str) -> str:
        if len(locator_key) <= 4:
            return "*" * len(locator_key)
        return f"{locator_key[:2]}***{locator_key[-2:]}"

    @classmethod
    def _instance_mapping_view(cls, entity) -> InstanceMappingView:
        return InstanceMappingView(
            binding_id=entity.target_source_binding_id,
            target_id=entity.target_id,
            source_id=entity.diagnostic_source_id,
            locator_hint=cls._locator_hint(entity.source_locator_key),
            status=entity.status,
            health_status=entity.health_status,
            row_version=int(entity.row_version),
        )

    async def _instance_discovery_adapter(
        self, *, uow, scope: ConfigurationScope, source
    ):
        if source.source_type not in {"PROMETHEUS", "ZABBIX"}:
            raise validation_failed("该 Diagnostic Source 不支持实例发现")
        if source.status != "ENABLED":
            raise state_conflict("Diagnostic Source 未启用，不能发现实例")
        if source.connectivity_status != "CONNECTED":
            raise state_conflict("Diagnostic Source 尚未通过连接验证")
        if self._diagnostic_source_registry is None:
            raise validation_failed("Diagnostic Source Adapter 注册表不可用")
        credentials: dict[str, object] = {}
        if source.auth_credential_id is not None:
            credentials = await self._managed_credentials.read(
                uow=uow,
                domain_id=scope.domain_id,
                credential_id=source.auth_credential_id,
                credential_kind="diagnostic_source",
                external_key=source.diagnostic_source_id,
            )
        normalized_credentials = {
            str(key): str(value) for key, value in credentials.items()
        }
        if "value" in normalized_credentials:
            normalized_credentials["token"] = normalized_credentials["value"]
        try:
            return self._diagnostic_source_registry.create(
                DiagnosticSourceContext(
                    source_id=str(source.diagnostic_source_id),
                    source_type=source.source_type,
                    adapter_id=source.adapter_id,
                    adapter_version=source.adapter_version,
                    config_version=int(source.row_version),
                    endpoint=source.endpoint,
                    credentials=normalized_credentials,
                    declared_capabilities=dict(
                        source.declared_capabilities_json or {}
                    ),
                    config=dict(source.config_json or {}),
                ),
                capability=CAPABILITY_INSTANCE_DISCOVERY,
            )
        except LookupError as exc:
            raise validation_failed(str(exc)) from exc
    @staticmethod
    def _apply_diagnostic_source_health_result(
        *, entity, result, checked_at: datetime
    ) -> None:
        entity.connectivity_status = (
            "CONNECTED" if result.ok else "UNREACHABLE"
        )
        entity.connectivity_check_request_id = None
        entity.connectivity_check_requested_at = None
        entity.last_connectivity_check_at = checked_at
        if result.ok:
            entity.last_connectivity_success_at = checked_at
        entity.last_error_code = result.error_code
        entity.discovered_capabilities_json = {
            capability: {
                "adapter_id": entity.adapter_id,
                "adapter_version": entity.adapter_version,
            }
            for capability in result.discovered_capabilities
        }

    async def _check_diagnostic_source_now(
        self,
        *,
        uow: AIOpsUnitOfWork,
        scope: ConfigurationScope,
        entity,
        credentials: dict[str, object] | None = None,
    ) -> None:
        if credentials is None:
            credential_id = (
                entity.auth_credential_id
                if entity.endpoint
                else entity.webhook_credential_id
            )
            credential_kind = (
                "diagnostic_source" if entity.endpoint else "source_webhook"
            )
            credentials = (
                await self._managed_credentials.read(
                    uow=uow,
                    domain_id=scope.domain_id,
                    credential_id=credential_id,
                    credential_kind=credential_kind,
                    external_key=entity.diagnostic_source_id,
                )
                if credential_id is not None
                else {}
            )
        result = await run_diagnostic_source_health_check(
            source_id=str(entity.diagnostic_source_id),
            source_type=entity.source_type,
            adapter_id=entity.adapter_id,
            adapter_version=entity.adapter_version,
            config_version=int(entity.row_version),
            endpoint=entity.endpoint,
            credentials=credentials,
            declared_capabilities=dict(
                entity.declared_capabilities_json or {}
            ),
            config=dict(entity.config_json or {}),
            trace_id=scope.trace_id,
            diagnostic_source_registry=self._diagnostic_source_registry,
        )
        self._apply_diagnostic_source_health_result(
            entity=entity,
            result=result,
            checked_at=datetime.now(UTC),
        )

    def _normalize_source_config(
        self, *, source_type: str, config: dict[str, object]
    ) -> dict[str, object]:
        catalog = self._diagnostic_source_catalog
        if catalog is None:
            raise validation_failed("Diagnostic Source Adapter 目录不可用")
        try:
            return catalog.normalize_config(
                source_type=source_type, config=config
            )
        except ValueError as exc:
            raise validation_failed(str(exc)) from exc

    def _resolve_source_descriptor(self, *, source_type: str):
        catalog = self._diagnostic_source_catalog
        if catalog is None:
            raise validation_failed("Diagnostic Source Adapter 目录不可用")
        try:
            return catalog.describe_source_type(source_type=source_type)
        except LookupError as exc:
            raise validation_failed(str(exc)) from exc

    @staticmethod
    def _validate_source_binding_locator(
        *, source, source_locator: dict[str, object]
    ) -> None:
        if CAPABILITY_LOG_QUERY not in (
            source.declared_capabilities_json or {}
        ):
            return
        try:
            LogSourceLocator.model_validate(source_locator)
        except ValueError as exc:
            raise validation_failed(
                "日志 Source Binding 必须提供受控 labels 定位"
            ) from exc

    def _validate_diagnostic_source_adapter(
        self,
        *,
        source_type: str,
        adapter_id: str,
        adapter_version: str,
        declared_capabilities: dict[str, object],
    ) -> None:
        catalog = self._diagnostic_source_catalog
        if catalog is None:
            return
        try:
            descriptor = catalog.describe(
                adapter_id=adapter_id,
                adapter_version=adapter_version,
            )
        except LookupError as exc:
            raise validation_failed(str(exc)) from exc
        if source_type not in descriptor.source_types:
            raise validation_failed(
                f"Adapter {adapter_id} 不支持 Source Type {source_type}"
            )
        unsupported = sorted(
            set(declared_capabilities) - descriptor.capabilities
        )
        if unsupported:
            raise validation_failed(
                "Diagnostic Source 声明了 Adapter 不支持的能力："
                + ", ".join(unsupported)
            )

    async def create_diagnostic_source(
        self,
        *,
        scope: ConfigurationScope,
        request: DiagnosticSourceCreate,
        idempotency_key: str,
    ) -> DiagnosticSourceDetail:
        descriptor = self._resolve_source_descriptor(
            source_type=request.source_type
        )
        declared_capabilities = {
            capability: {}
            for capability in descriptor.capabilities
            if capability != CAPABILITY_HEALTH_CHECK
        }
        config = self._normalize_source_config(
            source_type=request.source_type,
            config=dict(request.config),
        )

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> DiagnosticSourceDetail:
            assert uow.diagnostic_sources is not None
            assert uow.managed_credentials is not None
            source_id = uuid7()
            auth_credential_id = webhook_credential_id = None
            for kind, values in (
                ("diagnostic_source", request.credentials),
                ("source_webhook", request.webhook_credentials),
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
                if kind == "diagnostic_source":
                    auth_credential_id = credential.credential_id
                else:
                    webhook_credential_id = credential.credential_id
            entity = DiagnosticSourceEntity(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
                display_name=request.display_name,
                source_type=request.source_type,
                adapter_id=descriptor.adapter_id,
                adapter_version=descriptor.adapter_version,
                endpoint=(str(request.endpoint) if request.endpoint else None),
                auth_credential_id=auth_credential_id,
                webhook_credential_id=webhook_credential_id,
                tls_profile_ref=None,
                declared_capabilities_json=declared_capabilities,
                discovered_capabilities_json=None,
                config_json=config,
                status="DISABLED",
                connectivity_status="UNKNOWN",
                connectivity_check_request_id=None,
                connectivity_check_requested_at=None,
                row_version=1,
                connectivity_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await self._check_diagnostic_source_now(
                uow=uow,
                scope=scope,
                entity=entity,
                credentials=dict(
                    (
                        request.credentials
                        if request.endpoint
                        else request.webhook_credentials
                    )
                    or {}
                ),
            )
            await uow.diagnostic_sources.add(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="DIAGNOSTIC_SOURCE",
                aggregate_id=entity.diagnostic_source_id,
                event_type="DIAGNOSTIC_SOURCE_CREATED",
                row_version=1,
            )
            return _diagnostic_source_detail(entity)

        return await self._idempotent(
            scope=scope,
            operation="DIAGNOSTIC_SOURCE_CREATE",
            parent_resource="diagnostic-sources",
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=DiagnosticSourceDetail,
            handler=handler,
        )

    async def get_diagnostic_source(
        self, *, scope: ConfigurationScope, source_id: UUID
    ) -> DiagnosticSourceDetail:
        async with self._uow_factory() as uow:
            assert uow.diagnostic_sources is not None
            entity = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
            )
            if entity is None:
                raise resource_not_found("Diagnostic Source")
            return _diagnostic_source_detail(entity)

    async def list_diagnostic_sources(
        self,
        *,
        scope: ConfigurationScope,
        status: str | None,
        cursor: str | None,
        limit: int,
    ) -> DiagnosticSourcePage:
        if status is not None and status not in {"ENABLED", "DISABLED"}:
            raise validation_failed("Diagnostic Source status 过滤条件无效")
        filters = {"status": status}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._cursor_codec.decode(
                token=cursor, scope=scope, filters=filters
            )
        async with self._uow_factory() as uow:
            assert uow.diagnostic_sources is not None
            entities = await uow.diagnostic_sources.page_scoped(
                domain_id=scope.domain_id,
                statuses=(status,) if status else None,
                before_created_at=before_at,
                before_id=before_id,
                limit=limit + 1,
            )
            page_entities = entities[:limit]
            next_cursor = None
            if len(entities) > limit and page_entities:
                last = page_entities[-1]
                next_cursor = self._cursor_codec.encode(
                    scope=scope,
                    sort_at=last.created_at,
                    resource_id=last.diagnostic_source_id,
                    filters=filters,
                )
            return DiagnosticSourcePage(
                items=tuple(
                    _diagnostic_source_summary(item) for item in page_entities
                ),
                next_cursor=next_cursor,
                has_more=len(entities) > limit,
            )

    async def patch_diagnostic_source(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        request: DiagnosticSourcePatch,
        expected_version: int,
    ) -> DiagnosticSourceDetail:
        fields = request.model_dump(exclude_unset=True, mode="json")
        fields.pop("schema_version", None)
        if not fields:
            raise validation_failed("PATCH 至少需要一个可修改字段")
        credential_updates = {
            "diagnostic_source": fields.pop("credentials", ...),
            "source_webhook": fields.pop("webhook_credentials", ...),
        }
        if "config" in fields:
            fields["config_json"] = fields.pop("config")
        if "endpoint" in fields:
            fields["endpoint"] = (
                str(request.endpoint) if request.endpoint is not None else None
            )
        connectivity_changed = bool(
            {
                "endpoint",
                "credentials",
                "webhook_credentials",
                "config",
            }
            & request.model_fields_set
        )
        async with self._uow_factory() as uow:
            assert uow.diagnostic_sources is not None
            assert uow.managed_credentials is not None
            entity = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Diagnostic Source")
            self._check_version(entity.row_version, expected_version)
            if (
                entity.source_type not in {"ALERTMANAGER", "ZABBIX"}
                and credential_updates["source_webhook"] is not ...
            ):
                raise validation_failed(
                    "只有 Alertmanager 或 Zabbix 可以配置 Webhook 凭据"
                )
            if "config_json" in fields:
                fields["config_json"] = self._normalize_source_config(
                    source_type=entity.source_type,
                    config=fields["config_json"],
                )
            self._validate_diagnostic_source_adapter(
                source_type=entity.source_type,
                adapter_id=entity.adapter_id,
                adapter_version=entity.adapter_version,
                declared_capabilities=entity.declared_capabilities_json,
            )
            for kind, values in credential_updates.items():
                if values is ...:
                    continue
                field = (
                    "auth_credential_id"
                    if kind == "diagnostic_source"
                    else "webhook_credential_id"
                )
                current_id = getattr(entity, field)
                if values is None:
                    if current_id:
                        await self._managed_credentials.revoke(
                            uow=uow,
                            domain_id=scope.domain_id,
                            credential_id=current_id,
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
                setattr(entity, field, credential.credential_id)
            for name, value in fields.items():
                setattr(entity, name, value)
            if entity.endpoint is None and entity.webhook_credential_id is None:
                raise validation_failed(
                    "Diagnostic Source 必须保留 Endpoint 或 Webhook 凭据"
                )
            if connectivity_changed:
                entity.status = "DISABLED"
                entity.connectivity_version = (
                    int(entity.connectivity_version) + 1
                )
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now(UTC)
            if connectivity_changed:
                await self._check_diagnostic_source_now(
                    uow=uow,
                    scope=scope,
                    entity=entity,
                )
                entity.updated_at = datetime.now(UTC)
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="DIAGNOSTIC_SOURCE",
                aggregate_id=source_id,
                event_type="DIAGNOSTIC_SOURCE_UPDATED",
                row_version=int(entity.row_version),
            )
            response = _diagnostic_source_detail(entity)
            await uow.commit()
            return response

    async def command_diagnostic_source(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        command: str,
        expected_version: int,
        idempotency_key: str,
    ) -> DiagnosticSourceDetail:
        destination = {"enable": "ENABLED", "disable": "DISABLED"}.get(command)
        if destination is None:
            raise validation_failed("未知 Diagnostic Source 命令")

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> DiagnosticSourceDetail:
            assert uow.diagnostic_sources is not None
            entity = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Diagnostic Source")
            self._check_version(entity.row_version, expected_version)
            if entity.status == destination:
                raise state_conflict(f"Diagnostic Source 已处于 {destination}")
            if (
                destination == "ENABLED"
                and entity.connectivity_status
                not in {"CONNECTED", "DEGRADED"}
            ):
                raise validation_failed("启用前必须通过至少一次连通性检查")
            if (
                destination == "ENABLED"
                and (
                    entity.last_connectivity_success_at is None
                    or entity.last_connectivity_success_at
                    < now - timedelta(hours=2)
                )
            ):
                raise validation_failed("连通性结果已过期，请重新检查")
            entity.status = destination
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="DIAGNOSTIC_SOURCE",
                aggregate_id=source_id,
                event_type=f"DIAGNOSTIC_SOURCE_{destination}",
                row_version=int(entity.row_version),
            )
            return _diagnostic_source_detail(entity)

        return await self._idempotent(
            scope=scope,
            operation=f"DIAGNOSTIC_SOURCE_{command.upper()}",
            parent_resource=str(source_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=DiagnosticSourceDetail,
            handler=handler,
        )

    async def delete_diagnostic_source(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        expected_version: int,
        idempotency_key: str,
    ) -> DiagnosticSourceDetail:
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> DiagnosticSourceDetail:
            assert uow.diagnostic_sources is not None
            assert uow.managed_credentials is not None
            entity = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Diagnostic Source")
            self._check_version(entity.row_version, expected_version)
            if entity.status != "DISABLED":
                raise state_conflict("仅允许删除已停用的 Diagnostic Source")

            result = _diagnostic_source_detail(entity)
            for credential_id, kind in (
                (entity.auth_credential_id, "diagnostic_source"),
                (entity.webhook_credential_id, "source_webhook"),
            ):
                if credential_id:
                    await self._managed_credentials.revoke(
                        uow=uow,
                        domain_id=scope.domain_id,
                        credential_id=credential_id,
                        credential_kind=kind,
                        actor_id=scope.actor_id,
                    )
            try:
                await uow.diagnostic_sources.delete_source(entity)
            except IntegrityError as exc:
                raise state_conflict(
                    "Diagnostic Source 仍有关联的 Target 或运行历史，不能删除"
                ) from exc
            return result

        return await self._idempotent(
            scope=scope,
            operation="DIAGNOSTIC_SOURCE_DELETE",
            parent_resource=str(source_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=DiagnosticSourceDetail,
            handler=handler,
        )

    async def check_diagnostic_source_connectivity(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        expected_version: int,
        idempotency_key: str,
    ) -> DiagnosticSourceDetail:
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> DiagnosticSourceDetail:
            assert uow.diagnostic_sources is not None
            entity = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Diagnostic Source")
            self._check_version(entity.row_version, expected_version)
            await self._check_diagnostic_source_now(
                uow=uow,
                scope=scope,
                entity=entity,
            )
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now(UTC)
            entity.connectivity_version = int(entity.connectivity_version) + 1
            await uow.session.flush()  # type: ignore[union-attr]
            return _diagnostic_source_detail(entity)

        return await self._idempotent(
            scope=scope,
            operation="SOURCE_CONNECTIVITY_CHECK",
            parent_resource=str(source_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=DiagnosticSourceDetail,
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
            assert uow.diagnostic_sources is not None
            entity = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Diagnostic Source")
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
                aggregate_type="DIAGNOSTIC_SOURCE",
                aggregate_id=source_id,
                event_type="SOURCE_WEBHOOK_KEY_ROTATED",
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
            operation="SOURCE_WEBHOOK_KEY_ROTATE",
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

    async def discover_source_instances(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        request: InstanceDiscoveryRequest,
    ) -> InstanceDiscoveryPage:
        filter_types = tuple(sorted(request.db_types))
        after_locator_key = None
        if request.cursor is not None:
            cursor_value = self._instance_candidate_refs.decode(
                request.cursor,
                domain_id=scope.domain_id,
                source_id=source_id,
                actor_id=scope.actor_id,
                purpose="cursor",
            )
            if tuple(cursor_value.get("db_types") or ()) != filter_types:
                raise validation_failed("实例发现游标与当前过滤条件不匹配")
            after_locator_key = str(
                cursor_value.get("after_locator_key") or ""
            )
            if not after_locator_key:
                raise validation_failed("实例发现游标缺少定位信息")
        async with self._uow_factory() as uow:
            assert uow.diagnostic_sources is not None
            assert uow.targets is not None
            source = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
            )
            if source is None:
                raise resource_not_found("Diagnostic Source")
            adapter = await self._instance_discovery_adapter(
                uow=uow, scope=scope, source=source
            )
            try:
                discovered = await adapter.discover_instances(
                    ProviderInstanceDiscoveryRequest(
                        db_types=request.db_types,
                        after_locator_key=after_locator_key,
                        limit=request.page_size + 1,
                        trace_id=scope.trace_id,
                    )
                )
            except DiagnosticSourceAdapterError as exc:
                raise AIOpsApplicationError(
                    code="INSTANCE_DISCOVERY_FAILED",
                    message="监控来源实例发现失败",
                    status_code=502,
                    retryable=exc.retryable,
                ) from exc
            bindings = await uow.targets.list_source_bindings_by_source(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
            )
        mapped_targets = {
            item.source_locator_key: item.target_id for item in bindings
        }
        page_candidates = discovered.candidates[: request.page_size]
        items = tuple(
            InstanceDiscoveryCandidate(
                candidate_ref=self._instance_candidate_refs.encode(
                        domain_id=scope.domain_id,
                    source_id=source_id,
                    actor_id=scope.actor_id,
                    purpose="candidate",
                    value={
                        "source_locator_key": item.source_locator_key,
                        "source_locator": item.source_locator,
                        "db_type": item.db_type,
                    },
                ),
                display_name=item.display_name,
                locator_hint=self._locator_hint(item.source_locator_key),
                db_type=item.db_type,
                mapping_status=(
                    "MAPPED"
                    if item.source_locator_key in mapped_targets
                    else "UNMAPPED"
                ),
                mapped_target_id=mapped_targets.get(item.source_locator_key),
            )
            for item in page_candidates
        )
        has_more = len(discovered.candidates) > request.page_size
        next_cursor = None
        if has_more and page_candidates:
            next_cursor = self._instance_candidate_refs.encode(
                domain_id=scope.domain_id,
                source_id=source_id,
                actor_id=scope.actor_id,
                purpose="cursor",
                value={
                    "after_locator_key": page_candidates[-1].source_locator_key,
                    "db_types": filter_types,
                },
            )
        return InstanceDiscoveryPage(
            source_id=source_id,
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    async def map_source_instances(
        self,
        *,
        scope: ConfigurationScope,
        source_id: UUID,
        request: InstanceMappingRequest,
        idempotency_key: str,
    ) -> InstanceMappingResult:
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> InstanceMappingResult:
            assert uow.diagnostic_sources is not None
            source = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if source is None:
                raise resource_not_found("Diagnostic Source")
            adapter = await self._instance_discovery_adapter(
                uow=uow, scope=scope, source=source
            )
            decoded = []
            for mapping in request.mappings:
                value = self._instance_candidate_refs.decode(
                    mapping.candidate_ref,
                        domain_id=scope.domain_id,
                    source_id=source_id,
                    actor_id=scope.actor_id,
                    purpose="candidate",
                    now=now,
                )
                locator_key = str(value.get("source_locator_key") or "")
                locator = value.get("source_locator")
                db_type = str(value.get("db_type") or "")
                if (
                    not locator_key
                    or not isinstance(locator, dict)
                    or db_type not in {"ORACLE", "MYSQL", "POSTGRESQL"}
                ):
                    raise validation_failed("实例候选引用内容无效")
                decoded.append((mapping, locator_key, locator, db_type))
            locator_keys = [item[1] for item in decoded]
            if len(set(locator_keys)) != len(locator_keys):
                raise validation_failed("批量映射中的外部实例不能重复")
            try:
                verified = await adapter.discover_instances(
                    ProviderInstanceDiscoveryRequest(
                        db_types=tuple(sorted({item[3] for item in decoded})),
                        locator_keys=tuple(locator_keys),
                        limit=len(decoded),
                        trace_id=scope.trace_id,
                    )
                )
            except DiagnosticSourceAdapterError as exc:
                raise AIOpsApplicationError(
                    code="INSTANCE_DISCOVERY_FAILED",
                    message="保存映射前重新验证实例失败",
                    status_code=502,
                    retryable=exc.retryable,
                ) from exc
            verified_by_key = {
                item.source_locator_key: item for item in verified.candidates
            }
            prepared_entities = []
            for mapping, locator_key, locator, db_type in decoded:
                current = verified_by_key.get(locator_key)
                if (
                    current is None
                    or current.db_type != db_type
                    or canonical_json(current.source_locator)
                    != canonical_json(locator)
                ):
                    raise state_conflict(
                        "实例候选已变化或不存在，请重新执行发现"
                    )
                entity = await self._prepare_source_binding_in_uow(
                    uow=uow,
                    scope=scope,
                    target_id=mapping.target_id,
                    request=SourceBindingCreate(
                        source_id=source_id,
                        source_locator_key=locator_key,
                        source_locator=current.source_locator,
                    ),
                    now=now,
                    source=source,
                    expected_db_type=db_type,
                    discovery_verified=True,
                )
                prepared_entities.append(entity)
            for entity in prepared_entities:
                await self._persist_source_binding_in_uow(
                    uow=uow,
                    scope=scope,
                    entity=entity,
                )
            return InstanceMappingResult(
                source_id=source_id,
                items=tuple(
                    self._instance_mapping_view(entity)
                    for entity in prepared_entities
                ),
            )

        return await self._idempotent(
            scope=scope,
            operation="DIAGNOSTIC_SOURCE_INSTANCE_MAPPING_CREATE",
            parent_resource=str(source_id),
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=InstanceMappingResult,
            handler=handler,
        )

    async def _prepare_source_binding_in_uow(
        self,
        *,
        uow: AIOpsUnitOfWork,
        scope: ConfigurationScope,
        target_id: UUID,
        request: SourceBindingCreate,
        now: datetime,
        source=None,
        expected_db_type: str | None = None,
        discovery_verified: bool = False,
    ) -> TargetSourceBindingEntity:
        assert uow.targets is not None
        assert uow.diagnostic_sources is not None
        target = await uow.targets.get_scoped(
            target_id=target_id,
            domain_id=scope.domain_id,
            lock=True,
        )
        if target is None:
            raise resource_not_found("Target")
        if expected_db_type is not None and target.db_type != expected_db_type:
            raise validation_failed("候选实例数据库类型与 Target 不一致")
        if source is None:
            source = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=request.source_id,
                domain_id=scope.domain_id,
            )
        if source is None:
            raise resource_not_found("Diagnostic Source")
        if (
            source.source_type in {"PROMETHEUS", "ZABBIX"}
            and not discovery_verified
        ):
            raise validation_failed(
                "Prometheus 与 Zabbix 映射必须通过实例发现候选创建"
            )
        existing = await uow.targets.list_source_bindings(
            target_id=target_id,
            domain_id=scope.domain_id,
            active_only=False,
        )
        if any(
            binding.diagnostic_source_id == request.source_id
            for binding in existing
        ):
            raise state_conflict("Target 与 Diagnostic Source 已存在映射")
        locator_conflict = (
            await uow.targets.get_source_binding_by_locator_any_status(
                diagnostic_source_id=request.source_id,
                source_locator_key=request.source_locator_key,
                lock=True,
            )
        )
        if locator_conflict is not None:
            raise state_conflict("Diagnostic Source 外部定位键已被使用")
        self._validate_source_binding_locator(
            source=source,
            source_locator=request.source_locator,
        )
        return TargetSourceBindingEntity(
            target_source_binding_id=uuid7(),
            target_id=target_id,
            diagnostic_source_id=request.source_id,
            source_locator_key=request.source_locator_key,
            source_locator_json=request.source_locator,
            role=request.role,
            priority=request.priority,
            capability_scope_json=request.capability_scope,
            mapping_overrides_json=request.mapping_overrides,
            query_budget_json=request.query_budget,
            status="ACTIVE",
            health_status="UNKNOWN",
            row_version=1,
            health_version=1,
            created_by=scope.actor_id,
            updated_by=scope.actor_id,
            created_at=now,
            updated_at=now,
        )

    async def _persist_source_binding_in_uow(
        self,
        *,
        uow: AIOpsUnitOfWork,
        scope: ConfigurationScope,
        entity: TargetSourceBindingEntity,
    ) -> None:
        assert uow.targets is not None
        await uow.targets.add_source_binding(entity)
        await add_configuration_event(
            uow=uow,
            scope=scope,
            aggregate_type="TARGET_SOURCE_BINDING",
            aggregate_id=entity.target_source_binding_id,
            event_type="TARGET_SOURCE_BINDING_CREATED",
            row_version=1,
            details={"target_id": str(entity.target_id)},
        )

    async def _create_source_binding_in_uow(
        self,
        *,
        uow: AIOpsUnitOfWork,
        scope: ConfigurationScope,
        target_id: UUID,
        request: SourceBindingCreate,
        now: datetime,
        source=None,
        expected_db_type: str | None = None,
        discovery_verified: bool = False,
    ) -> TargetSourceBindingEntity:
        entity = await self._prepare_source_binding_in_uow(
            uow=uow,
            scope=scope,
            target_id=target_id,
            request=request,
            now=now,
            source=source,
            expected_db_type=expected_db_type,
            discovery_verified=discovery_verified,
        )
        await self._persist_source_binding_in_uow(
            uow=uow,
            scope=scope,
            entity=entity,
        )
        return entity

    async def create_source_binding(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        request: SourceBindingCreate,
        idempotency_key: str,
    ) -> SourceBindingView:
        try:
            validate_event_class_map(request.mapping_overrides)
        except ValueError as exc:
            raise validation_failed(str(exc)) from exc

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> SourceBindingView:
            entity = await self._create_source_binding_in_uow(
                uow=uow,
                scope=scope,
                target_id=target_id,
                request=request,
                now=now,
            )
            return _source_binding_view(entity)

        return await self._idempotent(
            scope=scope,
            operation="TARGET_SOURCE_BINDING_CREATE",
            parent_resource=str(target_id),
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=SourceBindingView,
            handler=handler,
        )

    async def list_source_bindings(
        self, *, scope: ConfigurationScope, target_id: UUID
    ) -> tuple[SourceBindingView, ...]:
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
            )
            if target is None:
                raise resource_not_found("Target")
            entities = await uow.targets.list_source_bindings(
                target_id=target_id,
                domain_id=scope.domain_id,
                active_only=False,
            )
            return tuple(_source_binding_view(item) for item in entities)

    async def list_source_target_bindings(
        self, *, scope: ConfigurationScope, source_id: UUID
    ) -> tuple[InstanceMappingView, ...]:
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            assert uow.diagnostic_sources is not None
            source = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
            )
            if source is None:
                raise resource_not_found("Diagnostic Source")
            entities = await uow.targets.list_source_bindings_by_source(
                diagnostic_source_id=source_id,
                domain_id=scope.domain_id,
            )
            return tuple(self._instance_mapping_view(item) for item in entities)

    async def patch_source_binding(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        binding_id: UUID,
        request: SourceBindingPatch,
        expected_version: int,
    ) -> SourceBindingView:
        fields = request.model_dump(exclude_unset=True, mode="json")
        fields.pop("schema_version", None)
        if not fields:
            raise validation_failed("PATCH 至少需要一个可修改字段")
        if "source_locator" in fields:
            fields["source_locator_json"] = fields.pop("source_locator")
        if "capability_scope" in fields:
            fields["capability_scope_json"] = fields.pop("capability_scope")
        if "mapping_overrides" in fields:
            try:
                validate_event_class_map(request.mapping_overrides)
            except ValueError as exc:
                raise validation_failed(str(exc)) from exc
            fields["mapping_overrides_json"] = fields.pop("mapping_overrides")
        if "query_budget" in fields:
            fields["query_budget_json"] = fields.pop("query_budget")
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            assert uow.diagnostic_sources is not None
            entity = await uow.targets.get_source_binding_scoped(
                target_source_binding_id=binding_id,
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Source Binding")
            self._check_version(entity.row_version, expected_version)
            if "source_locator_json" in fields or "source_locator_key" in fields:
                source = await uow.diagnostic_sources.get_scoped(
                    diagnostic_source_id=entity.diagnostic_source_id,
                    domain_id=scope.domain_id,
                )
                if source is None:
                    raise resource_not_found("Diagnostic Source")
                if source.source_type in {"PROMETHEUS", "ZABBIX"}:
                    raise validation_failed(
                        "Prometheus 与 Zabbix 外部定位只能通过实例发现重新映射"
                    )
                if "source_locator_json" in fields:
                    self._validate_source_binding_locator(
                        source=source,
                        source_locator=fields["source_locator_json"],
                    )
            if "source_locator_key" in fields:
                locator_conflict = (
                    await uow.targets.get_source_binding_by_locator_any_status(
                        diagnostic_source_id=entity.diagnostic_source_id,
                        source_locator_key=fields["source_locator_key"],
                        lock=True,
                    )
                )
                if (
                    locator_conflict is not None
                    and locator_conflict.target_source_binding_id != binding_id
                ):
                    raise state_conflict(
                        "Diagnostic Source 外部定位键已被使用"
                    )
            for name, value in fields.items():
                setattr(entity, name, value)
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now(UTC)
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET_SOURCE_BINDING",
                aggregate_id=binding_id,
                event_type="TARGET_SOURCE_BINDING_UPDATED",
                row_version=int(entity.row_version),
            )
            response = _source_binding_view(entity)
            await uow.commit()
            return response

    async def command_source_binding(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        binding_id: UUID,
        command: str,
        expected_version: int,
        idempotency_key: str,
    ) -> SourceBindingView:
        destination = {"enable": "ACTIVE", "disable": "DISABLED"}.get(command)
        if destination is None:
            raise validation_failed("未知 Source Binding 命令")

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> SourceBindingView:
            assert uow.targets is not None
            entity = await uow.targets.get_source_binding_scoped(
                target_source_binding_id=binding_id,
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Source Binding")
            self._check_version(entity.row_version, expected_version)
            if entity.status == destination:
                raise state_conflict(f"Source Binding 已处于 {destination}")
            entity.status = destination
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET_SOURCE_BINDING",
                aggregate_id=binding_id,
                event_type=f"TARGET_SOURCE_BINDING_{destination}",
                row_version=int(entity.row_version),
            )
            return _source_binding_view(entity)

        return await self._idempotent(
            scope=scope,
            operation=f"TARGET_SOURCE_BINDING_{command.upper()}",
            parent_resource=str(binding_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=SourceBindingView,
            handler=handler,
        )
