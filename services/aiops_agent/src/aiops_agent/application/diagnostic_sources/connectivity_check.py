"""Diagnostic Source 显式连通性检查用例。"""

from __future__ import annotations

from uuid import UUID

from loguru import logger

from aiops_agent.adapters.diagnostic_sources.base import (
    DiagnosticSourceAdapterError,
)
from aiops_agent.application.managed_credentials import AIOpsManagedCredentialService

from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_HEALTH_CHECK,
    DiagnosticSourceContext,
    SourceHealthRequest,
    SourceHealthResult,
)


class DiagnosticSourceConnectivityCheckService:
    _MAX_VERSION_RETRIES = 2

    def __init__(
        self, *, uow_factory, diagnostic_source_registry, secret_store
    ):
        self._uow_factory = uow_factory
        self._diagnostic_sources = diagnostic_source_registry
        self._secrets = secret_store

    async def execute(self, payload: dict) -> None:
        source_id = UUID(payload["aggregate_id"])
        details = payload["details"]
        request_id = UUID(details["connectivity_check_request_id"])
        for attempt in range(1, self._MAX_VERSION_RETRIES + 1):
            snapshot = await self._load_snapshot(
                payload=payload,
                source_id=source_id,
                request_id=request_id,
            )
            if snapshot is None:
                return
            result = await self._check_source(
                payload=payload,
                snapshot=snapshot,
            )
            if await self._record_result(
                source_id=source_id,
                request_id=request_id,
                snapshot=snapshot,
                result=result,
            ):
                return
            if attempt < self._MAX_VERSION_RETRIES:
                logger.info(
                    "监控源连通性检查遇到并发版本变化，重新读取后重试："
                    "source_id={} request_id={} attempt={}",
                    source_id,
                    request_id,
                    attempt,
                )
        logger.warning(
            "监控源连通性检查结果未能回写，等待调度器补偿："
            "source_id={} request_id={}",
            source_id,
            request_id,
        )

    async def _load_snapshot(
        self,
        *,
        payload: dict,
        source_id: UUID,
        request_id: UUID,
    ) -> dict | None:
        async with self._uow_factory() as uow:
            source = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_id,
                domain_id=int(payload["domain_id"]),
            )
            if (
                source is None
                or source.connectivity_check_request_id != request_id
            ):
                return None
            credential_id = (
                source.auth_credential_id
                if source.endpoint
                else source.webhook_credential_id
            )
            credential_kind = (
                "diagnostic_source" if source.endpoint else "source_webhook"
            )
            return {
                "source_id": str(source.diagnostic_source_id),
                "source_type": source.source_type,
                "adapter_id": source.adapter_id,
                "adapter_version": source.adapter_version,
                "config_version": int(source.row_version),
                "connectivity_version": int(source.connectivity_version),
                "endpoint": source.endpoint,
                "secret_ref": (
                    AIOpsManagedCredentialService.reference(
                        domain_id=int(source.domain_id),
                        external_key=source.diagnostic_source_id,
                        credential_kind=credential_kind,
                        credential_id=credential_id,
                    )
                    if credential_id
                    else None
                ),
                "declared_capabilities": dict(
                    source.declared_capabilities_json or {}
                ),
                "config": dict(source.config_json or {}),
            }

    async def _check_source(
        self, *, payload: dict, snapshot: dict
    ) -> SourceHealthResult:
        try:
            credentials = {}
            if snapshot["secret_ref"]:
                secret = await self._secrets.resolve(snapshot["secret_ref"])
                credentials = dict(secret.values)
                if "value" in credentials:
                    credentials["token"] = credentials["value"]
            adapter = self._diagnostic_sources.create(
                DiagnosticSourceContext(
                    source_id=snapshot["source_id"],
                    source_type=snapshot["source_type"],
                    adapter_id=snapshot["adapter_id"],
                    adapter_version=snapshot["adapter_version"],
                    config_version=snapshot["config_version"],
                    endpoint=snapshot["endpoint"],
                    credentials=credentials,
                    declared_capabilities=snapshot[
                        "declared_capabilities"
                    ],
                    config=snapshot["config"],
                ),
                capability=CAPABILITY_HEALTH_CHECK,
            )
            result = await adapter.health_check(
                SourceHealthRequest(trace_id=payload["trace_id"])
            )
        except DiagnosticSourceAdapterError as exc:
            result = SourceHealthResult(
                healthy=False,
                error_code=exc.code,
                adapter_id=snapshot["adapter_id"],
                adapter_version=snapshot["adapter_version"],
            )
        except LookupError:
            result = SourceHealthResult(
                healthy=False,
                error_code="SOURCE_ADAPTER_INVALID",
                adapter_id=snapshot["adapter_id"],
                adapter_version=snapshot["adapter_version"],
            )
        except Exception:
            return SourceHealthResult(
                healthy=False,
                error_code="SOURCE_UNREACHABLE",
                adapter_id=snapshot["adapter_id"],
                adapter_version=snapshot["adapter_version"],
            )
        return result

    async def _record_result(
        self,
        *,
        source_id: UUID,
        request_id: UUID,
        snapshot: dict,
        result: SourceHealthResult,
    ) -> bool:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            changed = await uow.diagnostic_sources.update_connectivity(
                diagnostic_source_id=source_id,
                connectivity_check_request_id=request_id,
                expected_config_version=snapshot["config_version"],
                expected_connectivity_version=snapshot[
                    "connectivity_version"
                ],
                connectivity_status=(
                    "CONNECTED" if result.healthy else "UNREACHABLE"
                ),
                checked_at=now,
                last_error_code=result.error_code,
                discovered_capabilities={
                    capability: {
                        "adapter_id": result.adapter_id,
                        "adapter_version": result.adapter_version,
                    }
                    for capability in result.discovered_capabilities
                },
            )
            if changed:
                await uow.commit()
            return changed
