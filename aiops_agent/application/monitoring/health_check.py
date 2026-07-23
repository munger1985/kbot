"""Monitor Source 显式健康检查用例。"""

from __future__ import annotations

from uuid import UUID

from aiops_agent.ports.monitor import (
    MonitorHealthRequest,
    MonitorProviderContext,
)


class MonitorHealthCheckService:
    def __init__(self, *, uow_factory, provider_registry, secret_store):
        self._uow_factory = uow_factory
        self._providers = provider_registry
        self._secrets = secret_store

    async def execute(self, payload: dict) -> None:
        source_id = UUID(payload["aggregate_id"])
        details = payload["details"]
        request_id = UUID(details["health_check_request_id"])
        async with self._uow_factory() as uow:
            source = await uow.monitor_sources.get_scoped(
                monitor_source_id=source_id,
                app_id=int(payload["app_id"]),
                domain_id=int(payload["domain_id"]),
            )
            if (
                source is None
                or source.health_check_request_id != request_id
                or not source.endpoint
            ):
                return
            snapshot = {
                "source_id": str(source.monitor_source_id),
                "source_type": source.source_type,
                "source_version": int(source.row_version),
                "health_version": int(source.health_version),
                "endpoint": source.endpoint,
                "secret_ref": source.secret_ref,
                "capabilities": dict(source.capabilities_json or {}),
            }
        credentials = {}
        if snapshot["secret_ref"]:
            secret = await self._secrets.resolve(snapshot["secret_ref"])
            credentials = dict(secret.values)
            if "value" in credentials:
                credentials["token"] = credentials["value"]
        adapter = self._providers.create(
            MonitorProviderContext(
                source_id=snapshot["source_id"],
                source_type=snapshot["source_type"],
                source_version=snapshot["source_version"],
                endpoint=snapshot["endpoint"],
                credentials=credentials,
                capabilities=snapshot["capabilities"],
            )
        )
        result = await adapter.health_check(
            MonitorHealthRequest(trace_id=payload["trace_id"])
        )
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            changed = await uow.monitor_sources.update_health(
                monitor_source_id=source_id,
                health_check_request_id=request_id,
                expected_config_version=snapshot["source_version"],
                expected_health_version=snapshot["health_version"],
                health_status=(
                    "HEALTHY" if result.healthy else "UNREACHABLE"
                ),
                checked_at=now,
                last_error_code=result.error_code,
            )
            if changed:
                await uow.commit()
