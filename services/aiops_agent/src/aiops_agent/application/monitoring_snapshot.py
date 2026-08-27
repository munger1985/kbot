"""冻结监控绑定、指标目录和查询窗口的共享应用组件。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from uuid import UUID

from aiops_agent.application.errors import validation_failed
from aiops_agent.application.managed_credentials import (
    AIOpsManagedCredentialService,
)
from aiops_agent.domain.evidence import DEFAULT_BASELINE_METRICS
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_QUERY,
    CAPABILITY_LOG_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
)


class MonitoringSnapshotBuilder:
    """在事务内按 Agent 与 Target 权限冻结可重放的监控查询快照。"""

    def __init__(
        self,
        *,
        metric_catalog,
        default_window_seconds: int,
        max_response_bytes: int,
    ) -> None:
        self._metric_catalog = metric_catalog
        self._default_window_seconds = default_window_seconds
        self._max_response_bytes = max_response_bytes

    async def build(
        self,
        *,
        uow,
        domain_id: int,
        target,
        now: datetime,
        allowed_source_ids: tuple[UUID, ...] | None = None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
    ) -> dict:
        if (window_start is None) != (window_end is None):
            raise validation_failed("观测窗口起止时间必须同时提供")
        resolved_end = window_end or now
        resolved_start = window_start or (
            resolved_end - timedelta(seconds=self._default_window_seconds)
        )
        if resolved_start >= resolved_end or resolved_end > now + timedelta(
            seconds=5
        ):
            raise validation_failed("观测窗口无效或结束时间位于未来")

        monitors = await uow.targets.list_source_bindings(
            target_id=target.target_id,
            domain_id=domain_id,
            active_only=True,
        )
        if allowed_source_ids is not None:
            allowed = set(allowed_source_ids)
            monitors = [
                monitor
                for monitor in monitors
                if monitor.diagnostic_source_id in allowed
            ]

        snapshots = []
        initial_gaps = []
        observation_binding_ids = []
        log_binding_ids = []
        for monitor in monitors:
            source = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=monitor.diagnostic_source_id,
                domain_id=domain_id,
            )
            if source is None or source.status != "ENABLED":
                initial_gaps.append(
                    self._gap(
                        monitor,
                        "DIAGNOSTIC_SOURCE_INACTIVE",
                        "监控源不存在或未激活",
                    )
                )
                continue
            if source.connectivity_status not in {"CONNECTED", "DEGRADED"}:
                initial_gaps.append(
                    self._gap(
                        monitor,
                        "DIAGNOSTIC_SOURCE_UNAVAILABLE",
                        "监控源当前不可连接",
                    )
                )
                continue
            requested = (monitor.capability_scope_json or {}).get(
                "metric_codes", DEFAULT_BASELINE_METRICS
            )
            if (
                not isinstance(requested, (list, tuple))
                or not requested
                or len(requested) > 64
                or not all(isinstance(item, str) and item for item in requested)
            ):
                raise validation_failed("监控绑定的 metric_codes 格式无效")
            requested_codes = tuple(dict.fromkeys(requested))
            try:
                selected = self._metric_catalog.select(
                    requested_codes, db_type=target.db_type
                )
            except KeyError as exc:
                raise validation_failed("监控绑定引用了未知标准指标") from exc
            supported = tuple(
                item for item in selected if source.source_type in item.providers
            )
            declared = set((source.declared_capabilities_json or {}).keys())
            requested_capabilities = (monitor.capability_scope_json or {}).get(
                "capabilities"
            )
            if requested_capabilities is not None:
                if (
                    not isinstance(requested_capabilities, (list, tuple))
                    or not requested_capabilities
                    or not all(
                        isinstance(item, str) and item
                        for item in requested_capabilities
                    )
                ):
                    raise validation_failed(
                        "Source Binding capabilities 格式无效"
                    )
                effective = declared.intersection(requested_capabilities)
            else:
                effective = declared
            binding_id = str(monitor.target_source_binding_id)
            if effective.intersection(
                {CAPABILITY_METRIC_QUERY_RANGE, CAPABILITY_EVENT_QUERY}
            ):
                observation_binding_ids.append(binding_id)
            if CAPABILITY_LOG_QUERY in effective:
                log_binding_ids.append(binding_id)
            snapshots.append(
                {
                    "binding_id": binding_id,
                    "binding_version": int(monitor.row_version),
                    "role": monitor.role,
                    "priority": int(monitor.priority),
                    "source_locator_key": monitor.source_locator_key,
                    "source_locator": dict(monitor.source_locator_json),
                    "source_locator_fingerprint": hashlib.sha256(
                        monitor.source_locator_key.encode("utf-8")
                    ).hexdigest(),
                    "mapping_overrides": dict(
                        monitor.mapping_overrides_json or {}
                    ),
                    "query_budget": dict(monitor.query_budget_json or {}),
                    "effective_capabilities": sorted(effective),
                    "source": {
                        "source_id": str(source.diagnostic_source_id),
                        "source_type": source.source_type,
                        "adapter_id": source.adapter_id,
                        "adapter_version": source.adapter_version,
                        "config_version": int(source.row_version),
                        "endpoint": source.endpoint,
                        "secret_ref": (
                            AIOpsManagedCredentialService.reference(
                                domain_id=int(source.domain_id),
                                external_key=source.diagnostic_source_id,
                                credential_kind="diagnostic_source",
                                credential_id=source.auth_credential_id,
                            )
                            if source.auth_credential_id
                            else None
                        ),
                        "declared_capabilities": dict(
                            source.declared_capabilities_json or {}
                        ),
                        "config": dict(source.config_json or {}),
                    },
                    "metrics": [
                        item.model_dump(mode="json")
                        for item in supported
                        if CAPABILITY_METRIC_QUERY_RANGE in effective
                    ],
                    "unsupported_metrics": sorted(
                        set(requested_codes)
                        - {item.metric_code for item in supported}
                        if CAPABILITY_METRIC_QUERY_RANGE in effective
                        else ()
                    ),
                }
            )
        return {
            "window": {
                "start": resolved_start.isoformat(),
                "end": resolved_end.isoformat(),
            },
            "catalog_version": self._metric_catalog.version,
            "catalog_hash": self._metric_catalog.manifest_hash,
            "max_response_bytes": self._max_response_bytes,
            "bindings": snapshots,
            "observation_binding_ids": sorted(observation_binding_ids),
            "log_binding_ids": sorted(log_binding_ids),
            "initial_gaps": initial_gaps,
        }

    @staticmethod
    def _gap(monitor, code: str, detail: str) -> dict:
        return {
            "binding_id": str(monitor.target_source_binding_id),
            "source_id": str(monitor.diagnostic_source_id),
            "code": code,
            "detail": detail,
        }
