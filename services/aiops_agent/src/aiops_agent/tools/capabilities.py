"""把 Agent、Target 和诊断源配置冻结为调查能力快照。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone

from platform_core.contracts.aiops.playbooks import (
    DbaCapabilitySnapshot,
    SourceCapabilitySnapshot,
)


def build_capability_snapshot(
    *,
    agent_id: object,
    agent_version: object,
    target: object,
    sources: Iterable[object],
    now: datetime | None = None,
) -> DbaCapabilitySnapshot:
    """只根据已持久化且可审计的配置生成规划快照。"""
    now = now or datetime.now(timezone.utc)
    target_payload = dict(getattr(target, "capabilities_json", None) or {})
    target_capabilities = set(target_capability_names(target))
    privileges = set(_string_values(target_payload.get("privileges")))
    source_snapshots = tuple(
        SourceCapabilitySnapshot(
            source_id=str(source.diagnostic_source_id),
            source_type=str(source.source_type),
            **_source_freshness(source, now),
            enabled=str(source.status) == "ENABLED",
            reachable=str(source.connectivity_status)
            in {"CONNECTED", "DEGRADED"},
            capabilities=tuple(
                sorted(dict(getattr(source, "declared_capabilities_json", None) or {}))
            ),
        )
        for source in sorted(
            sources,
            key=lambda item: str(item.diagnostic_source_id),
        )
    )
    return DbaCapabilitySnapshot(
        agent_id=str(agent_id),
        agent_version_id=str(agent_version.agent_version_id),
        target_id=str(target.target_id),
        database_type=str(target.db_type),
        database_version=getattr(target, "version_code", None),
        target_enabled=str(target.status) == "ENABLED",
        target_reachable=str(target.connectivity_status)
        in {"CONNECTED", "DEGRADED"},
        target_capabilities=tuple(sorted(target_capabilities)),
        privileges=tuple(sorted(privileges)),
        entitlements=tuple(
            sorted(_string_values(target_payload.get("entitlements")))
        ),
        source_snapshots=source_snapshots,
    )


def _source_freshness(source, now):
    checked = getattr(source, "last_connectivity_check_at", None)
    updated = getattr(source, "updated_at", None)
    expires = checked + timedelta(minutes=5) if checked is not None else None
    fresh = bool(checked and expires and checked <= now < expires and (updated is None or updated <= checked))
    status = str(getattr(source, "connectivity_status", "UNKNOWN"))
    return {
        "configuration_version": getattr(source, "row_version", None),
        "checked_at": checked, "expires_at": expires,
        "health_state": ("AVAILABLE" if status in {"CONNECTED", "DEGRADED"} else "UNAVAILABLE") if fresh else "UNKNOWN",
    }


def target_capability_names(target: object) -> tuple[str, ...]:
    """统一解析 Target 声明、探测和配置所确定的数据库能力。"""
    payload = dict(getattr(target, "capabilities_json", None) or {})
    capabilities = set(_capability_names(payload))
    readonly_ready = (
        bool(getattr(target, "readonly_connection_enabled", False))
        and getattr(target, "diagnostic_credential_id", None) is not None
        and bool(getattr(target, "endpoint_json", None))
    )
    if readonly_ready:
        capabilities.add("DB_READONLY")
    mutation_ready = (
        bool(getattr(target, "controlled_change_enabled", False))
        and getattr(target, "execution_credential_id", None) is not None
    )
    if mutation_ready:
        capabilities.add("DB_MUTATION_CREDENTIAL")
    return tuple(sorted(capabilities))


def _capability_names(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        return ()
    names = set(_string_values(payload.get("capabilities")))
    names.update(_string_values(payload.get("features")))
    for key, value in payload.items():
        if key in {
            "capabilities", "features", "privileges", "entitlements"
        }:
            continue
        if value is True:
            names.add(str(key))
        elif isinstance(value, Mapping) and bool(
            value.get("enabled") or value.get("supported")
        ):
            names.add(str(key))
    return tuple(sorted(names))


def _string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())
