"""把 Agent、Target 和诊断源配置冻结为 Skill Planner 能力快照。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from platform_core.contracts.aiops.skills import (
    DbaCapabilitySnapshot,
    SourceCapabilitySnapshot,
)


def build_capability_snapshot(
    *,
    agent_id: object,
    agent_version: object,
    target: object,
    sources: Iterable[object],
) -> DbaCapabilitySnapshot:
    """只根据已持久化且可审计的配置生成规划快照。"""
    target_payload = dict(getattr(target, "capabilities_json", None) or {})
    target_capabilities = set(_capability_names(target_payload))
    if (
        getattr(target, "diagnostic_credential_id", None) is not None
        and bool(getattr(target, "endpoint_json", None))
    ):
        target_capabilities.add("DB_READONLY")
        if str(getattr(target, "db_type", "")) == "ORACLE":
            # 能力表示允许尝试受控只读 Tool；具体对象授权仍由数据库执行结果确认。
            target_capabilities.update(
                {
                    "dynamic_performance_views",
                    "dba_catalog_views",
                    "replication_views",
                }
            )
    if getattr(target, "execution_credential_id", None) is not None:
        target_capabilities.add("DB_MUTATION_CREDENTIAL")

    source_snapshots = tuple(
        SourceCapabilitySnapshot(
            source_id=str(source.diagnostic_source_id),
            source_type=str(source.source_type),
            enabled=str(source.status) == "ENABLED",
            reachable=str(source.connectivity_status)
            in {"CONNECTED", "DEGRADED"},
            capabilities=tuple(
                sorted(
                    set(
                        _capability_names(
                            getattr(
                                source,
                                "declared_capabilities_json",
                                None,
                            )
                        )
                    )
                    | set(
                        _capability_names(
                            getattr(
                                source,
                                "discovered_capabilities_json",
                                None,
                            )
                        )
                    )
                )
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
        privileges=tuple(sorted(_string_values(target_payload.get("privileges")))),
        entitlements=tuple(
            sorted(_string_values(target_payload.get("entitlements")))
        ),
        source_snapshots=source_snapshots,
    )


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
