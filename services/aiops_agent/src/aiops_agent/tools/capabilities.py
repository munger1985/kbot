"""把 Agent、Target 和诊断源配置冻结为调查能力快照。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

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
) -> DbaCapabilitySnapshot:
    """只根据已持久化且可审计的配置生成规划快照。"""
    target_payload = dict(getattr(target, "capabilities_json", None) or {})
    target_capabilities = set(target_capability_names(target))
    privileges = set(_string_values(target_payload.get("privileges")))
    if (
        "DB_READONLY" in target_capabilities
        and str(getattr(target, "db_type", "")) == "ORACLE"
    ):
        privileges.update({"CREATE SESSION", "SELECT ANY DICTIONARY"})

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
        privileges=tuple(sorted(privileges)),
        entitlements=tuple(
            sorted(_string_values(target_payload.get("entitlements")))
        ),
        source_snapshots=source_snapshots,
    )


def target_capability_names(target: object) -> tuple[str, ...]:
    """统一解析 Target 声明、探测和配置所确定的数据库能力。"""
    payload = dict(getattr(target, "capabilities_json", None) or {})
    capabilities = set(_capability_names(payload))
    db_type = str(getattr(target, "db_type", ""))
    readonly_ready = (
        bool(getattr(target, "readonly_connection_enabled", False))
        and getattr(target, "diagnostic_credential_id", None) is not None
        and bool(getattr(target, "endpoint_json", None))
    )
    if readonly_ready:
        capabilities.add("DB_READONLY")
        if db_type == "ORACLE":
            # 能力表示已配置 Oracle 只读路径；具体对象访问权仍以数据库结果为准。
            capabilities.update(
                {
                    "dynamic_performance_views",
                    "dba_catalog_views",
                    "replication_views",
                }
            )
        elif db_type == "MYSQL":
            # Performance Schema consumer、锁视图和复制视图均可能被关闭或拒绝；
            # MySQL 能力只能来自连接预检持久化的实际发现结果。
            pass
    mutation_ready = (
        bool(getattr(target, "controlled_change_enabled", False))
        and getattr(target, "execution_credential_id", None) is not None
    )
    if mutation_ready:
        capabilities.add("DB_MUTATION_CREDENTIAL")
        if db_type == "ORACLE":
            # Oracle 支持会话控制和索引维护；执行账号权限仍由数据库鉴权。
            capabilities.update({"session_management", "index_maintenance"})
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
