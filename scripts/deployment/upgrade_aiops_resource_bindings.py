"""停写后定向升级资源监控绑定；不初始化账号、配置或其他业务表。"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import UUID

from sqlalchemy import inspect, text

from platform_core.identity import uuid7


ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {
    "binding": "target_source_binding", "resource": "managed_resource",
    "database": "database_resource", "compute": "compute_resource",
    "link": "business_system_resource", "relation": "resource_relation",
    "participant": "relation_participant", "system": "business_system",
    "source": "diagnostic_source", "target": "target", "agent": "agent",
    "agent_version": "agent_version", "agent_system": "agent_version_system",
    "agent_policy": "agent_version_resource_action_policy",
}
PROTECTED = ("system", "source", "target", "agent", "agent_version", "agent_system", "agent_policy")


def table_name(key: str, backend: str) -> str:
    suffix = SUFFIXES.get(key, key)
    if backend == "postgresql" and key == "agent_system":
        suffix = "agent_version_business_system"
    return "KBOT_OPS_" + suffix.upper() if backend == "oracle" else "aiops_" + suffix


def archive_name(backend: str) -> str:
    return "KBOT_BAK_RESOURCE_BINDING_57" if backend == "oracle" else "aiops_bak_resource_binding_57"


def normalize(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        key = key.lower()
        if isinstance(value, (bytes, memoryview)) and key.endswith("_id"):
            value = UUID(bytes=bytes(value))
        elif isinstance(value, Decimal):
            value = int(value) if value == int(value) else float(value)
        elif isinstance(value, datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        elif hasattr(value, "read"):
            value = value.read()
        if key.endswith("_json") and isinstance(value, str):
            value = json.loads(value)
        result[key] = value
    return result


def encode(value):
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, (bytes, memoryview)):
        return {"$bytes": bytes(value).hex()}
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    raise TypeError(type(value).__name__)


def serialized(value) -> str:
    return json.dumps(value, default=encode, ensure_ascii=False, sort_keys=True)


def digest(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(serialized(row) for row in rows)).encode()).hexdigest()


def private_json(path: Path, payload) -> None:
    """备份只允许新建，避免重复执行覆盖恢复点。"""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_mode & 0o077:
        raise ValueError("备份目录必须为 0700")
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(serialized(payload))
        stream.flush()
        os.fsync(stream.fileno())


async def read_rows(conn, name: str) -> list[dict]:
    result = await conn.execute(text(f"SELECT * FROM {name}"))
    return [normalize(dict(row)) for row in result.mappings()]


async def inventory(conn, backend: str, *, archived: bool = False) -> dict:
    data = {}
    for key in SUFFIXES:
        name = archive_name(backend) if key == "binding" and archived else table_name(key, backend)
        data[key] = await read_rows(conn, name)
    return data


def build_plan(data: dict, host_types: dict[str, str], *, defer_unknown: bool = False) -> dict:
    """只按显式数据库关联和主机定位拆分，不按地址或名称猜测硬件身份。"""
    resources = {row["resource_id"]: row for row in data["resource"]}
    dbs: dict = {}
    for row in data["database"]:
        dbs.setdefault(row["target_id"], []).append(row["resource_id"])
    sources = {row["diagnostic_source_id"]: row for row in data["source"]}
    systems = {row["business_system_id"]: row for row in data["system"]}
    now = datetime.now(UTC)
    plan = {key: [] for key in ("bindings", "resources", "computes", "links", "relations", "participants", "deferred")}
    hosts, host_links, seen = {}, {}, set()
    for old in sorted(data["binding"], key=lambda row: str(row["target_source_binding_id"])):
        matched = dbs.get(old["target_id"], [])
        if len(matched) != 1 or matched[0] not in resources:
            raise ValueError("旧绑定未唯一关联数据库资源，停止升级")
        resource = resources[matched[0]]
        source = sources[old["diagnostic_source_id"]]
        if resource["domain_id"] != source["domain_id"] or resource.get("tenant_id") != source.get("tenant_id"):
            raise ValueError("旧绑定跨越 Domain/Tenant，停止升级")
        if resource["resource_type_key"] != "DATABASE":
            raise ValueError("数据库扩展关联了非数据库资源")
        identity = (resource["resource_id"], old["diagnostic_source_id"])
        if identity in seen:
            raise ValueError("同资源同监控源存在重复绑定，不能自动合并")
        seen.add(identity)
        binding = dict(old)
        binding["resource_source_binding_id"] = binding.pop("target_source_binding_id")
        binding.pop("target_id")
        binding["resource_id"] = resource["resource_id"]
        locator = dict(binding["source_locator_json"] or {})
        host_key = locator.pop("host_target_key", None)
        binding["source_locator_json"] = locator
        plan["bindings"].append(binding)
        if not host_key:
            continue
        if source["source_type"] != "PROMETHEUS":
            raise ValueError("非 Prometheus 绑定含主机定位，需要人工核验")
        if host_key not in host_types:
            if not defer_unknown:
                raise ValueError("主机类型未核验；提供 host-types 或明确暂存未知主机")
            plan["deferred"].append({"binding_id": old["target_source_binding_id"], "host_target_key": host_key,
                                     "reason": "HARDWARE_TYPE_UNVERIFIED"})
            continue
        kind = host_types[host_key]
        if kind not in {"PHYSICAL_SERVER", "VIRTUAL_MACHINE"}:
            raise ValueError("独立主机类型仅支持物理服务器或虚拟机")
        host_identity = (resource.get("tenant_id"), resource["domain_id"], old["diagnostic_source_id"], host_key)
        if host_identity not in hosts:
            host = {key: resource[key] for key in ("domain_id", "owner_group_id", "created_by", "updated_by")}
            if "tenant_id" in resource:
                host["tenant_id"] = resource["tenant_id"]
            host.update(resource_id=uuid7(), resource_code=None, resource_type_key=kind, display_name=host_key,
                        identity_kind="STABLE", lifecycle_status="ACTIVE", row_version=1,
                        attributes_json={"migration_origin": "resource-monitoring-v57", "host_target_key": host_key},
                        created_at=now, updated_at=now)
            hosts[host_identity] = host
            plan["resources"].append(host)
            plan["computes"].append({"resource_id": host["resource_id"], "compute_kind": kind})
            host_binding = {key: old[key] for key in ("diagnostic_source_id", "role", "priority", "query_budget_json", "status", "created_by", "updated_by")}
            host_binding.update(resource_source_binding_id=uuid7(), resource_id=host["resource_id"],
                                source_locator_key=host_key, source_locator_json={"target_key": host_key, "resource_scope": "HOST"},
                                capability_scope_json={}, mapping_overrides_json={}, health_status="UNKNOWN",
                                row_version=1, health_version=1, created_at=now, updated_at=now)
            plan["bindings"].append(host_binding)
        host = hosts[host_identity]
        for db_link in data["link"]:
            if db_link["resource_id"] != resource["resource_id"]:
                continue
            system = systems[db_link["business_system_id"]]
            if system["domain_id"] != resource["domain_id"] or system.get("tenant_id") != resource.get("tenant_id"):
                raise ValueError("数据库资源的系统归属跨域")
            link_key = (db_link["business_system_id"], host["resource_id"])
            if link_key not in host_links:
                link = {key: db_link[key] for key in ("business_system_id", "responsibility_group_id", "display_order", "status", "created_by", "updated_by")}
                link.update(system_resource_id=uuid7(), resource_id=host["resource_id"],
                            architecture_layer_key="INFRASTRUCTURE", row_version=1, created_at=now, updated_at=now)
                host_links[link_key] = link
                plan["links"].append(link)
            elif db_link["status"] == "ACTIVE":
                host_links[link_key]["status"] = "ACTIVE"
            relation = {key: db_link[key] for key in ("business_system_id", "responsibility_group_id", "created_by", "updated_by")}
            relation.update(relation_id=uuid7(), relation_type_key="RUNS_ON", origin="MANUAL",
                            confirmation_status="CONFIRMED", verification_status="UNKNOWN", row_version=1,
                            attributes_json={"migration_origin": "resource-monitoring-v57"}, created_at=now, updated_at=now)
            plan["relations"].append(relation)
            for order, (role, link_id) in enumerate((("WORKLOAD", db_link["system_resource_id"]), ("HOST", host_links[link_key]["system_resource_id"]))):
                plan["participants"].append({"participant_id": uuid7(), "relation_id": relation["relation_id"],
                                             "system_resource_id": link_id, "participant_role": role, "display_order": order})
    return plan


def summary(data: dict, plan: dict) -> dict:
    return {"old_database_bindings": len(data["binding"]), "new_total_bindings": len(plan["bindings"]),
            "host_resources": len(plan["resources"]), "host_system_links": len(plan["links"]),
            "runs_on_relations": len(plan["relations"]), "deferred_host_references": len(plan["deferred"])}


async def backup(conn, backend: str, data: dict, plan: dict, path: Path) -> None:
    ddl = await conn.run_sync(lambda c: {
        "tables": inspect(c).get_table_names(),
        "binding_columns": inspect(c).get_columns(table_name("binding", backend)),
        "binding_foreign_keys": inspect(c).get_foreign_keys(table_name("binding", backend)),
        "binding_indexes": inspect(c).get_indexes(table_name("binding", backend)),
    })
    # 反射类型转为稳定文本；数据值仍保留可恢复的类型标签。
    for column in ddl["binding_columns"]:
        column["type"] = str(column["type"])
    if backend == "oracle":
        ddl["oracle_objects"] = []
        for kind, names in (("TABLE", [table_name("binding", backend)]), ("VIEW", ["KBOT_V_OPS_TARGET", "KBOT_V_OPS_DIAGNOSTIC_SOURCE", "KBOT_V_OPS_SCHEMA_VERSION"])):
            for name in names:
                result = await conn.execute(text("SELECT DBMS_METADATA.GET_DDL(:kind,:name) FROM DUAL"), {"kind": kind, "name": name})
                value = result.scalar_one()
                ddl["oracle_objects"].append({"kind": kind, "name": name, "ddl": value.read() if hasattr(value, "read") else value})
        result = await conn.execute(text("SELECT DBMS_METADATA.GET_DEPENDENT_DDL('REF_CONSTRAINT',:name) FROM DUAL"), {"name": table_name("signal_event", backend)})
        value = result.scalar_one()
        ddl["signal_foreign_keys"] = value.read() if hasattr(value, "read") else value
    private_json(path / "snapshot.json", {"backend": backend, "created_at": datetime.now(UTC), "data": data, "ddl": ddl,
                                           "protected_hashes": {key: digest(data[key]) for key in PROTECTED}})
    private_json(path / "plan.json", plan)
    private_json(path / "summary.json", summary(data, plan))


def oracle_statements() -> list[str]:
    roots = (ROOT / "database/oracle/aiops_agent/001_ops_roots.sql").read_text()
    names = [r"CREATE TABLE KBOT_OPS_RESOURCE_SOURCE_BINDING\b.*?;",
             r"CREATE INDEX IX_OPS_RSRC_RESOURCE\b.*?;", r"CREATE INDEX IX_OPS_TSRC_MONITOR\b.*?;"]
    statements = []
    for pattern in names:
        match = re.search(pattern, roots, re.S)
        if match is None:
            raise ValueError("规范资源绑定 DDL 缺失")
        statements.append(match.group().rstrip(";"))
    return statements


def oracle_views() -> list[str]:
    content = (ROOT / "database/oracle/aiops_agent/006_ops_fks_views.sql").read_text()
    return [re.search(rf"CREATE OR REPLACE VIEW {name}\b.*?;", content, re.S).group().rstrip(";")
            for name in ("KBOT_V_OPS_TARGET", "KBOT_V_OPS_DIAGNOSTIC_SOURCE", "KBOT_V_OPS_SCHEMA_VERSION")]


async def apply_structure(conn, backend: str) -> None:
    old, new = table_name("binding", backend), table_name("resource_source_binding", backend)
    await conn.execute(text(f"ALTER TABLE {old} RENAME TO {archive_name(backend)}"))
    if backend == "oracle":
        await conn.execute(text(f"ALTER TABLE {archive_name(backend)} RENAME CONSTRAINT FK_OPS_TSRC_MONITOR TO FK_BAK_RSB57_MONITOR"))
        await conn.execute(text("ALTER INDEX IX_OPS_TSRC_MONITOR RENAME TO IX_BAK_RSB57_MONITOR"))
        for statement in oracle_statements():
            await conn.execute(text(statement))
    else:
        from aiops_agent.entities import ResourceSourceBindingEntity
        await conn.run_sync(lambda c: ResourceSourceBindingEntity.__table__.create(c))


async def insert_plan(conn, plan: dict) -> None:
    from aiops_agent.entities import (
        BusinessSystemResourceEntity, ComputeResourceEntity, ManagedResourceEntity,
        RelationParticipantEntity, ResourceRelationEntity, ResourceSourceBindingEntity,
    )
    for key, entity in (("resources", ManagedResourceEntity), ("computes", ComputeResourceEntity),
                        ("links", BusinessSystemResourceEntity), ("relations", ResourceRelationEntity),
                        ("participants", RelationParticipantEntity), ("bindings", ResourceSourceBindingEntity)):
        for row in plan[key]:
            await conn.execute(entity.__table__.insert().values(**row))


async def verify(conn, backend: str, original: dict, plan: dict) -> dict:
    current = await inventory(conn, backend, archived=True)
    for key in PROTECTED:
        if digest(current[key]) != digest(original[key]):
            raise ValueError(f"非升级对象发生变化：{key}")
    for key in ("resource", "compute", "link", "relation", "participant", "database"):
        original_ids = {serialized(row) for row in original[key]}
        if not original_ids.issubset({serialized(row) for row in current[key]}):
            raise ValueError(f"原记录发生改变：{key}")
    actual = await read_rows(conn, table_name("resource_source_binding", backend))
    if len(actual) != len(plan["bindings"]):
        raise ValueError("新绑定行数不符合计划")
    actual_by_id = {row["resource_source_binding_id"]: row for row in actual}
    for row in plan["bindings"]:
        saved = actual_by_id.get(row["resource_source_binding_id"])
        if saved is None or any(serialized(saved.get(key)) != serialized(value) for key, value in row.items()):
            raise ValueError("绑定内容或历史 ID 未完整保留")
    if any("host_target_key" in row["source_locator_json"] for row in actual):
        raise ValueError("新数据库绑定仍含主机定位")
    for key, additions in (("resource", "resources"), ("compute", "computes"), ("link", "links"), ("relation", "relations"), ("participant", "participants")):
        if len(current[key]) != len(original[key]) + len(plan[additions]):
            raise ValueError(f"资源拓扑数量不符：{key}")
    if backend == "oracle":
        invalid = (await conn.execute(text("SELECT OBJECT_NAME FROM USER_OBJECTS WHERE STATUS='INVALID' AND OBJECT_NAME LIKE 'KBOT%OPS%'"))).scalars().all()
        if invalid:
            raise ValueError("升级后存在无效 AIOps 对象")
        version = (await conn.execute(text("SELECT SCHEMA_VERSION FROM KBOT_V_OPS_SCHEMA_VERSION"))).scalar_one()
        if int(version) != 57:
            raise ValueError("Schema 版本不是 57")
    return {**summary(original, plan), "protected_data_unchanged": True, "verified": True}


def assert_stopped() -> None:
    """部署门禁只属于升级工具；不注入运行时。"""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if (entry / "cwd").resolve() != ROOT:
                continue
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if any(arg.startswith((b"aiops_agent.entrypoints.", b"main_api.entrypoints.")) for arg in args):
            raise ValueError("当前仓库的 AIOps/Main API 尚未停写")


async def run(args) -> None:
    from aiops_agent.config import get_aiops_settings
    if args.backend == "oracle":
        from platform_core.database.oracle import create_database_runtime
    else:
        from platform_core.database.postgresql import create_database_runtime_from_settings as create_database_runtime
    runtime = create_database_runtime(get_aiops_settings())
    runtime.engine.sync_engine.echo = False
    runtime.engine.sync_engine.hide_parameters = True
    try:
        async with runtime.engine.connect() as conn:
            data = await inventory(conn, args.backend)
            host_types = json.loads(args.host_types.read_text()) if args.host_types else {}
            plan = build_plan(data, host_types, defer_unknown=args.defer_unknown_hosts)
            print(json.dumps(summary(data, plan), ensure_ascii=False), flush=True)
            if not args.apply:
                return
            assert_stopped()
            if not args.backup_dir:
                raise ValueError("执行升级必须指定备份目录")
            if args.backend == "postgresql":
                await conn.execute(text("SET LOCAL lock_timeout='10s'"))
                # 所有迁移相关表在同一事务内锁定，避免备份与写入窗口漂移。
                names = ",".join(table_name(key, args.backend) for key in SUFFIXES)
                await conn.execute(text(f"LOCK TABLE {names} IN ACCESS EXCLUSIVE MODE"))
                data = await inventory(conn, args.backend)
                plan = build_plan(data, host_types, defer_unknown=args.defer_unknown_hosts)
            else:
                version = (await conn.execute(text("SELECT SCHEMA_VERSION FROM KBOT_V_OPS_SCHEMA_VERSION"))).scalar_one()
                if int(version) != 56:
                    raise ValueError("Oracle 只允许从 Schema 56 升级")
            names = await conn.run_sync(lambda c: inspect(c).get_table_names())
            if archive_name(args.backend).lower() in {name.lower() for name in names} or table_name("resource_source_binding", args.backend).lower() in {name.lower() for name in names}:
                raise ValueError("检测到已有新表或归档表，请先核验已有升级状态")
            await backup(conn, args.backend, data, plan, args.backup_dir)
            print("备份完成；开始定向升级", flush=True)
            await apply_structure(conn, args.backend)
            await insert_plan(conn, plan)
            if args.backend == "oracle":
                # Oracle DDL 隐式提交；先完整写入并提交，再切换事件外键和视图。
                await conn.commit()
                await conn.execute(text("ALTER TABLE KBOT_OPS_SIGNAL_EVENT DROP CONSTRAINT FK_OPS_SIG_BINDING"))
                await conn.execute(text("ALTER TABLE KBOT_OPS_SIGNAL_EVENT ADD CONSTRAINT FK_OPS_SIG_BINDING FOREIGN KEY (SOURCE_BINDING_ID) REFERENCES KBOT_OPS_RESOURCE_SOURCE_BINDING (RESOURCE_SOURCE_BINDING_ID)"))
                for statement in oracle_views():
                    await conn.execute(text(statement))
            result = await verify(conn, args.backend, data, plan)
            await conn.commit()
            private_json(args.backup_dir / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        await runtime.close()


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("oracle", "postgresql"), required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--host-types", type=Path)
    parser.add_argument("--defer-unknown-hosts", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.project_root:
        ROOT = args.project_root.resolve()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        # 不向终端输出驱动异常中的 SQL 参数或数据库凭据。
        print(f"升级未完成（{type(exc).__name__}）；保持相关服务停止，核对受保护备份和具体阶段。", flush=True)
        if args.backup_dir and args.backup_dir.exists():
            private_json(args.backup_dir / f"error-{datetime.now(UTC).strftime('%H%M%S%f')}.json", {"error_type": type(exc).__name__, "detail": str(exc)})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
