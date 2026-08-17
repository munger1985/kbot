"""将 KBot 4.0 规范 DDL 应用到空白 Oracle Schema。"""

from __future__ import annotations

import argparse
import asyncio
import re
from configparser import ConfigParser, Error as ConfigParserError
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "database" / "oracle"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "init_services.ini"
PLATFORM_FOUNDATION_SCRIPT = (
    ROOT / "scripts" / "db" / "bootstrap_platform_foundation.sql"
)
PLATFORM_FOUNDATION_TABLES = {
    "KBOT_PLATFORM_DOMAIN",
    "KBOT_PLATFORM_USER",
    "KBOT_PLATFORM_USER_CREDENTIAL",
    "KBOT_PERMISSION",
    "KBOT_APP_ROLE",
    "KBOT_APP_ROLE_PERMISSION",
    "KBOT_APP_MEMBER_ROLE",
}
PLATFORM_FOUNDATION_PERMISSIONS = {
    "platform:user_manage",
    "platform:role_manage",
    "knowledge_retrieval:use",
    "knowledge_retrieval:upload",
    "knowledge_retrieval:review",
    "knowledge_retrieval:member_manage",
    "knowledge_retrieval:knowledge_manage",
    "knowledge_retrieval:data_manage",
    "knowledge_retrieval:agent_manage",
    "knowledge_retrieval:operations_manage",
    "km_asset:use",
    "km_asset:source_manage",
    "km_asset:data_manage",
    "km_asset:agent_manage",
    "km_asset:operations_manage",
    "km_asset:member_manage",
    "aiops:use",
    "aiops:domain_manage",
    "aiops:member_manage",
    "aiops:operations_manage",
    "aiops:target_manage",
    "aiops:monitor_source_manage",
    "aiops:policy_manage",
    "aiops:plan_manage",
    "aiops:agent_manage",
    "aiops:proposal:approve",
}
PLATFORM_FOUNDATION_ROLES = {
    ("platform", "platform_admin"),
    ("knowledge_retrieval", "user"),
    ("knowledge_retrieval", "contributor"),
    ("knowledge_retrieval", "reviewer"),
    ("knowledge_retrieval", "manager"),
    ("km_asset", "user"),
    ("km_asset", "manager"),
    ("aiops", "operator"),
    ("aiops", "approver"),
    ("aiops", "manager"),
}

from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7
from platform_core.prompts import load_prompt_catalog


REQUIRED_SERVICES = (
    "platform_core",
)
OPTIONAL_SERVICE_ORDER = (
    "main_api",
    "model_serving",
    "knowledge_core",
    "knowledge_retrieval_app",
    "km_asset_app",
    "agent_runtime",
    "data_query",
    "aiops_agent",
)


@dataclass(frozen=True)
class SchemaStatement:
    """带来源位置的单条 DDL。"""

    script: Path
    ordinal: int
    sql: str

    @property
    def label(self) -> str:
        """返回适合日志展示且不包含连接凭据的 DDL 摘要。"""
        without_comments = re.sub(
            r"^\s*(?:(?:--[^\n]*(?:\n|$))|(?:/\*.*?\*/\s*))*",
            "",
            self.sql,
            flags=re.DOTALL,
        )
        normalized = re.sub(r"\s+", " ", without_comments).strip()
        match = re.match(
            r"(CREATE(?:\s+OR\s+REPLACE)?\s+(?:TABLE|VIEW|INDEX)"
            r"|ALTER\s+TABLE|COMMENT\s+ON\s+COLUMN)\s+([A-Z0-9_.$]+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            return f"{match.group(1).upper()} {match.group(2).upper()}"
        return normalized[:100]


@dataclass(frozen=True)
class ServiceSelection:
    """一次空库初始化所包含的必选层和业务服务。"""

    required: tuple[str, ...]
    enabled: tuple[str, ...]

    @property
    def ordered(self) -> tuple[str, ...]:
        return self.required + self.enabled


def load_service_selection(config_path: Path) -> ServiceSelection:
    """从 INI 读取业务服务，基础层始终自动加入。"""
    if not config_path.is_file():
        raise RuntimeError(f"初始化配置不存在：{config_path}")

    parser = ConfigParser(interpolation=None)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            parser.read_file(config_file)
    except (OSError, ConfigParserError) as exc:
        raise RuntimeError(f"无法读取初始化配置 {config_path}：{exc}") from exc

    if not parser.has_section("services"):
        raise RuntimeError("初始化配置缺少 [services] 段")

    configured = set(parser["services"])
    required_in_config = configured.intersection(REQUIRED_SERVICES)
    if required_in_config:
        raise RuntimeError(
            "必建基础层不需要配置："
            f"{', '.join(sorted(required_in_config))}"
        )

    unknown = configured - set(OPTIONAL_SERVICE_ORDER)
    if unknown:
        raise RuntimeError(
            f"初始化配置包含未知服务：{', '.join(sorted(unknown))}"
        )

    enabled: list[str] = []
    for service in OPTIONAL_SERVICE_ORDER:
        try:
            selected = parser.getboolean(
                "services",
                service,
                fallback=False,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"服务 {service} 必须配置为 true 或 false"
            ) from exc
        if selected:
            enabled.append(service)

    return ServiceSelection(
        required=REQUIRED_SERVICES,
        enabled=tuple(enabled),
    )


def split_oracle_statements(sql: str) -> list[str]:
    """按分号拆分普通 Oracle DDL，同时正确处理字符串和注释。"""
    statements: list[str] = []
    current: list[str] = []
    index = 0
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if in_line_comment:
            current.append(char)
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            current.append(char)
            if char == "*" and next_char == "/":
                current.append(next_char)
                index += 2
                in_block_comment = False
            else:
                index += 1
            continue

        if in_string:
            current.append(char)
            if char == "'" and next_char == "'":
                current.append(next_char)
                index += 2
                continue
            if char == "'":
                in_string = False
            index += 1
            continue

        if char == "-" and next_char == "-":
            current.extend((char, next_char))
            index += 2
            in_line_comment = True
            continue
        if char == "/" and next_char == "*":
            current.extend((char, next_char))
            index += 2
            in_block_comment = True
            continue
        if char == "'":
            current.append(char)
            index += 1
            in_string = True
            continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    if in_string or in_block_comment:
        raise ValueError("SQL 文件存在未闭合的字符串或块注释")
    return statements


def load_schema_statements(
    services: Sequence[str] | None = None,
) -> list[SchemaStatement]:
    """按服务依赖和文件名前缀读取全部规范 DDL。"""
    selected = tuple(services) if services is not None else (
        REQUIRED_SERVICES + OPTIONAL_SERVICE_ORDER
    )
    duplicates = {
        service for service in selected if selected.count(service) > 1
    }
    if duplicates:
        raise RuntimeError(f"服务重复：{', '.join(sorted(duplicates))}")

    allowed = set(REQUIRED_SERVICES + OPTIONAL_SERVICE_ORDER)
    unknown = set(selected) - allowed
    if unknown:
        raise RuntimeError(f"未知服务：{', '.join(sorted(unknown))}")

    canonical_order = REQUIRED_SERVICES + OPTIONAL_SERVICE_ORDER
    ordered_services = tuple(
        service for service in canonical_order if service in selected
    )
    result: list[SchemaStatement] = []
    for service in ordered_services:
        scripts = sorted((SCHEMA_ROOT / service).glob("[0-9][0-9][0-9]_*.sql"))
        if not scripts:
            raise RuntimeError(f"{service} 没有规范 DDL")
        for script in scripts:
            statements = split_oracle_statements(
                script.read_text(encoding="utf-8")
            )
            for ordinal, statement in enumerate(statements, start=1):
                result.append(
                    SchemaStatement(
                        script=script.relative_to(ROOT),
                        ordinal=ordinal,
                        sql=statement,
                    )
                )
    return result


async def _read_target(connection: AsyncConnection) -> tuple[str, str]:
    row = (
        await connection.execute(
            text(
                """
                SELECT
                    SYS_CONTEXT('USERENV', 'CON_NAME'),
                    SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                FROM dual
                """
            )
        )
    ).one()
    return str(row[0]), str(row[1])


async def _assert_empty_schema(connection: AsyncConnection) -> None:
    existing = (
        await connection.execute(
            text(
                """
                SELECT object_name, object_type
                FROM user_objects
                WHERE object_name LIKE 'KBOT\\_%' ESCAPE '\\'
                  AND object_type IN ('TABLE', 'VIEW')
                ORDER BY object_type, object_name
                """
            )
        )
    ).all()
    if existing:
        rendered = ", ".join(f"{name}({kind})" for name, kind in existing)
        raise RuntimeError(
            "目标 Schema 已存在 KBot 表或视图，初始化已拒绝："
            f"{rendered}"
        )


async def _validate_platform_foundation(
    connection: AsyncConnection,
) -> None:
    """校验首次登录所需的默认 Domain、ADMIN 与全权限授权。"""
    existing_tables = set(
        (
            await connection.execute(
                text(
                    "SELECT table_name FROM user_tables "
                    "WHERE table_name IN ("
                    + ",".join(
                        f"'{name}'" for name in sorted(PLATFORM_FOUNDATION_TABLES)
                    )
                    + ")"
                )
            )
        ).scalars()
    )
    missing_tables = PLATFORM_FOUNDATION_TABLES - existing_tables
    if missing_tables:
        raise RuntimeError(
            "平台基础表不完整：" + ", ".join(sorted(missing_tables))
        )

    default_domain_count = (
        await connection.execute(
            text(
                "SELECT COUNT(*) FROM KBOT_PLATFORM_DOMAIN "
                "WHERE NAME = 'default' AND STATUS = 'ACTIVE'"
            )
        )
    ).scalar_one()
    if int(default_domain_count) != 1:
        raise RuntimeError("缺少唯一且启用的默认业务域 default")

    admin_count = (
        await connection.execute(
            text(
                "SELECT COUNT(*) FROM KBOT_PLATFORM_USER user_account "
                "JOIN KBOT_PLATFORM_USER_CREDENTIAL credential "
                "ON credential.USER_ID = user_account.USER_ID "
                "WHERE user_account.USER_ID = 'ADMIN' "
                "AND user_account.STATUS = 'ACTIVE'"
            )
        )
    ).scalar_one()
    if int(admin_count) != 1:
        raise RuntimeError("缺少启用的 ADMIN 用户或登录凭据")

    permission_codes = set(
        (
            await connection.execute(
                text("SELECT PERMISSION_CODE FROM KBOT_PERMISSION")
            )
        ).scalars()
    )
    missing_permissions = (
        PLATFORM_FOUNDATION_PERMISSIONS - permission_codes
    )
    if missing_permissions:
        raise RuntimeError(
            "平台权限目录不完整：" + ", ".join(sorted(missing_permissions))
        )

    active_roles = {
        (str(app_id), str(role_code))
        for app_id, role_code in (
            await connection.execute(
                text(
                    "SELECT APP_ID, ROLE_CODE FROM KBOT_APP_ROLE "
                    "WHERE STATUS = 'ACTIVE'"
                )
            )
        ).all()
    }
    missing_roles = PLATFORM_FOUNDATION_ROLES - active_roles
    if missing_roles:
        raise RuntimeError(
            "平台角色模板不完整："
            + ", ".join(
                f"{app_id}/{role_code}"
                for app_id, role_code in sorted(missing_roles)
            )
        )

    permissions_by_app: dict[str, set[str]] = {}
    for permission_code in PLATFORM_FOUNDATION_PERMISSIONS:
        permissions_by_app.setdefault(
            permission_code.partition(":")[0], set()
        ).add(permission_code)
    expected_role_permissions = {
        ("platform", "platform_admin"): permissions_by_app["platform"],
        ("knowledge_retrieval", "user"): {
            "knowledge_retrieval:use"
        },
        ("knowledge_retrieval", "contributor"): {
            "knowledge_retrieval:use",
            "knowledge_retrieval:upload",
        },
        ("knowledge_retrieval", "reviewer"): {
            "knowledge_retrieval:use",
            "knowledge_retrieval:upload",
            "knowledge_retrieval:review",
        },
        ("knowledge_retrieval", "manager"): permissions_by_app[
            "knowledge_retrieval"
        ],
        ("km_asset", "user"): {"km_asset:use"},
        ("km_asset", "manager"): permissions_by_app["km_asset"],
        ("aiops", "operator"): {
            "aiops:use",
            "aiops:operations_manage",
            "aiops:target_manage",
            "aiops:monitor_source_manage",
            "aiops:policy_manage",
            "aiops:plan_manage",
        },
        ("aiops", "approver"): {
            "aiops:use",
            "aiops:operations_manage",
            "aiops:proposal:approve",
        },
        ("aiops", "manager"): permissions_by_app["aiops"],
    }
    actual_role_permissions: dict[tuple[str, str], set[str]] = {}
    for app_id, role_code, permission_code in (
        await connection.execute(
            text(
                "SELECT APP_ID, ROLE_CODE, PERMISSION_CODE "
                "FROM KBOT_APP_ROLE_PERMISSION"
            )
        )
    ).all():
        actual_role_permissions.setdefault(
            (str(app_id), str(role_code)), set()
        ).add(str(permission_code))
    incomplete_role_mappings = sorted(
        f"{app_id}/{role_code}"
        for (app_id, role_code), expected in expected_role_permissions.items()
        if not expected.issubset(
            actual_role_permissions.get((app_id, role_code), set())
        )
    )
    if incomplete_role_mappings:
        raise RuntimeError(
            "角色权限映射不完整：" + ", ".join(incomplete_role_mappings)
        )

    missing_role_permissions = (
        await connection.execute(
            text(
                "SELECT COUNT(*) FROM KBOT_PERMISSION permission "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM KBOT_APP_ROLE role_definition "
                "JOIN KBOT_APP_ROLE_PERMISSION role_permission "
                "ON role_permission.APP_ID = role_definition.APP_ID "
                "AND role_permission.ROLE_CODE = role_definition.ROLE_CODE "
                "WHERE role_definition.APP_ID = permission.APP_ID "
                "AND role_definition.ROLE_CODE = 'system_admin' "
                "AND role_definition.STATUS = 'ACTIVE' "
                "AND role_permission.PERMISSION_CODE = "
                "permission.PERMISSION_CODE)"
            )
        )
    ).scalar_one()
    if int(missing_role_permissions):
        raise RuntimeError(
            f"system_admin 缺少 {missing_role_permissions} 项权限映射"
        )

    missing_admin_memberships = (
        await connection.execute(
            text(
                "SELECT COUNT(DISTINCT permission.APP_ID) "
                "FROM KBOT_PERMISSION permission "
                "WHERE NOT EXISTS ("
                "SELECT 1 FROM KBOT_PLATFORM_DOMAIN domain_row "
                "JOIN KBOT_APP_MEMBER_ROLE member_role "
                "ON member_role.DOMAIN_ID = domain_row.DOMAIN_ID "
                "WHERE domain_row.NAME = 'default' "
                "AND domain_row.STATUS = 'ACTIVE' "
                "AND member_role.APP_ID = permission.APP_ID "
                "AND member_role.USER_ID = 'ADMIN' "
                "AND member_role.ROLE_CODE = 'system_admin' "
                "AND member_role.STATUS = 'ACTIVE')"
            )
        )
    ).scalar_one()
    if int(missing_admin_memberships):
        raise RuntimeError(
            f"ADMIN 在默认业务域缺少 {missing_admin_memberships} 个 App 授权"
        )


async def _apply_platform_foundation(
    connection: AsyncConnection,
) -> None:
    """幂等写入首次登录所需的平台基础数据。"""
    statements = split_oracle_statements(
        PLATFORM_FOUNDATION_SCRIPT.read_text(encoding="utf-8")
    )
    for statement in statements:
        await connection.exec_driver_sql(statement)
    await connection.commit()


async def maintain_platform_foundation(*, repair: bool) -> None:
    """检查或幂等修复既有 Schema 的平台首次登录基础数据。"""
    runtime = create_database_runtime()
    try:
        async with runtime.engine.connect() as connection:
            pdb_name, schema_name = await _read_target(connection)
            if repair:
                await _apply_platform_foundation(connection)
            await _validate_platform_foundation(connection)
            action = "修复并校验" if repair else "校验"
            print(
                f"平台基础数据{action}通过："
                f"PDB={pdb_name}，Schema={schema_name}，Domain=default，用户=ADMIN"
            )
    finally:
        await runtime.close()


async def _assert_ddl_privileges(connection: AsyncConnection) -> None:
    """在执行第一条 DDL 前检查最小权限和默认表空间额度。"""
    privileges = set(
        (
            await connection.execute(
                text(
                    """
                    SELECT privilege
                    FROM session_privs
                    WHERE privilege IN (
                        'CREATE TABLE',
                        'CREATE VIEW',
                        'UNLIMITED TABLESPACE'
                    )
                    """
                )
            )
        ).scalars()
    )
    missing = {"CREATE TABLE", "CREATE VIEW"} - privileges
    if missing:
        raise RuntimeError(
            "当前用户缺少 DDL 权限："
            f"{', '.join(sorted(missing))}；请由 PDB 管理员授权"
        )

    default_tablespace = (
        await connection.execute(
            text("SELECT default_tablespace FROM user_users")
        )
    ).scalar_one()
    if "UNLIMITED TABLESPACE" not in privileges:
        quota = (
            await connection.execute(
                text(
                    """
                    SELECT max_bytes
                    FROM user_ts_quotas
                    WHERE tablespace_name = :tablespace_name
                    """
                ),
                {"tablespace_name": default_tablespace},
            )
        ).scalar_one_or_none()
        if quota is None or quota == 0:
            raise RuntimeError(
                f"当前用户在默认表空间 {default_tablespace} 没有额度；"
                "请由 PDB 管理员设置应用表空间和 QUOTA"
            )

    segment_management = (
        await connection.execute(
            text(
                """
                SELECT segment_space_management
                FROM user_tablespaces
                WHERE tablespace_name = :tablespace_name
                """
            ),
            {"tablespace_name": default_tablespace},
        )
    ).scalar_one_or_none()
    if segment_management != "AUTO":
        raise RuntimeError(
            f"默认表空间 {default_tablespace} 不是 ASSM；"
            "Oracle VECTOR 要求 SEGMENT SPACE MANAGEMENT AUTO"
        )


async def _assert_kc_runtime_privileges(
    connection: AsyncConnection,
) -> None:
    """确认 Knowledge Core 服务账号能够发送 DBMS_ALERT。"""
    try:
        await connection.exec_driver_sql(
            "BEGIN DBMS_ALERT.SIGNAL("
            "'KBOT_KC_INSTALL_CHECK', 'READY'); END;"
        )
    except Exception as exc:
        raise RuntimeError(
            "Knowledge Core 需要执行 SYS.DBMS_ALERT；"
            "请由 PDB 管理员授权后重试"
        ) from exc


async def _validate_schema(
    connection: AsyncConnection,
    expected_tables: set[str],
    expected_views: set[str],
) -> None:
    rows = (
        await connection.execute(
            text(
                """
                SELECT object_name, object_type, status
                FROM user_objects
                WHERE object_name LIKE 'KBOT\\_%' ESCAPE '\\'
                  AND object_type IN ('TABLE', 'VIEW')
                ORDER BY object_type, object_name
                """
            )
        )
    ).all()
    actual_tables = {name for name, kind, _ in rows if kind == "TABLE"}
    actual_views = {name for name, kind, _ in rows if kind == "VIEW"}
    invalid = [(name, kind) for name, kind, status in rows if status != "VALID"]
    if actual_tables != expected_tables:
        raise RuntimeError(
            "建表结果不完整："
            f"缺少={sorted(expected_tables - actual_tables)}，"
            f"多出={sorted(actual_tables - expected_tables)}"
        )
    if actual_views != expected_views:
        raise RuntimeError(
            "视图结果不完整："
            f"缺少={sorted(expected_views - actual_views)}，"
            f"多出={sorted(actual_views - expected_views)}"
        )
    if invalid:
        raise RuntimeError(f"存在无效对象：{invalid}")

    bad_constraints = (
        await connection.execute(
            text(
                """
                SELECT constraint_name, status, validated
                FROM user_constraints
                WHERE table_name LIKE 'KBOT\\_%' ESCAPE '\\'
                  AND (status <> 'ENABLED' OR validated <> 'VALIDATED')
                ORDER BY constraint_name
                """
            )
        )
    ).all()
    if bad_constraints:
        raise RuntimeError(f"存在未启用或未验证约束：{bad_constraints}")

    bad_indexes = (
        await connection.execute(
            text(
                """
                SELECT index_name, status
                FROM user_indexes
                WHERE table_name LIKE 'KBOT\\_%' ESCAPE '\\'
                  AND status <> 'VALID'
                ORDER BY index_name
                """
            )
        )
    ).all()
    if bad_indexes:
        raise RuntimeError(f"存在无效索引：{bad_indexes}")

    if "KBOT_OPS_TARGET" not in expected_tables:
        return

    deferred_artifact_fks = (
        await connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM user_constraints
                WHERE table_name LIKE 'KBOT_OPS\\_%' ESCAPE '\\'
                  AND constraint_type = 'R'
                  AND deferrable = 'DEFERRABLE'
                  AND deferred = 'DEFERRED'
                """
            )
        )
    ).scalar_one()
    if deferred_artifact_fks != 5:
        raise RuntimeError(
            "AIOps 延后 Artifact 外键数量错误："
            f"{deferred_artifact_fks}，预期 5"
        )

    function_indexes = set(
        (
            await connection.execute(
                text(
                    """
                    SELECT index_name
                    FROM user_indexes
                    WHERE index_name IN (
                        'UX_OPS_POLICY_ACTIVE',
                        'UX_OPS_ALERT_ACTIVE',
                        'UX_OPS_HITL_PENDING',
                        'UX_OPS_REPORT_CURRENT'
                    )
                    """
                )
            )
        ).scalars()
    )
    expected_function_indexes = {
        "UX_OPS_POLICY_ACTIVE",
        "UX_OPS_ALERT_ACTIVE",
        "UX_OPS_HITL_PENDING",
        "UX_OPS_REPORT_CURRENT",
    }
    if function_indexes != expected_function_indexes:
        raise RuntimeError(
            "AIOps 函数唯一索引不完整："
            f"{sorted(expected_function_indexes - function_indexes)}"
        )

    unindexed_foreign_keys = (
        await connection.execute(
            text(
                """
                WITH fk_cols AS (
                    SELECT
                        c.constraint_name,
                        c.table_name,
                        LISTAGG(cc.column_name, ',')
                            WITHIN GROUP (ORDER BY cc.position) AS columns_csv
                    FROM user_constraints c
                    JOIN user_cons_columns cc
                      ON cc.constraint_name = c.constraint_name
                    WHERE c.constraint_type = 'R'
                      AND c.table_name LIKE 'KBOT_OPS\\_%' ESCAPE '\\'
                    GROUP BY c.constraint_name, c.table_name
                ),
                index_cols AS (
                    SELECT
                        table_name,
                        index_name,
                        LISTAGG(column_name, ',')
                            WITHIN GROUP (ORDER BY column_position)
                            AS columns_csv
                    FROM user_ind_columns
                    GROUP BY table_name, index_name
                )
                SELECT fk.constraint_name
                FROM fk_cols fk
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM index_cols ix
                    WHERE ix.table_name = fk.table_name
                      AND (
                          ix.columns_csv = fk.columns_csv
                          OR ix.columns_csv LIKE fk.columns_csv || ',%'
                      )
                )
                ORDER BY fk.constraint_name
                """
            )
        )
    ).scalars().all()
    if unindexed_foreign_keys:
        raise RuntimeError(
            "AIOps 存在未覆盖前导列的外键："
            f"{unindexed_foreign_keys}"
        )

    schema_version = (
        await connection.execute(
            text(
                """
                SELECT component, schema_version, contract_version
                FROM KBOT_V_OPS_SCHEMA_VERSION
                """
            )
        )
    ).one()
    if tuple(schema_version) != ("AIOPS", 8, "aiops-oracle-v1"):
        raise RuntimeError(f"AIOps Schema 版本错误：{tuple(schema_version)}")


async def _seed_prompt_catalog(
    connection: AsyncConnection,
    *,
    selected_services: set[str],
) -> int:
    """幂等写入统一 Prompt Catalog，并保持数据库较新 Active 版本。"""
    catalog = load_prompt_catalog()
    entries = catalog.for_services(selected_services)
    for entry in entries:
        row = (
            await connection.execute(
                text(
                    """
                    SELECT prompt_id, active_version_id
                    FROM KBOT_PLATFORM_PROMPT
                    WHERE prompt_key = :prompt_key
                    FOR UPDATE
                    """
                ),
                {"prompt_key": entry.prompt_key},
            )
        ).one_or_none()
        if row is None:
            prompt_id = uuid7().bytes
            await connection.execute(
                text(
                    """
                    INSERT INTO KBOT_PLATFORM_PROMPT (
                        prompt_id, prompt_key, owner_service, purpose,
                        active_version_id, row_version, created_by, updated_by
                    ) VALUES (
                        :prompt_id, :prompt_key, :owner_service, :purpose,
                        NULL, 1, 'schema-initializer', 'schema-initializer'
                    )
                    """
                ),
                {
                    "prompt_id": prompt_id,
                    "prompt_key": entry.prompt_key,
                    "owner_service": entry.owner_service,
                    "purpose": entry.purpose,
                },
            )
            active_version_id = None
        else:
            prompt_id, active_version_id = row

        version_row = (
            await connection.execute(
                text(
                    """
                    SELECT prompt_version_id, content_sha256
                    FROM KBOT_PLATFORM_PROMPT_VERSION
                    WHERE prompt_id = :prompt_id
                      AND version = :version
                    """
                ),
                {"prompt_id": prompt_id, "version": entry.version},
            )
        ).one_or_none()
        if version_row is not None:
            prompt_version_id, existing_hash = version_row
            if str(existing_hash) != entry.sha256:
                raise RuntimeError(
                    "Prompt 相同版本正文 Hash 冲突："
                    f"{entry.prompt_key}@{entry.version}"
                )
            await connection.execute(
                text(
                    """
                    UPDATE KBOT_PLATFORM_PROMPT_VERSION
                    SET status = :status
                    WHERE prompt_version_id = :prompt_version_id
                    """
                ),
                {
                    "status": "ACTIVE" if entry.active else "RETIRED",
                    "prompt_version_id": prompt_version_id,
                },
            )
        else:
            prompt_version_id = uuid7().bytes
            await connection.execute(
                text(
                    """
                    INSERT INTO KBOT_PLATFORM_PROMPT_VERSION (
                        prompt_version_id, prompt_id, version, content,
                        content_sha256, input_variables_json,
                        output_schema_ref, status, source, created_by
                    ) VALUES (
                        :prompt_version_id, :prompt_id, :version, :content,
                        :content_sha256, :input_variables_json,
                        :output_schema_ref, :status, 'FILE_SEED',
                        'schema-initializer'
                    )
                    """
                ),
                {
                    "prompt_version_id": prompt_version_id,
                    "prompt_id": prompt_id,
                    "version": entry.version,
                    "content": entry.content,
                    "content_sha256": entry.sha256,
                    "input_variables_json": (
                        "["
                        + ",".join(
                            f'"{value}"' for value in entry.input_variables
                        )
                        + "]"
                    ),
                    "output_schema_ref": entry.output_schema,
                    "status": "ACTIVE" if entry.active else "RETIRED",
                },
            )
        if entry.active:
            await connection.execute(
                text(
                    """
                    UPDATE KBOT_PLATFORM_PROMPT_VERSION
                    SET status = 'RETIRED'
                    WHERE prompt_id = :prompt_id
                      AND prompt_version_id <> :prompt_version_id
                      AND status = 'ACTIVE'
                    """
                ),
                {
                    "prompt_id": prompt_id,
                    "prompt_version_id": prompt_version_id,
                },
            )
            await connection.execute(
                text(
                    """
                    UPDATE KBOT_PLATFORM_PROMPT
                    SET active_version_id = :prompt_version_id,
                        row_version = row_version + 1,
                        updated_by = 'schema-initializer',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE prompt_id = :prompt_id
                    """
                ),
                {
                    "prompt_version_id": prompt_version_id,
                    "prompt_id": prompt_id,
                },
            )
    await connection.commit()
    return len(entries)


async def apply_schema(*, dry_run: bool, config_path: Path) -> None:
    """执行空库检查、DDL 和对象完整性校验。"""
    selection = load_service_selection(config_path)
    statements = load_schema_statements(selection.ordered)
    expected_tables = {
        match.group(1).upper()
        for item in statements
        if (
            match := re.match(
                r"\s*(?:--[^\n]*\n\s*)*CREATE\s+TABLE\s+([A-Z0-9_]+)",
                item.sql,
                flags=re.IGNORECASE,
            )
        )
    }
    expected_views = {
        match.group(1).upper()
        for item in statements
        if (
            match := re.match(
                r"\s*(?:--[^\n]*\n\s*)*CREATE\s+OR\s+REPLACE\s+VIEW"
                r"\s+([A-Z0-9_]+)",
                item.sql,
                flags=re.IGNORECASE,
            )
        )
    }
    enabled_label = ", ".join(selection.enabled) if selection.enabled else "无"
    prompt_catalog = load_prompt_catalog()
    selected_prompt_entries = prompt_catalog.for_services(
        set(selection.ordered)
    )
    print(
        f"初始化范围：必建={', '.join(selection.required)}；"
        f"已选服务={enabled_label}"
    )
    if dry_run:
        print(
            f"DDL 解析通过：{len(statements)} 条语句，"
            f"{len(expected_tables)} 张表，{len(expected_views)} 个视图；"
            f"Prompt={len(selected_prompt_entries)}，"
            f"Catalog Hash={prompt_catalog.catalog_sha256}"
        )
        return

    runtime = create_database_runtime()
    try:
        async with runtime.engine.connect() as connection:
            pdb_name, schema_name = await _read_target(connection)
            await _assert_empty_schema(connection)
            await _assert_ddl_privileges(connection)
            if "knowledge_core" in selection.enabled:
                await _assert_kc_runtime_privileges(connection)
            print(f"目标确认：PDB={pdb_name}，Schema={schema_name}")

            for position, statement in enumerate(statements, start=1):
                try:
                    await connection.exec_driver_sql(statement.sql)
                except Exception:
                    print(
                        "DDL 执行失败："
                        f"{statement.script} 第 {statement.ordinal} 条，"
                        f"{statement.label}"
                    )
                    raise
                print(
                    f"[{position}/{len(statements)}] "
                    f"{statement.script}: {statement.label}"
                )

            await _validate_schema(
                connection,
                expected_tables=expected_tables,
                expected_views=expected_views,
            )
            prompt_count = await _seed_prompt_catalog(
                connection,
                selected_services=set(selection.ordered),
            )
            foundation_label = "未选择 Main API"
            if "main_api" in selection.enabled:
                await _apply_platform_foundation(connection)
                await _validate_platform_foundation(connection)
                foundation_label = "default/ADMIN/system_admin"
            print(
                f"Schema 初始化完成：{len(expected_tables)} 张表，"
                f"{len(expected_views)} 个视图，"
                f"{prompt_count} 个 Prompt，平台基础数据={foundation_label}"
            )
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将 KBot 4.0 规范 DDL 应用到空白 Oracle Schema"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"服务选择 INI，默认 {DEFAULT_CONFIG_PATH.relative_to(ROOT)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析并统计 DDL，不连接或修改数据库",
    )
    foundation_group = parser.add_mutually_exclusive_group()
    foundation_group.add_argument(
        "--foundation-only",
        action="store_true",
        help="不执行 DDL，仅幂等修复并校验平台首次登录基础数据",
    )
    foundation_group.add_argument(
        "--check-foundation",
        action="store_true",
        help="只读校验平台首次登录基础数据",
    )
    args = parser.parse_args()
    try:
        if args.foundation_only or args.check_foundation:
            if args.dry_run:
                raise RuntimeError(
                    "--dry-run 不能与平台基础数据维护参数同时使用"
                )
            asyncio.run(
                maintain_platform_foundation(repair=args.foundation_only)
            )
        else:
            asyncio.run(
                apply_schema(
                    dry_run=args.dry_run,
                    config_path=args.config.expanduser().resolve(),
                )
            )
    except RuntimeError as exc:
        print(f"Schema 初始化拒绝：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
