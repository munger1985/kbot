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
FOUNDATION_VALIDATION_EXIT_CODE = 3


class FoundationValidationError(RuntimeError):
    """平台基础数据未满足启动约束。"""


PLATFORM_FOUNDATION_TABLES = {
    "KBOT_PLATFORM_DOMAIN",
    "KBOT_PLATFORM_APP",
    "KBOT_PLATFORM_USER",
    "KBOT_PLATFORM_USER_CREDENTIAL",
    "KBOT_PERMISSION",
    "KBOT_APP_ROLE",
    "KBOT_APP_ROLE_PERMISSION",
    "KBOT_PLATFORM_USER_ROLE",
    "KBOT_APP_DOMAIN",
    "KBOT_APP_MEMBER",
    "KBOT_APP_MEMBER_ROLE",
    "KBOT_APP_MEMBER_ROLE_SCOPE",
}
PLATFORM_FOUNDATION_PERMISSIONS = {
    "platform:user_manage",
    "platform:role_manage",
    "platform:domain_manage",
    "platform:app_manage",
    "platform:app_grant_manage",
    "knowledge_retrieval:use",
    "knowledge_retrieval:upload",
    "knowledge_retrieval:review",
    "knowledge_retrieval:member_manage",
    "knowledge_retrieval:role_manage",
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
    "km_asset:role_manage",
    "aiops:use",
    "aiops:domain_manage",
    "aiops:member_manage",
    "aiops:role_manage",
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
    ("knowledge_retrieval", "app_admin"),
    ("km_asset", "user"),
    ("km_asset", "app_admin"),
    ("aiops", "operator"),
    ("aiops", "approver"),
    ("aiops", "app_admin"),
}


def _expected_foundation_role_permissions() -> dict[tuple[str, str], set[str]]:
    """返回平台内置角色必须具备的权限映射。"""
    permissions_by_app: dict[str, set[str]] = {}
    for permission_code in PLATFORM_FOUNDATION_PERMISSIONS:
        permissions_by_app.setdefault(
            permission_code.partition(":")[0], set()
        ).add(permission_code)
    return {
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
        ("knowledge_retrieval", "app_admin"): permissions_by_app[
            "knowledge_retrieval"
        ],
        ("km_asset", "user"): {"km_asset:use"},
        ("km_asset", "app_admin"): permissions_by_app["km_asset"],
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
        ("aiops", "app_admin"): permissions_by_app["aiops"],
    }

from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7
from platform_core.prompts import load_prompt_catalog, sync_prompt_catalog


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
        raise FoundationValidationError(
            "平台基础表不完整：" + ", ".join(sorted(missing_tables))
        )

    security_column_count = (
        await connection.execute(
            text(
                "SELECT COUNT(*) FROM user_tab_columns "
                "WHERE table_name = 'KBOT_PLATFORM_USER' "
                "AND column_name = 'MAX_SECURITY_LEVEL' "
                "AND nullable = 'N'"
            )
        )
    ).scalar_one()
    if int(security_column_count) != 1:
        raise FoundationValidationError(
            "KBOT_PLATFORM_USER 缺少非空字段 MAX_SECURITY_LEVEL"
        )

    security_constraint_count = (
        await connection.execute(
            text(
                "SELECT COUNT(*) FROM user_constraints "
                "WHERE table_name = 'KBOT_PLATFORM_USER' "
                "AND constraint_name = 'CK_PLATFORM_USER_SECURITY' "
                "AND status = 'ENABLED'"
            )
        )
    ).scalar_one()
    if int(security_constraint_count) != 1:
        raise FoundationValidationError(
            "KBOT_PLATFORM_USER 缺少启用的安全等级范围约束"
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
        raise FoundationValidationError("缺少唯一且启用的默认业务域 default")

    admin_count = (
        await connection.execute(
            text(
                "SELECT COUNT(*) FROM KBOT_PLATFORM_USER user_account "
                "JOIN KBOT_PLATFORM_USER_CREDENTIAL credential "
                "ON credential.USER_ID = user_account.USER_ID "
                "WHERE user_account.USER_ID = 'ADMIN' "
                "AND user_account.STATUS = 'ACTIVE' "
                "AND user_account.ACCOUNT_ORIGIN = 'PLATFORM' "
                "AND user_account.OWNER_APP_ID IS NULL "
                "AND user_account.IS_PROTECTED = 'Y' "
                "AND user_account.MAX_SECURITY_LEVEL = 3"
            )
        )
    ).scalar_one()
    if int(admin_count) != 1:
        raise FoundationValidationError(
            "缺少启用且安全等级为 3 的 ADMIN 用户或登录凭据"
        )

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
        raise FoundationValidationError(
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
        raise FoundationValidationError(
            "平台角色模板不完整："
            + ", ".join(
                f"{app_id}/{role_code}"
                for app_id, role_code in sorted(missing_roles)
            )
        )

    expected_role_permissions = _expected_foundation_role_permissions()
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
        raise FoundationValidationError(
            "角色权限映射不完整：" + ", ".join(incomplete_role_mappings)
        )

    admin_platform_role_count = (
        await connection.execute(
            text(
                "SELECT COUNT(*) FROM KBOT_PLATFORM_USER_ROLE "
                "WHERE USER_ID = 'ADMIN' "
                "AND ROLE_CODE = 'platform_admin' "
                "AND STATUS = 'ACTIVE'"
            )
        )
    ).scalar_one()
    if int(admin_platform_role_count) != 1:
        raise FoundationValidationError(
            "ADMIN 缺少启用的 platform_admin 平台角色"
        )

    admin_app_memberships = (
        await connection.execute(
            text(
                "SELECT COUNT(*) FROM KBOT_APP_MEMBER "
                "WHERE USER_ID = 'ADMIN' "
                "AND (MEMBER_SOURCE <> 'PLATFORM_GRANT' "
                "OR IS_INITIAL_ADMIN <> 'N')"
            )
        )
    ).scalar_one()
    if int(admin_app_memberships):
        raise FoundationValidationError(
            "ADMIN 不得拥有非显式业务 App 成员资格"
        )


async def _repair_foundation_role_permissions(
    connection: AsyncConnection,
) -> None:
    """只补齐内置角色缺失的权限，不删除现有扩展映射。"""
    mappings = [
        {
            "app_id": app_id,
            "role_code": role_code,
            "permission_code": permission_code,
        }
        for (app_id, role_code), permission_codes in sorted(
            _expected_foundation_role_permissions().items()
        )
        for permission_code in sorted(permission_codes)
    ]
    await connection.execute(
        text(
            "INSERT INTO KBOT_APP_ROLE_PERMISSION "
            "(APP_ID, ROLE_CODE, PERMISSION_CODE) "
            "SELECT :app_id, :role_code, :permission_code FROM DUAL "
            "WHERE EXISTS ("
            "SELECT 1 FROM KBOT_APP_ROLE "
            "WHERE APP_ID = :app_id AND ROLE_CODE = :role_code"
            ") AND EXISTS ("
            "SELECT 1 FROM KBOT_PERMISSION "
            "WHERE PERMISSION_CODE = :permission_code AND APP_ID = :app_id"
            ") AND NOT EXISTS ("
            "SELECT 1 FROM KBOT_APP_ROLE_PERMISSION "
            "WHERE APP_ID = :app_id AND ROLE_CODE = :role_code "
            "AND PERMISSION_CODE = :permission_code"
            ")"
        ),
        mappings,
    )


async def _apply_platform_foundation(
    connection: AsyncConnection,
) -> None:
    """幂等写入首次登录所需的平台基础数据。"""
    security_column = (
        await connection.execute(
            text(
                "SELECT nullable, data_default FROM user_tab_columns "
                "WHERE table_name = 'KBOT_PLATFORM_USER' "
                "AND column_name = 'MAX_SECURITY_LEVEL'"
            )
        )
    ).first()
    if security_column is None:
        await connection.exec_driver_sql(
            "ALTER TABLE KBOT_PLATFORM_USER ADD ("
            "MAX_SECURITY_LEVEL NUMBER(3) DEFAULT 1 NOT NULL)"
        )
    else:
        nullable, data_default = security_column
        normalized_default = re.sub(
            r"\s+", "", str(data_default or "")
        ).strip("()")
        if normalized_default != "1":
            await connection.exec_driver_sql(
                "ALTER TABLE KBOT_PLATFORM_USER MODIFY ("
                "MAX_SECURITY_LEVEL DEFAULT 1)"
            )
        if str(nullable).upper() != "N":
            await connection.exec_driver_sql(
                "UPDATE KBOT_PLATFORM_USER SET MAX_SECURITY_LEVEL = 1 "
                "WHERE MAX_SECURITY_LEVEL IS NULL"
            )
            await connection.exec_driver_sql(
                "ALTER TABLE KBOT_PLATFORM_USER MODIFY ("
                "MAX_SECURITY_LEVEL NOT NULL)"
            )

    security_constraint_status = (
        await connection.execute(
            text(
                "SELECT status FROM user_constraints "
                "WHERE table_name = 'KBOT_PLATFORM_USER' "
                "AND constraint_name = 'CK_PLATFORM_USER_SECURITY'"
            )
        )
    ).scalar_one_or_none()
    if security_constraint_status is None:
        await connection.exec_driver_sql(
            "ALTER TABLE KBOT_PLATFORM_USER ADD CONSTRAINT "
            "CK_PLATFORM_USER_SECURITY CHECK "
            "(MAX_SECURITY_LEVEL BETWEEN 0 AND 3)"
        )
    elif str(security_constraint_status).upper() != "ENABLED":
        await connection.exec_driver_sql(
            "ALTER TABLE KBOT_PLATFORM_USER ENABLE CONSTRAINT "
            "CK_PLATFORM_USER_SECURITY"
        )
    statements = split_oracle_statements(
        PLATFORM_FOUNDATION_SCRIPT.read_text(encoding="utf-8")
    )
    for statement in statements:
        await connection.exec_driver_sql(statement)
    await _repair_foundation_role_permissions(connection)
    await connection.commit()


async def maintain_platform_foundation(*, repair: bool) -> None:
    """检查或幂等修复既有 Schema 的平台首次登录基础数据。"""
    runtime = create_database_runtime()
    try:
        async with runtime.engine.connect() as connection:
            pdb_name, schema_name = await _read_target(connection)
            if repair:
                await _apply_platform_foundation(connection)
            try:
                await _validate_platform_foundation(connection)
            except FoundationValidationError as exc:
                raise FoundationValidationError(
                    f"PDB={pdb_name}，Schema={schema_name}：{exc}"
                ) from exc
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

    settings = get_settings()
    runtime = create_database_runtime(settings)
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
            prompt_count = await sync_prompt_catalog(
                connection,
                selected_services=set(selection.ordered),
                environment=settings.environment,
                actor_id="schema-initializer",
            )
            foundation_label = "未选择 Main API"
            if "main_api" in selection.enabled:
                await _apply_platform_foundation(connection)
                await _validate_platform_foundation(connection)
                foundation_label = "default/ADMIN/platform_admin"
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
    except FoundationValidationError as exc:
        print(f"Schema 初始化拒绝：{exc}")
        if args.check_foundation:
            return FOUNDATION_VALIDATION_EXIT_CODE
        return 1
    except RuntimeError as exc:
        print(f"Schema 初始化拒绝：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
