"""幂等初始化智能工作台引导 Domain、权限和初始管理员。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BOOTSTRAP_SQL = (
    ROOT
    / "database"
    / "oracle"
    / "bootstrap"
    / "assistant"
    / "initial_admin.sql"
)
APP_ID = "assistant"
DOMAIN_NAME = "assistant_portal"
ADMIN_USER = "assistantadmin"
INITIAL_PASSWORD = "AssistantAdmin@2026!"


@dataclass(frozen=True)
class AssistantInitializationResult:
    """智能工作台初始化后的关键资源快照。"""

    pdb_name: str
    schema_name: str
    domain_id: int
    permission_count: int
    initial_password_created: bool | None


def _split_oracle_statements(sql: str) -> list[str]:
    """按分号拆分普通 SQL，并保留字符串和注释中的分号。"""
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
            in_string = True
            index += 1
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


def load_assistant_bootstrap_statements(
    path: Path = BOOTSTRAP_SQL,
) -> list[str]:
    """将 SQL Developer 脚本转换为驱动可执行的 SQL/PLSQL 语句。"""
    if not path.is_file():
        raise RuntimeError(f"智能工作台初始化 SQL 不存在：{path}")

    statements: list[str] = []
    ordinary_lines: list[str] = []
    plsql_lines: list[str] = []
    in_plsql = False

    def flush_ordinary() -> None:
        if not ordinary_lines:
            return
        statements.extend(_split_oracle_statements("\n".join(ordinary_lines)))
        ordinary_lines.clear()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        upper = stripped.upper()
        if not in_plsql and (
            not stripped
            or stripped.startswith("--")
            or upper.startswith("SET ")
            or upper.startswith("WHENEVER ")
        ):
            continue
        if not in_plsql and upper in {"DECLARE", "BEGIN"}:
            flush_ordinary()
            in_plsql = True
            plsql_lines.append(raw_line)
            continue
        if in_plsql:
            if stripped == "/":
                statement = "\n".join(plsql_lines).strip()
                if not statement:
                    raise RuntimeError("智能工作台初始化 SQL 包含空的 PLSQL 块")
                statements.append(statement)
                plsql_lines.clear()
                in_plsql = False
            else:
                plsql_lines.append(raw_line)
            continue
        ordinary_lines.append(raw_line)

    if in_plsql:
        raise RuntimeError("智能工作台初始化 SQL 存在未闭合的 PLSQL 块")
    flush_ordinary()
    return statements


async def _credential_exists(connection) -> bool:
    """读取初始管理员凭据是否已存在，不读取或输出凭据内容。"""
    from sqlalchemy import text

    return bool(
        (
            await connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM KBOT_PLATFORM_USER_CREDENTIAL
                    WHERE USER_ID = :user_id
                    """
                ),
                {"user_id": ADMIN_USER},
            )
        ).scalar_one()
    )


async def _validate_assistant_initialization(connection) -> AssistantInitializationResult:
    """校验初始管理员、引导 Domain 和 App 权限覆盖关系。"""
    from sqlalchemy import text

    target = (
        await connection.execute(
            text(
                """
                SELECT SYS_CONTEXT('USERENV', 'CON_NAME'),
                       SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                FROM DUAL
                """
            )
        )
    ).one()
    resource = (
        await connection.execute(
            text(
                """
                SELECT domain.DOMAIN_ID
                FROM KBOT_PLATFORM_DOMAIN domain
                JOIN KBOT_PLATFORM_APP app
                  ON app.APP_ID = :app_id
                 AND app.STATUS = 'ACTIVE'
                 AND app.MEMBER_ASSIGNABLE = 'Y'
                JOIN KBOT_PLATFORM_USER app_user
                  ON app_user.USER_ID = :admin_user
                 AND app_user.STATUS = 'ACTIVE'
                 AND app_user.ACCOUNT_ORIGIN = 'APP'
                 AND app_user.OWNER_APP_ID = app.APP_ID
                 AND app_user.IS_PROTECTED = 'Y'
                 AND app_user.MAX_SECURITY_LEVEL = 3
                JOIN KBOT_PLATFORM_USER_CREDENTIAL credential
                  ON credential.USER_ID = app_user.USER_ID
                 AND credential.PASSWORD_HASH IS NOT NULL
                JOIN KBOT_APP_DOMAIN app_domain
                  ON app_domain.APP_ID = app.APP_ID
                 AND app_domain.DOMAIN_ID = domain.DOMAIN_ID
                 AND app_domain.STATUS = 'ACTIVE'
                JOIN KBOT_APP_MEMBER member
                  ON member.APP_ID = app.APP_ID
                 AND member.USER_ID = app_user.USER_ID
                 AND member.MEMBER_SOURCE = 'APP_INITIAL_ADMIN'
                 AND member.IS_INITIAL_ADMIN = 'Y'
                 AND member.STATUS = 'ACTIVE'
                JOIN KBOT_APP_MEMBER_ROLE member_role
                  ON member_role.APP_ID = member.APP_ID
                 AND member_role.USER_ID = member.USER_ID
                 AND member_role.ROLE_CODE = 'app_admin'
                 AND member_role.SCOPE_MODE = 'ALL_APP_DOMAINS'
                 AND member_role.STATUS = 'ACTIVE'
                WHERE domain.NAME = :domain_name
                  AND domain.STATUS = 'ACTIVE'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM KBOT_APP_MEMBER_ROLE_SCOPE scope
                      WHERE scope.APP_ID = member_role.APP_ID
                        AND scope.USER_ID = member_role.USER_ID
                        AND scope.ROLE_CODE = member_role.ROLE_CODE
                  )
                """
            ),
            {
                "app_id": APP_ID,
                "admin_user": ADMIN_USER,
                "domain_name": DOMAIN_NAME,
            },
        )
    ).one_or_none()
    if resource is None:
        raise RuntimeError(
            "assistantadmin、assistant_portal、app_admin 或凭据初始化不完整"
        )

    missing_permissions = (
        await connection.execute(
            text(
                """
                SELECT permission.PERMISSION_CODE
                FROM KBOT_PERMISSION permission
                WHERE permission.APP_ID = :app_id
                MINUS
                SELECT role_permission.PERMISSION_CODE
                FROM KBOT_APP_ROLE_PERMISSION role_permission
                WHERE role_permission.APP_ID = :app_id
                  AND role_permission.ROLE_CODE = 'app_admin'
                """
            ),
            {"app_id": APP_ID},
        )
    ).scalars().all()
    if missing_permissions:
        raise RuntimeError(
            "assistantadmin 的 app_admin 角色缺少智能工作台权限："
            f"{', '.join(sorted(missing_permissions))}"
        )
    permission_count = int(
        (
            await connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM KBOT_PERMISSION
                    WHERE APP_ID = :app_id
                    """
                ),
                {"app_id": APP_ID},
            )
        ).scalar_one()
    )
    if permission_count == 0:
        raise RuntimeError("智能工作台权限目录为空")

    return AssistantInitializationResult(
        pdb_name=str(target[0]),
        schema_name=str(target[1]),
        domain_id=int(resource[0]),
        permission_count=permission_count,
        initial_password_created=None,
    )


async def initialize_assistant(
    *, check_only: bool = False,
) -> AssistantInitializationResult:
    """执行或只读校验智能工作台首次使用数据。"""
    from platform_core.config import get_settings
    from platform_core.database.oracle import create_database_runtime

    settings = get_settings()
    runtime = create_database_runtime(settings)
    try:
        async with runtime.engine.connect() as connection:
            credential_existed = await _credential_exists(connection)
            if not check_only:
                try:
                    for statement in load_assistant_bootstrap_statements():
                        if statement.strip().upper() == "COMMIT":
                            await connection.commit()
                        else:
                            await connection.exec_driver_sql(statement)
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
            result = await _validate_assistant_initialization(connection)
            return AssistantInitializationResult(
                pdb_name=result.pdb_name,
                schema_name=result.schema_name,
                domain_id=result.domain_id,
                permission_count=result.permission_count,
                initial_password_created=(
                    False if check_only else not credential_existed
                ),
            )
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="幂等初始化智能工作台引导 Domain、权限和初始管理员"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="不修改数据库，只读校验智能工作台初始化结果",
    )
    parser.add_argument(
        "--show-initial-password",
        action="store_true",
        help="仅在本次首次创建凭据时显示初始密码",
    )
    args = parser.parse_args()
    try:
        result = asyncio.run(initialize_assistant(check_only=args.check_only))
    except Exception as exc:
        print(f"智能工作台初始化失败：{exc}")
        return 1

    action = "校验通过" if args.check_only else "初始化完成"
    print(
        f"智能工作台{action}：PDB={result.pdb_name}，Schema={result.schema_name}，"
        f"Domain={DOMAIN_NAME}({result.domain_id})，权限={result.permission_count}，"
        f"用户={ADMIN_USER}"
    )
    if args.show_initial_password and result.initial_password_created:
        print(f"初始密码：{INITIAL_PASSWORD}；首次登录后请立即修改")
    elif args.show_initial_password and not args.check_only:
        print("初始管理员凭据已存在；为避免误导，不显示历史初始密码")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
