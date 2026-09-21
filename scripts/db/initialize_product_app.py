"""幂等初始化知识检索或多媒体创作 App 的引导 Domain 与初始管理员。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class AppBootstrap:
    app_id: str
    display_name: str
    domain_name: str
    admin_user: str
    initial_password: str

    @property
    def sql_path(self) -> Path:
        return (
            ROOT / "database" / "oracle" / "bootstrap"
            / self.app_id / "initial_admin.sql"
        )


APP_BOOTSTRAPS = {
    "knowledge_retrieval": AppBootstrap(
        app_id="knowledge_retrieval",
        display_name="知识检索",
        domain_name="knowledge_retrieval_portal",
        admin_user="knowledgeadmin",
        initial_password="KnowledgeAdmin@2026!",
    ),
    "media_studio": AppBootstrap(
        app_id="media_studio",
        display_name="多媒体创作工作台",
        domain_name="media_studio_portal",
        admin_user="mediaadmin",
        initial_password="MediaAdmin@2026!",
    ),
}


@dataclass(frozen=True)
class AppInitializationResult:
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


def load_bootstrap_statements(config: AppBootstrap) -> list[str]:
    """把 SQL Developer 脚本转换为驱动可执行的 SQL/PLSQL 语句。"""
    path = config.sql_path
    if not path.is_file():
        raise RuntimeError(f"{config.display_name}初始化 SQL 不存在：{path}")
    statements: list[str] = []
    ordinary_lines: list[str] = []
    plsql_lines: list[str] = []
    in_plsql = False

    def flush_ordinary() -> None:
        if ordinary_lines:
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
                statements.append("\n".join(plsql_lines).strip())
                plsql_lines.clear()
                in_plsql = False
            else:
                plsql_lines.append(raw_line)
            continue
        ordinary_lines.append(raw_line)
    if in_plsql:
        raise RuntimeError(f"{config.display_name}初始化 SQL 存在未闭合的 PLSQL 块")
    flush_ordinary()
    return statements


async def _credential_exists(connection, config: AppBootstrap) -> bool:
    from sqlalchemy import text

    return bool((await connection.execute(
        text("SELECT COUNT(*) FROM KBOT_PLATFORM_USER_CREDENTIAL WHERE USER_ID = :user_id"),
        {"user_id": config.admin_user},
    )).scalar_one())


async def _validate(connection, config: AppBootstrap) -> AppInitializationResult:
    from sqlalchemy import text

    target = (await connection.execute(text(
        "SELECT SYS_CONTEXT('USERENV', 'CON_NAME'), "
        "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') FROM DUAL"
    ))).one()
    resource = (await connection.execute(text("""
        SELECT domain.DOMAIN_ID
          FROM KBOT_PLATFORM_DOMAIN domain
          JOIN KBOT_PLATFORM_APP app
            ON app.APP_ID = :app_id AND app.STATUS = 'ACTIVE'
          JOIN KBOT_PLATFORM_USER app_user
            ON app_user.USER_ID = :admin_user
           AND app_user.STATUS = 'ACTIVE'
           AND app_user.OWNER_APP_ID = app.APP_ID
          JOIN KBOT_PLATFORM_USER_CREDENTIAL credential
            ON credential.USER_ID = app_user.USER_ID
          JOIN KBOT_APP_DOMAIN app_domain
            ON app_domain.APP_ID = app.APP_ID
           AND app_domain.DOMAIN_ID = domain.DOMAIN_ID
           AND app_domain.STATUS = 'ACTIVE'
          JOIN KBOT_APP_MEMBER member
            ON member.APP_ID = app.APP_ID
           AND member.USER_ID = app_user.USER_ID
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
               SELECT 1 FROM KBOT_APP_MEMBER_ROLE_SCOPE scope
                WHERE scope.APP_ID = member_role.APP_ID
                  AND scope.USER_ID = member_role.USER_ID
                  AND scope.ROLE_CODE = member_role.ROLE_CODE
           )
    """), {
        "app_id": config.app_id,
        "admin_user": config.admin_user,
        "domain_name": config.domain_name,
    })).one_or_none()
    if resource is None:
        raise RuntimeError(f"{config.admin_user}、{config.domain_name} 或 app_admin 初始化不完整")
    missing = (await connection.execute(text("""
        SELECT PERMISSION_CODE FROM KBOT_PERMISSION WHERE APP_ID = :app_id
        MINUS
        SELECT PERMISSION_CODE FROM KBOT_APP_ROLE_PERMISSION
         WHERE APP_ID = :app_id AND ROLE_CODE = 'app_admin'
    """), {"app_id": config.app_id})).scalars().all()
    if missing:
        raise RuntimeError(f"{config.admin_user} 的 app_admin 缺少权限：{', '.join(sorted(missing))}")
    permission_count = int((await connection.execute(
        text("SELECT COUNT(*) FROM KBOT_PERMISSION WHERE APP_ID = :app_id"),
        {"app_id": config.app_id},
    )).scalar_one())
    return AppInitializationResult(
        pdb_name=str(target[0]),
        schema_name=str(target[1]),
        domain_id=int(resource[0]),
        permission_count=permission_count,
        initial_password_created=None,
    )


async def initialize_app(config: AppBootstrap, *, check_only: bool = False) -> AppInitializationResult:
    """执行或只读校验一个产品 App 的首次使用数据。"""
    from platform_core.config import get_settings
    from platform_core.database.oracle import create_database_runtime

    runtime = create_database_runtime(get_settings())
    try:
        async with runtime.engine.connect() as connection:
            credential_existed = await _credential_exists(connection, config)
            if not check_only:
                try:
                    for statement in load_bootstrap_statements(config):
                        if statement.strip().upper() == "COMMIT":
                            await connection.commit()
                        else:
                            await connection.exec_driver_sql(statement)
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
            result = await _validate(connection, config)
            return AppInitializationResult(
                pdb_name=result.pdb_name,
                schema_name=result.schema_name,
                domain_id=result.domain_id,
                permission_count=result.permission_count,
                initial_password_created=False if check_only else not credential_existed,
            )
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", choices=tuple(APP_BOOTSTRAPS))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--show-initial-password", action="store_true")
    args = parser.parse_args()
    config = APP_BOOTSTRAPS[args.app]
    try:
        result = asyncio.run(initialize_app(config, check_only=args.check_only))
    except Exception as exc:
        print(f"{config.display_name}初始化失败：{exc}")
        return 1
    action = "校验通过" if args.check_only else "初始化完成"
    print(
        f"{config.display_name}{action}：PDB={result.pdb_name}，Schema={result.schema_name}，"
        f"Domain={config.domain_name}({result.domain_id})，权限={result.permission_count}，"
        f"用户={config.admin_user}"
    )
    if args.show_initial_password and result.initial_password_created:
        print(f"初始密码：{config.initial_password}；首次登录后请立即修改")
    elif args.show_initial_password and not args.check_only:
        print("初始管理员凭据已存在；为避免误导，不显示历史初始密码")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
