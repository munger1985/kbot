"""幂等初始化 KM 管理员、权限和固定 Knowledge Core Collection。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime

from scripts.db.apply_oracle_schema import split_oracle_statements


BOOTSTRAP_SQL = ROOT / "scripts" / "db" / "bootstrap_km_initial_admin.sql"


@dataclass(frozen=True)
class KmInitializationResult:
    """KM 初始化后的关键资源快照。"""

    pdb_name: str
    schema_name: str
    domain_id: int
    collection_id: str
    permission_count: int


def load_km_bootstrap_statements(path: Path = BOOTSTRAP_SQL) -> list[str]:
    """将 SQL Developer 脚本转换为驱动可执行的 SQL/PLSQL 语句。"""
    if not path.is_file():
        raise RuntimeError(f"KM 初始化 SQL 不存在：{path}")

    statements: list[str] = []
    ordinary_lines: list[str] = []
    plsql_lines: list[str] = []
    in_plsql = False

    def flush_ordinary() -> None:
        if not ordinary_lines:
            return
        statements.extend(split_oracle_statements("\n".join(ordinary_lines)))
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
                    raise RuntimeError("KM 初始化 SQL 包含空的 PLSQL 块")
                statements.append(statement)
                plsql_lines.clear()
                in_plsql = False
            else:
                plsql_lines.append(raw_line)
            continue
        ordinary_lines.append(raw_line)

    if in_plsql:
        raise RuntimeError("KM 初始化 SQL 存在未闭合的 PLSQL 块")
    flush_ordinary()
    return statements


async def _validate_km_initialization(connection) -> KmInitializationResult:
    """校验 kmadmin、全部 KM 权限和固定 KC Collection。"""
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
                SELECT domain.DOMAIN_ID,
                       LOWER(
                           SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 1, 8)
                           || '-' || SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 9, 4)
                           || '-' || SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 13, 4)
                           || '-' || SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 17, 4)
                           || '-' || SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 21, 12)
                       ) AS COLLECTION_ID
                FROM KBOT_PLATFORM_DOMAIN domain
                JOIN KBOT_KC_COLLECTION collection
                  ON collection.DOMAIN_ID = domain.DOMAIN_ID
                JOIN KBOT_PLATFORM_USER app_user
                  ON app_user.USER_ID = 'kmadmin'
                 AND app_user.STATUS = 'ACTIVE'
                 AND app_user.OWNER_APP_ID = 'km_asset'
                JOIN KBOT_PLATFORM_USER_CREDENTIAL credential
                  ON credential.USER_ID = app_user.USER_ID
                 AND credential.PASSWORD_HASH IS NOT NULL
                JOIN KBOT_APP_DOMAIN app_domain
                  ON app_domain.APP_ID = 'km_asset'
                 AND app_domain.DOMAIN_ID = domain.DOMAIN_ID
                 AND app_domain.STATUS = 'ACTIVE'
                JOIN KBOT_APP_MEMBER member
                  ON member.APP_ID = 'km_asset'
                 AND member.USER_ID = app_user.USER_ID
                 AND member.STATUS = 'ACTIVE'
                JOIN KBOT_APP_MEMBER_ROLE member_role
                  ON member_role.APP_ID = member.APP_ID
                 AND member_role.USER_ID = member.USER_ID
                 AND member_role.ROLE_CODE = 'app_admin'
                 AND member_role.STATUS = 'ACTIVE'
                 AND member_role.SCOPE_MODE = 'ALL_APP_DOMAINS'
                WHERE domain.NAME = 'km_portal'
                  AND domain.STATUS = 'ACTIVE'
                  AND collection.DISPLAY_NAME = 'assets'
                  AND collection.STATUS = 'ACTIVE'
                  AND collection.MODELS_JSON IS NOT NULL
                """
            )
        )
    ).one_or_none()
    if resource is None:
        raise RuntimeError(
            "kmadmin、km_portal、app_admin 或 assets Collection 初始化不完整"
        )

    missing_permissions = (
        await connection.execute(
            text(
                """
                SELECT permission.PERMISSION_CODE
                FROM KBOT_PERMISSION permission
                WHERE permission.APP_ID = 'km_asset'
                MINUS
                SELECT role_permission.PERMISSION_CODE
                FROM KBOT_APP_ROLE_PERMISSION role_permission
                WHERE role_permission.APP_ID = 'km_asset'
                  AND role_permission.ROLE_CODE = 'app_admin'
                """
            )
        )
    ).scalars().all()
    if missing_permissions:
        raise RuntimeError(
            "kmadmin 的 app_admin 角色缺少 KM 权限："
            f"{', '.join(sorted(missing_permissions))}"
        )
    permission_count = int(
        (
            await connection.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM KBOT_PERMISSION
                    WHERE APP_ID = 'km_asset'
                    """
                )
            )
        ).scalar_one()
    )
    if permission_count == 0:
        raise RuntimeError("KM 权限目录为空")

    return KmInitializationResult(
        pdb_name=str(target[0]),
        schema_name=str(target[1]),
        domain_id=int(resource[0]),
        collection_id=str(resource[1]),
        permission_count=permission_count,
    )


async def initialize_km(*, check_only: bool = False) -> KmInitializationResult:
    """执行或只读校验 KM 首次使用数据。"""
    settings = get_settings()
    runtime = create_database_runtime(settings)
    try:
        async with runtime.engine.connect() as connection:
            if not check_only:
                statements = load_km_bootstrap_statements()
                try:
                    for statement in statements:
                        if statement.strip().upper() == "COMMIT":
                            await connection.commit()
                        else:
                            await connection.exec_driver_sql(statement)
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
            return await _validate_km_initialization(connection)
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="幂等初始化 kmadmin、KM 全部权限和固定 KC Collection"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="不修改数据库，只读校验 KM 初始化结果",
    )
    args = parser.parse_args()
    try:
        result = asyncio.run(initialize_km(check_only=args.check_only))
    except Exception as exc:
        print(f"KM 初始化失败：{exc}")
        return 1
    action = "校验通过" if args.check_only else "初始化完成"
    print(
        f"KM {action}：PDB={result.pdb_name}，Schema={result.schema_name}，"
        f"Domain=km_portal({result.domain_id})，"
        f"Collection=assets({result.collection_id})，"
        f"权限={result.permission_count}，用户=kmadmin"
    )
    if not args.check_only:
        print("初始密码：KmAdmin@2026!；首次登录后请立即修改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
