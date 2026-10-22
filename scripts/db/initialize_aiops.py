"""幂等初始化 AIOps 管理员、固定 Domain、默认知识库和诊断案例集合。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime
from scripts.db.apply_oracle_schema import split_oracle_statements


BOOTSTRAP_SQL = (
    ROOT
    / "database"
    / "oracle"
    / "bootstrap"
    / "aiops"
    / "initial_admin.sql"
)
DOMAIN_NAME = "aiops_portal"
COLLECTION_NAME = "operations-manuals"
DEFAULT_KNOWLEDGE_BASE_ID = "019ffff0-0000-7000-8000-000000001001"
CASE_COLLECTION_NAME = "diagnosis-cases"
ADMIN_USER = "aiopsadmin"
INITIAL_PASSWORD = "AIOpsAdmin@2026!"


@dataclass(frozen=True)
class AIOpsInitializationResult:
    """AIOps 初始化后的关键资源快照。"""

    pdb_name: str
    schema_name: str
    domain_id: int
    knowledge_base_id: str
    collection_id: str
    case_collection_id: str
    permission_count: int


def load_aiops_bootstrap_statements(path: Path = BOOTSTRAP_SQL) -> list[str]:
    """将 SQL Developer 脚本转换为驱动可执行语句。"""
    if not path.is_file():
        raise RuntimeError(f"AIOps 初始化 SQL 不存在：{path}")
    statements: list[str] = []
    ordinary: list[str] = []
    plsql: list[str] = []
    in_plsql = False

    def flush() -> None:
        if ordinary:
            statements.extend(split_oracle_statements("\n".join(ordinary)))
            ordinary.clear()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        upper = stripped.upper()
        if not in_plsql and (
            not stripped or stripped.startswith("--")
            or upper.startswith("SET ") or upper.startswith("WHENEVER ")
        ):
            continue
        if not in_plsql and upper in {"DECLARE", "BEGIN"}:
            flush()
            in_plsql = True
            plsql.append(raw_line)
            continue
        if in_plsql:
            if stripped == "/":
                statements.append("\n".join(plsql).strip())
                plsql.clear()
                in_plsql = False
            else:
                plsql.append(raw_line)
            continue
        ordinary.append(raw_line)
    if in_plsql:
        raise RuntimeError("AIOps 初始化 SQL 存在未闭合的 PLSQL 块")
    flush()
    return statements


async def _validate(connection) -> AIOpsInitializationResult:
    target = (await connection.execute(text(
        "SELECT SYS_CONTEXT('USERENV','CON_NAME'), "
        "SYS_CONTEXT('USERENV','CURRENT_SCHEMA') FROM DUAL"
    ))).one()
    resource = (await connection.execute(text("""
        SELECT domain.DOMAIN_ID,
               LOWER(SUBSTR(RAWTOHEX(collection.COLLECTION_ID),1,8)||'-'||
                     SUBSTR(RAWTOHEX(collection.COLLECTION_ID),9,4)||'-'||
                     SUBSTR(RAWTOHEX(collection.COLLECTION_ID),13,4)||'-'||
                     SUBSTR(RAWTOHEX(collection.COLLECTION_ID),17,4)||'-'||
                     SUBSTR(RAWTOHEX(collection.COLLECTION_ID),21,12))
          FROM KBOT_PLATFORM_DOMAIN domain
          JOIN KBOT_KC_COLLECTION collection ON collection.DOMAIN_ID=domain.DOMAIN_ID
          JOIN KBOT_PLATFORM_USER app_user ON app_user.USER_ID='aiopsadmin'
             AND app_user.STATUS='ACTIVE' AND app_user.OWNER_APP_ID='aiops'
          JOIN KBOT_PLATFORM_USER_CREDENTIAL credential
             ON credential.USER_ID=app_user.USER_ID AND credential.PASSWORD_HASH IS NOT NULL
          JOIN KBOT_APP_DOMAIN app_domain ON app_domain.APP_ID='aiops'
             AND app_domain.DOMAIN_ID=domain.DOMAIN_ID AND app_domain.STATUS='ACTIVE'
          JOIN KBOT_APP_MEMBER member ON member.APP_ID='aiops'
             AND member.USER_ID=app_user.USER_ID AND member.STATUS='ACTIVE'
          JOIN KBOT_APP_MEMBER_ROLE member_role ON member_role.APP_ID='aiops'
             AND member_role.USER_ID=app_user.USER_ID
             AND member_role.ROLE_CODE='app_admin'
             AND member_role.STATUS='ACTIVE'
             AND member_role.SCOPE_MODE='ALL_APP_DOMAINS'
          JOIN KBOT_OPS_KNOWLEDGE_BASE knowledge_base
            ON knowledge_base.DOMAIN_ID=domain.DOMAIN_ID
           AND knowledge_base.COLLECTION_ID=collection.COLLECTION_ID
           AND knowledge_base.KNOWLEDGE_BASE_ID=HEXTORAW(REPLACE(
               '019ffff0-0000-7000-8000-000000001001','-',''
           ))
           AND knowledge_base.STATUS='ACTIVE'
         WHERE domain.NAME='aiops_portal' AND domain.STATUS='ACTIVE'
           AND collection.DISPLAY_NAME='operations-manuals'
           AND collection.STATUS='ACTIVE' AND collection.MODELS_JSON IS NOT NULL
    """))).one_or_none()
    if resource is None:
        raise RuntimeError("aiopsadmin、aiops_portal 或默认运维知识库初始化不完整")
    case_collection_id = (await connection.execute(text("""
        SELECT LOWER(SUBSTR(RAWTOHEX(collection.COLLECTION_ID),1,8)||'-'||
                     SUBSTR(RAWTOHEX(collection.COLLECTION_ID),9,4)||'-'||
                     SUBSTR(RAWTOHEX(collection.COLLECTION_ID),13,4)||'-'||
                     SUBSTR(RAWTOHEX(collection.COLLECTION_ID),17,4)||'-'||
                     SUBSTR(RAWTOHEX(collection.COLLECTION_ID),21,12))
          FROM KBOT_KC_COLLECTION collection
         WHERE collection.DOMAIN_ID=:domain_id
           AND collection.DISPLAY_NAME='diagnosis-cases'
           AND collection.STATUS='ACTIVE'
           AND JSON_VALUE(collection.METADATA_JSON, '$.owner_app_id')='aiops'
           AND JSON_VALUE(collection.METADATA_JSON, '$.fixed_resource')='true'
    """), {"domain_id": int(resource[0])})).scalar_one_or_none()
    if case_collection_id is None:
        raise RuntimeError("固定诊断案例集合初始化不完整")
    missing = (await connection.execute(text("""
        SELECT PERMISSION_CODE FROM KBOT_PERMISSION WHERE APP_ID='aiops'
        MINUS
        SELECT PERMISSION_CODE FROM KBOT_APP_ROLE_PERMISSION
         WHERE APP_ID='aiops' AND ROLE_CODE='app_admin'
    """))).scalars().all()
    if missing:
        raise RuntimeError(f"AIOps app_admin 缺少权限：{', '.join(sorted(missing))}")
    permission_count = int((await connection.execute(text(
        "SELECT COUNT(*) FROM KBOT_PERMISSION WHERE APP_ID='aiops'"
    ))).scalar_one())
    return AIOpsInitializationResult(
        pdb_name=str(target[0]), schema_name=str(target[1]),
        domain_id=int(resource[0]),
        knowledge_base_id=DEFAULT_KNOWLEDGE_BASE_ID,
        collection_id=str(resource[1]),
        case_collection_id=str(case_collection_id),
        permission_count=permission_count,
    )


async def initialize_aiops(
    *, check_only: bool = False,
) -> AIOpsInitializationResult:
    settings = get_settings()
    runtime = create_database_runtime(settings)
    try:
        async with runtime.engine.connect() as connection:
            if not check_only:
                try:
                    for statement in load_aiops_bootstrap_statements():
                        if statement.strip().upper() == "COMMIT":
                            await connection.commit()
                        else:
                            await connection.exec_driver_sql(statement)
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
            result = await _validate(connection)
        return result
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只读校验初始化结果")
    args = parser.parse_args()
    try:
        result = asyncio.run(initialize_aiops(
            check_only=args.check_only,
        ))
    except Exception as exc:
        print(f"AIOps 初始化失败：{exc}")
        return 1
    action = "校验通过" if args.check_only else "初始化完成"
    print(
        f"AIOps {action}：PDB={result.pdb_name}，Schema={result.schema_name}，"
        f"Domain={DOMAIN_NAME}({result.domain_id})，"
        f"KnowledgeBase=默认运维知识库({result.knowledge_base_id})，"
        f"Collection={COLLECTION_NAME}({result.collection_id})，"
        f"CaseCollection={CASE_COLLECTION_NAME}({result.case_collection_id})，"
        f"权限={result.permission_count}，用户={ADMIN_USER}"
    )
    if not args.check_only:
        print(f"初始密码：{INITIAL_PASSWORD}；首次登录后请立即修改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
