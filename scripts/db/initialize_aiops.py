"""幂等初始化 AIOps 管理员、固定 Domain、KC Collection 和运维手册。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

import aiohttp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from sqlalchemy import text

from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime
from scripts.db.apply_oracle_schema import split_oracle_statements


BOOTSTRAP_SQL = ROOT / "scripts" / "db" / "bootstrap_aiops_initial_admin.sql"
MANUAL_PATH = (
    ROOT / "services" / "aiops_agent" / "resources" / "knowledge"
    / "database-operations-manual.md"
)
DOMAIN_NAME = "aiops_portal"
COLLECTION_NAME = "operations-manuals"
ADMIN_USER = "aiopsadmin"
INITIAL_PASSWORD = "AIOpsAdmin@2026!"
MANUAL_SOURCE_ID = "aiops-database-operations-manual"


@dataclass(frozen=True)
class AIOpsInitializationResult:
    """AIOps 初始化后的关键资源快照。"""

    pdb_name: str
    schema_name: str
    domain_id: int
    collection_id: str
    permission_count: int
    manual_status: str | None


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
         WHERE domain.NAME='aiops_portal' AND domain.STATUS='ACTIVE'
           AND collection.DISPLAY_NAME='operations-manuals'
           AND collection.STATUS='ACTIVE' AND collection.MODELS_JSON IS NOT NULL
    """))).one_or_none()
    if resource is None:
        raise RuntimeError("aiopsadmin、aiops_portal 或固定 KC 初始化不完整")
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
    manual_status = (await connection.execute(text("""
        SELECT revision.APPROVAL_STATUS
          FROM KBOT_KC_BUNDLE bundle
          JOIN KBOT_KC_BUNDLE_REVISION revision
            ON revision.BUNDLE_ID=bundle.BUNDLE_ID
         WHERE bundle.COLLECTION_ID=HEXTORAW(REPLACE(:collection_id,'-',''))
           AND bundle.SOURCE_SYSTEM='kbot'
           AND bundle.SOURCE_TYPE='USER_UPLOAD'
           AND bundle.SOURCE_ID=:source_id
         ORDER BY revision.REVISION_NO DESC
         FETCH FIRST 1 ROWS ONLY
    """), {
        "collection_id": str(resource[1]), "source_id": MANUAL_SOURCE_ID,
    })).scalar_one_or_none()
    return AIOpsInitializationResult(
        pdb_name=str(target[0]), schema_name=str(target[1]),
        domain_id=int(resource[0]), collection_id=str(resource[1]),
        permission_count=permission_count,
        manual_status=str(manual_status) if manual_status else None,
    )


def _configured_main_api_url() -> str:
    configured = Path(os.getenv("KBOT_CONFIG_FILE") or ROOT / "configuration" / "kbot.toml")
    if not configured.is_absolute():
        configured = (ROOT / configured).resolve()
    with configured.open("rb") as stream:
        value = str((tomllib.load(stream).get("ui") or {}).get("main_api_base_url") or "")
    if not value.startswith(("http://", "https://")):
        raise RuntimeError("kbot.toml 缺少合法的 [ui].main_api_base_url")
    return value.rstrip("/")


async def _upload_manual(*, base_url: str) -> None:
    content = MANUAL_PATH.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    idempotency_key = f"aiops-manual:{digest}"
    declaration = [{
        "part_name": "manual",
        "client_file_id": MANUAL_SOURCE_ID,
        "display_name": "KBot AIOps 数据库运维手册",
        "declared_mime_type": "text/markdown",
        "byte_size": len(content),
        "content_sha256": digest,
        "ordinal": 0,
        "role": "CONTENT",
        "required_flag": True,
    }]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async with session.post(
            f"{base_url}/api/v1/apps/aiops/auth/login",
            json={"user_id": ADMIN_USER, "password": INITIAL_PASSWORD},
        ) as response:
            login = await response.json()
            if response.status != 200:
                raise RuntimeError(f"AIOps 初始化用户登录失败：HTTP {response.status} {login}")
        token = str(login["access_token"])
        form = aiohttp.FormData()
        form.add_field("grouping_mode", "EACH_FILE")
        form.add_field("files", json.dumps(declaration, ensure_ascii=False))
        form.add_field(
            "manual", content, filename=MANUAL_PATH.name,
            content_type="text/markdown",
        )
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": idempotency_key}
        async with session.post(
            f"{base_url}/api/v1/apps/aiops/knowledge-core/manuals",
            data=form, headers=headers,
        ) as response:
            payload = await response.json()
            if response.status != 202:
                raise RuntimeError(f"AIOps 运维手册入库失败：HTTP {response.status} {payload}")
        items = payload.get("items") or []
        if not items or items[0].get("status") == "REJECTED":
            raise RuntimeError(f"AIOps 运维手册未被 KC 受理：{payload}")
        revision_id = items[0].get("bundle_revision_id")
        async with session.post(
            f"{base_url}/api/v1/apps/aiops/knowledge-core/manuals/{revision_id}/approve",
            json={"comment": "AIOps 初始化脚本批准固定运维手册"},
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            approval = await response.json()
            if response.status != 200:
                raise RuntimeError(f"AIOps 运维手册审批失败：HTTP {response.status} {approval}")


async def initialize_aiops(
    *, check_only: bool = False, skip_manual_upload: bool = False,
    main_api_url: str | None = None,
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
        if not check_only and not skip_manual_upload and result.manual_status != "APPROVED":
            await _upload_manual(base_url=main_api_url or _configured_main_api_url())
            async with runtime.engine.connect() as connection:
                result = await _validate(connection)
        if not skip_manual_upload and result.manual_status != "APPROVED":
            raise RuntimeError("固定运维手册尚未在 KC 中批准")
        return result
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="只读校验初始化结果")
    parser.add_argument("--skip-manual-upload", action="store_true", help="仅初始化数据库资源，不调用 KC")
    parser.add_argument("--main-api-url", help="覆盖 kbot.toml 中的 Main API URL")
    args = parser.parse_args()
    try:
        result = asyncio.run(initialize_aiops(
            check_only=args.check_only,
            skip_manual_upload=args.skip_manual_upload,
            main_api_url=args.main_api_url,
        ))
    except Exception as exc:
        print(f"AIOps 初始化失败：{exc}")
        return 1
    action = "校验通过" if args.check_only else "初始化完成"
    print(
        f"AIOps {action}：PDB={result.pdb_name}，Schema={result.schema_name}，"
        f"Domain={DOMAIN_NAME}({result.domain_id})，"
        f"Collection={COLLECTION_NAME}({result.collection_id})，"
        f"权限={result.permission_count}，用户={ADMIN_USER}，"
        f"运维手册={result.manual_status or '未上传'}"
    )
    if not args.check_only:
        print(f"初始密码：{INITIAL_PASSWORD}；首次登录后请立即修改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
