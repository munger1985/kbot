"""仅初始化指定运维空间的固定案例资源；不改账号、资料库或模型绑定。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT, ROOT / "packages/platform_core/src", ROOT / "services/aiops_agent/src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from sqlalchemy import text

from aiops_agent.application.case_collection import case_collection_configuration
from platform_core.identity import uuid7


class CaseResourceProvisioningError(ValueError):
    """仅携带无敏感数据的业务诊断，不包装数据库异常。"""


def _json_object(value):
    """兼容 Oracle 驱动对 JSON 的对象与文本返回形式，不猜测业务字段。"""
    if hasattr(value, "read"):
        value = value.read()
    if isinstance(value, (str, bytes)):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise CaseResourceProvisioningError("固定案例资源 JSON 必须是对象")
    return value


async def provision_case_collection(
    connection, *, domain_id: int, actor_id: str, check_only: bool = False,
) -> dict:
    """锁定可信父空间，只补缺失资源；已有配置必须原样保留。"""
    if domain_id <= 0 or not actor_id.strip() or len(actor_id) > 256:
        raise CaseResourceProvisioningError("必须显式指定有效 Domain 和审计操作者")
    domain = (await connection.execute(text("""
        SELECT domain.DOMAIN_ID
          FROM KBOT_PLATFORM_DOMAIN domain
         WHERE domain.DOMAIN_ID=:domain_id AND domain.STATUS='ACTIVE'
           AND EXISTS (SELECT 1 FROM KBOT_APP_DOMAIN app_domain
                WHERE app_domain.DOMAIN_ID=domain.DOMAIN_ID
                  AND app_domain.APP_ID='aiops' AND app_domain.STATUS='ACTIVE')
           AND EXISTS (SELECT 1 FROM KBOT_PLATFORM_APP app
                WHERE app.APP_ID='aiops' AND app.STATUS='ACTIVE')
           AND NOT EXISTS (SELECT 1 FROM KBOT_APP_DOMAIN other_app
                WHERE other_app.DOMAIN_ID=domain.DOMAIN_ID
                  AND other_app.APP_ID<>'aiops' AND other_app.STATUS='ACTIVE')
    """ + ("" if check_only else " FOR UPDATE OF domain.DOMAIN_ID")),
        {"domain_id": domain_id})).scalar_one_or_none()
    if domain is None:
        raise CaseResourceProvisioningError("目标 Domain 不是已启用 AIOps 的活动独立空间")
    rows = (await connection.execute(text("""
        SELECT COLLECTION_ID AS "collection_id", DOMAIN_ID AS "domain_id",
               DISPLAY_NAME AS "display_name", STATUS AS "status",
               MODELS_JSON AS "models_json", PARSE_POLICY_JSON AS "parse_policy_json",
               METADATA_JSON AS "metadata_json"
          FROM KBOT_KC_COLLECTION
         WHERE DOMAIN_ID=:domain_id AND DISPLAY_NAME='diagnosis-cases'
    """), {"domain_id": domain_id})).mappings().all()
    if len(rows) > 1:
        raise CaseResourceProvisioningError("固定诊断案例资源重复；拒绝选择或合并")
    created = not rows
    if created:
        if check_only:
            raise CaseResourceProvisioningError("固定诊断案例资源缺失，请先执行定向初始化")
        row = {"collection_id": uuid7(), "domain_id": domain_id,
            "display_name": "diagnosis-cases", "status": "ACTIVE", "models_json": {},
            "parse_policy_json": {"parse_strategy": "TEXT"},
            "metadata_json": {"owner_app_id": "aiops", "fixed_resource": True,
                "content_kind": "diagnosis_case"}}
        await connection.execute(text("""
            INSERT INTO KBOT_KC_COLLECTION
              (COLLECTION_ID, DOMAIN_ID, DISPLAY_NAME, DESCRIPTION, MODELS_JSON,
               PARSE_POLICY_JSON, STATUS, DEFAULT_SECURITY_LEVEL, METADATA_JSON,
               ROW_VERSION, CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT)
            VALUES (HEXTORAW(:collection_hex), :domain_id, 'diagnosis-cases',
               'AIOps 固定诊断案例索引资源', :models_json, :parse_policy_json,
               'ACTIVE', 1, :metadata_json, 1, :actor_id, :actor_id,
               SYSTIMESTAMP, SYSTIMESTAMP)
        """), {"collection_hex": row["collection_id"].hex, "domain_id": domain_id,
            "actor_id": actor_id, **{key: json.dumps(row[key]) for key in
                ("models_json", "parse_policy_json", "metadata_json")}})
    else:
        row = dict(rows[0])
        for key in ("models_json", "parse_policy_json", "metadata_json"):
            row[key] = _json_object(row[key])
        if isinstance(row["collection_id"], bytes):
            from uuid import UUID
            row["collection_id"] = UUID(bytes=row["collection_id"])
    metadata = row["metadata_json"]
    if (int(row["domain_id"]) != domain_id or row["display_name"] != "diagnosis-cases"
        or row["status"] != "ACTIVE" or metadata.get("owner_app_id") != "aiops"
        or metadata.get("fixed_resource") is not True
        or metadata.get("content_kind") != "diagnosis_case"):
        raise CaseResourceProvisioningError("固定诊断案例资源身份或状态冲突；不会自动修复或重新激活")
    return {"created": created, "domain_id": str(domain_id),
        **case_collection_configuration({"collection_id": row["collection_id"],
            "models": row["models_json"], "parse_policy": row["parse_policy_json"]})}


async def run(*, domain_id: int, actor_id: str, check_only=False):
    """复用部署配置，只在明确指定的业务范围内提交。"""
    from platform_core.config import get_settings
    from platform_core.database.oracle import create_database_runtime

    runtime = create_database_runtime(get_settings())
    try:
        async with runtime.engine.connect() as connection:
            try:
                result = await provision_case_collection(connection, domain_id=domain_id,
                    actor_id=actor_id, check_only=check_only)
                if not check_only:
                    await connection.commit()
                return result
            finally:
                await connection.rollback()
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--check-only", action="store_true", help="只读核验资源与静态配置")
    args = parser.parse_args()
    try:
        result = asyncio.run(run(**vars(args)))
    except CaseResourceProvisioningError as exc:
        print(f"案例资源核验／初始化失败：{exc}")
        return 1
    except Exception:
        # 驱动异常可能含连接详情或参数，不向终端直接打印原始异常。
        print("案例资源核验／初始化失败；请核对目标、资源身份和部署连接，不执行全库初始化")
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
