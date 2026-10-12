"""只读审计既有 Oracle App 授权数据，不输出用户标识或凭据。"""

from __future__ import annotations

import asyncio
import json

from sqlalchemy import text

from platform_core.authorization import (
    APP_AUTHORIZATION_POLICIES,
    BUILT_IN_APP_ROLE_CODES,
    built_in_role_permissions,
)
from platform_core.database.oracle import create_database_runtime


BUSINESS_APP_IDS = tuple(APP_AUTHORIZATION_POLICIES)


async def _rows(connection, sql: str) -> list[dict[str, object]]:
    result = await connection.execute(text(sql))
    return [
        {str(key).lower(): value for key, value in row.items()}
        for row in result.mappings().all()
    ]


async def audit() -> dict[str, object]:
    """读取聚合授权状态并生成可用于迁移决策的安全报告。"""
    runtime = create_database_runtime()
    try:
        async with runtime.engine.connect() as connection:
            target_row = (
                await connection.execute(text(
                    "SELECT SYS_CONTEXT('USERENV', 'CON_NAME') PDB_NAME, "
                    "SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') SCHEMA_NAME "
                    "FROM DUAL"
                ))
            ).mappings().one()
            target = {
                str(key).lower(): value for key, value in target_row.items()
            }
            required_tables = {
                "KBOT_APP_ROLE", "KBOT_APP_ROLE_PERMISSION",
                "KBOT_APP_MEMBER_ROLE", "KBOT_APP_MEMBER_ROLE_SCOPE",
                "KBOT_APP_DOMAIN", "KBOT_PERMISSION",
            }
            existing_tables = set((await connection.execute(text(
                "SELECT TABLE_NAME FROM USER_TABLES"
            ))).scalars().all())
            missing = sorted(required_tables - existing_tables)
            if missing:
                raise RuntimeError(
                    "当前 Schema 缺少授权表：" + ", ".join(missing)
                )

            roles = await _rows(connection, """
                SELECT role_row.APP_ID,
                       role_row.ROLE_CODE,
                       role_row.DISPLAY_NAME,
                       role_row.IS_SYSTEM,
                       role_row.SCOPE_POLICY,
                       role_row.STATUS,
                       role_row.ROW_VERSION,
                       COUNT(DISTINCT CASE
                           WHEN binding.STATUS = 'ACTIVE' THEN binding.USER_ID
                       END) ACTIVE_MEMBER_COUNT,
                       COUNT(DISTINCT CASE
                           WHEN binding.STATUS = 'DISABLED' THEN binding.USER_ID
                       END) DISABLED_MEMBER_COUNT,
                       COUNT(DISTINCT role_permission.PERMISSION_CODE)
                           PERMISSION_COUNT
                  FROM KBOT_APP_ROLE role_row
                  LEFT JOIN KBOT_APP_MEMBER_ROLE binding
                    ON binding.APP_ID = role_row.APP_ID
                   AND binding.ROLE_CODE = role_row.ROLE_CODE
                  LEFT JOIN KBOT_APP_ROLE_PERMISSION role_permission
                    ON role_permission.APP_ID = role_row.APP_ID
                   AND role_permission.ROLE_CODE = role_row.ROLE_CODE
                 WHERE role_row.APP_ID IN (
                     'knowledge_retrieval', 'km_asset',
                     'media_studio', 'aiops'
                 )
                 GROUP BY role_row.APP_ID, role_row.ROLE_CODE,
                          role_row.DISPLAY_NAME, role_row.IS_SYSTEM,
                          role_row.SCOPE_POLICY, role_row.STATUS,
                          role_row.ROW_VERSION
                 ORDER BY role_row.APP_ID, role_row.ROLE_CODE
            """)
            permissions = await _rows(connection, """
                SELECT role_permission.APP_ID,
                       role_permission.ROLE_CODE,
                       role_permission.PERMISSION_CODE
                  FROM KBOT_APP_ROLE_PERMISSION role_permission
                 WHERE role_permission.APP_ID IN (
                     'knowledge_retrieval', 'km_asset',
                     'media_studio', 'aiops'
                 )
                 ORDER BY role_permission.APP_ID,
                          role_permission.ROLE_CODE,
                          role_permission.PERMISSION_CODE
            """)
            scope_counts = await _rows(connection, """
                SELECT binding.APP_ID,
                       binding.ROLE_CODE,
                       binding.SCOPE_MODE,
                       COUNT(*) BINDING_COUNT,
                       COUNT(scope_row.DOMAIN_ID) SCOPE_ROW_COUNT
                  FROM KBOT_APP_MEMBER_ROLE binding
                  LEFT JOIN KBOT_APP_MEMBER_ROLE_SCOPE scope_row
                    ON scope_row.APP_ID = binding.APP_ID
                   AND scope_row.USER_ID = binding.USER_ID
                   AND scope_row.ROLE_CODE = binding.ROLE_CODE
                 WHERE binding.APP_ID IN (
                     'knowledge_retrieval', 'km_asset',
                     'media_studio', 'aiops'
                 )
                 GROUP BY binding.APP_ID, binding.ROLE_CODE,
                          binding.SCOPE_MODE
                 ORDER BY binding.APP_ID, binding.ROLE_CODE,
                          binding.SCOPE_MODE
            """)
            findings = await _rows(connection, """
                SELECT 'ORPHAN_BINDING' FINDING, COUNT(*) ITEM_COUNT
                  FROM KBOT_APP_MEMBER_ROLE binding
                 WHERE NOT EXISTS (
                     SELECT 1 FROM KBOT_APP_ROLE role_row
                      WHERE role_row.APP_ID = binding.APP_ID
                        AND role_row.ROLE_CODE = binding.ROLE_CODE
                 )
                UNION ALL
                SELECT 'ORPHAN_SCOPE', COUNT(*)
                  FROM KBOT_APP_MEMBER_ROLE_SCOPE scope_row
                 WHERE NOT EXISTS (
                     SELECT 1 FROM KBOT_APP_MEMBER_ROLE binding
                      WHERE binding.APP_ID = scope_row.APP_ID
                        AND binding.USER_ID = scope_row.USER_ID
                        AND binding.ROLE_CODE = scope_row.ROLE_CODE
                 )
                    OR NOT EXISTS (
                     SELECT 1 FROM KBOT_APP_DOMAIN app_domain
                      WHERE app_domain.APP_ID = scope_row.APP_ID
                        AND app_domain.DOMAIN_ID = scope_row.DOMAIN_ID
                 )
                UNION ALL
                SELECT 'CANONICAL_CUSTOM_ROLE_COLLISION', COUNT(*)
                  FROM KBOT_APP_ROLE role_row
                 WHERE role_row.APP_ID IN (
                     'knowledge_retrieval', 'km_asset',
                     'media_studio', 'aiops'
                 )
                   AND role_row.ROLE_CODE IN (
                     'app_user', 'app_manager', 'app_admin'
                 )
                   AND role_row.IS_SYSTEM <> 'Y'
                UNION ALL
                SELECT 'CUSTOM_RESERVED_PERMISSION', COUNT(*)
                  FROM KBOT_APP_ROLE_PERMISSION role_permission
                 WHERE role_permission.APP_ID IN (
                     'knowledge_retrieval', 'km_asset',
                     'media_studio', 'aiops'
                 )
                   AND role_permission.ROLE_CODE NOT IN (
                     'user', 'app_user', 'app_manager', 'app_admin'
                 )
                   AND role_permission.PERMISSION_CODE IN (
                     role_permission.APP_ID || ':member_manage',
                     role_permission.APP_ID || ':role_manage'
                 )
            """)
            active_domains = await _rows(connection, """
                SELECT APP_ID, COUNT(*) ACTIVE_DOMAIN_COUNT
                  FROM KBOT_APP_DOMAIN
                 WHERE STATUS = 'ACTIVE'
                   AND APP_ID IN (
                     'knowledge_retrieval', 'km_asset',
                     'media_studio', 'aiops'
                 )
                 GROUP BY APP_ID
                 ORDER BY APP_ID
            """)

        actual_permissions: dict[tuple[str, str], set[str]] = {}
        for row in permissions:
            key = (str(row["app_id"]), str(row["role_code"]))
            actual_permissions.setdefault(key, set()).add(
                str(row["permission_code"])
            )
        expected_roles = []
        for app_id in BUSINESS_APP_IDS:
            for role_code in BUILT_IN_APP_ROLE_CODES:
                expected = set(built_in_role_permissions(app_id, role_code))
                actual = actual_permissions.get((app_id, role_code), set())
                expected_roles.append({
                    "app_id": app_id,
                    "role_code": role_code,
                    "missing_permissions": sorted(expected - actual),
                    "excess_permissions": sorted(actual - expected),
                })

        return {
            "target": {
                "pdb": str(target["pdb_name"]),
                "schema": str(target["schema_name"]),
            },
            "roles": roles,
            "role_permissions": permissions,
            "scope_counts": scope_counts,
            "active_domains": active_domains,
            "findings": findings,
            "expected_system_role_diff": expected_roles,
            "safe_output": True,
        }
    finally:
        await runtime.close()


def main() -> int:
    try:
        report = asyncio.run(audit())
    except Exception as exc:
        print(f"App 权限审计失败：{exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
