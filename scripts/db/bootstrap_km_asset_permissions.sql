-- 为现有 KBot Schema 幂等补充 KM Asset 权限、角色及角色权限关系。
-- 本脚本用于 SQL Developer：粘贴后使用 Run Script（F5）执行。
-- 不创建用户或 App 成员；幂等修复初始管理员及旧 manager 的 app_admin 角色绑定。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_table_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME IN (
        'KBOT_PERMISSION',
        'KBOT_APP_ROLE',
        'KBOT_APP_ROLE_PERMISSION',
        'KBOT_APP_MEMBER',
        'KBOT_APP_MEMBER_ROLE',
        'KBOT_APP_MEMBER_ROLE_SCOPE'
     );

    IF l_table_count <> 6 THEN
        raise_application_error(
            -20001,
            'Main API 权限基础表不完整，不能初始化 KM Asset 权限。'
        );
    END IF;
END;
/

MERGE INTO KBOT_PERMISSION target
USING (
    SELECT 'km_asset:use' AS PERMISSION_CODE,
           'km_asset' AS APP_ID,
           '使用 KM Asset' AS DISPLAY_NAME FROM DUAL
    UNION ALL
    SELECT 'km_asset:source_manage', 'km_asset', '管理 KM Asset 来源' FROM DUAL
    UNION ALL
    SELECT 'km_asset:data_manage', 'km_asset', '管理 KM Asset 问数模型' FROM DUAL
    UNION ALL
    SELECT 'km_asset:agent_manage', 'km_asset', '管理 KM Asset Agent' FROM DUAL
    UNION ALL
    SELECT 'km_asset:operations_manage', 'km_asset', '管理 KM Asset 同步运行' FROM DUAL
    UNION ALL
    SELECT 'km_asset:member_manage', 'km_asset', '管理 KM Asset 成员' FROM DUAL
    UNION ALL
    SELECT 'km_asset:role_manage', 'km_asset', '管理 KM Asset 角色' FROM DUAL
    UNION ALL
    SELECT 'km_asset:api_key_manage', 'km_asset', '管理 KM Asset API Client' FROM DUAL
) source
ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN MATCHED THEN
    UPDATE SET
        target.APP_ID = source.APP_ID,
        target.DISPLAY_NAME = source.DISPLAY_NAME
WHEN NOT MATCHED THEN
    INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
    VALUES (source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME);

MERGE INTO KBOT_APP_ROLE target
USING (
    SELECT 'km_asset' AS APP_ID,
           'user' AS ROLE_CODE,
           '用户' AS DISPLAY_NAME,
           'Y' AS IS_SYSTEM,
           'SELECTABLE' AS SCOPE_POLICY,
           'ACTIVE' AS STATUS
      FROM DUAL
    UNION ALL
    SELECT 'km_asset', 'app_admin', 'KM Asset 初始管理员',
           'Y', 'ALL_APP_DOMAINS', 'ACTIVE' FROM DUAL
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.ROLE_CODE = source.ROLE_CODE
)
WHEN MATCHED THEN
    UPDATE SET
        target.DISPLAY_NAME = source.DISPLAY_NAME,
        target.IS_SYSTEM = source.IS_SYSTEM,
        target.SCOPE_POLICY = source.SCOPE_POLICY,
        target.STATUS = source.STATUS
WHEN NOT MATCHED THEN
    INSERT (
        APP_ID, ROLE_CODE, DISPLAY_NAME, IS_SYSTEM,
        SCOPE_POLICY, STATUS, ROW_VERSION
    )
    VALUES (
        source.APP_ID, source.ROLE_CODE,
        source.DISPLAY_NAME, source.IS_SYSTEM,
        source.SCOPE_POLICY, source.STATUS, 1
    );

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT
        'km_asset' AS APP_ID,
        'app_admin' AS ROLE_CODE,
        permission.PERMISSION_CODE
    FROM KBOT_PERMISSION permission
    WHERE permission.APP_ID = 'km_asset'
    UNION ALL
    SELECT
        'km_asset' AS APP_ID,
        'user' AS ROLE_CODE,
        'km_asset:use' AS PERMISSION_CODE
    FROM DUAL
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.ROLE_CODE = source.ROLE_CODE
    AND target.PERMISSION_CODE = source.PERMISSION_CODE
)
WHEN NOT MATCHED THEN
    INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
    VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

-- 权限目录脚本可能在初始管理员早于 app_admin 角色创建时执行。
-- 旧 Schema 中 manager 即 App 管理员；统一迁移为当前标准 app_admin。
MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (
    SELECT DISTINCT
        member.APP_ID,
        member.USER_ID,
        'app_admin' AS ROLE_CODE
    FROM KBOT_APP_MEMBER member
    WHERE member.APP_ID = 'km_asset'
      AND member.STATUS = 'ACTIVE'
      AND (
          member.IS_INITIAL_ADMIN = 'Y'
          OR EXISTS (
              SELECT 1
              FROM KBOT_APP_MEMBER_ROLE legacy_role
              WHERE legacy_role.APP_ID = member.APP_ID
                AND legacy_role.USER_ID = member.USER_ID
                AND legacy_role.ROLE_CODE = 'manager'
                AND legacy_role.STATUS = 'ACTIVE'
          )
      )
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.USER_ID = source.USER_ID
    AND target.ROLE_CODE = source.ROLE_CODE
)
WHEN MATCHED THEN
    UPDATE SET
        target.SCOPE_MODE = 'ALL_APP_DOMAINS',
        target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN
    INSERT (
        APP_ID, USER_ID, ROLE_CODE, SCOPE_MODE,
        STATUS, CREATED_BY, CREATED_AT
    )
    VALUES (
        source.APP_ID, source.USER_ID, source.ROLE_CODE,
        'ALL_APP_DOMAINS', 'ACTIVE',
        'bootstrap:km_asset_permissions', SYSTIMESTAMP
    );

DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE scope
WHERE scope.APP_ID = 'km_asset'
  AND scope.ROLE_CODE = 'app_admin'
  AND EXISTS (
      SELECT 1
      FROM KBOT_APP_MEMBER_ROLE member_role
      WHERE member_role.APP_ID = scope.APP_ID
        AND member_role.USER_ID = scope.USER_ID
        AND member_role.ROLE_CODE = scope.ROLE_CODE
        AND member_role.STATUS = 'ACTIVE'
  );

-- app_admin 为全 App Domain 角色，迁移完成后移除已废弃的 manager 定义。
DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE
 WHERE APP_ID = 'km_asset'
   AND ROLE_CODE = 'manager';

DELETE FROM KBOT_APP_MEMBER_ROLE
 WHERE APP_ID = 'km_asset'
   AND ROLE_CODE = 'manager';

DELETE FROM KBOT_APP_ROLE_PERMISSION
 WHERE APP_ID = 'km_asset'
   AND ROLE_CODE = 'manager';

DELETE FROM KBOT_APP_ROLE
 WHERE APP_ID = 'km_asset'
   AND ROLE_CODE = 'manager';

COMMIT;

PROMPT === KM Asset 角色 ===

SELECT APP_ID, ROLE_CODE, DISPLAY_NAME, STATUS
FROM KBOT_APP_ROLE
WHERE APP_ID = 'km_asset'
ORDER BY ROLE_CODE;

PROMPT === KM Asset 角色权限 ===

SELECT ROLE_CODE, PERMISSION_CODE
FROM KBOT_APP_ROLE_PERMISSION
WHERE APP_ID = 'km_asset'
ORDER BY ROLE_CODE, PERMISSION_CODE;

PROMPT === KM Asset App 管理员最终授权 ===

SELECT
    member.USER_ID,
    member_role.ROLE_CODE,
    role_permission.PERMISSION_CODE
FROM KBOT_APP_MEMBER member
JOIN KBOT_APP_MEMBER_ROLE member_role
  ON member_role.APP_ID = member.APP_ID
 AND member_role.USER_ID = member.USER_ID
 AND member_role.STATUS = 'ACTIVE'
JOIN KBOT_APP_ROLE_PERMISSION role_permission
  ON role_permission.APP_ID = member_role.APP_ID
 AND role_permission.ROLE_CODE = member_role.ROLE_CODE
WHERE member.APP_ID = 'km_asset'
  AND member.STATUS = 'ACTIVE'
  AND member_role.ROLE_CODE = 'app_admin'
ORDER BY member.USER_ID, role_permission.PERMISSION_CODE;
