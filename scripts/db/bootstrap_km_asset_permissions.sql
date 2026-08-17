-- 为现有 KBot Schema 幂等补充 KM Asset 权限、角色及角色权限关系。
-- 本脚本用于 SQL Developer：粘贴后使用 Run Script（F5）执行。
-- 不创建用户和成员授权；初始管理员由平台接口或 bootstrap_km_initial_admin.sql 创建。

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
        'KBOT_APP_ROLE_PERMISSION'
     );

    IF l_table_count <> 3 THEN
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
