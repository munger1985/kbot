-- 为既有新权限模型补齐平台管理权限；不授予业务 App 权限。
-- SQL Developer 使用 Run Script（F5）执行。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
BEGIN
    MERGE INTO KBOT_PERMISSION target
    USING (
        SELECT 'platform:user_manage' PERMISSION_CODE, 'platform' APP_ID, '管理平台用户' DISPLAY_NAME FROM DUAL
        UNION ALL SELECT 'platform:role_manage', 'platform', '管理平台角色' FROM DUAL
        UNION ALL SELECT 'platform:domain_manage', 'platform', '管理平台 Domain' FROM DUAL
        UNION ALL SELECT 'platform:app_manage', 'platform', '管理 App 生命周期和初始管理员' FROM DUAL
        UNION ALL SELECT 'platform:app_grant_manage', 'platform', '显式授予平台用户 App 权限' FROM DUAL
    ) source
    ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
    WHEN MATCHED THEN UPDATE SET
        target.APP_ID = source.APP_ID, target.DISPLAY_NAME = source.DISPLAY_NAME
    WHEN NOT MATCHED THEN INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
        VALUES (source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME);

    MERGE INTO KBOT_APP_ROLE target
    USING (SELECT 'platform' APP_ID, 'platform_admin' ROLE_CODE, '平台管理员' DISPLAY_NAME FROM DUAL) source
    ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE)
    WHEN MATCHED THEN UPDATE SET
        target.DISPLAY_NAME = source.DISPLAY_NAME, target.IS_SYSTEM = 'Y',
        target.SCOPE_POLICY = 'PLATFORM', target.STATUS = 'ACTIVE'
    WHEN NOT MATCHED THEN INSERT (
        APP_ID, ROLE_CODE, DISPLAY_NAME, IS_SYSTEM, SCOPE_POLICY, STATUS, ROW_VERSION
    ) VALUES (
        source.APP_ID, source.ROLE_CODE, source.DISPLAY_NAME,
        'Y', 'PLATFORM', 'ACTIVE', 1
    );

    MERGE INTO KBOT_APP_ROLE_PERMISSION target
    USING (
        SELECT 'platform' APP_ID, 'platform_admin' ROLE_CODE, PERMISSION_CODE
          FROM KBOT_PERMISSION WHERE APP_ID = 'platform'
    ) source
    ON (
        target.APP_ID = source.APP_ID
        AND target.ROLE_CODE = source.ROLE_CODE
        AND target.PERMISSION_CODE = source.PERMISSION_CODE
    )
    WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
        VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

    MERGE INTO KBOT_PLATFORM_USER_ROLE target
    USING (SELECT 'ADMIN' USER_ID, 'platform_admin' ROLE_CODE FROM DUAL) source
    ON (target.USER_ID = source.USER_ID AND target.ROLE_CODE = source.ROLE_CODE)
    WHEN MATCHED THEN UPDATE SET target.STATUS = 'ACTIVE'
    WHEN NOT MATCHED THEN INSERT (
        USER_ID, ROLE_CODE, STATUS, CREATED_BY, CREATED_AT
    ) VALUES (
        source.USER_ID, source.ROLE_CODE, 'ACTIVE',
        'bootstrap:platform_access', SYSTIMESTAMP
    );

    COMMIT;
    dbms_output.put_line('平台管理权限初始化完成，未修改业务 App Grant。');
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        dbms_output.put_line('初始化失败：' || SQLERRM);
        RAISE;
END;
/
