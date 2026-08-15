-- 为既有 KBot 4.0 数据库补充普通登录所需的平台用户和角色管理权限。
-- 在 SQL Developer 中使用 Run Script（F5）直接执行；不读取外部文件、不要求输入参数。
-- 本脚本不会创建用户、不会修改 ADMIN 密码，也不会修改现有业务角色。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    c_admin_user_id CONSTANT VARCHAR2(256 CHAR) := 'ADMIN';
    c_actor_id      CONSTANT VARCHAR2(256 CHAR) := 'bootstrap:platform_access';
    l_admin_count   PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_admin_count
      FROM KBOT_PLATFORM_USER
     WHERE USER_ID = c_admin_user_id
       AND STATUS = 'ACTIVE';

    IF l_admin_count <> 1 THEN
        raise_application_error(
            -20001,
            '未找到启用的 ADMIN，请先执行 bootstrap_global_admin.sql。'
        );
    END IF;

    MERGE INTO KBOT_PERMISSION target
    USING (
        SELECT 'platform:user_manage' AS PERMISSION_CODE,
               'platform' AS APP_ID,
               '管理平台用户与成员授权' AS DISPLAY_NAME
          FROM DUAL
        UNION ALL
        SELECT 'platform:role_manage',
               'platform',
               '管理平台应用角色与权限'
          FROM DUAL
    ) source
    ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
    WHEN MATCHED THEN
        UPDATE SET
            target.APP_ID = source.APP_ID,
            target.DISPLAY_NAME = source.DISPLAY_NAME
    WHEN NOT MATCHED THEN
        INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
        VALUES (
            source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME
        );

    MERGE INTO KBOT_APP_ROLE target
    USING (
        SELECT 'platform' AS APP_ID,
               'platform_admin' AS ROLE_CODE,
               '平台管理员' AS DISPLAY_NAME
          FROM DUAL
        UNION ALL
        SELECT 'platform',
               'system_admin',
               '系统管理员'
          FROM DUAL
    ) source
    ON (
        target.APP_ID = source.APP_ID
        AND target.ROLE_CODE = source.ROLE_CODE
    )
    WHEN MATCHED THEN
        UPDATE SET
            target.DISPLAY_NAME = source.DISPLAY_NAME,
            target.STATUS = 'ACTIVE'
    WHEN NOT MATCHED THEN
        INSERT (APP_ID, ROLE_CODE, DISPLAY_NAME, STATUS)
        VALUES (
            source.APP_ID, source.ROLE_CODE,
            source.DISPLAY_NAME, 'ACTIVE'
        );

    MERGE INTO KBOT_APP_ROLE_PERMISSION target
    USING (
        SELECT role.ROLE_CODE,
               permission.APP_ID,
               permission.PERMISSION_CODE
          FROM KBOT_APP_ROLE role
          JOIN KBOT_PERMISSION permission
            ON permission.APP_ID = role.APP_ID
         WHERE role.APP_ID = 'platform'
           AND role.ROLE_CODE IN ('platform_admin', 'system_admin')
    ) source
    ON (
        target.APP_ID = source.APP_ID
        AND target.ROLE_CODE = source.ROLE_CODE
        AND target.PERMISSION_CODE = source.PERMISSION_CODE
    )
    WHEN NOT MATCHED THEN
        INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
        VALUES (
            source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE
        );

    MERGE INTO KBOT_APP_MEMBER_ROLE target
    USING (
        SELECT 'platform' AS APP_ID,
               domain.DOMAIN_ID,
               c_admin_user_id AS USER_ID,
               'system_admin' AS ROLE_CODE
          FROM KBOT_PLATFORM_DOMAIN domain
         WHERE domain.STATUS = 'ACTIVE'
    ) source
    ON (
        target.APP_ID = source.APP_ID
        AND target.DOMAIN_ID = source.DOMAIN_ID
        AND target.USER_ID = source.USER_ID
        AND target.ROLE_CODE = source.ROLE_CODE
    )
    WHEN MATCHED THEN
        UPDATE SET target.STATUS = 'ACTIVE'
    WHEN NOT MATCHED THEN
        INSERT (
            APP_ID, DOMAIN_ID, USER_ID, ROLE_CODE,
            STATUS, CREATED_BY, CREATED_AT
        )
        VALUES (
            source.APP_ID, source.DOMAIN_ID, source.USER_ID,
            source.ROLE_CODE, 'ACTIVE', c_actor_id, SYSTIMESTAMP
        );

    COMMIT;
    dbms_output.put_line('平台用户与角色管理权限初始化完成。');
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        dbms_output.put_line('初始化失败：' || SQLERRM);
        RAISE;
END;
/

SELECT APP_ID, ROLE_CODE, STATUS
  FROM KBOT_APP_ROLE
 WHERE APP_ID = 'platform'
 ORDER BY ROLE_CODE;

SELECT PERMISSION_CODE, DISPLAY_NAME
  FROM KBOT_PERMISSION
 WHERE APP_ID = 'platform'
 ORDER BY PERMISSION_CODE;

SELECT USER_ID, DOMAIN_ID, APP_ID, ROLE_CODE, STATUS
  FROM KBOT_APP_MEMBER_ROLE
 WHERE USER_ID = 'ADMIN'
   AND APP_ID = 'platform'
 ORDER BY DOMAIN_ID;
