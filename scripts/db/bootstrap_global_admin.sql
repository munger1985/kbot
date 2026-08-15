-- 一次性创建或重置 KBot 全局管理员，并授予当前全部 App、全部启用 Domain 的所有权限。
-- 在 SQL Developer 中使用 Run Script（F5）执行；不使用绑定变量、替换变量或外部 SQL 文件。
-- 固定登录账号：ADMIN
-- 固定初始密码：Admin@2026!
-- 重新执行本脚本会将 ADMIN 密码恢复为上述初始密码。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    c_user_id       CONSTANT VARCHAR2(256 CHAR) := 'ADMIN';
    c_display_name  CONSTANT VARCHAR2(256 CHAR) := 'KBot 全局管理员';
    c_password_hash CONSTANT VARCHAR2(128 CHAR) :=
        '$2b$12$5WDXCasJPPANzr/QGlwbA.WOQxDa5Jq.RGGuFupPG1oossZZYKS3W';
    c_role_code     CONSTANT VARCHAR2(64 CHAR) := 'system_admin';
    c_actor_id      CONSTANT VARCHAR2(256 CHAR) := 'bootstrap:global_admin';

    l_table_count      PLS_INTEGER;
    l_app_count        PLS_INTEGER;
    l_domain_count     PLS_INTEGER;
    l_permission_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME IN (
        'KBOT_PLATFORM_DOMAIN',
        'KBOT_PLATFORM_USER',
        'KBOT_PLATFORM_USER_CREDENTIAL',
        'KBOT_PERMISSION',
        'KBOT_APP_ROLE',
        'KBOT_APP_ROLE_PERMISSION',
        'KBOT_APP_MEMBER_ROLE'
     );

    IF l_table_count <> 7 THEN
        raise_application_error(
            -20001,
            '平台用户、Domain 或应用权限基础表不完整，请先执行 Main API 数据库初始化。'
        );
    END IF;

    -- 确保已部署环境具备通用用户和角色管理权限目录。
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

    SELECT COUNT(DISTINCT APP_ID), COUNT(*)
      INTO l_app_count, l_permission_count
      FROM KBOT_PERMISSION;

    IF l_app_count = 0 OR l_permission_count = 0 THEN
        raise_application_error(-20002, '权限目录为空，无法创建全局管理员。');
    END IF;

    SELECT COUNT(*)
      INTO l_domain_count
      FROM KBOT_PLATFORM_DOMAIN
     WHERE STATUS = 'ACTIVE';

    IF l_domain_count = 0 THEN
        raise_application_error(-20003, '不存在启用的 Domain，无法授权。');
    END IF;

    MERGE INTO KBOT_PLATFORM_USER target
    USING (
        SELECT c_user_id AS USER_ID,
               c_display_name AS DISPLAY_NAME
          FROM DUAL
    ) source
    ON (target.USER_ID = source.USER_ID)
    WHEN MATCHED THEN
        UPDATE SET
            target.DISPLAY_NAME = source.DISPLAY_NAME,
            target.STATUS = 'ACTIVE',
            target.UPDATED_AT = SYSTIMESTAMP
    WHEN NOT MATCHED THEN
        INSERT (
            USER_ID, DISPLAY_NAME, STATUS, CREATED_AT, UPDATED_AT
        )
        VALUES (
            source.USER_ID, source.DISPLAY_NAME, 'ACTIVE',
            SYSTIMESTAMP, SYSTIMESTAMP
        );

    MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL target
    USING (
        SELECT c_user_id AS USER_ID,
               c_password_hash AS PASSWORD_HASH
          FROM DUAL
    ) source
    ON (target.USER_ID = source.USER_ID)
    WHEN MATCHED THEN
        UPDATE SET
            target.PASSWORD_HASH = source.PASSWORD_HASH,
            target.MUST_CHANGE_PASSWORD = 'N',
            target.PASSWORD_UPDATED_AT = SYSTIMESTAMP,
            target.UPDATED_AT = SYSTIMESTAMP
    WHEN NOT MATCHED THEN
        INSERT (
            USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD,
            PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT
        )
        VALUES (
            source.USER_ID, source.PASSWORD_HASH, 'N',
            SYSTIMESTAMP, SYSTIMESTAMP, SYSTIMESTAMP
        );

    -- 每个 App 都拥有独立的 system_admin 角色。
    MERGE INTO KBOT_APP_ROLE target
    USING (
        SELECT DISTINCT
               permission.APP_ID,
               c_role_code AS ROLE_CODE,
               '系统管理员' AS DISPLAY_NAME
          FROM KBOT_PERMISSION permission
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

    -- 把每个 App 当前登记的全部 Permission 授予 system_admin。
    MERGE INTO KBOT_APP_ROLE_PERMISSION target
    USING (
        SELECT
            permission.APP_ID,
            c_role_code AS ROLE_CODE,
            permission.PERMISSION_CODE
        FROM KBOT_PERMISSION permission
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

    -- 向全部启用 Domain 授予每个 App 的 system_admin 角色。
    MERGE INTO KBOT_APP_MEMBER_ROLE target
    USING (
        SELECT
            role.APP_ID,
            domain.DOMAIN_ID,
            c_user_id AS USER_ID,
            role.ROLE_CODE
        FROM KBOT_PLATFORM_DOMAIN domain
        CROSS JOIN KBOT_APP_ROLE role
        WHERE domain.STATUS = 'ACTIVE'
          AND role.STATUS = 'ACTIVE'
          AND role.ROLE_CODE = c_role_code
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
            source.APP_ID, source.DOMAIN_ID,
            source.USER_ID, source.ROLE_CODE,
            'ACTIVE', c_actor_id, SYSTIMESTAMP
        );

    COMMIT;

    dbms_output.put_line('全局管理员初始化完成。');
    dbms_output.put_line('USER_ID: ' || c_user_id);
    dbms_output.put_line('已授权 App 数: ' || l_app_count);
    dbms_output.put_line('已授权 Domain 数: ' || l_domain_count);
    dbms_output.put_line('已授予 Permission 数: ' || l_permission_count);
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        dbms_output.put_line('全局管理员初始化失败：' || SQLERRM);
        RAISE;
END;
/

PROMPT === ADMIN 全局角色授权 ===

SELECT
    member_role.USER_ID,
    member_role.DOMAIN_ID,
    domain.NAME AS DOMAIN_NAME,
    member_role.APP_ID,
    member_role.ROLE_CODE,
    member_role.STATUS
FROM KBOT_APP_MEMBER_ROLE member_role
JOIN KBOT_PLATFORM_DOMAIN domain
  ON domain.DOMAIN_ID = member_role.DOMAIN_ID
WHERE member_role.USER_ID = 'ADMIN'
  AND member_role.ROLE_CODE = 'system_admin'
ORDER BY member_role.DOMAIN_ID, member_role.APP_ID;

PROMPT === ADMIN 当前有效权限 ===

SELECT DISTINCT
    member_role.DOMAIN_ID,
    domain.NAME AS DOMAIN_NAME,
    role_permission.APP_ID,
    role_permission.PERMISSION_CODE
FROM KBOT_APP_MEMBER_ROLE member_role
JOIN KBOT_PLATFORM_DOMAIN domain
  ON domain.DOMAIN_ID = member_role.DOMAIN_ID
JOIN KBOT_APP_ROLE_PERMISSION role_permission
  ON role_permission.APP_ID = member_role.APP_ID
 AND role_permission.ROLE_CODE = member_role.ROLE_CODE
WHERE member_role.USER_ID = 'ADMIN'
  AND member_role.STATUS = 'ACTIVE'
  AND member_role.ROLE_CODE = 'system_admin'
ORDER BY member_role.DOMAIN_ID,
         role_permission.APP_ID,
         role_permission.PERMISSION_CODE;
