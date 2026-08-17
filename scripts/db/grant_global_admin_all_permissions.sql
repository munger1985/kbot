-- 为现有 ADMIN 补齐当前全部 App、全部启用 Domain 的所有权限。
-- 在 SQL Developer 中使用 Run Script（F5）直接执行。
-- 本脚本不创建用户、不修改 ADMIN 密码，可重复执行。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    c_admin_user_id CONSTANT VARCHAR2(256 CHAR) := 'ADMIN';
    c_role_code     CONSTANT VARCHAR2(64 CHAR) := 'system_admin';
    c_actor_id      CONSTANT VARCHAR2(256 CHAR) :=
        'bootstrap:grant_all_permissions';

    l_admin_count      PLS_INTEGER;
    l_domain_count     PLS_INTEGER;
    l_app_count        PLS_INTEGER;
    l_permission_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_admin_count
      FROM KBOT_PLATFORM_USER
     WHERE USER_ID = c_admin_user_id;

    IF l_admin_count <> 1 THEN
        raise_application_error(
            -20001,
            '未找到 ADMIN 用户，请先执行 bootstrap_global_admin.sql。'
        );
    END IF;

    UPDATE KBOT_PLATFORM_USER
       SET STATUS = 'ACTIVE',
           UPDATED_AT = SYSTIMESTAMP
     WHERE USER_ID = c_admin_user_id
       AND STATUS <> 'ACTIVE';

    -- 补齐创建用户和管理角色所需的平台权限目录。
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
            source.PERMISSION_CODE,
            source.APP_ID,
            source.DISPLAY_NAME
        );

    -- 每个已登记 App 都拥有独立的 system_admin 角色。
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
            source.APP_ID,
            source.ROLE_CODE,
            source.DISPLAY_NAME,
            'ACTIVE'
        );

    -- 将每个 App 当前登记的全部 Permission 授予 system_admin。
    MERGE INTO KBOT_APP_ROLE_PERMISSION target
    USING (
        SELECT permission.APP_ID,
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
            source.APP_ID,
            source.ROLE_CODE,
            source.PERMISSION_CODE
        );

    -- 在全部启用 Domain 中为 ADMIN 分配各 App 的 system_admin。
    MERGE INTO KBOT_APP_MEMBER_ROLE target
    USING (
        SELECT role.APP_ID,
               domain.DOMAIN_ID,
               c_admin_user_id AS USER_ID,
               c_role_code AS ROLE_CODE
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
            APP_ID,
            DOMAIN_ID,
            USER_ID,
            ROLE_CODE,
            STATUS,
            CREATED_BY,
            CREATED_AT
        )
        VALUES (
            source.APP_ID,
            source.DOMAIN_ID,
            source.USER_ID,
            source.ROLE_CODE,
            'ACTIVE',
            c_actor_id,
            SYSTIMESTAMP
        );

    SELECT COUNT(*)
      INTO l_domain_count
      FROM KBOT_PLATFORM_DOMAIN
     WHERE STATUS = 'ACTIVE';

    SELECT COUNT(DISTINCT APP_ID)
      INTO l_app_count
      FROM KBOT_PERMISSION;

    SELECT COUNT(*)
      INTO l_permission_count
      FROM KBOT_PERMISSION;

    COMMIT;

    dbms_output.put_line('ADMIN 全部权限授权完成。');
    dbms_output.put_line('启用 Domain 数：' || l_domain_count);
    dbms_output.put_line('已授权 App 数：' || l_app_count);
    dbms_output.put_line('权限目录总数：' || l_permission_count);
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        dbms_output.put_line('ADMIN 授权失败：' || SQLERRM);
        RAISE;
END;
/

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
