-- 为 KBot 业务用户 kbotui_dev 和 KBOTUI_DEV 授予全部已启用 Domain 的应用管理员权限。
-- 本脚本用于 SQL Developer：粘贴后使用 Run Script（F5）执行。
-- 两个用户标识用于匹配 APEX 可能传递的小写或大写 APP_USER，不是 Oracle 数据库账号。
-- 当前权限模型没有跨应用的 system_admin 角色；本脚本授予
-- knowledge_retrieval 和 aiops 两个应用的 manager 角色，等同当前产品范围的系统管理员。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_role_count PLS_INTEGER;
    l_domain_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_role_count
      FROM KBOT_APP_ROLE
     WHERE (
            (APP_ID = 'knowledge_retrieval' AND ROLE_CODE = 'manager')
            OR (APP_ID = 'aiops' AND ROLE_CODE = 'manager')
        )
       AND STATUS = 'ACTIVE';

    IF l_role_count <> 2 THEN
        raise_application_error(
            -20001,
            '缺少启用的应用管理员角色，请先完成 Main API 权限表初始化。'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_domain_count
      FROM KBOT_PLATFORM_DOMAIN
     WHERE STATUS = 'ACTIVE';

    IF l_domain_count = 0 THEN
        raise_application_error(-20002, '不存在启用的 Domain，无法授权。');
    END IF;

    dbms_output.put_line(
        '将向 ' || l_domain_count
        || ' 个启用 Domain 的用户 kbotui_dev、KBOTUI_DEV 授予两个应用管理员角色。'
    );
END;
/

MERGE INTO KBOT_PLATFORM_USER target
USING (
    SELECT 'kbotui_dev' AS user_id, 'KBot 系统管理员' AS display_name FROM DUAL
    UNION ALL
    SELECT 'KBOTUI_DEV' AS user_id, 'KBot 系统管理员' AS display_name FROM DUAL
) source
ON (target.USER_ID = source.user_id)
WHEN MATCHED THEN
    UPDATE SET
        target.DISPLAY_NAME = source.display_name,
        target.STATUS = 'ACTIVE',
        target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (USER_ID, DISPLAY_NAME, STATUS, CREATED_AT, UPDATED_AT)
    VALUES (
        source.user_id, source.display_name, 'ACTIVE',
        SYSTIMESTAMP, SYSTIMESTAMP
    );

MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (
    SELECT
        role.APP_ID,
        domain.DOMAIN_ID,
        app_user.USER_ID,
        role.ROLE_CODE
    FROM KBOT_PLATFORM_DOMAIN domain
    CROSS JOIN KBOT_APP_ROLE role
    CROSS JOIN (
        SELECT 'kbotui_dev' AS USER_ID FROM DUAL
        UNION ALL
        SELECT 'KBOTUI_DEV' AS USER_ID FROM DUAL
    ) app_user
    WHERE domain.STATUS = 'ACTIVE'
      AND role.STATUS = 'ACTIVE'
      AND (
          (role.APP_ID = 'knowledge_retrieval' AND role.ROLE_CODE = 'manager')
          OR (role.APP_ID = 'aiops' AND role.ROLE_CODE = 'manager')
      )
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
        source.APP_ID, source.DOMAIN_ID, source.USER_ID, source.ROLE_CODE,
        'ACTIVE', 'bootstrap:system_admin', SYSTIMESTAMP
    );

COMMIT;

PROMPT === kbotui_dev 当前管理员授权 ===

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
WHERE member_role.USER_ID IN ('kbotui_dev', 'KBOTUI_DEV')
  AND member_role.ROLE_CODE = 'manager'
  AND member_role.APP_ID IN ('knowledge_retrieval', 'aiops')
ORDER BY member_role.USER_ID, member_role.DOMAIN_ID, member_role.APP_ID;
