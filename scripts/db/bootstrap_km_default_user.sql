-- 一次性初始化 KM Asset 默认操作用户，并授予当前全部启用 Domain 的 manager 角色。
-- 用法：在 SQL Developer 中修改下方变量，然后使用 Run Script（F5）执行。
-- USER_ID 必须与 APEX 登录后的 APP_USER 完全一致（包括大小写）。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
DEFINE KM_DEFAULT_USER_ID = kbotui_dev
DEFINE KM_DEFAULT_DISPLAY_NAME = KM默认操作用户

DECLARE
    l_role_count PLS_INTEGER;
    l_domain_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_role_count
      FROM KBOT_APP_ROLE
     WHERE APP_ID = 'km_asset'
       AND ROLE_CODE = 'manager'
       AND STATUS = 'ACTIVE';

    IF l_role_count <> 1 THEN
        raise_application_error(
            -20001,
            '缺少启用的 km_asset/manager 角色，请先初始化 Main API 权限表。'
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
        '将用户 &&KM_DEFAULT_USER_ID 授权到 '
        || l_domain_count || ' 个启用 Domain。'
    );
END;
/

MERGE INTO KBOT_PLATFORM_USER target
USING (
    SELECT
        '&&KM_DEFAULT_USER_ID' AS USER_ID,
        '&&KM_DEFAULT_DISPLAY_NAME' AS DISPLAY_NAME
    FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN
    UPDATE SET
        target.DISPLAY_NAME = source.DISPLAY_NAME,
        target.STATUS = 'ACTIVE',
        target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (USER_ID, DISPLAY_NAME, STATUS, CREATED_AT, UPDATED_AT)
    VALUES (
        source.USER_ID, source.DISPLAY_NAME, 'ACTIVE',
        SYSTIMESTAMP, SYSTIMESTAMP
    );

MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (
    SELECT
        'km_asset' AS APP_ID,
        domain.DOMAIN_ID,
        '&&KM_DEFAULT_USER_ID' AS USER_ID,
        'manager' AS ROLE_CODE
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
        source.APP_ID, source.DOMAIN_ID, source.USER_ID, source.ROLE_CODE,
        'ACTIVE', 'bootstrap:km_default_user', SYSTIMESTAMP
    );

COMMIT;

PROMPT === KM Asset 默认用户授权结果 ===

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
WHERE member_role.USER_ID = '&&KM_DEFAULT_USER_ID'
  AND member_role.APP_ID = 'km_asset'
  AND member_role.ROLE_CODE = 'manager'
ORDER BY member_role.DOMAIN_ID;

UNDEFINE KM_DEFAULT_USER_ID
UNDEFINE KM_DEFAULT_DISPLAY_NAME
