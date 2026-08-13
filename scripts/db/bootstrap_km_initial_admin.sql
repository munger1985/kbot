-- KM Asset 首次使用的一步初始化脚本。
-- 在 SQL Developer 中修改下方 BEGIN 块的用户赋值，然后使用 Run Script（F5）执行。
-- 本脚本会补齐 KM 权限与角色、创建或启用平台用户，并在全部启用 Domain
-- 中授予 km_asset/manager。USER_ID 区分大小写，必须与页面登录用户完全一致。
-- 初始账号：kmadmin，初始密码：KmAdmin@2026!
-- 首次登录必须修改初始密码后才能访问其他 KM API。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

VARIABLE KM_ADMIN_USER_ID VARCHAR2(256)
VARIABLE KM_ADMIN_DISPLAY_NAME VARCHAR2(256)

BEGIN
    :KM_ADMIN_USER_ID := 'kmadmin';
    :KM_ADMIN_DISPLAY_NAME := 'KM Asset 管理员';
END;
/

DECLARE
    l_table_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_PLATFORM_USER_CREDENTIAL';

    IF l_table_count = 0 THEN
        EXECUTE IMMEDIATE q'[
            CREATE TABLE KBOT_PLATFORM_USER_CREDENTIAL (
                USER_ID VARCHAR2(256 CHAR) PRIMARY KEY,
                PASSWORD_HASH VARCHAR2(128 CHAR) NOT NULL,
                MUST_CHANGE_PASSWORD CHAR(1 CHAR) DEFAULT 'Y' NOT NULL,
                PASSWORD_UPDATED_AT TIMESTAMP(6) WITH TIME ZONE
                    DEFAULT CURRENT_TIMESTAMP NOT NULL,
                CREATED_AT TIMESTAMP(6) WITH TIME ZONE
                    DEFAULT CURRENT_TIMESTAMP NOT NULL,
                UPDATED_AT TIMESTAMP(6) WITH TIME ZONE
                    DEFAULT CURRENT_TIMESTAMP NOT NULL,
                CONSTRAINT FK_PLATFORM_USER_CRED_USER
                    FOREIGN KEY (USER_ID)
                    REFERENCES KBOT_PLATFORM_USER (USER_ID),
                CONSTRAINT CK_PLATFORM_USER_CRED_CHANGE
                    CHECK (MUST_CHANGE_PASSWORD IN ('Y', 'N'))
            )
        ]';
        dbms_output.put_line('已创建 KM 用户凭据表。');
    END IF;
END;
/

DECLARE
    l_table_count PLS_INTEGER;
    l_domain_count PLS_INTEGER;
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
            'Main API 用户与权限基础表不完整，不能初始化 KM 管理员。'
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
        '将初始化用户 ' || :KM_ADMIN_USER_ID || '，并授权到 '
        || l_domain_count || ' 个启用 Domain。'
    );
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
           'ACTIVE' AS STATUS
      FROM DUAL
    UNION ALL
    SELECT 'km_asset', 'manager', '管理员', 'ACTIVE' FROM DUAL
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.ROLE_CODE = source.ROLE_CODE
)
WHEN MATCHED THEN
    UPDATE SET
        target.DISPLAY_NAME = source.DISPLAY_NAME,
        target.STATUS = source.STATUS
WHEN NOT MATCHED THEN
    INSERT (APP_ID, ROLE_CODE, DISPLAY_NAME, STATUS)
    VALUES (
        source.APP_ID, source.ROLE_CODE,
        source.DISPLAY_NAME, source.STATUS
    );

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT
        'km_asset' AS APP_ID,
        'manager' AS ROLE_CODE,
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

MERGE INTO KBOT_PLATFORM_USER target
USING (
    SELECT
        :KM_ADMIN_USER_ID AS USER_ID,
        :KM_ADMIN_DISPLAY_NAME AS DISPLAY_NAME
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

MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL target
USING (
    SELECT
        :KM_ADMIN_USER_ID AS USER_ID,
        '$2b$12$QyA/YRNs6.JVOLh9saV7oeW7wskZ0qDEioAgV8oMBO7jOkxchNDQa'
            AS PASSWORD_HASH
    FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN NOT MATCHED THEN
    INSERT (
        USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD,
        PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT
    )
    VALUES (
        source.USER_ID, source.PASSWORD_HASH, 'Y',
        SYSTIMESTAMP, SYSTIMESTAMP, SYSTIMESTAMP
    );

MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (
    SELECT
        'km_asset' AS APP_ID,
        domain.DOMAIN_ID,
        :KM_ADMIN_USER_ID AS USER_ID,
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
        'ACTIVE', 'bootstrap:km_initial_admin', SYSTIMESTAMP
    );

COMMIT;