-- 由 scripts/db/apply_oracle_schema.py 调用，幂等初始化平台 ADMIN。
-- 本资产只写基础数据，不创建或修改 Schema 对象，也不向业务 App 自动授权。

MERGE INTO KBOT_PLATFORM_DOMAIN target
USING (SELECT 'default' NAME, 'KBot 默认业务域' DESCRIPTION FROM DUAL) source
ON (target.NAME = source.NAME)
WHEN MATCHED THEN UPDATE SET
    target.STATUS = 'ACTIVE', target.DESCRIPTION = source.DESCRIPTION,
    target.UPDATED_BY = 'bootstrap:platform', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    NAME, STATUS, DESCRIPTION, ROW_VERSION, CREATED_BY, UPDATED_BY,
    CREATED_AT, UPDATED_AT
) VALUES (
    source.NAME, 'ACTIVE', source.DESCRIPTION, 1,
    'bootstrap:platform', 'bootstrap:platform', SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_PLATFORM_USER target
USING (SELECT 'ADMIN' USER_ID, 'KBot 全局管理员' DISPLAY_NAME FROM DUAL) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN UPDATE SET
    target.DISPLAY_NAME = source.DISPLAY_NAME,
    target.ACCOUNT_ORIGIN = 'PLATFORM', target.OWNER_APP_ID = NULL,
    target.IS_PROTECTED = 'Y', target.MAX_SECURITY_LEVEL = 3,
    target.STATUS = 'ACTIVE', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    USER_ID, DISPLAY_NAME, ACCOUNT_ORIGIN, OWNER_APP_ID, IS_PROTECTED,
    MAX_SECURITY_LEVEL, STATUS, CREATED_AT, UPDATED_AT
) VALUES (
    source.USER_ID, source.DISPLAY_NAME, 'PLATFORM', NULL, 'Y',
    3, 'ACTIVE', SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL target
USING (
    SELECT 'ADMIN' USER_ID,
           '$2b$12$5WDXCasJPPANzr/QGlwbA.WOQxDa5Jq.RGGuFupPG1oossZZYKS3W' PASSWORD_HASH
    FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN NOT MATCHED THEN INSERT (
    USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD,
    PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT
) VALUES (
    source.USER_ID, source.PASSWORD_HASH, 'N',
    SYSTIMESTAMP, SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_PLATFORM_USER_ROLE target
USING (SELECT 'ADMIN' USER_ID, 'platform_admin' ROLE_CODE FROM DUAL) source
ON (target.USER_ID = source.USER_ID AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN UPDATE SET target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN INSERT (
    USER_ID, ROLE_CODE, STATUS, CREATED_BY, CREATED_AT
) VALUES (
    source.USER_ID, source.ROLE_CODE, 'ACTIVE', 'bootstrap:platform', SYSTIMESTAMP
);

-- ADMIN 可以通过平台接口获得显式 App Grant，但不能成为 App 创建用户或初始管理员。
DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE scope_row
 WHERE scope_row.USER_ID = 'ADMIN'
   AND EXISTS (
       SELECT 1
         FROM KBOT_APP_MEMBER member_row
        WHERE member_row.APP_ID = scope_row.APP_ID
          AND member_row.USER_ID = scope_row.USER_ID
          AND (
              member_row.MEMBER_SOURCE <> 'PLATFORM_GRANT'
              OR member_row.IS_INITIAL_ADMIN <> 'N'
          )
   );

DELETE FROM KBOT_APP_MEMBER_ROLE role_row
 WHERE role_row.USER_ID = 'ADMIN'
   AND EXISTS (
       SELECT 1
         FROM KBOT_APP_MEMBER member_row
        WHERE member_row.APP_ID = role_row.APP_ID
          AND member_row.USER_ID = role_row.USER_ID
          AND (
              member_row.MEMBER_SOURCE <> 'PLATFORM_GRANT'
              OR member_row.IS_INITIAL_ADMIN <> 'N'
          )
   );

DELETE FROM KBOT_APP_MEMBER member_row
 WHERE member_row.USER_ID = 'ADMIN'
   AND (
       member_row.MEMBER_SOURCE <> 'PLATFORM_GRANT'
       OR member_row.IS_INITIAL_ADMIN <> 'N'
   );
