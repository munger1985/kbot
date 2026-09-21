-- 由 scripts/db/apply_oracle_schema.py 调用，幂等同步平台目录并初始化 ADMIN。
-- 本资产只写基础数据，不创建或修改 Schema 对象，也不向业务 App 自动授权。

MERGE INTO KBOT_PLATFORM_APP target
USING (SELECT 'media_studio' APP_ID, '多媒体创作工作台' DISPLAY_NAME FROM DUAL) source
ON (target.APP_ID = source.APP_ID)
WHEN MATCHED THEN UPDATE SET
    target.DISPLAY_NAME = source.DISPLAY_NAME,
    target.STATUS = 'ACTIVE', target.MEMBER_ASSIGNABLE = 'Y',
    target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    APP_ID, DISPLAY_NAME, STATUS, MEMBER_ASSIGNABLE, ROW_VERSION, CREATED_AT, UPDATED_AT
) VALUES (
    source.APP_ID, source.DISPLAY_NAME, 'ACTIVE', 'Y', 1, SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_PERMISSION target
USING (
    SELECT 'knowledge_retrieval:knowledge_chat' PERMISSION_CODE, 'knowledge_retrieval' APP_ID, '使用知识问答' DISPLAY_NAME FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:x_search', 'knowledge_retrieval', '使用 X 实时搜索' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:domain_manage', 'knowledge_retrieval', '管理知识 Domain' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:knowledge_core_manage', 'knowledge_retrieval', '管理 Knowledge Core' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:data_model_manage', 'knowledge_retrieval', '管理问数模型' FROM DUAL UNION ALL
    SELECT 'media_studio:use', 'media_studio', '使用多媒体创作工作台' FROM DUAL UNION ALL
    SELECT 'media_studio:image_generate', 'media_studio', '使用图片生成' FROM DUAL UNION ALL
    SELECT 'media_studio:media_read', 'media_studio', '查看媒体资产' FROM DUAL UNION ALL
    SELECT 'media_studio:model_binding_manage', 'media_studio', '管理媒体模型绑定' FROM DUAL UNION ALL
    SELECT 'media_studio:run_read', 'media_studio', '查看生成运行和用量' FROM DUAL
) source
ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN MATCHED THEN UPDATE SET target.APP_ID = source.APP_ID, target.DISPLAY_NAME = source.DISPLAY_NAME
WHEN NOT MATCHED THEN INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
VALUES (source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME);

MERGE INTO KBOT_APP_ROLE target
USING (
    SELECT 'media_studio' APP_ID, 'user' ROLE_CODE, '创作用户' DISPLAY_NAME,
           'Y' IS_SYSTEM, 'SELECTABLE' SCOPE_POLICY, 'ACTIVE' STATUS FROM DUAL
    UNION ALL
    SELECT 'media_studio', 'app_admin', '多媒体创作工作台初始管理员',
           'Y', 'ALL_APP_DOMAINS', 'ACTIVE' FROM DUAL
    UNION ALL
    SELECT 'aiops', 'user', '用户',
           'Y', 'SELECTABLE', 'ACTIVE' FROM DUAL
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN UPDATE SET
    target.DISPLAY_NAME = source.DISPLAY_NAME, target.IS_SYSTEM = source.IS_SYSTEM,
    target.SCOPE_POLICY = source.SCOPE_POLICY, target.STATUS = source.STATUS
WHEN NOT MATCHED THEN INSERT (
    APP_ID, ROLE_CODE, DISPLAY_NAME, IS_SYSTEM, SCOPE_POLICY, STATUS, ROW_VERSION
) VALUES (
    source.APP_ID, source.ROLE_CODE, source.DISPLAY_NAME, source.IS_SYSTEM,
    source.SCOPE_POLICY, source.STATUS, 1
);

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
