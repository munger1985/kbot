-- 由 scripts/db/initialize_assistant.py 调用的智能工作台首次使用数据资产。
-- 本资产只写基础数据，不创建或修改 Schema 对象。
-- 固定资源：assistant_portal 引导 Domain、assistantadmin 初始管理员。
-- 不创建 Knowledge Core、问数模型、Agent、模型绑定或任何 OCI 配置。
-- 初始账号：assistantadmin，初始密码：AssistantAdmin@2026!
-- 重复执行不会重置已存在的初始管理员密码。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_table_count PLS_INTEGER;
    l_conflicting_user_count PLS_INTEGER;
    l_other_initial_admin_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME IN (
        'KBOT_PLATFORM_DOMAIN', 'KBOT_PLATFORM_APP', 'KBOT_PLATFORM_USER',
        'KBOT_PLATFORM_USER_CREDENTIAL', 'KBOT_PERMISSION', 'KBOT_APP_ROLE',
        'KBOT_APP_ROLE_PERMISSION', 'KBOT_APP_DOMAIN', 'KBOT_APP_MEMBER',
        'KBOT_APP_MEMBER_ROLE', 'KBOT_APP_MEMBER_ROLE_SCOPE',
        'KBOT_ASST_AGENT', 'KBOT_ASST_AGENT_VERSION'
     );
    IF l_table_count <> 13 THEN
        raise_application_error(
            -20001,
            '智能工作台 Domain、权限或 Agent Schema 基础表不完整。'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_conflicting_user_count
      FROM KBOT_PLATFORM_USER
     WHERE USER_ID = 'assistantadmin'
       AND (ACCOUNT_ORIGIN <> 'APP' OR OWNER_APP_ID <> 'assistant');
    IF l_conflicting_user_count > 0 THEN
        raise_application_error(
            -20002,
            'assistantadmin 已属于其他账户来源，拒绝接管。'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_other_initial_admin_count
      FROM KBOT_APP_MEMBER
     WHERE APP_ID = 'assistant'
       AND IS_INITIAL_ADMIN = 'Y'
       AND USER_ID <> 'assistantadmin';
    IF l_other_initial_admin_count > 0 THEN
        raise_application_error(
            -20003,
            '智能工作台已存在其他初始管理员，拒绝覆盖。'
        );
    END IF;
END;
/

MERGE INTO KBOT_PLATFORM_APP target
USING (SELECT 'assistant' APP_ID, '智能工作台' DISPLAY_NAME FROM DUAL) source
ON (target.APP_ID = source.APP_ID)
WHEN MATCHED THEN UPDATE SET
    target.DISPLAY_NAME = source.DISPLAY_NAME,
    target.STATUS = 'ACTIVE', target.MEMBER_ASSIGNABLE = 'Y',
    target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    APP_ID, DISPLAY_NAME, STATUS, MEMBER_ASSIGNABLE, ROW_VERSION,
    CREATED_AT, UPDATED_AT
) VALUES (
    source.APP_ID, source.DISPLAY_NAME, 'ACTIVE', 'Y', 1,
    SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_PLATFORM_DOMAIN target
USING (
    SELECT 'assistant_portal' NAME,
           '智能工作台初始空白 Domain；可在 App 内创建业务 Domain' DESCRIPTION
      FROM DUAL
) source
ON (target.NAME = source.NAME)
WHEN MATCHED THEN UPDATE SET
    target.STATUS = 'ACTIVE', target.DESCRIPTION = source.DESCRIPTION,
    target.UPDATED_BY = 'bootstrap:assistant_initial_admin',
    target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    NAME, STATUS, DESCRIPTION, ROW_VERSION,
    CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT
) VALUES (
    source.NAME, 'ACTIVE', source.DESCRIPTION, 1,
    'bootstrap:assistant_initial_admin', 'bootstrap:assistant_initial_admin',
    SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_PERMISSION target
USING (
    SELECT 'assistant:access' PERMISSION_CODE, 'assistant' APP_ID, '进入智能工作台' DISPLAY_NAME FROM DUAL UNION ALL
    SELECT 'assistant:knowledge_chat', 'assistant', '使用知识问答' FROM DUAL UNION ALL
    SELECT 'assistant:x_search', 'assistant', '使用 X 实时搜索' FROM DUAL UNION ALL
    SELECT 'assistant:image_generate', 'assistant', '使用文生图' FROM DUAL UNION ALL
    SELECT 'assistant:media_read', 'assistant', '查看图片资产' FROM DUAL UNION ALL
    SELECT 'assistant:domain_manage', 'assistant', '管理智能工作台 Domain' FROM DUAL UNION ALL
    SELECT 'assistant:knowledge_core_manage', 'assistant', '管理 Knowledge Core 关联' FROM DUAL UNION ALL
    SELECT 'assistant:data_model_manage', 'assistant', '管理问数模型关联' FROM DUAL UNION ALL
    SELECT 'assistant:agent_manage', 'assistant', '管理智能工作台 Agent' FROM DUAL UNION ALL
    SELECT 'assistant:model_binding_manage', 'assistant', '管理模型绑定' FROM DUAL UNION ALL
    SELECT 'assistant:run_read', 'assistant', '查看运行和用量' FROM DUAL
) source
ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN MATCHED THEN UPDATE SET
    target.APP_ID = source.APP_ID, target.DISPLAY_NAME = source.DISPLAY_NAME
WHEN NOT MATCHED THEN INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
VALUES (source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME);

MERGE INTO KBOT_APP_ROLE target
USING (
    SELECT 'assistant' APP_ID, 'user' ROLE_CODE, '用户' DISPLAY_NAME,
           'Y' IS_SYSTEM, 'SELECTABLE' SCOPE_POLICY, 'ACTIVE' STATUS FROM DUAL
    UNION ALL
    SELECT 'assistant', 'app_admin', '智能工作台初始管理员',
           'Y', 'ALL_APP_DOMAINS', 'ACTIVE' FROM DUAL
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN UPDATE SET
    target.DISPLAY_NAME = source.DISPLAY_NAME,
    target.IS_SYSTEM = source.IS_SYSTEM,
    target.SCOPE_POLICY = source.SCOPE_POLICY,
    target.STATUS = source.STATUS
WHEN NOT MATCHED THEN INSERT (
    APP_ID, ROLE_CODE, DISPLAY_NAME, IS_SYSTEM,
    SCOPE_POLICY, STATUS, ROW_VERSION
) VALUES (
    source.APP_ID, source.ROLE_CODE, source.DISPLAY_NAME, source.IS_SYSTEM,
    source.SCOPE_POLICY, source.STATUS, 1
);

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT 'assistant' APP_ID, 'app_admin' ROLE_CODE, PERMISSION_CODE
      FROM KBOT_PERMISSION
     WHERE APP_ID = 'assistant'
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.ROLE_CODE = source.ROLE_CODE
    AND target.PERMISSION_CODE = source.PERMISSION_CODE
)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT 'assistant' APP_ID, 'user' ROLE_CODE, PERMISSION_CODE
      FROM KBOT_PERMISSION
     WHERE PERMISSION_CODE IN (
        'assistant:access', 'assistant:knowledge_chat', 'assistant:x_search',
        'assistant:image_generate', 'assistant:media_read', 'assistant:run_read'
     )
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.ROLE_CODE = source.ROLE_CODE
    AND target.PERMISSION_CODE = source.PERMISSION_CODE
)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_PLATFORM_USER target
USING (
    SELECT 'assistantadmin' USER_ID, '智能工作台管理员' DISPLAY_NAME FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN UPDATE SET
    target.DISPLAY_NAME = source.DISPLAY_NAME,
    target.ACCOUNT_ORIGIN = 'APP', target.OWNER_APP_ID = 'assistant',
    target.IS_PROTECTED = 'Y', target.MAX_SECURITY_LEVEL = 3,
    target.STATUS = 'ACTIVE', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    USER_ID, DISPLAY_NAME, ACCOUNT_ORIGIN, OWNER_APP_ID,
    IS_PROTECTED, MAX_SECURITY_LEVEL, STATUS, CREATED_AT, UPDATED_AT
) VALUES (
    source.USER_ID, source.DISPLAY_NAME, 'APP', 'assistant',
    'Y', 3, 'ACTIVE', SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL target
USING (
    SELECT 'assistantadmin' USER_ID,
           '$2b$12$sD3IfkQyIfZcU7H07W9uCu1Vlzb.4PLpgwnmvQMr4Ok2/5PQ7xylK' PASSWORD_HASH
      FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN NOT MATCHED THEN INSERT (
    USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD,
    PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT
) VALUES (
    source.USER_ID, source.PASSWORD_HASH, 'Y',
    SYSTIMESTAMP, SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_APP_DOMAIN target
USING (
    SELECT 'assistant' APP_ID, DOMAIN_ID
      FROM KBOT_PLATFORM_DOMAIN
     WHERE NAME = 'assistant_portal' AND STATUS = 'ACTIVE'
) source
ON (target.APP_ID = source.APP_ID AND target.DOMAIN_ID = source.DOMAIN_ID)
WHEN MATCHED THEN UPDATE SET target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN INSERT (
    APP_ID, DOMAIN_ID, STATUS, CREATED_BY, CREATED_AT
) VALUES (
    source.APP_ID, source.DOMAIN_ID, 'ACTIVE',
    'bootstrap:assistant_initial_admin', SYSTIMESTAMP
);

MERGE INTO KBOT_APP_MEMBER target
USING (SELECT 'assistant' APP_ID, 'assistantadmin' USER_ID FROM DUAL) source
ON (target.APP_ID = source.APP_ID AND target.USER_ID = source.USER_ID)
WHEN MATCHED THEN UPDATE SET
    target.MEMBER_SOURCE = 'APP_INITIAL_ADMIN',
    target.IS_INITIAL_ADMIN = 'Y', target.STATUS = 'ACTIVE',
    target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    APP_ID, USER_ID, MEMBER_SOURCE, IS_INITIAL_ADMIN,
    STATUS, GRANTED_BY, CREATED_AT, UPDATED_AT
) VALUES (
    source.APP_ID, source.USER_ID, 'APP_INITIAL_ADMIN', 'Y',
    'ACTIVE', 'bootstrap:assistant_initial_admin', SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (
    SELECT 'assistant' APP_ID, 'assistantadmin' USER_ID,
           'app_admin' ROLE_CODE FROM DUAL
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.USER_ID = source.USER_ID
    AND target.ROLE_CODE = source.ROLE_CODE
)
WHEN MATCHED THEN UPDATE SET
    target.SCOPE_MODE = 'ALL_APP_DOMAINS', target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN INSERT (
    APP_ID, USER_ID, ROLE_CODE, SCOPE_MODE, STATUS, CREATED_BY, CREATED_AT
) VALUES (
    source.APP_ID, source.USER_ID, source.ROLE_CODE,
    'ALL_APP_DOMAINS', 'ACTIVE', 'bootstrap:assistant_initial_admin', SYSTIMESTAMP
);

DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE
 WHERE APP_ID = 'assistant'
   AND USER_ID = 'assistantadmin'
   AND ROLE_CODE = 'app_admin';

COMMIT;
