-- 知识检索首次使用数据资产；只写基础数据，不创建或修改 Schema。
-- 固定资源：knowledge_retrieval_portal 引导 Domain、knowledgeadmin 初始管理员。
-- 不创建业务 Agent、模型绑定或模型配置；重复执行不会重置已有管理员密码。
-- 初始密码：KnowledgeAdmin@2026!，首次登录必须修改。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_missing VARCHAR2(4000);
    l_conflicting_user_count PLS_INTEGER;
    l_other_initial_admin_count PLS_INTEGER;
BEGIN
    SELECT LISTAGG(required.TABLE_NAME, ', ') WITHIN GROUP (ORDER BY required.TABLE_NAME)
      INTO l_missing
      FROM (
        SELECT 'KBOT_PLATFORM_DOMAIN' TABLE_NAME FROM DUAL UNION ALL
        SELECT 'KBOT_PLATFORM_APP' FROM DUAL UNION ALL
        SELECT 'KBOT_PLATFORM_USER' FROM DUAL UNION ALL
        SELECT 'KBOT_PLATFORM_USER_CREDENTIAL' FROM DUAL UNION ALL
        SELECT 'KBOT_PERMISSION' FROM DUAL UNION ALL
        SELECT 'KBOT_APP_ROLE' FROM DUAL UNION ALL
        SELECT 'KBOT_APP_ROLE_PERMISSION' FROM DUAL UNION ALL
        SELECT 'KBOT_APP_DOMAIN' FROM DUAL UNION ALL
        SELECT 'KBOT_APP_MEMBER' FROM DUAL UNION ALL
        SELECT 'KBOT_APP_MEMBER_ROLE' FROM DUAL UNION ALL
        SELECT 'KBOT_APP_MEMBER_ROLE_SCOPE' FROM DUAL UNION ALL
        SELECT 'KBOT_KR_AGENT' FROM DUAL UNION ALL
        SELECT 'KBOT_KR_AGENT_VERSION' FROM DUAL UNION ALL
        SELECT 'KBOT_KR_RESEARCH_RUN' FROM DUAL UNION ALL
        SELECT 'KBOT_KR_RESEARCH_EVENT' FROM DUAL UNION ALL
        SELECT 'KBOT_KR_X_SOURCE' FROM DUAL
      ) required
      LEFT JOIN USER_TABLES existing ON existing.TABLE_NAME = required.TABLE_NAME
     WHERE existing.TABLE_NAME IS NULL;
    IF l_missing IS NOT NULL THEN
        raise_application_error(-20001, '知识检索基础表不完整，缺表：' || l_missing);
    END IF;

    SELECT COUNT(*) INTO l_conflicting_user_count
      FROM KBOT_PLATFORM_USER
     WHERE USER_ID = 'knowledgeadmin'
       AND (ACCOUNT_ORIGIN <> 'APP' OR OWNER_APP_ID <> 'knowledge_retrieval');
    IF l_conflicting_user_count > 0 THEN
        raise_application_error(-20002, 'knowledgeadmin 已属于其他账户来源，拒绝接管。');
    END IF;

    SELECT COUNT(*) INTO l_other_initial_admin_count
      FROM KBOT_APP_MEMBER
     WHERE APP_ID = 'knowledge_retrieval' AND IS_INITIAL_ADMIN = 'Y' AND USER_ID <> 'knowledgeadmin';
    IF l_other_initial_admin_count > 0 THEN
        raise_application_error(-20003, '知识检索已存在其他初始管理员，拒绝覆盖。');
    END IF;
END;
/

MERGE INTO KBOT_PLATFORM_APP target
USING (SELECT 'knowledge_retrieval' APP_ID, '知识检索' DISPLAY_NAME FROM DUAL) source
ON (target.APP_ID = source.APP_ID)
WHEN MATCHED THEN UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME, target.STATUS = 'ACTIVE', target.MEMBER_ASSIGNABLE = 'Y', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (APP_ID, DISPLAY_NAME, STATUS, MEMBER_ASSIGNABLE, ROW_VERSION, CREATED_AT, UPDATED_AT)
VALUES (source.APP_ID, source.DISPLAY_NAME, 'ACTIVE', 'Y', 1, SYSTIMESTAMP, SYSTIMESTAMP);

MERGE INTO KBOT_PLATFORM_DOMAIN target
USING (SELECT 'knowledge_retrieval_portal' NAME, '知识检索初始空白 Domain；可在 App 内创建业务 Domain' DESCRIPTION FROM DUAL) source
ON (target.NAME = source.NAME)
WHEN MATCHED THEN UPDATE SET target.STATUS = 'ACTIVE', target.DESCRIPTION = source.DESCRIPTION, target.UPDATED_BY = 'bootstrap:knowledge_retrieval_initial_admin', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (NAME, STATUS, DESCRIPTION, ROW_VERSION, CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT)
VALUES (source.NAME, 'ACTIVE', source.DESCRIPTION, 1, 'bootstrap:knowledge_retrieval_initial_admin', 'bootstrap:knowledge_retrieval_initial_admin', SYSTIMESTAMP, SYSTIMESTAMP);

MERGE INTO KBOT_PERMISSION target
USING (
    SELECT 'knowledge_retrieval:use' PERMISSION_CODE, 'knowledge_retrieval' APP_ID, '使用知识检索' DISPLAY_NAME FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:upload', 'knowledge_retrieval', '上传知识文件' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:review', 'knowledge_retrieval', '审核知识文件' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:member_manage', 'knowledge_retrieval', '管理应用成员' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:role_manage', 'knowledge_retrieval', '管理应用角色' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:knowledge_manage', 'knowledge_retrieval', '管理知识库' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:data_manage', 'knowledge_retrieval', '管理问数资源' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:agent_manage', 'knowledge_retrieval', '管理知识检索 Agent' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:api_key_manage', 'knowledge_retrieval', '管理知识检索 API Client' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:knowledge_chat', 'knowledge_retrieval', '使用知识问答' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:x_search', 'knowledge_retrieval', '使用 X 实时搜索' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:domain_manage', 'knowledge_retrieval', '管理知识 Domain' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:knowledge_core_manage', 'knowledge_retrieval', '管理 Knowledge Core' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval:data_model_manage', 'knowledge_retrieval', '管理问数模型' FROM DUAL
) source
ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN MATCHED THEN UPDATE SET target.APP_ID = source.APP_ID, target.DISPLAY_NAME = source.DISPLAY_NAME
WHEN NOT MATCHED THEN INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
VALUES (source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME);

MERGE INTO KBOT_APP_ROLE target
USING (
    SELECT 'knowledge_retrieval' APP_ID, 'user' ROLE_CODE, '用户' DISPLAY_NAME, 'Y' IS_SYSTEM, 'SELECTABLE' SCOPE_POLICY, 'ACTIVE' STATUS FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval', 'contributor', '贡献者', 'Y', 'SELECTABLE', 'ACTIVE' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval', 'reviewer', '审核人', 'Y', 'SELECTABLE', 'ACTIVE' FROM DUAL UNION ALL
    SELECT 'knowledge_retrieval', 'app_admin', '知识检索初始管理员', 'Y', 'ALL_APP_DOMAINS', 'ACTIVE' FROM DUAL
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME, target.IS_SYSTEM = source.IS_SYSTEM, target.SCOPE_POLICY = source.SCOPE_POLICY, target.STATUS = source.STATUS
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, DISPLAY_NAME, IS_SYSTEM, SCOPE_POLICY, STATUS, ROW_VERSION)
VALUES (source.APP_ID, source.ROLE_CODE, source.DISPLAY_NAME, source.IS_SYSTEM, source.SCOPE_POLICY, source.STATUS, 1);


MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT 'knowledge_retrieval' APP_ID, 'user' ROLE_CODE, PERMISSION_CODE
      FROM KBOT_PERMISSION
     WHERE PERMISSION_CODE IN ('knowledge_retrieval:use', 'knowledge_retrieval:knowledge_chat', 'knowledge_retrieval:x_search')
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE AND target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT 'knowledge_retrieval' APP_ID, 'contributor' ROLE_CODE, PERMISSION_CODE
      FROM KBOT_PERMISSION
     WHERE PERMISSION_CODE IN ('knowledge_retrieval:use', 'knowledge_retrieval:upload')
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE AND target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT 'knowledge_retrieval' APP_ID, 'reviewer' ROLE_CODE, PERMISSION_CODE
      FROM KBOT_PERMISSION
     WHERE PERMISSION_CODE IN ('knowledge_retrieval:use', 'knowledge_retrieval:upload', 'knowledge_retrieval:review')
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE AND target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT 'knowledge_retrieval' APP_ID, 'app_admin' ROLE_CODE, PERMISSION_CODE
      FROM KBOT_PERMISSION
     WHERE APP_ID = 'knowledge_retrieval'
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE AND target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_PLATFORM_USER target
USING (SELECT 'knowledgeadmin' USER_ID, '知识检索管理员' DISPLAY_NAME FROM DUAL) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME, target.ACCOUNT_ORIGIN = 'APP', target.OWNER_APP_ID = 'knowledge_retrieval', target.IS_PROTECTED = 'Y', target.MAX_SECURITY_LEVEL = 3, target.STATUS = 'ACTIVE', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (USER_ID, DISPLAY_NAME, ACCOUNT_ORIGIN, OWNER_APP_ID, IS_PROTECTED, MAX_SECURITY_LEVEL, STATUS, CREATED_AT, UPDATED_AT)
VALUES (source.USER_ID, source.DISPLAY_NAME, 'APP', 'knowledge_retrieval', 'Y', 3, 'ACTIVE', SYSTIMESTAMP, SYSTIMESTAMP);

MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL target
USING (SELECT 'knowledgeadmin' USER_ID, '$2b$12$8cE1PJglw40By3I.RlWHAeiQDtEBARUk8g3P8yKva9m.XqQ38ENwa' PASSWORD_HASH FROM DUAL) source
ON (target.USER_ID = source.USER_ID)
WHEN NOT MATCHED THEN INSERT (USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD, PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT)
VALUES (source.USER_ID, source.PASSWORD_HASH, 'Y', SYSTIMESTAMP, SYSTIMESTAMP, SYSTIMESTAMP);

MERGE INTO KBOT_APP_DOMAIN target
USING (SELECT 'knowledge_retrieval' APP_ID, DOMAIN_ID FROM KBOT_PLATFORM_DOMAIN WHERE NAME = 'knowledge_retrieval_portal' AND STATUS = 'ACTIVE') source
ON (target.APP_ID = source.APP_ID AND target.DOMAIN_ID = source.DOMAIN_ID)
WHEN MATCHED THEN UPDATE SET target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN INSERT (APP_ID, DOMAIN_ID, STATUS, CREATED_BY, CREATED_AT)
VALUES (source.APP_ID, source.DOMAIN_ID, 'ACTIVE', 'bootstrap:knowledge_retrieval_initial_admin', SYSTIMESTAMP);

MERGE INTO KBOT_APP_MEMBER target
USING (SELECT 'knowledge_retrieval' APP_ID, 'knowledgeadmin' USER_ID FROM DUAL) source
ON (target.APP_ID = source.APP_ID AND target.USER_ID = source.USER_ID)
WHEN MATCHED THEN UPDATE SET target.MEMBER_SOURCE = 'APP_INITIAL_ADMIN', target.IS_INITIAL_ADMIN = 'Y', target.STATUS = 'ACTIVE', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (APP_ID, USER_ID, MEMBER_SOURCE, IS_INITIAL_ADMIN, STATUS, GRANTED_BY, CREATED_AT, UPDATED_AT)
VALUES (source.APP_ID, source.USER_ID, 'APP_INITIAL_ADMIN', 'Y', 'ACTIVE', 'bootstrap:knowledge_retrieval_initial_admin', SYSTIMESTAMP, SYSTIMESTAMP);

MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (SELECT 'knowledge_retrieval' APP_ID, 'knowledgeadmin' USER_ID, 'app_admin' ROLE_CODE FROM DUAL) source
ON (target.APP_ID = source.APP_ID AND target.USER_ID = source.USER_ID AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN UPDATE SET target.SCOPE_MODE = 'ALL_APP_DOMAINS', target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN INSERT (APP_ID, USER_ID, ROLE_CODE, SCOPE_MODE, STATUS, CREATED_BY, CREATED_AT)
VALUES (source.APP_ID, source.USER_ID, source.ROLE_CODE, 'ALL_APP_DOMAINS', 'ACTIVE', 'bootstrap:knowledge_retrieval_initial_admin', SYSTIMESTAMP);

DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE
 WHERE APP_ID = 'knowledge_retrieval' AND USER_ID = 'knowledgeadmin' AND ROLE_CODE = 'app_admin';

COMMIT;
