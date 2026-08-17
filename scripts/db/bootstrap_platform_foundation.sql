-- KBot 4.0 首次登录所需的平台基础数据。
-- 本脚本只维护默认 Domain、权限目录、角色模板、ADMIN 和成员授权，可重复执行。
-- ADMIN 初始密码：Admin@2026!；既有凭据不会被覆盖。

MERGE INTO KBOT_PLATFORM_DOMAIN target
USING (
    SELECT 'default' AS NAME,
           'KBot 默认业务域' AS DESCRIPTION
      FROM DUAL
) source
ON (target.NAME = source.NAME)
WHEN MATCHED THEN
    UPDATE SET target.STATUS = 'ACTIVE',
               target.DESCRIPTION = source.DESCRIPTION,
               target.UPDATED_BY = 'schema-initializer',
               target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (
        NAME, STATUS, DESCRIPTION, ROW_VERSION,
        CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT
    )
    VALUES (
        source.NAME, 'ACTIVE', source.DESCRIPTION, 1,
        'schema-initializer', 'schema-initializer',
        SYSTIMESTAMP, SYSTIMESTAMP
    );

MERGE INTO KBOT_PERMISSION target
USING (
    SELECT 'platform:user_manage' PERMISSION_CODE, 'platform' APP_ID,
           '管理平台用户与成员授权' DISPLAY_NAME FROM DUAL
    UNION ALL SELECT 'platform:role_manage', 'platform', '管理平台应用角色与权限' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:use', 'knowledge_retrieval', '使用知识检索' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:upload', 'knowledge_retrieval', '上传知识文件' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:review', 'knowledge_retrieval', '审核知识文件' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:member_manage', 'knowledge_retrieval', '管理应用成员' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:knowledge_manage', 'knowledge_retrieval', '管理知识库' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:data_manage', 'knowledge_retrieval', '管理问数资源' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:agent_manage', 'knowledge_retrieval', '管理知识检索 Agent' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval:operations_manage', 'knowledge_retrieval', '管理知识检索运行' FROM DUAL
    UNION ALL SELECT 'km_asset:use', 'km_asset', '使用 KM Asset' FROM DUAL
    UNION ALL SELECT 'km_asset:source_manage', 'km_asset', '管理 KM Asset 来源' FROM DUAL
    UNION ALL SELECT 'km_asset:data_manage', 'km_asset', '管理 KM Asset 问数模型' FROM DUAL
    UNION ALL SELECT 'km_asset:agent_manage', 'km_asset', '管理 KM Asset Agent' FROM DUAL
    UNION ALL SELECT 'km_asset:operations_manage', 'km_asset', '管理 KM Asset 同步运行' FROM DUAL
    UNION ALL SELECT 'km_asset:member_manage', 'km_asset', '管理 KM Asset 成员' FROM DUAL
    UNION ALL SELECT 'aiops:use', 'aiops', '使用 AIOps' FROM DUAL
    UNION ALL SELECT 'aiops:domain_manage', 'aiops', '管理 AIOps Domain 配置' FROM DUAL
    UNION ALL SELECT 'aiops:member_manage', 'aiops', '管理 AIOps 成员' FROM DUAL
    UNION ALL SELECT 'aiops:operations_manage', 'aiops', '管理 AIOps 运行' FROM DUAL
    UNION ALL SELECT 'aiops:target_manage', 'aiops', '管理诊断目标' FROM DUAL
    UNION ALL SELECT 'aiops:monitor_source_manage', 'aiops', '管理监控源' FROM DUAL
    UNION ALL SELECT 'aiops:policy_manage', 'aiops', '管理诊断策略' FROM DUAL
    UNION ALL SELECT 'aiops:plan_manage', 'aiops', '管理变更计划' FROM DUAL
    UNION ALL SELECT 'aiops:agent_manage', 'aiops', '管理 AIOps Agent' FROM DUAL
    UNION ALL SELECT 'aiops:proposal:approve', 'aiops', '审批执行提案' FROM DUAL
) source
ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN MATCHED THEN
    UPDATE SET target.APP_ID = source.APP_ID,
               target.DISPLAY_NAME = source.DISPLAY_NAME
WHEN NOT MATCHED THEN
    INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
    VALUES (source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME);

MERGE INTO KBOT_APP_ROLE target
USING (
    SELECT 'platform' APP_ID, 'platform_admin' ROLE_CODE,
           '平台管理员' DISPLAY_NAME FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval', 'user', '用户' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval', 'contributor', '贡献者' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval', 'reviewer', '审核人' FROM DUAL
    UNION ALL SELECT 'knowledge_retrieval', 'manager', '管理员' FROM DUAL
    UNION ALL SELECT 'km_asset', 'user', '用户' FROM DUAL
    UNION ALL SELECT 'km_asset', 'manager', '管理员' FROM DUAL
    UNION ALL SELECT 'aiops', 'operator', '运维操作员' FROM DUAL
    UNION ALL SELECT 'aiops', 'approver', '审批人' FROM DUAL
    UNION ALL SELECT 'aiops', 'manager', '管理员' FROM DUAL
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN
    UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME,
               target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN
    INSERT (APP_ID, ROLE_CODE, DISPLAY_NAME, STATUS)
    VALUES (source.APP_ID, source.ROLE_CODE, source.DISPLAY_NAME, 'ACTIVE');

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT role_definition.APP_ID,
           role_definition.ROLE_CODE,
           permission.PERMISSION_CODE
      FROM KBOT_APP_ROLE role_definition
      JOIN KBOT_PERMISSION permission
        ON permission.APP_ID = role_definition.APP_ID
     WHERE (role_definition.APP_ID = 'platform'
            AND role_definition.ROLE_CODE = 'platform_admin')
        OR (role_definition.APP_ID = 'knowledge_retrieval'
            AND role_definition.ROLE_CODE = 'manager')
        OR (role_definition.APP_ID = 'km_asset'
            AND role_definition.ROLE_CODE = 'manager')
        OR (role_definition.APP_ID = 'aiops'
            AND role_definition.ROLE_CODE = 'manager')
        OR (role_definition.APP_ID = 'knowledge_retrieval'
            AND role_definition.ROLE_CODE = 'user'
            AND permission.PERMISSION_CODE = 'knowledge_retrieval:use')
        OR (role_definition.APP_ID = 'knowledge_retrieval'
            AND role_definition.ROLE_CODE = 'contributor'
            AND permission.PERMISSION_CODE IN (
                'knowledge_retrieval:use', 'knowledge_retrieval:upload'
            ))
        OR (role_definition.APP_ID = 'knowledge_retrieval'
            AND role_definition.ROLE_CODE = 'reviewer'
            AND permission.PERMISSION_CODE IN (
                'knowledge_retrieval:use', 'knowledge_retrieval:upload',
                'knowledge_retrieval:review'
            ))
        OR (role_definition.APP_ID = 'km_asset'
            AND role_definition.ROLE_CODE = 'user'
            AND permission.PERMISSION_CODE = 'km_asset:use')
        OR (role_definition.APP_ID = 'aiops'
            AND role_definition.ROLE_CODE = 'operator'
            AND permission.PERMISSION_CODE IN (
                'aiops:use', 'aiops:operations_manage', 'aiops:target_manage',
                'aiops:monitor_source_manage', 'aiops:policy_manage',
                'aiops:plan_manage'
            ))
        OR (role_definition.APP_ID = 'aiops'
            AND role_definition.ROLE_CODE = 'approver'
            AND permission.PERMISSION_CODE IN (
                'aiops:use', 'aiops:operations_manage',
                'aiops:proposal:approve'
            ))
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.ROLE_CODE = source.ROLE_CODE
    AND target.PERMISSION_CODE = source.PERMISSION_CODE
)
WHEN NOT MATCHED THEN
    INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
    VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_APP_ROLE target
USING (
    SELECT DISTINCT APP_ID, 'system_admin' ROLE_CODE,
           '系统管理员' DISPLAY_NAME
      FROM KBOT_PERMISSION
) source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN
    UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME,
               target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN
    INSERT (APP_ID, ROLE_CODE, DISPLAY_NAME, STATUS)
    VALUES (source.APP_ID, source.ROLE_CODE, source.DISPLAY_NAME, 'ACTIVE');

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT permission.APP_ID, 'system_admin' ROLE_CODE,
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
    VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_PLATFORM_USER target
USING (
    SELECT 'ADMIN' USER_ID, 'KBot 全局管理员' DISPLAY_NAME FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN
    UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME,
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
    SELECT 'ADMIN' USER_ID,
           '$2b$12$5WDXCasJPPANzr/QGlwbA.WOQxDa5Jq.RGGuFupPG1oossZZYKS3W'
               PASSWORD_HASH
      FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN NOT MATCHED THEN
    INSERT (
        USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD,
        PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT
    )
    VALUES (
        source.USER_ID, source.PASSWORD_HASH, 'N',
        SYSTIMESTAMP, SYSTIMESTAMP, SYSTIMESTAMP
    );

MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (
    SELECT role_definition.APP_ID, domain_row.DOMAIN_ID, 'ADMIN' USER_ID,
           role_definition.ROLE_CODE
      FROM KBOT_PLATFORM_DOMAIN domain_row
      CROSS JOIN KBOT_APP_ROLE role_definition
     WHERE domain_row.NAME = 'default'
       AND domain_row.STATUS = 'ACTIVE'
       AND role_definition.ROLE_CODE = 'system_admin'
       AND role_definition.STATUS = 'ACTIVE'
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
        'ACTIVE', 'schema-initializer', SYSTIMESTAMP
    );
