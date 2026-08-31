-- 由 scripts/db/initialize_aiops.py 调用的 AIOps 首次使用数据资产。
-- 本资产只写基础数据，不创建或修改 Schema 对象。
-- 固定资源：aiops_portal Domain、operations-manuals KC Collection、aiopsadmin。
-- 初始密码：AIOpsAdmin@2026!；重复执行会恢复该密码。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    l_table_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*) INTO l_table_count FROM USER_TABLES WHERE TABLE_NAME IN (
        'KBOT_PLATFORM_DOMAIN', 'KBOT_PLATFORM_APP', 'KBOT_PLATFORM_USER',
        'KBOT_PLATFORM_USER_CREDENTIAL', 'KBOT_AI_MODEL', 'KBOT_KC_COLLECTION',
        'KBOT_PERMISSION', 'KBOT_APP_ROLE', 'KBOT_APP_ROLE_PERMISSION',
        'KBOT_APP_DOMAIN', 'KBOT_APP_MEMBER', 'KBOT_APP_MEMBER_ROLE',
        'KBOT_APP_MEMBER_ROLE_SCOPE'
    );
    IF l_table_count <> 13 THEN
        raise_application_error(-20001, 'AIOps Domain、模型、KC 或权限基础表不完整。');
    END IF;
END;
/

MERGE INTO KBOT_PLATFORM_APP target
USING (SELECT 'aiops' APP_ID, 'AIOps' DISPLAY_NAME FROM DUAL) source
ON (target.APP_ID = source.APP_ID)
WHEN MATCHED THEN UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME,
    target.STATUS = 'ACTIVE', target.MEMBER_ASSIGNABLE = 'Y', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    APP_ID, DISPLAY_NAME, STATUS, MEMBER_ASSIGNABLE, ROW_VERSION, CREATED_AT, UPDATED_AT
) VALUES (source.APP_ID, source.DISPLAY_NAME, 'ACTIVE', 'Y', 1, SYSTIMESTAMP, SYSTIMESTAMP);

MERGE INTO KBOT_PLATFORM_DOMAIN target
USING (SELECT 'aiops_portal' NAME, 'AIOps 固定业务 Domain' DESCRIPTION FROM DUAL) source
ON (target.NAME = source.NAME)
WHEN MATCHED THEN UPDATE SET target.STATUS = 'ACTIVE', target.DESCRIPTION = source.DESCRIPTION,
    target.UPDATED_BY = 'bootstrap:aiops_initial_admin', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    NAME, STATUS, DESCRIPTION, ROW_VERSION, CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT
) VALUES (
    source.NAME, 'ACTIVE', source.DESCRIPTION, 1,
    'bootstrap:aiops_initial_admin', 'bootstrap:aiops_initial_admin', SYSTIMESTAMP, SYSTIMESTAMP
);

DECLARE
    l_domain_id NUMBER(38);
    l_collection_count PLS_INTEGER;
    l_embedding_model_id RAW(16);
    l_embedding_uuid VARCHAR2(36 CHAR);
BEGIN
    SELECT DOMAIN_ID INTO l_domain_id FROM KBOT_PLATFORM_DOMAIN
     WHERE NAME = 'aiops_portal' AND STATUS = 'ACTIVE';
    SELECT COUNT(*) INTO l_collection_count FROM KBOT_KC_COLLECTION
     WHERE DOMAIN_ID = l_domain_id AND DISPLAY_NAME = 'operations-manuals';
    IF l_collection_count > 1 THEN
        raise_application_error(-20002, 'aiops_portal 存在多个 operations-manuals Collection。');
    END IF;
    IF l_collection_count = 0 THEN
        BEGIN
            SELECT MODEL_ID INTO l_embedding_model_id FROM (
                SELECT MODEL_ID FROM KBOT_AI_MODEL
                 WHERE CATEGORY = 2 AND STATUS = 1
                 ORDER BY UPDATED_AT DESC, CREATED_AT DESC
            ) WHERE ROWNUM = 1;
        EXCEPTION WHEN NO_DATA_FOUND THEN
            raise_application_error(-20003, '缺少启用的文本 Embedding 模型。');
        END;
        l_embedding_uuid := LOWER(
            SUBSTR(RAWTOHEX(l_embedding_model_id), 1, 8) || '-' ||
            SUBSTR(RAWTOHEX(l_embedding_model_id), 9, 4) || '-' ||
            SUBSTR(RAWTOHEX(l_embedding_model_id), 13, 4) || '-' ||
            SUBSTR(RAWTOHEX(l_embedding_model_id), 17, 4) || '-' ||
            SUBSTR(RAWTOHEX(l_embedding_model_id), 21, 12)
        );
        INSERT INTO KBOT_KC_COLLECTION (
            COLLECTION_ID, DOMAIN_ID, DISPLAY_NAME, DESCRIPTION, MODELS_JSON,
            PARSE_POLICY_JSON, STATUS, DEFAULT_SECURITY_LEVEL, METADATA_JSON,
            ROW_VERSION, CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT
        ) VALUES (
            HEXTORAW(REPLACE('019ffff0-0000-7000-8000-000000000001', '-', '')),
            l_domain_id, 'operations-manuals', 'AIOps 固定数据库运维手册 Collection',
            '{"embedding":"' || l_embedding_uuid || '"}',
            '{"parse_strategy":"AUTO","do_ocr":true,"ocr_engine":"tesseract","image_scale":2.0,"extract_page_images":true,"extract_picture_images":true,"detect_table_structure":true}',
            'ACTIVE', 1, '{"owner_app_id":"aiops","fixed_resource":true,"content_kind":"operations_manual"}',
            1, 'bootstrap:aiops_initial_admin', 'bootstrap:aiops_initial_admin',
            SYSTIMESTAMP, SYSTIMESTAMP
        );
    ELSE
        UPDATE KBOT_KC_COLLECTION SET STATUS = 'ACTIVE',
            DESCRIPTION = 'AIOps 固定数据库运维手册 Collection',
            UPDATED_BY = 'bootstrap:aiops_initial_admin', UPDATED_AT = SYSTIMESTAMP
         WHERE DOMAIN_ID = l_domain_id AND DISPLAY_NAME = 'operations-manuals';
    END IF;
END;
/

MERGE INTO KBOT_PERMISSION target
USING (
    SELECT 'aiops:use' PERMISSION_CODE, 'aiops' APP_ID, '使用 AIOps' DISPLAY_NAME FROM DUAL UNION ALL
    SELECT 'aiops:domain_manage', 'aiops', '管理 AIOps Domain 配置' FROM DUAL UNION ALL
    SELECT 'aiops:member_manage', 'aiops', '管理 AIOps 成员' FROM DUAL UNION ALL
    SELECT 'aiops:role_manage', 'aiops', '管理 AIOps 角色' FROM DUAL UNION ALL
    SELECT 'aiops:operations_manage', 'aiops', '管理 AIOps 运行' FROM DUAL UNION ALL
    SELECT 'aiops:target_manage', 'aiops', '管理诊断目标' FROM DUAL UNION ALL
    SELECT 'aiops:diagnostic_source_manage', 'aiops', '管理诊断源' FROM DUAL UNION ALL
    SELECT 'aiops:policy_manage', 'aiops', '管理诊断策略' FROM DUAL UNION ALL
    SELECT 'aiops:plan_manage', 'aiops', '管理巡检计划' FROM DUAL UNION ALL
    SELECT 'aiops:agent_manage', 'aiops', '管理 AIOps Agent' FROM DUAL UNION ALL
    SELECT 'aiops:knowledge_manage', 'aiops', '管理 AIOps Knowledge Core' FROM DUAL UNION ALL
    SELECT 'aiops:api_key_manage', 'aiops', '管理 AIOps API Client' FROM DUAL UNION ALL
    SELECT 'aiops:proposal:approve', 'aiops', '审批执行提案' FROM DUAL
) source ON (target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN MATCHED THEN UPDATE SET target.APP_ID = source.APP_ID, target.DISPLAY_NAME = source.DISPLAY_NAME
WHEN NOT MATCHED THEN INSERT (PERMISSION_CODE, APP_ID, DISPLAY_NAME)
VALUES (source.PERMISSION_CODE, source.APP_ID, source.DISPLAY_NAME);

MERGE INTO KBOT_APP_ROLE target
USING (
    SELECT 'aiops' APP_ID, 'app_admin' ROLE_CODE, 'AIOps 初始管理员' DISPLAY_NAME,
           'Y' IS_SYSTEM, 'ALL_APP_DOMAINS' SCOPE_POLICY, 'ACTIVE' STATUS FROM DUAL
    UNION ALL SELECT 'aiops', 'operator', '运维操作员', 'Y', 'SELECTABLE', 'ACTIVE' FROM DUAL
    UNION ALL SELECT 'aiops', 'approver', '审批人', 'Y', 'SELECTABLE', 'ACTIVE' FROM DUAL
) source ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME,
    target.IS_SYSTEM = source.IS_SYSTEM, target.SCOPE_POLICY = source.SCOPE_POLICY,
    target.STATUS = source.STATUS
WHEN NOT MATCHED THEN INSERT (
    APP_ID, ROLE_CODE, DISPLAY_NAME, IS_SYSTEM, SCOPE_POLICY, STATUS, ROW_VERSION
) VALUES (
    source.APP_ID, source.ROLE_CODE, source.DISPLAY_NAME, source.IS_SYSTEM,
    source.SCOPE_POLICY, source.STATUS, 1
);

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (SELECT 'aiops' APP_ID, 'app_admin' ROLE_CODE, PERMISSION_CODE
         FROM KBOT_PERMISSION WHERE APP_ID = 'aiops') source
ON (target.APP_ID = source.APP_ID AND target.ROLE_CODE = source.ROLE_CODE
    AND target.PERMISSION_CODE = source.PERMISSION_CODE)
WHEN NOT MATCHED THEN INSERT (APP_ID, ROLE_CODE, PERMISSION_CODE)
VALUES (source.APP_ID, source.ROLE_CODE, source.PERMISSION_CODE);

MERGE INTO KBOT_PLATFORM_USER target
USING (SELECT 'aiopsadmin' USER_ID, 'AIOps 管理员' DISPLAY_NAME FROM DUAL) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN UPDATE SET target.DISPLAY_NAME = source.DISPLAY_NAME,
    target.ACCOUNT_ORIGIN = 'APP', target.OWNER_APP_ID = 'aiops',
    target.IS_PROTECTED = 'Y', target.MAX_SECURITY_LEVEL = 3,
    target.STATUS = 'ACTIVE', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    USER_ID, DISPLAY_NAME, ACCOUNT_ORIGIN, OWNER_APP_ID,
    IS_PROTECTED, MAX_SECURITY_LEVEL, STATUS, CREATED_AT, UPDATED_AT
) VALUES (
    source.USER_ID, source.DISPLAY_NAME, 'APP', 'aiops',
    'Y', 3, 'ACTIVE', SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL target
USING (SELECT 'aiopsadmin' USER_ID,
    '$2b$12$Ampx0AbuXRgkmWUXjfqMCOTbA32h2vmtYqFOfBCGlXwHrblm0JHse' PASSWORD_HASH FROM DUAL) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN UPDATE SET target.PASSWORD_HASH = source.PASSWORD_HASH,
    target.MUST_CHANGE_PASSWORD = 'N', target.PASSWORD_UPDATED_AT = SYSTIMESTAMP,
    target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD, PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT
) VALUES (source.USER_ID, source.PASSWORD_HASH, 'N', SYSTIMESTAMP, SYSTIMESTAMP, SYSTIMESTAMP);

MERGE INTO KBOT_APP_DOMAIN target
USING (SELECT 'aiops' APP_ID, DOMAIN_ID FROM KBOT_PLATFORM_DOMAIN
        WHERE NAME = 'aiops_portal' AND STATUS = 'ACTIVE') source
ON (target.APP_ID = source.APP_ID AND target.DOMAIN_ID = source.DOMAIN_ID)
WHEN MATCHED THEN UPDATE SET target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN INSERT (APP_ID, DOMAIN_ID, STATUS, CREATED_BY, CREATED_AT)
VALUES (source.APP_ID, source.DOMAIN_ID, 'ACTIVE', 'bootstrap:aiops_initial_admin', SYSTIMESTAMP);

MERGE INTO KBOT_APP_MEMBER target
USING (SELECT 'aiops' APP_ID, 'aiopsadmin' USER_ID FROM DUAL) source
ON (target.APP_ID = source.APP_ID AND target.USER_ID = source.USER_ID)
WHEN MATCHED THEN UPDATE SET target.MEMBER_SOURCE = 'APP_INITIAL_ADMIN',
    target.IS_INITIAL_ADMIN = 'Y', target.STATUS = 'ACTIVE', target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN INSERT (
    APP_ID, USER_ID, MEMBER_SOURCE, IS_INITIAL_ADMIN, STATUS, GRANTED_BY, CREATED_AT, UPDATED_AT
) VALUES (
    source.APP_ID, source.USER_ID, 'APP_INITIAL_ADMIN', 'Y', 'ACTIVE',
    'bootstrap:aiops_initial_admin', SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (SELECT 'aiops' APP_ID, 'aiopsadmin' USER_ID, 'app_admin' ROLE_CODE FROM DUAL) source
ON (target.APP_ID = source.APP_ID AND target.USER_ID = source.USER_ID
    AND target.ROLE_CODE = source.ROLE_CODE)
WHEN MATCHED THEN UPDATE SET target.SCOPE_MODE = 'ALL_APP_DOMAINS', target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN INSERT (
    APP_ID, USER_ID, ROLE_CODE, SCOPE_MODE, STATUS, CREATED_BY, CREATED_AT
) VALUES (
    source.APP_ID, source.USER_ID, source.ROLE_CODE, 'ALL_APP_DOMAINS',
    'ACTIVE', 'bootstrap:aiops_initial_admin', SYSTIMESTAMP
);

DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE
 WHERE APP_ID = 'aiops' AND USER_ID = 'aiopsadmin' AND ROLE_CODE = 'app_admin';

COMMIT;

DECLARE
    l_domain_id NUMBER(38);
    l_collection_id VARCHAR2(36 CHAR);
BEGIN
    SELECT domain.DOMAIN_ID,
           LOWER(SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 1, 8) || '-' ||
                 SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 9, 4) || '-' ||
                 SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 13, 4) || '-' ||
                 SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 17, 4) || '-' ||
                 SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 21, 12))
      INTO l_domain_id, l_collection_id
      FROM KBOT_PLATFORM_DOMAIN domain
      JOIN KBOT_KC_COLLECTION collection ON collection.DOMAIN_ID = domain.DOMAIN_ID
     WHERE domain.NAME = 'aiops_portal' AND collection.DISPLAY_NAME = 'operations-manuals';
    dbms_output.put_line('AIOps 初始化完成。');
    dbms_output.put_line('登录用户：aiopsadmin');
    dbms_output.put_line('固定 Domain：aiops_portal（ID=' || l_domain_id || '）');
    dbms_output.put_line('固定 Collection：operations-manuals（ID=' || l_collection_id || '）');
END;
/
