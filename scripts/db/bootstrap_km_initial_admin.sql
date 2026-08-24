-- KM Asset 首次使用的一步初始化脚本。
-- 在 SQL Developer 中使用 Run Script（F5）执行；脚本不要求输入绑定变量。
-- 本脚本会创建或启用固定 Domain km_portal、创建固定 Collection assets、
-- 补齐 KM 权限与角色，并创建或启用受保护的 KM 初始 App 管理员。
-- USER_ID 区分大小写，必须与页面登录用户完全一致。
-- 初始账号：kmadmin，初始密码：KmAdmin@2026!
-- 登录后可直接使用；重新执行本脚本会将密码恢复为上述固定值。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

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
BEGIN
    SELECT COUNT(*)
      INTO l_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME IN (
        'KBOT_PLATFORM_DOMAIN',
        'KBOT_PLATFORM_APP',
        'KBOT_PLATFORM_USER',
        'KBOT_PLATFORM_USER_CREDENTIAL',
        'KBOT_AI_MODEL',
        'KBOT_KC_COLLECTION',
        'KBOT_PERMISSION',
        'KBOT_APP_ROLE',
        'KBOT_APP_ROLE_PERMISSION',
        'KBOT_APP_DOMAIN',
        'KBOT_APP_MEMBER',
        'KBOT_APP_MEMBER_ROLE',
        'KBOT_APP_MEMBER_ROLE_SCOPE'
     );

    IF l_table_count <> 13 THEN
        raise_application_error(
            -20001,
            'Domain、模型、KC Collection 或用户权限基础表不完整，不能初始化 KM。'
        );
    END IF;
    dbms_output.put_line('KM 初始化依赖表检查通过。');
END;
/

MERGE INTO KBOT_PLATFORM_APP target
USING (SELECT 'km_asset' APP_ID, 'KM Asset' DISPLAY_NAME FROM DUAL) source
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
    SELECT
        'km_portal' AS NAME,
        'KM Portal 固定业务 Domain' AS DESCRIPTION
    FROM DUAL
) source
ON (target.NAME = source.NAME)
WHEN MATCHED THEN
    UPDATE SET
        target.STATUS = 'ACTIVE',
        target.DESCRIPTION = source.DESCRIPTION,
        target.UPDATED_BY = 'bootstrap:km_initial_admin',
        target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (
        NAME, STATUS, DESCRIPTION, ROW_VERSION,
        CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT
    )
    VALUES (
        source.NAME, 'ACTIVE', source.DESCRIPTION, 1,
        'bootstrap:km_initial_admin', 'bootstrap:km_initial_admin',
        SYSTIMESTAMP, SYSTIMESTAMP
    );

DECLARE
    l_domain_id NUMBER(38);
    l_collection_count PLS_INTEGER;
    l_llm_model_id RAW(16);
    l_embedding_model_id RAW(16);
    l_llm_uuid VARCHAR2(36 CHAR);
    l_embedding_uuid VARCHAR2(36 CHAR);
    l_models_json VARCHAR2(1000 CHAR);

    FUNCTION raw_uuid(value RAW) RETURN VARCHAR2 IS
        hex_value VARCHAR2(32 CHAR) := LOWER(RAWTOHEX(value));
    BEGIN
        RETURN SUBSTR(hex_value, 1, 8) || '-'
            || SUBSTR(hex_value, 9, 4) || '-'
            || SUBSTR(hex_value, 13, 4) || '-'
            || SUBSTR(hex_value, 17, 4) || '-'
            || SUBSTR(hex_value, 21, 12);
    END;
BEGIN
    SELECT DOMAIN_ID
      INTO l_domain_id
      FROM KBOT_PLATFORM_DOMAIN
     WHERE NAME = 'km_portal'
       AND STATUS = 'ACTIVE';

    SELECT COUNT(*)
      INTO l_collection_count
      FROM KBOT_KC_COLLECTION
     WHERE DOMAIN_ID = l_domain_id
       AND DISPLAY_NAME = 'assets';

    IF l_collection_count > 1 THEN
        raise_application_error(
            -20002,
            'km_portal Domain 中存在多个 assets Collection，请先清理重复数据。'
        );
    END IF;

    IF l_collection_count = 0 THEN
        BEGIN
            SELECT MODEL_ID
              INTO l_llm_model_id
              FROM (
                  SELECT MODEL_ID
                    FROM KBOT_AI_MODEL
                   WHERE CATEGORY = 1
                     AND STATUS = 1
                   ORDER BY UPDATED_AT DESC, CREATED_AT DESC
              )
             WHERE ROWNUM = 1;
        EXCEPTION
            WHEN NO_DATA_FOUND THEN
                raise_application_error(
                    -20003,
                    '缺少启用的 LLM 模型，无法创建 assets Collection。'
                );
        END;

        BEGIN
            SELECT MODEL_ID
              INTO l_embedding_model_id
              FROM (
                  SELECT MODEL_ID
                    FROM KBOT_AI_MODEL
                   WHERE CATEGORY = 2
                     AND STATUS = 1
                   ORDER BY UPDATED_AT DESC, CREATED_AT DESC
              )
             WHERE ROWNUM = 1;
        EXCEPTION
            WHEN NO_DATA_FOUND THEN
                raise_application_error(
                    -20004,
                    '缺少启用的文本 Embedding 模型，无法创建 assets Collection。'
                );
        END;

        l_llm_uuid := raw_uuid(l_llm_model_id);
        l_embedding_uuid := raw_uuid(l_embedding_model_id);
        l_models_json := '{"parser_llm":"' || l_llm_uuid
            || '","embedding":"' || l_embedding_uuid || '"}';

        INSERT INTO KBOT_KC_COLLECTION (
            COLLECTION_ID, DOMAIN_ID, DISPLAY_NAME, DESCRIPTION,
            MODELS_JSON, PARSE_POLICY_JSON, STATUS,
            DEFAULT_SECURITY_LEVEL, METADATA_JSON, ROW_VERSION,
            CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT
        ) VALUES (
            HEXTORAW(REPLACE('019ff8d2-ea17-765e-b167-6014a28d157b', '-', '')),
            l_domain_id,
            'assets',
            'KM Portal Asset 文档固定 Collection',
            l_models_json,
            '{"parse_strategy":"AUTO","do_ocr":true,"ocr_engine":"tesseract","image_scale":2.0,"extract_page_images":true,"extract_picture_images":true,"detect_table_structure":true,"visual_min_text_characters":80,"visual_min_mean_confidence":0.65,"visual_max_gibberish_ratio":0.08}',
            'ACTIVE',
            1,
            '{"owner_app_id":"km_asset","fixed_resource":true}',
            1,
            'bootstrap:km_initial_admin',
            'bootstrap:km_initial_admin',
            SYSTIMESTAMP,
            SYSTIMESTAMP
        );
        dbms_output.put_line('已创建 km_portal/assets 固定 Collection。');
    ELSE
        UPDATE KBOT_KC_COLLECTION
           SET STATUS = 'ACTIVE',
               DESCRIPTION = 'KM Portal Asset 文档固定 Collection',
               UPDATED_BY = 'bootstrap:km_initial_admin',
               UPDATED_AT = SYSTIMESTAMP
         WHERE DOMAIN_ID = l_domain_id
           AND DISPLAY_NAME = 'assets';
        dbms_output.put_line('已复用并启用 km_portal/assets 固定 Collection。');
    END IF;
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
    SELECT 'km_asset:knowledge_manage', 'km_asset', '管理 KM Portal Knowledge Core' FROM DUAL
    UNION ALL
    SELECT 'km_asset:data_manage', 'km_asset', '管理 KM Asset 问数模型' FROM DUAL
    UNION ALL
    SELECT 'km_asset:agent_manage', 'km_asset', '管理 KM Asset Agent' FROM DUAL
    UNION ALL
    SELECT 'km_asset:operations_manage', 'km_asset', '管理 KM Asset 同步运行' FROM DUAL
    UNION ALL
    SELECT 'km_asset:member_manage', 'km_asset', '管理 KM Asset 成员' FROM DUAL
    UNION ALL
    SELECT 'km_asset:role_manage', 'km_asset', '管理 KM Asset 角色' FROM DUAL
    UNION ALL
    SELECT 'km_asset:api_key_manage', 'km_asset', '管理 KM Asset API Client' FROM DUAL
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
           'Y' AS IS_SYSTEM,
           'SELECTABLE' AS SCOPE_POLICY,
           'ACTIVE' AS STATUS
      FROM DUAL
    UNION ALL
    SELECT 'km_asset', 'app_admin', 'KM Asset 初始管理员',
           'Y', 'ALL_APP_DOMAINS', 'ACTIVE' FROM DUAL
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.ROLE_CODE = source.ROLE_CODE
)
WHEN MATCHED THEN
    UPDATE SET
        target.DISPLAY_NAME = source.DISPLAY_NAME,
        target.IS_SYSTEM = source.IS_SYSTEM,
        target.SCOPE_POLICY = source.SCOPE_POLICY,
        target.STATUS = source.STATUS
WHEN NOT MATCHED THEN
    INSERT (
        APP_ID, ROLE_CODE, DISPLAY_NAME, IS_SYSTEM,
        SCOPE_POLICY, STATUS, ROW_VERSION
    )
    VALUES (
        source.APP_ID, source.ROLE_CODE,
        source.DISPLAY_NAME, source.IS_SYSTEM,
        source.SCOPE_POLICY, source.STATUS, 1
    );

MERGE INTO KBOT_APP_ROLE_PERMISSION target
USING (
    SELECT
        'km_asset' AS APP_ID,
        'app_admin' AS ROLE_CODE,
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
        'kmadmin' AS USER_ID,
        'KM Asset 管理员' AS DISPLAY_NAME
    FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN
    UPDATE SET
        target.DISPLAY_NAME = source.DISPLAY_NAME,
        target.ACCOUNT_ORIGIN = 'APP',
        target.OWNER_APP_ID = 'km_asset',
        target.IS_PROTECTED = 'Y',
        target.MAX_SECURITY_LEVEL = 3,
        target.STATUS = 'ACTIVE',
        target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (
        USER_ID, DISPLAY_NAME, ACCOUNT_ORIGIN, OWNER_APP_ID,
        IS_PROTECTED, MAX_SECURITY_LEVEL, STATUS, CREATED_AT, UPDATED_AT
    )
    VALUES (
        source.USER_ID, source.DISPLAY_NAME, 'APP', 'km_asset',
        'Y', 3, 'ACTIVE',
        SYSTIMESTAMP, SYSTIMESTAMP
    );

MERGE INTO KBOT_PLATFORM_USER_CREDENTIAL target
USING (
    SELECT
        'kmadmin' AS USER_ID,
        '$2b$12$QyA/YRNs6.JVOLh9saV7oeW7wskZ0qDEioAgV8oMBO7jOkxchNDQa'
            AS PASSWORD_HASH
    FROM DUAL
) source
ON (target.USER_ID = source.USER_ID)
WHEN MATCHED THEN
    UPDATE SET
        target.PASSWORD_HASH = source.PASSWORD_HASH,
        target.MUST_CHANGE_PASSWORD = 'N',
        target.PASSWORD_UPDATED_AT = SYSTIMESTAMP,
        target.UPDATED_AT = SYSTIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (
        USER_ID, PASSWORD_HASH, MUST_CHANGE_PASSWORD,
        PASSWORD_UPDATED_AT, CREATED_AT, UPDATED_AT
    )
    VALUES (
        source.USER_ID, source.PASSWORD_HASH, 'N',
        SYSTIMESTAMP, SYSTIMESTAMP, SYSTIMESTAMP
    );

MERGE INTO KBOT_APP_DOMAIN target
USING (
    SELECT
        'km_asset' AS APP_ID,
        domain.DOMAIN_ID
    FROM KBOT_PLATFORM_DOMAIN domain
    WHERE domain.NAME = 'km_portal'
      AND domain.STATUS = 'ACTIVE'
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.DOMAIN_ID = source.DOMAIN_ID
)
WHEN MATCHED THEN
    UPDATE SET target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN
    INSERT (
        APP_ID, DOMAIN_ID, STATUS, CREATED_BY, CREATED_AT
    )
    VALUES (
        source.APP_ID, source.DOMAIN_ID, 'ACTIVE',
        'bootstrap:km_initial_admin', SYSTIMESTAMP
    );

MERGE INTO KBOT_APP_MEMBER target
USING (
    SELECT 'km_asset' APP_ID, 'kmadmin' USER_ID FROM DUAL
) source
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
    'ACTIVE', 'bootstrap:km_initial_admin', SYSTIMESTAMP, SYSTIMESTAMP
);

MERGE INTO KBOT_APP_MEMBER_ROLE target
USING (
    SELECT 'km_asset' APP_ID, 'kmadmin' USER_ID, 'app_admin' ROLE_CODE FROM DUAL
) source
ON (
    target.APP_ID = source.APP_ID
    AND target.USER_ID = source.USER_ID
    AND target.ROLE_CODE = source.ROLE_CODE
)
WHEN MATCHED THEN UPDATE SET
    target.SCOPE_MODE = 'ALL_APP_DOMAINS', target.STATUS = 'ACTIVE'
WHEN NOT MATCHED THEN INSERT (
    APP_ID, USER_ID, ROLE_CODE, SCOPE_MODE,
    STATUS, CREATED_BY, CREATED_AT
) VALUES (
    source.APP_ID, source.USER_ID, source.ROLE_CODE, 'ALL_APP_DOMAINS',
    'ACTIVE', 'bootstrap:km_initial_admin', SYSTIMESTAMP
);

DELETE FROM KBOT_APP_MEMBER_ROLE_SCOPE
 WHERE APP_ID = 'km_asset'
   AND USER_ID = 'kmadmin'
   AND ROLE_CODE = 'app_admin';

COMMIT;

DECLARE
    l_domain_id NUMBER(38);
    l_collection_id VARCHAR2(36 CHAR);
BEGIN
    SELECT domain.DOMAIN_ID,
           LOWER(SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 1, 8) || '-'
               || SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 9, 4) || '-'
               || SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 13, 4) || '-'
               || SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 17, 4) || '-'
               || SUBSTR(RAWTOHEX(collection.COLLECTION_ID), 21, 12))
      INTO l_domain_id, l_collection_id
      FROM KBOT_PLATFORM_DOMAIN domain
      JOIN KBOT_KC_COLLECTION collection
        ON collection.DOMAIN_ID = domain.DOMAIN_ID
     WHERE domain.NAME = 'km_portal'
       AND collection.DISPLAY_NAME = 'assets';

    dbms_output.put_line('KM 初始化完成。');
    dbms_output.put_line('登录用户：kmadmin');
    dbms_output.put_line('固定 Domain：km_portal（ID=' || l_domain_id || '）');
    dbms_output.put_line('固定 Collection：assets（ID=' || l_collection_id || '）');
END;
/
