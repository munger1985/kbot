-- AIOps Schema 46 -> 47：增加 Domain 级多运维知识库及 Agent 多选绑定。
-- 使用 Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- Oracle DDL 会自动提交；执行前必须备份并停止全部 AIOps 写入服务。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
SET SERVEROUTPUT ON
SET VERIFY OFF
SET SQLBLANKLINES ON
SET DEFINE OFF

ALTER SESSION SET TIME_ZONE = '+00:00';
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

PROMPT === 检查 Schema 46 升级前置条件 ===

DECLARE
    l_component VARCHAR2(32 CHAR);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_default_collection_count PLS_INTEGER;
    l_unmapped_domain_count PLS_INTEGER;
BEGIN
    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_component <> 'AIOPS' OR l_schema_version <> 46
       OR l_contract_version <> 'aiops-oracle-v46-knowledge-tags' THEN
        raise_application_error(
            -20500,
            '本脚本仅支持 AIOPS/46/aiops-oracle-v46-knowledge-tags，当前为 '
            || l_component || '/' || l_schema_version || '/' || l_contract_version
        );
    END IF;

    SELECT COUNT(*) INTO l_default_collection_count
      FROM KBOT_PLATFORM_DOMAIN domain
      JOIN KBOT_KC_COLLECTION collection
        ON collection.DOMAIN_ID = domain.DOMAIN_ID
     WHERE domain.NAME = 'aiops_portal'
       AND domain.STATUS = 'ACTIVE'
       AND collection.DISPLAY_NAME = 'operations-manuals'
       AND collection.STATUS = 'ACTIVE';

    IF l_default_collection_count <> 1 THEN
        raise_application_error(
            -20501,
            'aiops_portal 必须且只能有一个启用的 operations-manuals Collection，当前为 '
            || l_default_collection_count
        );
    END IF;

    SELECT COUNT(*) INTO l_unmapped_domain_count
      FROM (
            SELECT DOMAIN_ID FROM KBOT_OPS_KNOWLEDGE_ASSET
            UNION
            SELECT DOMAIN_ID FROM KBOT_OPS_AGENT
      ) owned_domain
     WHERE (
            SELECT COUNT(*)
              FROM KBOT_KC_COLLECTION collection
             WHERE collection.DOMAIN_ID = owned_domain.DOMAIN_ID
               AND collection.DISPLAY_NAME = 'operations-manuals'
               AND collection.STATUS = 'ACTIVE'
     ) <> 1;

    IF l_unmapped_domain_count <> 0 THEN
        raise_application_error(
            -20502,
            '存在无法唯一映射到 operations-manuals Collection 的既有 AIOps Domain，数量='
            || l_unmapped_domain_count
        );
    END IF;
END;
/

PROMPT === 创建运维知识库与固定知识问答 Agent 绑定表 ===

CREATE TABLE KBOT_OPS_KNOWLEDGE_BASE (
    KNOWLEDGE_BASE_ID RAW(16) PRIMARY KEY,
    DOMAIN_ID NUMBER(38) NOT NULL,
    DISPLAY_NAME VARCHAR2(256 CHAR) NOT NULL,
    DESCRIPTION VARCHAR2(1000 CHAR),
    COLLECTION_ID RAW(16) NOT NULL,
    STATUS VARCHAR2(16 CHAR) DEFAULT 'ACTIVE' NOT NULL,
    CREATED_BY VARCHAR2(256 CHAR) NOT NULL,
    UPDATED_BY VARCHAR2(256 CHAR) NOT NULL,
    CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    UPDATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    ROW_VERSION NUMBER(19) DEFAULT 1 NOT NULL,
    CONSTRAINT FK_OPS_KNOW_BASE_DOMAIN FOREIGN KEY (DOMAIN_ID)
        REFERENCES KBOT_PLATFORM_DOMAIN (DOMAIN_ID)
);

CREATE INDEX IX_OPS_KNOW_BASE_SCOPE ON KBOT_OPS_KNOWLEDGE_BASE
    (DOMAIN_ID, STATUS, UPDATED_AT);

CREATE TABLE KBOT_OPS_KNOW_ASSISTANT_BIND (
    DOMAIN_ID NUMBER(38) PRIMARY KEY,
    ASSISTANT_ID RAW(16) NOT NULL,
    ASSISTANT_VERSION_ID RAW(16) NOT NULL,
    CREATED_BY VARCHAR2(256 CHAR) NOT NULL,
    UPDATED_BY VARCHAR2(256 CHAR) NOT NULL,
    CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    UPDATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    ROW_VERSION NUMBER(19) DEFAULT 1 NOT NULL,
    CONSTRAINT FK_OPS_KNOW_ASSIST_DOMAIN FOREIGN KEY (DOMAIN_ID)
        REFERENCES KBOT_PLATFORM_DOMAIN (DOMAIN_ID)
);

CREATE TABLE KBOT_OPS_KNOW_ASSISTANT_BASE (
    DOMAIN_ID NUMBER(38) NOT NULL,
    KNOWLEDGE_BASE_ID RAW(16) NOT NULL,
    CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT PK_OPS_KNOW_ASSIST_BASE
        PRIMARY KEY (DOMAIN_ID, KNOWLEDGE_BASE_ID),
    CONSTRAINT FK_OPS_KNOW_ASSIST_BIND FOREIGN KEY (DOMAIN_ID)
        REFERENCES KBOT_OPS_KNOW_ASSISTANT_BIND (DOMAIN_ID) ON DELETE CASCADE,
    CONSTRAINT FK_OPS_KNOW_ASSIST_BASE FOREIGN KEY (KNOWLEDGE_BASE_ID)
        REFERENCES KBOT_OPS_KNOWLEDGE_BASE (KNOWLEDGE_BASE_ID)
);

CREATE INDEX IX_OPS_KNOW_ASSIST_BASE
    ON KBOT_OPS_KNOW_ASSISTANT_BASE (KNOWLEDGE_BASE_ID);

PROMPT === 迁移既有 Collection、知识资产与固定知识问答 Agent ===

INSERT INTO KBOT_OPS_KNOWLEDGE_BASE (
    KNOWLEDGE_BASE_ID, DOMAIN_ID, DISPLAY_NAME, DESCRIPTION, COLLECTION_ID,
    STATUS, CREATED_BY, UPDATED_BY, CREATED_AT, UPDATED_AT, ROW_VERSION
)
SELECT CASE
           WHEN domain.NAME = 'aiops_portal'
           THEN HEXTORAW('019FFFF0000070008000000000001001')
           ELSE collection.COLLECTION_ID
       END,
       domain.DOMAIN_ID,
       '默认运维知识库',
       'Schema 47 升级时由既有 operations-manuals Collection 迁移',
       collection.COLLECTION_ID,
       'ACTIVE',
       'migration:aiops_schema_47',
       'migration:aiops_schema_47',
       SYSTIMESTAMP,
       SYSTIMESTAMP,
       1
  FROM KBOT_PLATFORM_DOMAIN domain
  JOIN KBOT_KC_COLLECTION collection
    ON collection.DOMAIN_ID = domain.DOMAIN_ID
   AND collection.DISPLAY_NAME = 'operations-manuals'
   AND collection.STATUS = 'ACTIVE'
 WHERE domain.NAME = 'aiops_portal'
    OR EXISTS (
        SELECT 1 FROM KBOT_OPS_KNOWLEDGE_ASSET asset
         WHERE asset.DOMAIN_ID = domain.DOMAIN_ID
    )
    OR EXISTS (
        SELECT 1 FROM KBOT_OPS_AGENT agent
         WHERE agent.DOMAIN_ID = domain.DOMAIN_ID
    );

ALTER TABLE KBOT_OPS_KNOWLEDGE_ASSET ADD (
    KNOWLEDGE_BASE_ID RAW(16),
    CONSTRAINT FK_OPS_KNOW_ASSET_BASE FOREIGN KEY (KNOWLEDGE_BASE_ID)
        REFERENCES KBOT_OPS_KNOWLEDGE_BASE (KNOWLEDGE_BASE_ID)
);

UPDATE KBOT_OPS_KNOWLEDGE_ASSET asset
   SET asset.KNOWLEDGE_BASE_ID = (
       SELECT knowledge_base.KNOWLEDGE_BASE_ID
         FROM KBOT_OPS_KNOWLEDGE_BASE knowledge_base
        WHERE knowledge_base.DOMAIN_ID = asset.DOMAIN_ID
   );

INSERT INTO KBOT_OPS_KNOW_ASSISTANT_BIND (
    DOMAIN_ID, ASSISTANT_ID, ASSISTANT_VERSION_ID, CREATED_BY, UPDATED_BY,
    CREATED_AT, UPDATED_AT, ROW_VERSION
)
SELECT knowledge_base.DOMAIN_ID,
       HEXTORAW('019D2A10000070008000000000000001'),
       HEXTORAW('019D2A10000070008000000000000002'),
       'migration:aiops_schema_47',
       'migration:aiops_schema_47',
       SYSTIMESTAMP,
       SYSTIMESTAMP,
       1
  FROM KBOT_OPS_KNOWLEDGE_BASE knowledge_base;

INSERT INTO KBOT_OPS_KNOW_ASSISTANT_BASE (
    DOMAIN_ID, KNOWLEDGE_BASE_ID, CREATED_AT
)
SELECT knowledge_base.DOMAIN_ID,
       knowledge_base.KNOWLEDGE_BASE_ID,
       SYSTIMESTAMP
  FROM KBOT_OPS_KNOWLEDGE_BASE knowledge_base;

PROMPT === 创建诊断 Agent 版本与运维知识库多选绑定 ===

CREATE TABLE KBOT_OPS_AGENT_VERSION_KNOW_BASE (
    AGENT_VERSION_ID RAW(16) NOT NULL,
    KNOWLEDGE_BASE_ID RAW(16) NOT NULL,
    CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT PK_OPS_AGENT_VER_KNOW_BASE
        PRIMARY KEY (AGENT_VERSION_ID, KNOWLEDGE_BASE_ID),
    CONSTRAINT FK_OPS_AGENT_VER_KNOW_VER FOREIGN KEY (AGENT_VERSION_ID)
        REFERENCES KBOT_OPS_AGENT_VERSION (AGENT_VERSION_ID) ON DELETE CASCADE,
    CONSTRAINT FK_OPS_AGENT_VER_KNOW_BASE FOREIGN KEY (KNOWLEDGE_BASE_ID)
        REFERENCES KBOT_OPS_KNOWLEDGE_BASE (KNOWLEDGE_BASE_ID)
);

CREATE INDEX IX_OPS_AGENT_VER_KNOW_BASE
    ON KBOT_OPS_AGENT_VERSION_KNOW_BASE (KNOWLEDGE_BASE_ID);

INSERT INTO KBOT_OPS_AGENT_VERSION_KNOW_BASE (
    AGENT_VERSION_ID, KNOWLEDGE_BASE_ID, CREATED_AT
)
SELECT version.AGENT_VERSION_ID,
       knowledge_base.KNOWLEDGE_BASE_ID,
       SYSTIMESTAMP
  FROM KBOT_OPS_AGENT_VERSION version
  JOIN KBOT_OPS_AGENT agent ON agent.AGENT_ID = version.AGENT_ID
  JOIN KBOT_OPS_KNOWLEDGE_BASE knowledge_base
    ON knowledge_base.DOMAIN_ID = agent.DOMAIN_ID;

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    47 AS SCHEMA_VERSION,
    'aiops-oracle-v47-knowledge-bases' AS CONTRACT_VERSION
FROM DUAL;

PROMPT === 验证 Schema 47 ===

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_table_count PLS_INTEGER;
    l_column_count PLS_INTEGER;
    l_unmapped_asset_count PLS_INTEGER;
    l_unbound_version_count PLS_INTEGER;
    l_incomplete_assistant_count PLS_INTEGER;
BEGIN
    SELECT schema_version, contract_version
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    SELECT COUNT(*) INTO l_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME IN (
        'KBOT_OPS_KNOWLEDGE_BASE',
        'KBOT_OPS_KNOW_ASSISTANT_BIND',
        'KBOT_OPS_KNOW_ASSISTANT_BASE',
        'KBOT_OPS_AGENT_VERSION_KNOW_BASE'
     );

    SELECT COUNT(*) INTO l_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_KNOWLEDGE_ASSET'
       AND COLUMN_NAME = 'KNOWLEDGE_BASE_ID'
       AND DATA_TYPE = 'RAW'
       AND DATA_LENGTH = 16;

    SELECT COUNT(*) INTO l_unmapped_asset_count
      FROM KBOT_OPS_KNOWLEDGE_ASSET
     WHERE KNOWLEDGE_BASE_ID IS NULL;

    SELECT COUNT(*) INTO l_unbound_version_count
      FROM KBOT_OPS_AGENT_VERSION version
     WHERE NOT EXISTS (
        SELECT 1
          FROM KBOT_OPS_AGENT_VERSION_KNOW_BASE binding
         WHERE binding.AGENT_VERSION_ID = version.AGENT_VERSION_ID
     );

    SELECT COUNT(*) INTO l_incomplete_assistant_count
      FROM KBOT_OPS_KNOWLEDGE_BASE knowledge_base
     WHERE NOT EXISTS (
        SELECT 1 FROM KBOT_OPS_KNOW_ASSISTANT_BIND binding
         WHERE binding.DOMAIN_ID = knowledge_base.DOMAIN_ID
     )
        OR NOT EXISTS (
        SELECT 1 FROM KBOT_OPS_KNOW_ASSISTANT_BASE binding_base
         WHERE binding_base.DOMAIN_ID = knowledge_base.DOMAIN_ID
           AND binding_base.KNOWLEDGE_BASE_ID = knowledge_base.KNOWLEDGE_BASE_ID
     );

    IF l_schema_version <> 47
       OR l_contract_version <> 'aiops-oracle-v47-knowledge-bases'
       OR l_table_count <> 4
       OR l_column_count <> 1
       OR l_unmapped_asset_count <> 0
       OR l_unbound_version_count <> 0
       OR l_incomplete_assistant_count <> 0 THEN
        raise_application_error(
            -20503,
            'Schema 47 验证失败：tables=' || l_table_count
            || ', columns=' || l_column_count
            || ', unmapped_assets=' || l_unmapped_asset_count
            || ', unbound_versions=' || l_unbound_version_count
            || ', incomplete_assistants=' || l_incomplete_assistant_count
        );
    END IF;

    dbms_output.put_line('Schema 47 验证通过：多运维知识库与 Agent 绑定已就绪');
END;
/

COMMIT;

PROMPT === AIOps Schema 47 升级完成 ===
