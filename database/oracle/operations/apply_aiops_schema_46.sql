-- AIOps Schema 45 -> 46：增加运维知识受控标签及审批前识别结果。
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

PROMPT === 检查 Schema 45 升级前置条件 ===

DECLARE
    l_component VARCHAR2(32 CHAR);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_component <> 'AIOPS' OR l_schema_version <> 45
       OR l_contract_version <> 'aiops-oracle-v45-shared-monitor-locator' THEN
        raise_application_error(
            -20500,
            '本脚本仅支持 AIOPS/45/aiops-oracle-v45-shared-monitor-locator，当前为 '
            || l_component || '/' || l_schema_version || '/' || l_contract_version
        );
    END IF;
END;
/

PROMPT === 增加受控知识标签与审批分析字段 ===

ALTER TABLE KBOT_OPS_KNOWLEDGE_VERSION ADD (
    TAG_ANALYSIS_JSON JSON
);

CREATE TABLE KBOT_OPS_KNOWLEDGE_TAG (
    TAG_ID RAW(16) PRIMARY KEY,
    DOMAIN_ID NUMBER(38) NOT NULL,
    CATEGORY VARCHAR2(32 CHAR) NOT NULL,
    DISPLAY_NAME VARCHAR2(128 CHAR) NOT NULL,
    NORMALIZED_NAME VARCHAR2(128 CHAR) NOT NULL,
    DESCRIPTION VARCHAR2(1000 CHAR),
    STATUS VARCHAR2(16 CHAR) NOT NULL,
    SORT_ORDER NUMBER(19) DEFAULT 0 NOT NULL,
    CREATED_BY VARCHAR2(256 CHAR) NOT NULL,
    UPDATED_BY VARCHAR2(256 CHAR) NOT NULL,
    CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    UPDATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    ROW_VERSION NUMBER(19) DEFAULT 1 NOT NULL,
    CONSTRAINT FK_OPS_KNOW_TAG_DOMAIN FOREIGN KEY (DOMAIN_ID)
        REFERENCES KBOT_PLATFORM_DOMAIN (DOMAIN_ID)
);

CREATE INDEX IX_OPS_KNOW_TAG_CATALOG ON KBOT_OPS_KNOWLEDGE_TAG
    (DOMAIN_ID, CATEGORY, STATUS, SORT_ORDER);

ALTER TABLE KBOT_OPS_KNOWLEDGE_SCOPE ADD (
    TAG_ID RAW(16),
    CONSTRAINT FK_OPS_KNOW_SCOPE_TAG FOREIGN KEY (TAG_ID)
        REFERENCES KBOT_OPS_KNOWLEDGE_TAG (TAG_ID)
);

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    46 AS SCHEMA_VERSION,
    'aiops-oracle-v46-knowledge-tags' AS CONTRACT_VERSION
FROM DUAL;

PROMPT === 验证 Schema 46 ===

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_table_count PLS_INTEGER;
    l_column_count PLS_INTEGER;
BEGIN
    SELECT schema_version, contract_version
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    SELECT COUNT(*) INTO l_table_count
      FROM user_tables
     WHERE table_name = 'KBOT_OPS_KNOWLEDGE_TAG';

    SELECT COUNT(*) INTO l_column_count
      FROM user_tab_columns
     WHERE (table_name = 'KBOT_OPS_KNOWLEDGE_VERSION' AND column_name = 'TAG_ANALYSIS_JSON')
        OR (table_name = 'KBOT_OPS_KNOWLEDGE_SCOPE' AND column_name = 'TAG_ID');

    IF l_schema_version <> 46
       OR l_contract_version <> 'aiops-oracle-v46-knowledge-tags'
       OR l_table_count <> 1
       OR l_column_count <> 2 THEN
        raise_application_error(-20501, 'Schema 46 验证失败');
    END IF;

    dbms_output.put_line('Schema 46 验证通过：运维知识受控标签已就绪');
END;
/

COMMIT;

PROMPT === AIOps Schema 46 升级完成 ===
