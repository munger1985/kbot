-- AIOps Schema 47 -> 48：知识问答 Agent 的回答模型改为 Domain 级用户配置。
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

PROMPT === 检查 Schema 47 升级前置条件 ===

DECLARE
    l_component VARCHAR2(32 CHAR);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_component <> 'AIOPS' OR l_schema_version <> 47
       OR l_contract_version <> 'aiops-oracle-v47-knowledge-bases' THEN
        raise_application_error(
            -20510,
            '本脚本仅支持 AIOPS/47/aiops-oracle-v47-knowledge-bases，当前为 '
            || l_component || '/' || l_schema_version || '/' || l_contract_version
        );
    END IF;
END;
/

PROMPT === 增加知识问答 Agent 回答模型绑定 ===

ALTER TABLE KBOT_OPS_KNOW_ASSISTANT_BIND ADD (
    COMPOSER_MODEL_ID RAW(16)
);

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    48 AS SCHEMA_VERSION,
    'aiops-oracle-v48-knowledge-assistant' AS CONTRACT_VERSION
FROM DUAL;

PROMPT === 验证 Schema 48 ===

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_column_count PLS_INTEGER;
BEGIN
    SELECT schema_version, contract_version
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    SELECT COUNT(*) INTO l_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_KNOW_ASSISTANT_BIND'
       AND COLUMN_NAME = 'COMPOSER_MODEL_ID'
       AND DATA_TYPE = 'RAW'
       AND DATA_LENGTH = 16
       AND NULLABLE = 'Y';

    IF l_schema_version <> 48
       OR l_contract_version <> 'aiops-oracle-v48-knowledge-assistant'
       OR l_column_count <> 1 THEN
        raise_application_error(
            -20511,
            'Schema 48 验证失败：composer_model_column=' || l_column_count
        );
    END IF;

    dbms_output.put_line('Schema 48 验证通过：知识问答 Agent 可由用户绑定回答模型');
END;
/

COMMIT;

PROMPT === AIOps Schema 48 升级完成 ===
