-- AIOps Schema 26 -> 27 原地升级。
-- 影响范围：扩展回答块类型约束，允许持久化 IMPLEMENTATION_RUNBOOK。
-- 数据保护：不删除表、不删除或改写任何业务行。
-- 前置条件：当前必须是 Schema 26 / aiops-oracle-v16。
-- 恢复方式：Oracle DDL 会隐式提交；回退时恢复旧约束和版本视图。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

ALTER SESSION SET TIME_ZONE = '+00:00';

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF l_schema_version <> 26
       OR l_contract_version <> 'aiops-oracle-v16' THEN
        RAISE_APPLICATION_ERROR(
            -20060,
            '只允许从 AIOPS Schema 26 / aiops-oracle-v16 升级'
        );
    END IF;
END;
/

ALTER TABLE KBOT_OPS_ANSWER_BLOCK ADD CONSTRAINT CK_OPS_ANSWER_BLOCK_TYPE_V17
    CHECK (BLOCK_TYPE IN (
        'MARKDOWN', 'TABLE', 'CHART', 'EVIDENCE_REFERENCES',
        'CLARIFICATION', 'EVIDENCE_REQUEST', 'PROPOSAL_SUMMARY',
        'VERIFICATION_COMPARISON', 'FINDING_CARDS', 'ANALYSIS_MARKDOWN',
        'SOLUTION_MARKDOWN', 'FACT_CONFIRMATION', 'IMPLEMENTATION_RUNBOOK',
        'HTML_REPORT_LINKS'
    )) ENABLE VALIDATE;

ALTER TABLE KBOT_OPS_ANSWER_BLOCK DROP CONSTRAINT CK_OPS_ANSWER_BLOCK_TYPE;

ALTER TABLE KBOT_OPS_ANSWER_BLOCK
    RENAME CONSTRAINT CK_OPS_ANSWER_BLOCK_TYPE_V17 TO CK_OPS_ANSWER_BLOCK_TYPE;

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    27 AS SCHEMA_VERSION,
    'aiops-oracle-v17' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_constraint_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*)
      INTO l_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_OPS_ANSWER_BLOCK'
       AND CONSTRAINT_NAME = 'CK_OPS_ANSWER_BLOCK_TYPE'
       AND CONSTRAINT_TYPE = 'C'
       AND STATUS = 'ENABLED'
       AND VALIDATED = 'VALIDATED'
       AND SEARCH_CONDITION_VC LIKE '%''IMPLEMENTATION_RUNBOOK''%';

    IF l_schema_version <> 27
       OR l_contract_version <> 'aiops-oracle-v17'
       OR l_constraint_count <> 1 THEN
        RAISE_APPLICATION_ERROR(
            -20061,
            'Schema 27 升级校验失败，请核对约束和版本视图'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 27 / aiops-oracle-v17。'
    );
END;
/
