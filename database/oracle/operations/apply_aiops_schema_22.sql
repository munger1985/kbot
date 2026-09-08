-- AIOps Schema 21 -> 22 原地升级。
-- 影响范围：扩展工具调用分类约束，并更新 Schema 版本视图。
-- 数据保护：不删除表、不删除业务行、不修改既有诊断、附件、命令或审计哈希。
-- 前置条件：停止 AIOps API、Worker、Scheduler 和 DB Executor，避免 DDL 与运行时写入并发。
-- 恢复方式：Oracle DDL 会隐式提交；执行前必须完成 Schema 备份，失败时按备份恢复。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_constraint_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF l_schema_version <> 21
       OR l_contract_version <> 'aiops-oracle-v11' THEN
        RAISE_APPLICATION_ERROR(
            -20040,
            '只允许从 AIOPS Schema 21 / aiops-oracle-v11 升级'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_OPS_TOOL_INVOCATION'
       AND CONSTRAINT_NAME = 'CK_OPS_TOOL_INV_CLASS'
       AND CONSTRAINT_TYPE = 'C'
       AND STATUS = 'ENABLED'
       AND VALIDATED = 'VALIDATED';

    IF l_constraint_count <> 1 THEN
        RAISE_APPLICATION_ERROR(
            -20041,
            'KBOT_OPS_TOOL_INVOCATION.CK_OPS_TOOL_INV_CLASS 不符合 Schema 21 前置条件'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE('Schema 21 前置检查通过。');
END;
/

ALTER TABLE KBOT_OPS_TOOL_INVOCATION DROP CONSTRAINT CK_OPS_TOOL_INV_CLASS;

ALTER TABLE KBOT_OPS_TOOL_INVOCATION
    ADD CONSTRAINT CK_OPS_TOOL_INV_CLASS CHECK (TOOL_CLASS IN (
        'PROMETHEUS', 'LOKI', 'ORACLE_SQL', 'ORACLE_SQL_DYNAMIC',
        'USER_EVIDENCE', 'HOST', 'MEDIA', 'REASONING'
    ));

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    22 AS SCHEMA_VERSION,
    'aiops-oracle-v12' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_tool_class_constraint_count PLS_INTEGER;
    l_invalid_object_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*)
      INTO l_tool_class_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_OPS_TOOL_INVOCATION'
       AND CONSTRAINT_NAME = 'CK_OPS_TOOL_INV_CLASS'
       AND CONSTRAINT_TYPE = 'C'
       AND STATUS = 'ENABLED'
       AND VALIDATED = 'VALIDATED'
       AND SEARCH_CONDITION_VC LIKE '%''PROMETHEUS''%'
       AND SEARCH_CONDITION_VC LIKE '%''LOKI''%'
       AND SEARCH_CONDITION_VC LIKE '%''ORACLE_SQL''%'
       AND SEARCH_CONDITION_VC LIKE '%''ORACLE_SQL_DYNAMIC''%'
       AND SEARCH_CONDITION_VC LIKE '%''USER_EVIDENCE''%';

    SELECT COUNT(*)
      INTO l_invalid_object_count
      FROM USER_OBJECTS
     WHERE OBJECT_NAME IN (
         'KBOT_OPS_TOOL_INVOCATION',
         'KBOT_V_OPS_SCHEMA_VERSION'
     )
       AND STATUS <> 'VALID';

    IF l_schema_version <> 22
       OR l_contract_version <> 'aiops-oracle-v12'
       OR l_tool_class_constraint_count <> 1
       OR l_invalid_object_count <> 0 THEN
        RAISE_APPLICATION_ERROR(
            -20042,
            'Schema 22 升级校验失败，请从备份恢复并核对输出'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 22 / aiops-oracle-v12。'
    );
END;
/
