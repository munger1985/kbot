-- AIOps Schema 27 -> 28 原地升级。
-- 影响范围：移除 AIOps 业务表的命名 CHECK 约束，业务规则回归应用层合同。
-- 数据保护：不删除表、不删除或改写任何业务行。
-- 前置条件：当前必须是 Schema 27 / aiops-oracle-v17。
-- 恢复方式：Oracle DDL 会隐式提交；如需回退，应从变更前备份恢复旧约束和版本视图。

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

    IF l_schema_version <> 27
       OR l_contract_version <> 'aiops-oracle-v17' THEN
        RAISE_APPLICATION_ERROR(
            -20062,
            '只允许从 AIOPS Schema 27 / aiops-oracle-v17 升级'
        );
    END IF;
END;
/

BEGIN
    FOR constraint_row IN (
        SELECT TABLE_NAME, CONSTRAINT_NAME
          FROM USER_CONSTRAINTS
         WHERE TABLE_NAME LIKE 'KBOT\_OPS\_%' ESCAPE '\'
           AND CONSTRAINT_TYPE = 'C'
           AND GENERATED = 'USER NAME'
         ORDER BY TABLE_NAME, CONSTRAINT_NAME
    ) LOOP
        EXECUTE IMMEDIATE
            'ALTER TABLE '
            || DBMS_ASSERT.ENQUOTE_NAME(constraint_row.TABLE_NAME, FALSE)
            || ' DROP CONSTRAINT '
            || DBMS_ASSERT.ENQUOTE_NAME(
                constraint_row.CONSTRAINT_NAME,
                FALSE
            );
        DBMS_OUTPUT.PUT_LINE(
            '已移除业务 CHECK：'
            || constraint_row.TABLE_NAME
            || '.'
            || constraint_row.CONSTRAINT_NAME
        );
    END LOOP;
END;
/

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    28 AS SCHEMA_VERSION,
    'aiops-oracle-v18' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_business_check_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*)
      INTO l_business_check_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME LIKE 'KBOT\_OPS\_%' ESCAPE '\'
       AND CONSTRAINT_TYPE = 'C'
       AND GENERATED = 'USER NAME';

    IF l_schema_version <> 28
       OR l_contract_version <> 'aiops-oracle-v18'
       OR l_business_check_count <> 0 THEN
        RAISE_APPLICATION_ERROR(
            -20063,
            'Schema 28 升级校验失败，请核对业务 CHECK 和版本视图'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 28 / aiops-oracle-v18。'
    );
END;
/
