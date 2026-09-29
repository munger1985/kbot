-- AIOps Schema 28 -> 29 原地升级。
-- 影响范围：为运维 Target 增加 1 到 5 级的重要程度，供自动告警诊断过滤。
-- 数据保护：不删除表、不删除业务行；既有 Target 统一回填为 3（普通）。
-- 前置条件：当前必须是 Schema 28 / aiops-oracle-v18，或已部分/全部升级到 Schema 29。
-- 恢复方式：Oracle DDL 会隐式提交；如需回退，应从变更前备份恢复列和版本视图。
-- 并发要求：执行前停止 AIOps API、Worker、Scheduler 和 DB Executor；脚本允许中断后续跑。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

ALTER SESSION SET TIME_ZONE = '+00:00';
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF NOT (
        (l_schema_version = 28
         AND l_contract_version = 'aiops-oracle-v18')
        OR
        (l_schema_version = 29
         AND l_contract_version = 'aiops-oracle-v19')
    ) THEN
        RAISE_APPLICATION_ERROR(
            -20065,
            '只允许从 AIOPS Schema 28 / aiops-oracle-v18 升级，'
            || '或续跑 Schema 29 / aiops-oracle-v19'
        );
    END IF;
END;
/

DECLARE
    l_column_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_TARGET'
       AND COLUMN_NAME = 'IMPORTANCE_LEVEL';

    IF l_column_count = 0 THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_OPS_TARGET ADD ('
            || 'IMPORTANCE_LEVEL NUMBER(1) DEFAULT 3 NOT NULL)';
        DBMS_OUTPUT.PUT_LINE(
            '已新增 KBOT_OPS_TARGET.IMPORTANCE_LEVEL，既有 Target 回填为 3。'
        );
    END IF;
END;
/

CREATE OR REPLACE VIEW KBOT_V_OPS_TARGET AS
SELECT
    LOWER(
        SUBSTR(RAWTOHEX(t.TARGET_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(t.TARGET_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(t.TARGET_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(t.TARGET_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(t.TARGET_ID), 21, 12)
    ) AS TARGET_ID,
    t.DOMAIN_ID,
    t.DISPLAY_NAME,
    t.DB_TYPE,
    t.VERSION_CODE,
    t.ENVIRONMENT,
    t.DB_ROLE,
    t.ORACLE_CONTAINER_SCOPE,
    t.ORACLE_PDB_NAME,
    t.OBSERVED_ORACLE_CONTAINER_SCOPE,
    t.OBSERVED_ORACLE_CONTAINER_NAME,
    t.OBSERVED_ORACLE_CONTAINER_NUMBER,
    t.OBSERVED_ORACLE_DATABASE_NAME,
    t.IMPORTANCE_LEVEL,
    t.SECURITY_LEVEL,
    t.STATUS,
    t.CONNECTIVITY_STATUS,
    t.OBSERVED_STATUS,
    t.LAST_OBSERVED_AT,
    t.LAST_CONNECTIVITY_CHECK_AT,
    t.LAST_ERROR_CODE,
    t.ROW_VERSION,
    t.CREATED_AT,
    t.UPDATED_AT,
    COUNT(tm.TARGET_SOURCE_BINDING_ID) AS SOURCE_BINDING_COUNT,
    SUM(CASE WHEN tm.HEALTH_STATUS = 'HEALTHY' THEN 1 ELSE 0 END)
        AS HEALTHY_SOURCE_BINDING_COUNT,
    SUM(CASE WHEN tm.HEALTH_STATUS IN ('DEGRADED', 'UNREACHABLE')
             THEN 1 ELSE 0 END) AS UNHEALTHY_SOURCE_BINDING_COUNT
FROM KBOT_OPS_TARGET t
LEFT JOIN KBOT_OPS_TARGET_SOURCE_BINDING tm
    ON tm.TARGET_ID = t.TARGET_ID
   AND tm.STATUS = 'ACTIVE'
GROUP BY
    t.TARGET_ID, t.DOMAIN_ID, t.DISPLAY_NAME,
    t.DB_TYPE, t.VERSION_CODE, t.ENVIRONMENT, t.DB_ROLE,
    t.ORACLE_CONTAINER_SCOPE, t.ORACLE_PDB_NAME,
    t.OBSERVED_ORACLE_CONTAINER_SCOPE, t.OBSERVED_ORACLE_CONTAINER_NAME,
    t.OBSERVED_ORACLE_CONTAINER_NUMBER, t.OBSERVED_ORACLE_DATABASE_NAME,
    t.IMPORTANCE_LEVEL, t.SECURITY_LEVEL, t.STATUS,
    t.CONNECTIVITY_STATUS, t.OBSERVED_STATUS, t.LAST_OBSERVED_AT,
    t.LAST_CONNECTIVITY_CHECK_AT, t.LAST_ERROR_CODE, t.ROW_VERSION,
    t.CREATED_AT, t.UPDATED_AT;

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    29 AS SCHEMA_VERSION,
    'aiops-oracle-v19' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_column_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*)
      INTO l_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_TARGET'
       AND COLUMN_NAME = 'IMPORTANCE_LEVEL'
       AND NULLABLE = 'N'
       AND DATA_TYPE = 'NUMBER';

    IF l_schema_version <> 29
       OR l_contract_version <> 'aiops-oracle-v19'
       OR l_column_count <> 1 THEN
        RAISE_APPLICATION_ERROR(
            -20066,
            'Schema 29 升级校验失败，请核对 Target 重要程度列和版本视图'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 29 / aiops-oracle-v19。'
    );
END;
/
