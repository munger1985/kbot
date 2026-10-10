-- AIOps Schema 35 -> 36 原地升级。
-- 影响范围：从 Target 及其只读视图移除已废弃的安全级别；保留全部业务行。
-- 前置条件：当前必须是 Schema 35 / aiops-oracle-v25，或已部分升级到 Schema 36。
-- 执行要求：停止 AIOps API、Worker、Scheduler 和 DB Executor，并完成 Schema 备份。
-- 敏感数据：脚本不读取或输出凭据、连接串及其他受保护字段。
-- 失败恢复：修复报错后可直接续跑；需要回退时使用执行前的 Schema 备份。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

ALTER SESSION SET TIME_ZONE = '+00:00';
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

PROMPT === 正在检查 AIOps Schema 36 升级前置条件 ===

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF NOT (
        (l_schema_version = 35 AND l_contract_version = 'aiops-oracle-v25')
        OR
        (l_schema_version = 36 AND l_contract_version = 'aiops-oracle-v26')
    ) THEN
        RAISE_APPLICATION_ERROR(
            -20091,
            '只允许从 AIOPS Schema 35 升级或续跑 Schema 36'
        );
    END IF;
END;
/

PROMPT === 正在移除 Target 安全级别投影 ===

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
    t.IMPORTANCE_LEVEL, t.STATUS,
    t.CONNECTIVITY_STATUS, t.OBSERVED_STATUS,
    t.LAST_OBSERVED_AT,
    t.LAST_CONNECTIVITY_CHECK_AT, t.LAST_ERROR_CODE, t.ROW_VERSION,
    t.CREATED_AT, t.UPDATED_AT;

PROMPT === 正在移除 Target 安全级别列 ===

DECLARE
    l_column_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_TARGET'
       AND COLUMN_NAME = 'SECURITY_LEVEL';

    IF l_column_count > 0 THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_OPS_TARGET DROP COLUMN SECURITY_LEVEL';
        DBMS_OUTPUT.PUT_LINE('已删除 KBOT_OPS_TARGET.SECURITY_LEVEL。');
    ELSE
        DBMS_OUTPUT.PUT_LINE('KBOT_OPS_TARGET.SECURITY_LEVEL 已不存在，继续校验。');
    END IF;
END;
/

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    36 AS SCHEMA_VERSION,
    'aiops-oracle-v26' AS CONTRACT_VERSION
FROM DUAL;

COMMIT;

PROMPT === 正在验证 AIOps Schema 36 ===

DECLARE
    l_table_column_count PLS_INTEGER;
    l_view_column_count PLS_INTEGER;
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT COUNT(*)
      INTO l_table_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_TARGET'
       AND COLUMN_NAME = 'SECURITY_LEVEL';

    SELECT COUNT(*)
      INTO l_view_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_V_OPS_TARGET'
       AND COLUMN_NAME = 'SECURITY_LEVEL';

    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF l_table_column_count <> 0 OR l_view_column_count <> 0 THEN
        RAISE_APPLICATION_ERROR(-20092, 'Target 安全级别仍存在于表或视图中');
    END IF;
    IF l_schema_version <> 36
       OR l_contract_version <> 'aiops-oracle-v26' THEN
        RAISE_APPLICATION_ERROR(-20093, 'AIOps Schema 36 版本合同校验失败');
    END IF;

    DBMS_OUTPUT.PUT_LINE('AIOps Schema 36 升级完成，全部业务行已保留。');
END;
/

SELECT COMPONENT, SCHEMA_VERSION, CONTRACT_VERSION
  FROM KBOT_V_OPS_SCHEMA_VERSION;
