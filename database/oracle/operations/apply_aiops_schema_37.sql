-- AIOps Schema 36 -> 37 原地升级。
-- 影响范围：删除废弃的 Agent–监控源关系表；保留 Agent、Target、监控源、Target 映射和历史数据。
-- 迁移策略：不把旧监控源关系推断为 Target 授权；不满足新绑定模型的启用对象会被安全停用。
-- 前置条件：当前必须是 Schema 36 / aiops-oracle-v26，或已部分升级到 Schema 37。
-- 执行要求：停止 AIOps API、Worker、Scheduler 和 DB Executor，并完成 Schema 备份。
-- 敏感数据：脚本只输出受影响行数，不输出对象内容、凭据或连接信息。
-- 失败恢复：修复报错后可直接续跑；旧 Agent–监控源关系只能从执行前备份恢复。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

ALTER SESSION SET TIME_ZONE = '+00:00';
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

PROMPT === 正在检查 AIOps Schema 37 升级前置条件 ===

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF NOT (
        (l_schema_version = 36 AND l_contract_version = 'aiops-oracle-v26')
        OR
        (l_schema_version = 37 AND l_contract_version = 'aiops-oracle-v27')
    ) THEN
        RAISE_APPLICATION_ERROR(
            -20091,
            '只允许从 AIOPS Schema 36 升级或续跑 Schema 37'
        );
    END IF;
END;
/

PROMPT === 正在收敛 Target 与 Agent 启用状态 ===

DECLARE
    l_legacy_table_count PLS_INTEGER;
    l_legacy_relation_count PLS_INTEGER := 0;
    l_disabled_target_count PLS_INTEGER := 0;
    l_disabled_agent_count PLS_INTEGER := 0;
BEGIN
    UPDATE KBOT_OPS_TARGET target
       SET STATUS = 'DISABLED',
           ROW_VERSION = target.ROW_VERSION + 1,
           UPDATED_AT = SYSTIMESTAMP,
           UPDATED_BY = 'schema-upgrade-37'
     WHERE target.STATUS = 'ENABLED'
       AND (
           NOT EXISTS (
               SELECT 1
                 FROM KBOT_OPS_TARGET_SOURCE_BINDING binding
                WHERE binding.TARGET_ID = target.TARGET_ID
                  AND binding.STATUS = 'ACTIVE'
           )
           OR EXISTS (
               SELECT 1
                 FROM KBOT_OPS_TARGET_SOURCE_BINDING binding
                 JOIN KBOT_OPS_DIAGNOSTIC_SOURCE source
                   ON source.DIAGNOSTIC_SOURCE_ID = binding.DIAGNOSTIC_SOURCE_ID
                WHERE binding.TARGET_ID = target.TARGET_ID
                  AND binding.STATUS = 'ACTIVE'
                  AND source.STATUS <> 'ENABLED'
           )
       );
    l_disabled_target_count := SQL%ROWCOUNT;

    UPDATE KBOT_OPS_AGENT agent
       SET STATUS = 'DISABLED',
           ROW_VERSION = agent.ROW_VERSION + 1,
           UPDATED_AT = SYSTIMESTAMP,
           UPDATED_BY = 'schema-upgrade-37'
     WHERE agent.STATUS = 'ACTIVE'
       AND (
           NOT EXISTS (
               SELECT 1
                 FROM KBOT_OPS_AGENT_VERSION_TARGET version_target
                WHERE version_target.AGENT_VERSION_ID = agent.CURRENT_VERSION_ID
           )
           OR EXISTS (
               SELECT 1
                 FROM KBOT_OPS_AGENT_VERSION_TARGET version_target
                 JOIN KBOT_OPS_TARGET target
                   ON target.TARGET_ID = version_target.TARGET_ID
                WHERE version_target.AGENT_VERSION_ID = agent.CURRENT_VERSION_ID
                  AND target.STATUS <> 'ENABLED'
           )
       );
    l_disabled_agent_count := SQL%ROWCOUNT;

    SELECT COUNT(*)
      INTO l_legacy_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_AGENT_VERSION_SOURCE';

    IF l_legacy_table_count > 0 THEN
        EXECUTE IMMEDIATE
            'SELECT COUNT(*) FROM KBOT_OPS_AGENT_VERSION_SOURCE'
            INTO l_legacy_relation_count;
        EXECUTE IMMEDIATE
            'DROP TABLE KBOT_OPS_AGENT_VERSION_SOURCE CASCADE CONSTRAINTS PURGE';
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        '因监控映射不满足新模型而停用的 Target：'
        || l_disabled_target_count
    );
    DBMS_OUTPUT.PUT_LINE(
        '因 Target 授权不满足新模型而停用的 Agent：'
        || l_disabled_agent_count
    );
    DBMS_OUTPUT.PUT_LINE(
        '已删除的旧 Agent–监控源关系：'
        || l_legacy_relation_count
    );
END;
/

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    37 AS SCHEMA_VERSION,
    'aiops-oracle-v27' AS CONTRACT_VERSION
FROM DUAL;

COMMIT;

PROMPT === 正在验证 AIOps Schema 37 ===

DECLARE
    l_legacy_table_count PLS_INTEGER;
    l_invalid_target_count PLS_INTEGER;
    l_invalid_agent_count PLS_INTEGER;
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT COUNT(*)
      INTO l_legacy_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_AGENT_VERSION_SOURCE';

    SELECT COUNT(*)
      INTO l_invalid_target_count
      FROM KBOT_OPS_TARGET target
     WHERE target.STATUS = 'ENABLED'
       AND (
           NOT EXISTS (
               SELECT 1
                 FROM KBOT_OPS_TARGET_SOURCE_BINDING binding
                WHERE binding.TARGET_ID = target.TARGET_ID
                  AND binding.STATUS = 'ACTIVE'
           )
           OR EXISTS (
               SELECT 1
                 FROM KBOT_OPS_TARGET_SOURCE_BINDING binding
                 JOIN KBOT_OPS_DIAGNOSTIC_SOURCE source
                   ON source.DIAGNOSTIC_SOURCE_ID = binding.DIAGNOSTIC_SOURCE_ID
                WHERE binding.TARGET_ID = target.TARGET_ID
                  AND binding.STATUS = 'ACTIVE'
                  AND source.STATUS <> 'ENABLED'
           )
       );

    SELECT COUNT(*)
      INTO l_invalid_agent_count
      FROM KBOT_OPS_AGENT agent
     WHERE agent.STATUS = 'ACTIVE'
       AND (
           NOT EXISTS (
               SELECT 1
                 FROM KBOT_OPS_AGENT_VERSION_TARGET version_target
                WHERE version_target.AGENT_VERSION_ID = agent.CURRENT_VERSION_ID
           )
           OR EXISTS (
               SELECT 1
                 FROM KBOT_OPS_AGENT_VERSION_TARGET version_target
                 JOIN KBOT_OPS_TARGET target
                   ON target.TARGET_ID = version_target.TARGET_ID
                WHERE version_target.AGENT_VERSION_ID = agent.CURRENT_VERSION_ID
                  AND target.STATUS <> 'ENABLED'
           )
       );

    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF l_legacy_table_count <> 0 THEN
        RAISE_APPLICATION_ERROR(-20092, '废弃的 Agent–监控源关系表仍然存在');
    END IF;
    IF l_invalid_target_count <> 0 OR l_invalid_agent_count <> 0 THEN
        RAISE_APPLICATION_ERROR(-20093, '仍有启用对象不满足新的 Target 绑定模型');
    END IF;
    IF l_schema_version <> 37
       OR l_contract_version <> 'aiops-oracle-v27' THEN
        RAISE_APPLICATION_ERROR(-20094, 'AIOps Schema 37 版本合同校验失败');
    END IF;

    DBMS_OUTPUT.PUT_LINE('AIOps Schema 37 升级完成。');
END;
/

SELECT COMPONENT, SCHEMA_VERSION, CONTRACT_VERSION
  FROM KBOT_V_OPS_SCHEMA_VERSION;
