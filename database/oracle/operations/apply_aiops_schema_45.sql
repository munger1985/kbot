-- AIOps Schema 44 -> 45：允许多个数据库 Target 复用同一监控 Locator。
-- 用于 RAC 测试或监控平台仅暴露集群级 Target 的场景。
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

PROMPT === 检查 Schema 44 升级前置条件 ===

DECLARE
    l_component VARCHAR2(32 CHAR);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_constraint_count PLS_INTEGER;
BEGIN
    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_component <> 'AIOPS' OR l_schema_version <> 44
       OR l_contract_version <> 'aiops-oracle-v44-resource-graph' THEN
        raise_application_error(
            -20500,
            '本脚本仅支持 AIOPS/44/aiops-oracle-v44-resource-graph，当前为 '
            || l_component || '/' || l_schema_version || '/' || l_contract_version
        );
    END IF;

    SELECT COUNT(*)
      INTO l_constraint_count
      FROM user_constraints
     WHERE table_name = 'KBOT_OPS_TARGET_SOURCE_BINDING'
       AND constraint_name = 'UK_OPS_TSRC_LOCATOR'
       AND constraint_type = 'U';

    IF l_constraint_count <> 1 THEN
        raise_application_error(
            -20501,
            '未找到预期唯一约束 UK_OPS_TSRC_LOCATOR，拒绝继续升级'
        );
    END IF;
END;
/

PROMPT === 允许 RAC 节点复用监控 Locator ===

ALTER TABLE KBOT_OPS_TARGET_SOURCE_BINDING
    DROP CONSTRAINT UK_OPS_TSRC_LOCATOR;

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    45 AS SCHEMA_VERSION,
    'aiops-oracle-v45-shared-monitor-locator' AS CONTRACT_VERSION
FROM DUAL;

PROMPT === 验证 Schema 45 ===

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_constraint_count PLS_INTEGER;
BEGIN
    SELECT schema_version, contract_version
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    SELECT COUNT(*)
      INTO l_constraint_count
      FROM user_constraints
     WHERE table_name = 'KBOT_OPS_TARGET_SOURCE_BINDING'
       AND constraint_name = 'UK_OPS_TSRC_LOCATOR';

    IF l_schema_version <> 45
       OR l_contract_version <> 'aiops-oracle-v45-shared-monitor-locator'
       OR l_constraint_count <> 0 THEN
        raise_application_error(-20502, 'Schema 45 验证失败');
    END IF;

    dbms_output.put_line(
        'Schema 45 验证通过：多个 Target 可复用同一 Diagnostic Source Locator'
    );
END;
/

COMMIT;

PROMPT === AIOps Schema 45 升级完成 ===
