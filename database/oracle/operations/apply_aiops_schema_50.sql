-- AIOps Schema 49 -> 50：退役旧固定知识问答 Agent 绑定。
-- 使用 Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- Oracle DDL 会自动提交；执行前必须备份并停止全部 AIOps 写入服务，
-- 且必须先部署不再读取旧表的 Schema 50 应用版本。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
SET SERVEROUTPUT ON
SET VERIFY OFF
SET SQLBLANKLINES ON
SET DEFINE OFF

ALTER SESSION SET TIME_ZONE = '+00:00';
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

PROMPT === 检查 Schema 49 升级前置条件 ===

DECLARE
    l_component VARCHAR2(32 CHAR);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_unified_binding_count NUMBER;
BEGIN
    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_component <> 'AIOPS' OR l_schema_version <> 49
       OR l_contract_version <> 'aiops-oracle-v49-unified-assistant' THEN
        raise_application_error(-20530, '本脚本仅支持 AIOPS Schema 49');
    END IF;

    SELECT COUNT(*)
      INTO l_unified_binding_count
      FROM KBOT_OPS_AGENT_VERSION_KNOW_BASE;
    dbms_output.put_line(
        '统一 Agent 版本知识库绑定保留数量=' || l_unified_binding_count
    );
END;
/

PROMPT === 删除旧固定知识问答 Agent 绑定表 ===

DROP TABLE KBOT_OPS_KNOW_ASSISTANT_BASE CASCADE CONSTRAINTS PURGE;
DROP TABLE KBOT_OPS_KNOW_ASSISTANT_BIND CASCADE CONSTRAINTS PURGE;

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    50 AS SCHEMA_VERSION,
    'aiops-oracle-v50-unified-agent-only' AS CONTRACT_VERSION
FROM DUAL;

PROMPT === 验证 Schema 50 ===

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_retired_table_count NUMBER;
    l_unified_table_count NUMBER;
BEGIN
    SELECT schema_version, contract_version
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    SELECT COUNT(*) INTO l_retired_table_count
      FROM user_tables
     WHERE table_name IN (
        'KBOT_OPS_KNOW_ASSISTANT_BIND',
        'KBOT_OPS_KNOW_ASSISTANT_BASE'
     );

    SELECT COUNT(*) INTO l_unified_table_count
      FROM user_tables
     WHERE table_name = 'KBOT_OPS_AGENT_VERSION_KNOW_BASE';

    IF l_schema_version <> 50
       OR l_contract_version <> 'aiops-oracle-v50-unified-agent-only'
       OR l_retired_table_count <> 0
       OR l_unified_table_count <> 1 THEN
        raise_application_error(-20531, 'Schema 50 验证失败');
    END IF;
END;
/

COMMIT;

PROMPT === AIOps Schema 50 升级完成 ===
