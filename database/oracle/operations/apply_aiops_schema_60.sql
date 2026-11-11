-- Exadata 专用新增表；只创建新对象，不修改/回填既有绑定。
-- 先备份并在发布窗口由 Schema 所有者执行，未由本次开发自动应用。
-- 只适用于完整且匹配的 Schema 59 基线，不能代替旧版本的整体切换。
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
DECLARE
    l_component VARCHAR2(64);
    l_version NUMBER;
BEGIN
    SELECT COMPONENT, SCHEMA_VERSION INTO l_component, l_version
    FROM KBOT_V_OPS_SCHEMA_VERSION;
    IF l_component <> 'AIOPS' OR l_version <> 59 THEN
        RAISE_APPLICATION_ERROR(-20060, 'Exadata 新增表脚本仅适用于匹配的 AIOps Schema 59 基线');
    END IF;
END;
/
@@../aiops_agent/015_ops_exadata.sql

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT 'AIOPS' AS COMPONENT,
       60 AS SCHEMA_VERSION,
       'aiops-oracle-v60-exadata-oem' AS CONTRACT_VERSION
FROM DUAL;
