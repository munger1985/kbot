-- AIOps Schema 52 -> 53：同一 Run 的各份报告来源独立保存，保留全部既有行。
-- 停止 AIOps 写入进程并备份，由 Schema 所有者使用 F5 执行；不操作运维目标库。
-- Oracle DDL 会隐式提交；如中断，保持停机并重新执行本脚本，不运行重建脚本。
WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
SET SERVEROUTPUT ON
SET DEFINE OFF
SET SQLBLANKLINES ON
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

DECLARE
    l_version NUMBER;
    l_key VARCHAR2(4000);
    l_name VARCHAR2(128);
    l_before NUMBER;
    l_after NUMBER;
BEGIN
    SELECT schema_version INTO l_version FROM KBOT_V_OPS_SCHEMA_VERSION;
    IF l_version NOT IN (52, 53) THEN
        raise_application_error(-20535, '仅支持 AIOps Schema 52 或已升级的 53');
    END IF;
    SELECT COUNT(*) INTO l_before FROM KBOT_OPS_REPORT_SOURCE;
    SELECT MAX(constraint_name) INTO l_name FROM user_constraints
    WHERE table_name = 'KBOT_OPS_REPORT_SOURCE' AND constraint_type = 'P';
    IF l_name IS NOT NULL THEN
        SELECT LISTAGG(column_name, ',') WITHIN GROUP (ORDER BY position)
        INTO l_key FROM user_cons_columns WHERE constraint_name = l_name;
        IF l_key = 'REPORT_ID,OPS_RUN_ID' THEN
            EXECUTE IMMEDIATE 'ALTER TABLE KBOT_OPS_REPORT_SOURCE DROP CONSTRAINT '
                || dbms_assert.enquote_name(l_name, FALSE);
        ELSIF l_key <> 'REPORT_ID,OPS_RUN_ID,SOURCE_ARTIFACT_ID' THEN
            raise_application_error(-20536, '报告来源主键不符合已知合同，拒绝改写');
        END IF;
    END IF;
    IF l_name IS NULL OR l_key = 'REPORT_ID,OPS_RUN_ID' THEN
        EXECUTE IMMEDIATE 'ALTER TABLE KBOT_OPS_REPORT_SOURCE ADD CONSTRAINT '
            || 'PK_OPS_REPORT_SOURCE PRIMARY KEY (REPORT_ID, OPS_RUN_ID, SOURCE_ARTIFACT_ID)';
    END IF;
    SELECT COUNT(*) INTO l_after FROM KBOT_OPS_REPORT_SOURCE;
    IF l_before <> l_after THEN
        raise_application_error(-20537, '报告来源行数变化，停止升级');
    END IF;
END;
/

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT 'AIOPS' AS COMPONENT, 53 AS SCHEMA_VERSION,
       'aiops-oracle-v53-report-artifact-sources' AS CONTRACT_VERSION FROM DUAL;

COMMIT;
PROMPT === AIOps Schema 53 升级完成；既有报告与来源保留，未修改目标库 ===
