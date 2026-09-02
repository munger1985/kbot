-- AIOps Schema 18 -> 19 增量升级。
-- 目标：报告摘要改为 CLOB，并修复历史终态 Run 遗留的非终态 Agent Turn。
-- 请使用 AIOps Schema Owner 执行；脚本不会删除业务报告内容。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

DECLARE
    l_summary_type USER_TAB_COLUMNS.DATA_TYPE%TYPE;
    l_summary_count PLS_INTEGER;
    l_clob_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*), MAX(DATA_TYPE)
      INTO l_summary_count, l_summary_type
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_REPORT'
       AND COLUMN_NAME = 'SUMMARY';

    SELECT COUNT(*)
      INTO l_clob_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_REPORT'
       AND COLUMN_NAME = 'SUMMARY_CLOB';

    IF l_summary_count = 1 AND l_summary_type = 'CLOB' THEN
        DBMS_OUTPUT.PUT_LINE('KBOT_OPS_REPORT.SUMMARY 已是 CLOB。');
    ELSIF l_summary_count = 1 THEN
        IF l_clob_count = 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE KBOT_OPS_REPORT ADD (SUMMARY_CLOB CLOB)';
        END IF;

        EXECUTE IMMEDIATE
            'UPDATE KBOT_OPS_REPORT SET SUMMARY_CLOB = TO_CLOB(SUMMARY)';
        COMMIT;

        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_OPS_REPORT DROP COLUMN SUMMARY';
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_OPS_REPORT RENAME COLUMN SUMMARY_CLOB TO SUMMARY';
        DBMS_OUTPUT.PUT_LINE('KBOT_OPS_REPORT.SUMMARY 已转换为 CLOB。');
    ELSIF l_summary_count = 0 AND l_clob_count = 1 THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_OPS_REPORT RENAME COLUMN SUMMARY_CLOB TO SUMMARY';
        DBMS_OUTPUT.PUT_LINE('已恢复中断的 SUMMARY CLOB 列重命名。');
    ELSE
        RAISE_APPLICATION_ERROR(
            -20019,
            'KBOT_OPS_REPORT.SUMMARY 列状态异常，停止升级'
        );
    END IF;
END;
/

CREATE OR REPLACE VIEW KBOT_V_OPS_RUN AS
SELECT
    LOWER(
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 21, 12)
    ) AS OPS_RUN_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.TARGET_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 21, 12)
    ) AS TARGET_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.SOURCE_PROPOSAL_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.SOURCE_PROPOSAL_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.SOURCE_PROPOSAL_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.SOURCE_PROPOSAL_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.SOURCE_PROPOSAL_ID), 21, 12)
    ) AS SOURCE_PROPOSAL_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.SOURCE_RESULT_ARTIFACT_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.SOURCE_RESULT_ARTIFACT_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.SOURCE_RESULT_ARTIFACT_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.SOURCE_RESULT_ARTIFACT_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.SOURCE_RESULT_ARTIFACT_ID), 21, 12)
    ) AS SOURCE_RESULT_ARTIFACT_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.PARENT_AGENT_RUN_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.PARENT_AGENT_RUN_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.PARENT_AGENT_RUN_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.PARENT_AGENT_RUN_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.PARENT_AGENT_RUN_ID), 21, 12)
    ) AS PARENT_AGENT_RUN_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.PARENT_DELEGATION_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.PARENT_DELEGATION_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.PARENT_DELEGATION_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.PARENT_DELEGATION_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.PARENT_DELEGATION_ID), 21, 12)
    ) AS PARENT_DELEGATION_ID,
    t.DOMAIN_ID,
    t.DISPLAY_NAME AS TARGET_NAME,
    r.TRIGGER_TYPE,
    r.WORKFLOW_KIND,
    r.ACTOR_ID,
    r.STATUS,
    r.ERROR_CODE,
    r.TRACE_ID,
    r.CREATED_AT,
    r.STARTED_AT,
    r.COMPLETED_AT,
    report_info.REPORT_COUNT,
    report_info.LATEST_REPORT_SUMMARY
FROM KBOT_OPS_RUN r
JOIN KBOT_OPS_TARGET t ON t.TARGET_ID = r.TARGET_ID
LEFT JOIN (
    SELECT
        OPS_RUN_ID,
        REPORT_COUNT,
        SUMMARY AS LATEST_REPORT_SUMMARY
    FROM (
        SELECT
            OPS_RUN_ID,
            SUMMARY,
            COUNT(*) OVER (PARTITION BY OPS_RUN_ID) AS REPORT_COUNT,
            ROW_NUMBER() OVER (
                PARTITION BY OPS_RUN_ID
                ORDER BY UPDATED_AT DESC, REPORT_ID DESC
            ) AS REPORT_ROW_NO
        FROM KBOT_OPS_REPORT
        WHERE IS_CURRENT = 1
    )
    WHERE REPORT_ROW_NO = 1
) report_info ON report_info.OPS_RUN_ID = r.OPS_RUN_ID;

CREATE OR REPLACE VIEW KBOT_V_OPS_REPORT AS
SELECT
    t.DOMAIN_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.REPORT_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.REPORT_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.REPORT_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.REPORT_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.REPORT_ID), 21, 12)
    ) AS REPORT_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 21, 12)
    ) AS OPS_RUN_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.TARGET_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 21, 12)
    ) AS TARGET_ID,
    t.DISPLAY_NAME AS TARGET_NAME,
    r.REPORT_KEY,
    r.REPORT_VERSION,
    r.REPORT_TYPE,
    r.TITLE,
    r.STATUS,
    r.PERIOD_START,
    r.PERIOD_END,
    r.BASELINE_START,
    r.BASELINE_END,
    r.AFTER_START,
    r.AFTER_END,
    r.RESULT,
    r.SUMMARY,
    r.SECURITY_LEVEL,
    r.CREATED_AT,
    r.UPDATED_AT
FROM KBOT_OPS_REPORT r
JOIN KBOT_OPS_TARGET t ON t.TARGET_ID = r.TARGET_ID
WHERE r.IS_CURRENT = 1;

-- Schema 18 期间若 Run 已终止但 Agent Turn 未同步终态，会持续阻塞巡检 Fire。
-- 先写入用户可见审计事件，再更新 Turn 权威投影。
INSERT INTO KBOT_OPS_TURN_EVENT (
    TURN_ID,
    SEQUENCE_NO,
    EVENT_TYPE,
    EVENT_KEY,
    VISIBILITY,
    PAYLOAD_JSON,
    CREATED_AT
)
SELECT
    ct.TURN_ID,
    ct.EVENT_CURSOR + 1,
    'turn.status',
    'schema19-recovery:' || RAWTOHEX(r.OPS_RUN_ID),
    'USER',
    JSON_OBJECT(
        'status' VALUE CASE
            WHEN r.STATUS = 'CANCELLED' THEN 'CANCELLED'
            WHEN r.STATUS = 'COMPLETED' THEN 'PARTIAL'
            ELSE 'FAILED'
        END,
        'error_domain' VALUE 'EXECUTION',
        'error_code' VALUE COALESCE(r.ERROR_CODE, 'OPS_RUN_TERMINAL_RECOVERED'),
        'public_summary' VALUE
            '系统已根据关联 Run 的终态恢复本轮诊断状态'
    ),
    CURRENT_TIMESTAMP
FROM KBOT_OPS_CONVERSATION_TURN ct
JOIN KBOT_OPS_TURN_RUN tr
  ON tr.TURN_ID = ct.TURN_ID
 AND tr.PURPOSE = 'PRIMARY'
JOIN KBOT_OPS_RUN r
  ON r.OPS_RUN_ID = tr.OPS_RUN_ID
WHERE r.WORKFLOW_KIND IN ('CHAT_TURN', 'ALERT_DIAGNOSIS', 'INSPECTION')
  AND r.STATUS IN ('COMPLETED', 'FAILED', 'CANCELLED', 'EXPIRED')
  AND ct.STATUS NOT IN ('COMPLETED', 'PARTIAL', 'FAILED', 'CANCELLED');

UPDATE KBOT_OPS_CONVERSATION_TURN ct
   SET (
       STATUS,
       ERROR_DOMAIN,
       ERROR_CODE,
       ERROR_MESSAGE,
       COMPLETED_AT,
       EVENT_CURSOR,
       ROW_VERSION,
       UPDATED_AT
   ) = (
       SELECT
           CASE
               WHEN r.STATUS = 'CANCELLED' THEN 'CANCELLED'
               WHEN r.STATUS = 'COMPLETED' THEN 'PARTIAL'
               ELSE 'FAILED'
           END,
           'EXECUTION',
           COALESCE(r.ERROR_CODE, 'OPS_RUN_TERMINAL_RECOVERED'),
           '系统已根据关联 Run 的终态恢复本轮诊断状态',
           COALESCE(r.COMPLETED_AT, CURRENT_TIMESTAMP),
           ct.EVENT_CURSOR + 1,
           ct.ROW_VERSION + 1,
           CURRENT_TIMESTAMP
       FROM KBOT_OPS_TURN_RUN tr
       JOIN KBOT_OPS_RUN r ON r.OPS_RUN_ID = tr.OPS_RUN_ID
       WHERE tr.TURN_ID = ct.TURN_ID
         AND tr.PURPOSE = 'PRIMARY'
         AND r.WORKFLOW_KIND IN (
             'CHAT_TURN', 'ALERT_DIAGNOSIS', 'INSPECTION'
         )
         AND r.STATUS IN ('COMPLETED', 'FAILED', 'CANCELLED', 'EXPIRED')
   )
 WHERE ct.STATUS NOT IN ('COMPLETED', 'PARTIAL', 'FAILED', 'CANCELLED')
   AND EXISTS (
       SELECT 1
       FROM KBOT_OPS_TURN_RUN tr
       JOIN KBOT_OPS_RUN r ON r.OPS_RUN_ID = tr.OPS_RUN_ID
       WHERE tr.TURN_ID = ct.TURN_ID
         AND tr.PURPOSE = 'PRIMARY'
         AND r.WORKFLOW_KIND IN (
             'CHAT_TURN', 'ALERT_DIAGNOSIS', 'INSPECTION'
         )
         AND r.STATUS IN ('COMPLETED', 'FAILED', 'CANCELLED', 'EXPIRED')
   );

COMMIT;

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    19 AS SCHEMA_VERSION,
    'aiops-oracle-v9' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_data_type USER_TAB_COLUMNS.DATA_TYPE%TYPE;
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_invalid_views PLS_INTEGER;
BEGIN
    SELECT DATA_TYPE
      INTO l_data_type
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_REPORT'
       AND COLUMN_NAME = 'SUMMARY';

    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*)
      INTO l_invalid_views
      FROM USER_OBJECTS
     WHERE OBJECT_NAME IN (
         'KBOT_V_OPS_RUN',
         'KBOT_V_OPS_REPORT',
         'KBOT_V_OPS_SCHEMA_VERSION'
     )
       AND STATUS <> 'VALID';

    IF l_data_type <> 'CLOB'
       OR l_schema_version <> 19
       OR l_contract_version <> 'aiops-oracle-v9'
       OR l_invalid_views <> 0 THEN
        RAISE_APPLICATION_ERROR(-20020, 'AIOps Schema 19 升级校验失败');
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 19，合同 aiops-oracle-v9。'
    );
END;
/
