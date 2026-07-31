-- KBot 4.0 一次性原地修改：删除冗余资源标识列。
-- 本脚本保留现有业务数据，不删除表，不重建 Schema。
-- 使用当前 KBot Schema 用户执行；DDL 会隐式提交，执行前请完成备份并停止 KBot 服务。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE

DECLARE
    l_column_count PLS_INTEGER := 0;
    l_removed_count PLS_INTEGER := 0;

    PROCEDURE drop_resource_column(
        p_table_name IN VARCHAR2,
        p_column_name IN VARCHAR2
    ) IS
    BEGIN
        SELECT COUNT(*)
          INTO l_column_count
          FROM user_tab_columns columns
          JOIN user_tables tables
            ON tables.table_name = columns.table_name
         WHERE columns.table_name = p_table_name
           AND columns.column_name = p_column_name;

        IF l_column_count = 0 THEN
            dbms_output.put_line(
                '已跳过不存在的列 ' || p_table_name || '.' || p_column_name
            );
            RETURN;
        END IF;

        -- 先删除引用该列所属主键或唯一键的外键，避免 ORA-02449。
        FOR foreign_key_row IN (
            SELECT child.table_name, child.constraint_name
              FROM user_constraints child
             WHERE child.constraint_type = 'R'
               AND child.r_constraint_name IN (
                    SELECT parent.constraint_name
                      FROM user_constraints parent
                      JOIN user_cons_columns parent_column
                        ON parent_column.constraint_name =
                           parent.constraint_name
                       AND parent_column.table_name = parent.table_name
                     WHERE parent.table_name = p_table_name
                       AND parent_column.column_name = p_column_name
               )
             ORDER BY child.table_name, child.constraint_name
        ) LOOP
            EXECUTE IMMEDIATE
                'ALTER TABLE '
                || dbms_assert.enquote_name(
                    foreign_key_row.table_name, FALSE
                )
                || ' DROP CONSTRAINT '
                || dbms_assert.enquote_name(
                    foreign_key_row.constraint_name, FALSE
                );
            dbms_output.put_line(
                '已删除引用外键 '
                || foreign_key_row.table_name || '.'
                || foreign_key_row.constraint_name
            );
        END LOOP;

        EXECUTE IMMEDIATE
            'ALTER TABLE '
            || dbms_assert.enquote_name(p_table_name, FALSE)
            || ' DROP COLUMN '
            || dbms_assert.enquote_name(p_column_name, FALSE)
            || ' CASCADE CONSTRAINTS';
        l_removed_count := l_removed_count + 1;
        dbms_output.put_line(
            '已删除列 ' || p_table_name || '.' || p_column_name
        );
    END;
BEGIN
    drop_resource_column('KBOT_AGENT_DEFINITION', 'AGENT_KEY');
    drop_resource_column('KBOT_KC_COLLECTION', 'COLLECTION_KEY');
    drop_resource_column('KBOT_OPS_TARGET', 'TARGET_KEY');
    drop_resource_column('KBOT_OPS_MONITOR_SOURCE', 'SOURCE_KEY');
    drop_resource_column('KBOT_OPS_INSPECTION_PLAN', 'PLAN_KEY');

    dbms_output.put_line(
        '冗余资源标识列处理完成，本次删除 ' || l_removed_count || ' 列。'
    );
END;
/

-- 重建所有直接依赖上述 AIOps 列的只读视图。
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
    t.SECURITY_LEVEL,
    t.STATUS,
    t.HEALTH_STATUS,
    t.LAST_HEALTH_CHECK_AT,
    t.LAST_ERROR_CODE,
    t.ROW_VERSION,
    t.CREATED_AT,
    t.UPDATED_AT,
    COUNT(tm.TARGET_MONITOR_ID) AS MONITOR_COUNT,
    SUM(CASE WHEN tm.HEALTH_STATUS = 'HEALTHY' THEN 1 ELSE 0 END)
        AS HEALTHY_MONITOR_COUNT,
    SUM(CASE WHEN tm.HEALTH_STATUS IN ('DEGRADED', 'UNREACHABLE')
             THEN 1 ELSE 0 END) AS UNHEALTHY_MONITOR_COUNT
FROM KBOT_OPS_TARGET t
LEFT JOIN KBOT_OPS_TARGET_MONITOR tm
    ON tm.TARGET_ID = t.TARGET_ID
   AND tm.STATUS = 'ACTIVE'
GROUP BY
    t.TARGET_ID, t.DOMAIN_ID, t.DISPLAY_NAME,
    t.DB_TYPE, t.VERSION_CODE, t.ENVIRONMENT, t.DB_ROLE,
    t.SECURITY_LEVEL, t.STATUS, t.HEALTH_STATUS,
    t.LAST_HEALTH_CHECK_AT, t.LAST_ERROR_CODE, t.ROW_VERSION,
    t.CREATED_AT, t.UPDATED_AT;

CREATE OR REPLACE VIEW KBOT_V_OPS_MONITOR_SOURCE AS
SELECT
    LOWER(
        SUBSTR(RAWTOHEX(m.MONITOR_SOURCE_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(m.MONITOR_SOURCE_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(m.MONITOR_SOURCE_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(m.MONITOR_SOURCE_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(m.MONITOR_SOURCE_ID), 21, 12)
    ) AS MONITOR_SOURCE_ID,
    m.DOMAIN_ID,
    m.DISPLAY_NAME,
    m.SOURCE_TYPE,
    m.STATUS,
    m.HEALTH_STATUS,
    m.LAST_HEALTH_CHECK_AT,
    m.LAST_ERROR_CODE,
    m.ROW_VERSION,
    m.CREATED_AT,
    m.UPDATED_AT,
    COUNT(tm.TARGET_MONITOR_ID) AS ACTIVE_TARGET_COUNT
FROM KBOT_OPS_MONITOR_SOURCE m
LEFT JOIN KBOT_OPS_TARGET_MONITOR tm
    ON tm.MONITOR_SOURCE_ID = m.MONITOR_SOURCE_ID
   AND tm.STATUS = 'ACTIVE'
GROUP BY
    m.MONITOR_SOURCE_ID, m.DOMAIN_ID,
    m.DISPLAY_NAME, m.SOURCE_TYPE, m.STATUS, m.HEALTH_STATUS,
    m.LAST_HEALTH_CHECK_AT, m.LAST_ERROR_CODE, m.ROW_VERSION,
    m.CREATED_AT, m.UPDATED_AT;

CREATE OR REPLACE VIEW KBOT_V_OPS_INSPECTION_PLAN AS
SELECT
    LOWER(
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 21, 12)
    ) AS INSPECTION_PLAN_ID,
    p.DOMAIN_ID,
    p.DISPLAY_NAME,
    p.SCHEDULE_TYPE,
    p.CRON_EXPRESSION,
    p.TIMEZONE,
    p.TEMPLATE_ID,
    p.TEMPLATE_VERSION,
    p.STATUS,
    p.NEXT_RUN_AT,
    p.LAST_RUN_AT,
    p.LAST_SCHEDULED_FOR,
    p.ROW_VERSION,
    p.CREATED_AT,
    p.UPDATED_AT
FROM KBOT_OPS_INSPECTION_PLAN p;

CREATE OR REPLACE VIEW KBOT_V_OPS_INSPECTION_FIRE AS
SELECT
    LOWER(
        SUBSTR(RAWTOHEX(f.INSPECTION_FIRE_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(f.INSPECTION_FIRE_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(f.INSPECTION_FIRE_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(f.INSPECTION_FIRE_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(f.INSPECTION_FIRE_ID), 21, 12)
    ) AS INSPECTION_FIRE_ID,
    LOWER(
        SUBSTR(RAWTOHEX(f.INSPECTION_PLAN_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(f.INSPECTION_PLAN_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(f.INSPECTION_PLAN_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(f.INSPECTION_PLAN_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(f.INSPECTION_PLAN_ID), 21, 12)
    ) AS INSPECTION_PLAN_ID,
    p.DOMAIN_ID,
    f.SCHEDULED_FOR,
    f.STATUS,
    f.TARGET_COUNT,
    f.RUN_COUNT,
    f.COMPLETED_COUNT,
    f.FAILED_COUNT,
    f.SKIP_REASON,
    f.STARTED_AT,
    f.COMPLETED_AT,
    f.CREATED_AT,
    f.UPDATED_AT
FROM KBOT_OPS_INSPECTION_FIRE f
JOIN KBOT_OPS_INSPECTION_PLAN p
    ON p.INSPECTION_PLAN_ID = f.INSPECTION_PLAN_ID;

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
    r.ACTOR_ID,
    r.STATUS,
    r.ROOT_CAUSE_LEVEL,
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
        COUNT(*) AS REPORT_COUNT,
        MAX(SUMMARY) KEEP (DENSE_RANK LAST ORDER BY UPDATED_AT)
            AS LATEST_REPORT_SUMMARY
    FROM KBOT_OPS_REPORT
    WHERE IS_CURRENT = 1
    GROUP BY OPS_RUN_ID
) report_info ON report_info.OPS_RUN_ID = r.OPS_RUN_ID;

CREATE OR REPLACE VIEW KBOT_V_OPS_PENDING_APPROVAL AS
SELECT
    t.DOMAIN_ID,
    LOWER(
        SUBSTR(RAWTOHEX(p.PROPOSAL_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(p.PROPOSAL_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.PROPOSAL_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.PROPOSAL_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.PROPOSAL_ID), 21, 12)
    ) AS PROPOSAL_ID,
    LOWER(
        SUBSTR(RAWTOHEX(h.HITL_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(h.HITL_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(h.HITL_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(h.HITL_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(h.HITL_ID), 21, 12)
    ) AS HITL_ID,
    LOWER(
        SUBSTR(RAWTOHEX(p.OPS_RUN_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(p.OPS_RUN_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.OPS_RUN_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.OPS_RUN_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.OPS_RUN_ID), 21, 12)
    ) AS OPS_RUN_ID,
    LOWER(
        SUBSTR(RAWTOHEX(p.TARGET_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(p.TARGET_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.TARGET_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.TARGET_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.TARGET_ID), 21, 12)
    ) AS TARGET_ID,
    t.DISPLAY_NAME AS TARGET_NAME,
    h.ASSIGNEE_USER_ID,
    p.ACTION_TYPE,
    p.ACTION_TEMPLATE_ID,
    p.ACTION_TEMPLATE_VERSION,
    p.RISK_LEVEL,
    p.RATIONALE,
    p.STATUS AS PROPOSAL_STATUS,
    h.STATUS AS HITL_STATUS,
    h.EXPIRES_AT
FROM KBOT_OPS_CHANGE_PROPOSAL p
JOIN KBOT_OPS_HITL h ON h.PROPOSAL_ID = p.PROPOSAL_ID
JOIN KBOT_OPS_TARGET t ON t.TARGET_ID = p.TARGET_ID
WHERE p.STATUS = 'PENDING_APPROVAL'
  AND h.STATUS = 'PENDING'
  AND h.REQUEST_TYPE = 'CHANGE_APPROVAL';

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

CREATE OR REPLACE VIEW KBOT_V_OPS_CHAT_PENDING AS
SELECT
    t.DOMAIN_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.OPS_RUN_ID), 21, 12)
    ) AS OPS_RUN_ID,
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
    LOWER(
        SUBSTR(RAWTOHEX(h.HITL_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(h.HITL_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(h.HITL_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(h.HITL_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(h.HITL_ID), 21, 12)
    ) AS HITL_ID,
    LOWER(
        SUBSTR(RAWTOHEX(r.TARGET_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(r.TARGET_ID), 21, 12)
    ) AS TARGET_ID,
    t.DISPLAY_NAME AS TARGET_NAME,
    h.ASSIGNEE_USER_ID,
    h.REQUEST_TYPE,
    h.PROMPT_TEXT,
    h.REQUESTED_AT,
    h.EXPIRES_AT
FROM KBOT_OPS_HITL h
JOIN KBOT_OPS_RUN r ON r.OPS_RUN_ID = h.OPS_RUN_ID
JOIN KBOT_OPS_TARGET t ON t.TARGET_ID = r.TARGET_ID
WHERE r.TRIGGER_TYPE IN ('CHAT', 'ROOT')
  AND h.STATUS = 'PENDING';

-- 验证一：执行后应返回 0 行。
SELECT table_name, column_name
FROM user_tab_columns columns
WHERE (columns.table_name, columns.column_name) IN (
    ('KBOT_AGENT_DEFINITION', 'AGENT_KEY'),
    ('KBOT_KC_COLLECTION', 'COLLECTION_KEY'),
    ('KBOT_OPS_TARGET', 'TARGET_KEY'),
    ('KBOT_OPS_MONITOR_SOURCE', 'SOURCE_KEY'),
    ('KBOT_OPS_INSPECTION_PLAN', 'PLAN_KEY')
)
  AND columns.table_name IN (SELECT table_name FROM user_tables)
ORDER BY columns.table_name, columns.column_name;

-- 验证二：以下 8 个视图都应为 VALID。
SELECT object_name, status
FROM user_objects
WHERE object_type = 'VIEW'
  AND object_name IN (
      'KBOT_V_OPS_TARGET',
      'KBOT_V_OPS_MONITOR_SOURCE',
      'KBOT_V_OPS_INSPECTION_PLAN',
      'KBOT_V_OPS_INSPECTION_FIRE',
      'KBOT_V_OPS_RUN',
      'KBOT_V_OPS_PENDING_APPROVAL',
      'KBOT_V_OPS_REPORT',
      'KBOT_V_OPS_CHAT_PENDING'
  )
ORDER BY object_name;
