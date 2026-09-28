-- AIOps Schema 23/24 -> 26 原地升级。
-- 影响范围：巡检计划增加 SELECTED_CHECKS_JSON；新增 Target 运维事实表；
--           更新巡检计划视图和 Schema 版本视图。
-- 数据保护：不删除表、不删除业务行，不修改历史运行、诊断、报告、附件或审计数据。
--           既有巡检计划若缺少检查项快照，回填当前 Check Catalog 的 18 个 READY
--           默认项（不含 user.* 合成报告类检查项）。
-- 前置条件：停止 AIOps API、Worker、Scheduler 和 DB Executor，避免 DDL 与运行时写入并发。
--           当前必须是 Schema 23 / aiops-oracle-v13 或 Schema 24 / aiops-oracle-v14。
--           若仍是 Schema 23，请先执行 converge_core_authorization.sql 删除
--           KBOT_OPS_AGENT_GRANT；本脚本不处理权限模型收敛。
-- 恢复方式：Oracle DDL 会隐式提交；执行前必须完成 Schema 备份，失败时按备份恢复。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

ALTER SESSION SET TIME_ZONE = '+00:00';

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_plan_count PLS_INTEGER;
    l_fact_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF NOT (
            (l_schema_version = 23 AND l_contract_version = 'aiops-oracle-v13')
         OR (l_schema_version = 24 AND l_contract_version = 'aiops-oracle-v14')
       ) THEN
        RAISE_APPLICATION_ERROR(
            -20050,
            '只允许从 AIOPS Schema 23 / aiops-oracle-v13 或 24 / aiops-oracle-v14 升级'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_plan_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_INSPECTION_PLAN';

    SELECT COUNT(*)
      INTO l_fact_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_TARGET_FACT';

    IF l_plan_count <> 1 OR l_fact_count <> 0 THEN
        RAISE_APPLICATION_ERROR(
            -20051,
            'KBOT_OPS_INSPECTION_PLAN / KBOT_OPS_TARGET_FACT 不符合 Schema 23/24 前置条件'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE('Schema ' || l_schema_version || ' 前置检查通过。');
END;
/

DECLARE
    l_column_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_INSPECTION_PLAN'
       AND COLUMN_NAME = 'SELECTED_CHECKS_JSON';

    IF l_column_count = 0 THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_OPS_INSPECTION_PLAN ADD (SELECTED_CHECKS_JSON JSON)';
        DBMS_OUTPUT.PUT_LINE('已增加 KBOT_OPS_INSPECTION_PLAN.SELECTED_CHECKS_JSON。');
    ELSE
        DBMS_OUTPUT.PUT_LINE('SELECTED_CHECKS_JSON 已存在，跳过加列。');
    END IF;
END;
/

UPDATE KBOT_OPS_INSPECTION_PLAN
   SET SELECTED_CHECKS_JSON = JSON(
           '["oracle.backup.rman_volume","oracle.backup.space_usage",'
           || '"oracle.backup.recent_jobs","oracle.storage.tablespace_headroom",'
           || '"oracle.storage.datafile_count","oracle.storage.growth",'
           || '"oracle.storage.undo","oracle.session.count",'
           || '"oracle.session.long_running","oracle.session.long_transaction",'
           || '"oracle.session.lock_wait","oracle.object.invalid",'
           || '"oracle.object.sparse_index","oracle.log.archive_generation",'
           || '"oracle.performance.datafile_io","oracle.performance.critical_waits",'
           || '"oracle.performance.heavy_sort_sql","oracle.performance.core_parameters"]'
       )
 WHERE SELECTED_CHECKS_JSON IS NULL;

COMMIT;

ALTER TABLE KBOT_OPS_INSPECTION_PLAN
    MODIFY SELECTED_CHECKS_JSON NOT NULL;

COMMENT ON COLUMN KBOT_OPS_INSPECTION_PLAN.SELECTED_CHECKS_JSON IS
    '计划勾选的 Check Catalog ID 列表；只存 READY 项，顺序与目录一致';

CREATE TABLE KBOT_OPS_TARGET_FACT (
    TARGET_FACT_ID RAW(16) PRIMARY KEY,
    TARGET_ID RAW(16) NOT NULL,
    DOMAIN_ID NUMBER(38) NOT NULL,
    FACT_TYPE VARCHAR2(64 CHAR) NOT NULL,
    FACT_KEY VARCHAR2(256 CHAR) NOT NULL,
    FACT_VALUE JSON NOT NULL,
    SOURCE VARCHAR2(32 CHAR) NOT NULL,
    STATUS VARCHAR2(16 CHAR) DEFAULT 'ACTIVE' NOT NULL,
    CONFIRMED_BY VARCHAR2(256 CHAR),
    CONFIRMED_AT TIMESTAMP(6) WITH TIME ZONE,
    RETIRED_BY VARCHAR2(256 CHAR),
    RETIRED_AT TIMESTAMP(6) WITH TIME ZONE,
    ROW_VERSION NUMBER(19) DEFAULT 1 NOT NULL,
    CREATED_BY VARCHAR2(256 CHAR) NOT NULL,
    UPDATED_BY VARCHAR2(256 CHAR) NOT NULL,
    CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    UPDATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT FK_OPS_TARGET_FACT_TARGET
        FOREIGN KEY (TARGET_ID) REFERENCES KBOT_OPS_TARGET (TARGET_ID),
    CONSTRAINT FK_OPS_TARGET_FACT_OWNER
        FOREIGN KEY (TARGET_ID, DOMAIN_ID)
        REFERENCES KBOT_OPS_TARGET (TARGET_ID, DOMAIN_ID),
    CONSTRAINT FK_OPS_TARGET_FACT_DOMAIN
        FOREIGN KEY (DOMAIN_ID)
        REFERENCES KBOT_PLATFORM_DOMAIN (DOMAIN_ID),
    CONSTRAINT CK_OPS_TARGET_FACT_TYPE
        CHECK (FACT_TYPE IN (
            'ASM_DISKGROUP', 'DATAFILE_PATH', 'TABLESPACE_PLACEMENT'
        )),
    CONSTRAINT CK_OPS_TARGET_FACT_SOURCE
        CHECK (SOURCE IN ('MANUAL_CONFIRMED', 'DISCOVERED')),
    CONSTRAINT CK_OPS_TARGET_FACT_STATUS
        CHECK (STATUS IN ('ACTIVE', 'RETIRED')),
    CONSTRAINT CK_OPS_TARGET_FACT_VERSION
        CHECK (ROW_VERSION >= 1),
    CONSTRAINT CK_OPS_TARGET_FACT_RETIRED CHECK (
        (STATUS = 'ACTIVE' AND RETIRED_BY IS NULL AND RETIRED_AT IS NULL)
        OR
        (STATUS = 'RETIRED' AND RETIRED_BY IS NOT NULL AND RETIRED_AT IS NOT NULL)
    ),
    CONSTRAINT CK_OPS_TARGET_FACT_CONFIRMED CHECK (
        (SOURCE = 'MANUAL_CONFIRMED'
         AND CONFIRMED_BY IS NOT NULL
         AND CONFIRMED_AT IS NOT NULL)
        OR
        (SOURCE = 'DISCOVERED'
         AND CONFIRMED_BY IS NULL
         AND CONFIRMED_AT IS NULL)
    )
);

CREATE INDEX IX_OPS_TARGET_FACT_TARGET
    ON KBOT_OPS_TARGET_FACT (TARGET_ID, STATUS);
CREATE INDEX IX_OPS_TARGET_FACT_DOMAIN
    ON KBOT_OPS_TARGET_FACT (DOMAIN_ID);

CREATE UNIQUE INDEX UX_OPS_TARGET_FACT_ACTIVE ON KBOT_OPS_TARGET_FACT (
    CASE WHEN STATUS = 'ACTIVE' THEN TARGET_ID END,
    CASE WHEN STATUS = 'ACTIVE' THEN FACT_TYPE END,
    CASE WHEN STATUS = 'ACTIVE' THEN FACT_KEY END
);

COMMENT ON COLUMN KBOT_OPS_TARGET_FACT.FACT_VALUE IS
    '已确认的运维事实值；仅保存白名单字段，不保存可执行 SQL';

CREATE OR REPLACE VIEW KBOT_V_OPS_INSPECTION_PLAN AS
SELECT
    LOWER(
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_PLAN_ID), 21, 12)
    ) AS INSPECTION_PLAN_ID,
    LOWER(
        SUBSTR(RAWTOHEX(p.AGENT_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(p.AGENT_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.AGENT_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.AGENT_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.AGENT_ID), 21, 12)
    ) AS AGENT_ID,
    p.DOMAIN_ID,
    p.DISPLAY_NAME,
    p.SCHEDULE_TYPE,
    p.CRON_EXPRESSION,
    p.TIMEZONE,
    p.TEMPLATE_ID,
    p.TEMPLATE_VERSION,
    p.SELECTED_CHECKS_JSON,
    p.STATUS,
    p.NEXT_RUN_AT,
    p.LAST_RUN_AT,
    p.LAST_SCHEDULED_FOR,
    p.ROW_VERSION,
    p.CREATED_AT,
    p.UPDATED_AT
FROM KBOT_OPS_INSPECTION_PLAN p;

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    26 AS SCHEMA_VERSION,
    'aiops-oracle-v16' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_column_nullable CHAR(1 CHAR);
    l_fact_count PLS_INTEGER;
    l_index_count PLS_INTEGER;
    l_view_column_count PLS_INTEGER;
    l_null_plan_count PLS_INTEGER;
    l_invalid_object_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT NULLABLE
      INTO l_column_nullable
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_INSPECTION_PLAN'
       AND COLUMN_NAME = 'SELECTED_CHECKS_JSON';

    SELECT COUNT(*)
      INTO l_fact_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_TARGET_FACT';

    SELECT COUNT(*)
      INTO l_index_count
      FROM USER_INDEXES
     WHERE TABLE_NAME = 'KBOT_OPS_TARGET_FACT'
       AND INDEX_NAME IN (
           'IX_OPS_TARGET_FACT_TARGET',
           'IX_OPS_TARGET_FACT_DOMAIN',
           'UX_OPS_TARGET_FACT_ACTIVE'
       );

    SELECT COUNT(*)
      INTO l_view_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_V_OPS_INSPECTION_PLAN'
       AND COLUMN_NAME = 'SELECTED_CHECKS_JSON';

    SELECT COUNT(*)
      INTO l_null_plan_count
      FROM KBOT_OPS_INSPECTION_PLAN
     WHERE SELECTED_CHECKS_JSON IS NULL;

    SELECT COUNT(*)
      INTO l_invalid_object_count
      FROM USER_OBJECTS
     WHERE OBJECT_NAME IN (
         'KBOT_OPS_INSPECTION_PLAN',
         'KBOT_OPS_TARGET_FACT',
         'KBOT_V_OPS_INSPECTION_PLAN',
         'KBOT_V_OPS_SCHEMA_VERSION'
     )
       AND STATUS <> 'VALID';

    IF l_schema_version <> 26
       OR l_contract_version <> 'aiops-oracle-v16'
       OR l_column_nullable <> 'N'
       OR l_fact_count <> 1
       OR l_index_count <> 3
       OR l_view_column_count <> 1
       OR l_null_plan_count <> 0
       OR l_invalid_object_count <> 0 THEN
        RAISE_APPLICATION_ERROR(
            -20052,
            'Schema 26 升级校验失败，请从备份恢复并核对输出'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 26 / aiops-oracle-v16。'
    );
END;
/
