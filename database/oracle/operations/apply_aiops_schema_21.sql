-- AIOps Schema 20 -> 21 原地升级。
-- 影响范围：扩展正式报告类型，新增报告来源关联表和索引，并更新 Schema 版本视图。
-- 数据保护：不删除表、不删除业务行、不修改既有报告、诊断、命令或审计哈希。
-- 前置条件：停止 AIOps API、Worker、Scheduler 和 DB Executor，避免 DDL 与运行时写入并发。
-- 恢复方式：Oracle DDL 会隐式提交；执行前必须完成 Schema 备份，失败时按备份恢复。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_report_source_count PLS_INTEGER;
    l_report_type_constraint_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF l_schema_version <> 20
       OR l_contract_version <> 'aiops-oracle-v10' THEN
        RAISE_APPLICATION_ERROR(
            -20030,
            '只允许从 AIOPS Schema 20 / aiops-oracle-v10 升级'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_report_source_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_REPORT_SOURCE';

    IF l_report_source_count <> 0 THEN
        RAISE_APPLICATION_ERROR(
            -20031,
            'KBOT_OPS_REPORT_SOURCE 已存在，当前 Schema 状态不适合执行本升级脚本'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_report_type_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_OPS_REPORT'
       AND CONSTRAINT_NAME = 'CK_OPS_REPORT_TYPE'
       AND CONSTRAINT_TYPE = 'C'
       AND STATUS = 'ENABLED'
       AND VALIDATED = 'VALIDATED';

    IF l_report_type_constraint_count <> 1 THEN
        RAISE_APPLICATION_ERROR(
            -20032,
            'KBOT_OPS_REPORT.CK_OPS_REPORT_TYPE 不符合 Schema 20 前置条件'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE('Schema 20 前置检查通过。');
END;
/

-- 只扩展允许值；既有报告类型和业务行保持不变。
ALTER TABLE KBOT_OPS_REPORT DROP CONSTRAINT CK_OPS_REPORT_TYPE;

ALTER TABLE KBOT_OPS_REPORT ADD CONSTRAINT CK_OPS_REPORT_TYPE CHECK (
    REPORT_TYPE IN (
        'INCIDENT', 'PERFORMANCE', 'INSPECTION_DAILY',
        'INSPECTION_WEEKLY', 'INSPECTION_MONTHLY',
        'INSPECTION_QUARTERLY', 'INSPECTION_ANNUAL',
        'INSPECTION_CUSTOM', 'COMPARISON'
    )
);

CREATE TABLE KBOT_OPS_REPORT_SOURCE (
    REPORT_ID RAW(16) NOT NULL,
    OPS_RUN_ID RAW(16) NOT NULL,
    SOURCE_ARTIFACT_ID RAW(16) NOT NULL,
    SOURCE_KIND VARCHAR2(16 CHAR) NOT NULL,
    CONTENT_HASH VARCHAR2(64 CHAR) NOT NULL,
    OBSERVED_AT TIMESTAMP(6) WITH TIME ZONE NOT NULL,
    CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT PK_OPS_REPORT_SOURCE PRIMARY KEY (REPORT_ID, OPS_RUN_ID),
    CONSTRAINT FK_OPS_RPT_SOURCE_REPORT FOREIGN KEY (REPORT_ID)
        REFERENCES KBOT_OPS_REPORT (REPORT_ID) ON DELETE CASCADE,
    CONSTRAINT FK_OPS_RPT_SOURCE_RUN FOREIGN KEY (OPS_RUN_ID)
        REFERENCES KBOT_OPS_RUN (OPS_RUN_ID),
    CONSTRAINT FK_OPS_RPT_SOURCE_ART FOREIGN KEY (SOURCE_ARTIFACT_ID)
        REFERENCES KBOT_OPS_ARTIFACT (ARTIFACT_ID),
    CONSTRAINT CK_OPS_RPT_SOURCE_KIND CHECK (
        SOURCE_KIND IN ('CHAT', 'ALERT', 'INSPECTION')
    ),
    CONSTRAINT CK_OPS_RPT_SOURCE_HASH CHECK (LENGTH(CONTENT_HASH) = 64)
);

CREATE INDEX IX_OPS_RPT_SOURCE_RUN
    ON KBOT_OPS_REPORT_SOURCE (OPS_RUN_ID);
CREATE INDEX IX_OPS_RPT_SOURCE_ART
    ON KBOT_OPS_REPORT_SOURCE (SOURCE_ARTIFACT_ID);

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    21 AS SCHEMA_VERSION,
    'aiops-oracle-v11' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_report_source_count PLS_INTEGER;
    l_index_count PLS_INTEGER;
    l_report_type_constraint_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*)
      INTO l_report_source_count
      FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_REPORT_SOURCE';

    SELECT COUNT(*)
      INTO l_index_count
      FROM USER_INDEXES
     WHERE TABLE_NAME = 'KBOT_OPS_REPORT_SOURCE'
       AND INDEX_NAME IN ('IX_OPS_RPT_SOURCE_RUN', 'IX_OPS_RPT_SOURCE_ART');

    SELECT COUNT(*)
      INTO l_report_type_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_OPS_REPORT'
       AND CONSTRAINT_NAME = 'CK_OPS_REPORT_TYPE'
       AND CONSTRAINT_TYPE = 'C'
       AND STATUS = 'ENABLED'
       AND VALIDATED = 'VALIDATED';

    IF l_schema_version <> 21
       OR l_contract_version <> 'aiops-oracle-v11'
       OR l_report_source_count <> 1
       OR l_index_count <> 2
       OR l_report_type_constraint_count <> 1 THEN
        RAISE_APPLICATION_ERROR(
            -20033,
            'Schema 21 升级校验失败，请从备份恢复并核对输出'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 21 / aiops-oracle-v11。'
    );
END;
/
