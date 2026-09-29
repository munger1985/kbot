-- AIOps Schema 29 -> 30 原地升级。
-- 影响范围：拆分巡检内容模板与 Session 报告展示模板；巡检计划改为引用冻结模板版本。
-- 数据保护：不删除业务行；每个既有巡检计划按原勾选项生成一个不可变模板版本。
-- 前置条件：当前必须是 Schema 29 / aiops-oracle-v19，或已部分/全部升级到 Schema 30。
-- 恢复方式：Oracle DDL 会隐式提交；执行前必须完成 Schema 备份。
-- 并发要求：执行前停止 AIOps API、Worker、Scheduler 和 DB Executor；不得存在运行中或排队中的巡检 Fire。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

ALTER SESSION SET TIME_ZONE = '+00:00';
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_open_fire_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF NOT (
        (l_schema_version = 29
         AND l_contract_version = 'aiops-oracle-v19')
        OR
        (l_schema_version = 30
         AND l_contract_version = 'aiops-oracle-v20')
    ) THEN
        RAISE_APPLICATION_ERROR(
            -20067,
            '只允许从 AIOPS Schema 29 / aiops-oracle-v19 升级，'
            || '或续跑 Schema 30 / aiops-oracle-v20'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_open_fire_count
      FROM KBOT_OPS_INSPECTION_FIRE
     WHERE STATUS IN ('RUNNING', 'QUEUED');

    IF l_open_fire_count > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20068,
            '仍有运行中或排队中的巡检 Fire，请等待其结束后重新执行升级'
        );
    END IF;
END;
/

DECLARE
    l_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*) INTO l_count FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_INSPECTION_TEMPLATE';
    IF l_count = 0 THEN
        EXECUTE IMMEDIATE q'~
            CREATE TABLE KBOT_OPS_INSPECTION_TEMPLATE (
                INSPECTION_TEMPLATE_ID RAW(16) PRIMARY KEY,
                DOMAIN_ID NUMBER(38) NOT NULL,
                DISPLAY_NAME VARCHAR2(256 CHAR) NOT NULL,
                STATUS VARCHAR2(16 CHAR) DEFAULT 'ACTIVE' NOT NULL,
                CURRENT_VERSION_ID RAW(16) NOT NULL,
                ROW_VERSION NUMBER(19) DEFAULT 1 NOT NULL,
                CREATED_BY VARCHAR2(256 CHAR) NOT NULL,
                UPDATED_BY VARCHAR2(256 CHAR) NOT NULL,
                CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                UPDATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                CONSTRAINT FK_OPS_INSP_TPL_DOMAIN FOREIGN KEY (DOMAIN_ID)
                    REFERENCES KBOT_PLATFORM_DOMAIN (DOMAIN_ID)
            )~';
    END IF;

    SELECT COUNT(*) INTO l_count FROM USER_TABLES
     WHERE TABLE_NAME = 'KBOT_OPS_INSPECTION_TEMPLATE_VER';
    IF l_count = 0 THEN
        EXECUTE IMMEDIATE q'~
            CREATE TABLE KBOT_OPS_INSPECTION_TEMPLATE_VER (
                INSPECTION_TEMPLATE_VERSION_ID RAW(16) PRIMARY KEY,
                DOMAIN_ID NUMBER(38) NOT NULL,
                INSPECTION_TEMPLATE_ID RAW(16) NOT NULL,
                VERSION_NO NUMBER(19) NOT NULL,
                DEFINITION_JSON JSON NOT NULL,
                CONTENT_HASH VARCHAR2(64 CHAR) NOT NULL,
                CREATED_BY VARCHAR2(256 CHAR) NOT NULL,
                CREATED_AT TIMESTAMP(6) WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                CONSTRAINT FK_OPS_INSP_TPL_VER_DOMAIN FOREIGN KEY (DOMAIN_ID)
                    REFERENCES KBOT_PLATFORM_DOMAIN (DOMAIN_ID),
                CONSTRAINT FK_OPS_INSP_TPL_VER_TEMPLATE
                    FOREIGN KEY (INSPECTION_TEMPLATE_ID)
                    REFERENCES KBOT_OPS_INSPECTION_TEMPLATE (INSPECTION_TEMPLATE_ID)
                    ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED
            )~';
    END IF;
END;
/

DECLARE
    PROCEDURE create_index_if_missing(
        p_index_name VARCHAR2,
        p_statement VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*) INTO l_count FROM USER_INDEXES
         WHERE INDEX_NAME = p_index_name;
        IF l_count = 0 THEN
            EXECUTE IMMEDIATE p_statement;
        END IF;
    END;

    PROCEDURE add_constraint_if_missing(
        p_constraint_name VARCHAR2,
        p_statement VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*) INTO l_count FROM USER_CONSTRAINTS
         WHERE CONSTRAINT_NAME = p_constraint_name;
        IF l_count = 0 THEN
            EXECUTE IMMEDIATE p_statement;
        END IF;
    END;
BEGIN
    create_index_if_missing(
        'IX_OPS_INSP_TEMPLATE_SCOPE',
        'CREATE INDEX IX_OPS_INSP_TEMPLATE_SCOPE '
        || 'ON KBOT_OPS_INSPECTION_TEMPLATE (DOMAIN_ID, STATUS)'
    );
    create_index_if_missing(
        'IX_OPS_INSP_TPL_VER_TEMPLATE',
        'CREATE INDEX IX_OPS_INSP_TPL_VER_TEMPLATE '
        || 'ON KBOT_OPS_INSPECTION_TEMPLATE_VER (INSPECTION_TEMPLATE_ID)'
    );
    create_index_if_missing(
        'IX_OPS_INSP_TPL_VER_DOMAIN',
        'CREATE INDEX IX_OPS_INSP_TPL_VER_DOMAIN '
        || 'ON KBOT_OPS_INSPECTION_TEMPLATE_VER (DOMAIN_ID)'
    );
    add_constraint_if_missing(
        'FK_OPS_INSP_TPL_CURRENT_VER',
        'ALTER TABLE KBOT_OPS_INSPECTION_TEMPLATE '
        || 'ADD CONSTRAINT FK_OPS_INSP_TPL_CURRENT_VER '
        || 'FOREIGN KEY (CURRENT_VERSION_ID) '
        || 'REFERENCES KBOT_OPS_INSPECTION_TEMPLATE_VER '
        || '(INSPECTION_TEMPLATE_VERSION_ID) '
        || 'DEFERRABLE INITIALLY DEFERRED'
    );
    create_index_if_missing(
        'IX_OPS_INSP_TPL_CURRENT_VER',
        'CREATE INDEX IX_OPS_INSP_TPL_CURRENT_VER '
        || 'ON KBOT_OPS_INSPECTION_TEMPLATE (CURRENT_VERSION_ID)'
    );
END;
/

DECLARE
    PROCEDURE add_column_if_missing(
        p_table_name VARCHAR2,
        p_column_name VARCHAR2,
        p_definition VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*) INTO l_count FROM USER_TAB_COLUMNS
         WHERE TABLE_NAME = p_table_name AND COLUMN_NAME = p_column_name;
        IF l_count = 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE ' || p_table_name || ' ADD ('
                || p_column_name || ' ' || p_definition || ')';
        END IF;
    END;
BEGIN
    add_column_if_missing(
        'KBOT_OPS_INSPECTION_PLAN',
        'INSPECTION_TEMPLATE_ID',
        'RAW(16)'
    );
    add_column_if_missing(
        'KBOT_OPS_INSPECTION_PLAN',
        'INSPECTION_TEMPLATE_VERSION_ID',
        'RAW(16)'
    );
    add_column_if_missing(
        'KBOT_OPS_INSPECTION_FIRE',
        'INSPECTION_TEMPLATE_ID',
        'RAW(16)'
    );
    add_column_if_missing(
        'KBOT_OPS_INSPECTION_FIRE',
        'INSPECTION_TEMPLATE_VERSION_ID',
        'RAW(16)'
    );
END;
/

DECLARE
    l_template_id RAW(16);
    l_version_id RAW(16);
    l_definition CLOB;
    l_template_name VARCHAR2(256 CHAR);
    l_plan_cursor SYS_REFCURSOR;
    l_plan_id RAW(16);
    l_domain_id NUMBER(38);
    l_plan_display_name VARCHAR2(256 CHAR);
    l_created_by VARCHAR2(256 CHAR);
    l_created_at TIMESTAMP(6) WITH TIME ZONE;
    l_selected_json CLOB;
    l_selected_column_count PLS_INTEGER;
    l_content_hash VARCHAR2(64 CHAR);

    FUNCTION uuid7_raw RETURN RAW IS
        l_hex VARCHAR2(32 CHAR) := RAWTOHEX(SYS_GUID());
    BEGIN
        RETURN HEXTORAW(
            SUBSTR(l_hex, 1, 12)
            || '7' || SUBSTR(l_hex, 14, 3)
            || '8' || SUBSTR(l_hex, 18, 15)
        );
    END;

    FUNCTION clob_fingerprint(p_value CLOB) RETURN VARCHAR2 IS
        l_offset PLS_INTEGER := 1;
        l_chunk VARCHAR2(3600 BYTE);
        l_chain RAW(32) := HEXTORAW(RPAD('0', 64, '0'));
    BEGIN
        LOOP
            l_chunk := DBMS_LOB.SUBSTR(p_value, 900, l_offset);
            EXIT WHEN l_chunk IS NULL;
            SELECT STANDARD_HASH(
                       RAWTOHEX(l_chain) || ':' || l_chunk,
                       'SHA256'
                   )
              INTO l_chain
              FROM DUAL;
            l_offset := l_offset + LENGTH(l_chunk);
        END LOOP;
        RETURN LOWER(RAWTOHEX(l_chain));
    END;

    FUNCTION build_definition(
        p_display_name VARCHAR2,
        p_selected_json CLOB
    ) RETURN CLOB IS
        l_definition JSON_OBJECT_T := JSON_OBJECT_T();
        l_selected JSON_ARRAY_T := JSON_ARRAY_T.parse(p_selected_json);
        l_steps JSON_ARRAY_T := JSON_ARRAY_T();
        l_check_id VARCHAR2(128 CHAR);

        PROCEDURE append_step(
            p_check_id VARCHAR2,
            p_title VARCHAR2,
            p_tool_id VARCHAR2,
            p_evidence_kind VARCHAR2,
            p_trend_required BOOLEAN
        ) IS
            l_step JSON_OBJECT_T := JSON_OBJECT_T();
            l_check_ids JSON_ARRAY_T := JSON_ARRAY_T();
        BEGIN
            l_check_ids.append(p_check_id);
            l_step.put('title', p_title);
            l_step.put('tool_id', p_tool_id);
            l_step.put('input', JSON_OBJECT_T());
            l_step.put('expected_evidence_kind', p_evidence_kind);
            l_step.put('measurement_semantics', 'CURRENT_ACTIVITY');
            l_step.put('optional', FALSE);
            l_step.put('check_ids', l_check_ids);
            l_step.put('trend_required', p_trend_required);
            l_steps.append(l_step);
        END;
    BEGIN
        FOR i IN 0 .. l_selected.get_size - 1 LOOP
            l_check_id := l_selected.get_string(i);
            CASE l_check_id
                WHEN 'oracle.backup.rman_volume' THEN
                    append_step(l_check_id, 'RMAN 备份量', 'db.backup.recent_jobs', 'DB_BACKUP_RECENT_JOBS', TRUE);
                WHEN 'oracle.backup.space_usage' THEN
                    append_step(l_check_id, '占用空间', 'db.backup.recent_jobs', 'DB_BACKUP_RECENT_JOBS', TRUE);
                WHEN 'oracle.backup.recent_jobs' THEN
                    append_step(l_check_id, '最近成功/失败', 'db.backup.recent_jobs', 'BACKUP_FAILED', TRUE);
                WHEN 'oracle.storage.tablespace_headroom' THEN
                    append_step(l_check_id, '表空间余量', 'db.storage.capacity', 'TABLESPACE', TRUE);
                WHEN 'oracle.storage.datafile_count' THEN
                    append_step(l_check_id, '数据文件数', 'db.storage.capacity', 'TABLESPACE', TRUE);
                WHEN 'oracle.storage.growth' THEN
                    append_step(l_check_id, '增长量', 'db.storage.capacity', 'TABLESPACE', TRUE);
                WHEN 'oracle.storage.undo' THEN
                    append_step(l_check_id, 'UNDO', 'db.storage.undo_usage', 'DB_STORAGE_UNDO_USAGE', TRUE);
                WHEN 'oracle.session.count' THEN
                    append_step(l_check_id, '会话数', 'db.resource.session_utilization', 'DB_RESOURCE_SESSION_UTILIZATION', FALSE);
                WHEN 'oracle.session.long_running' THEN
                    append_step(l_check_id, '长会话', 'db.session.active', 'LONG_SESSION', FALSE);
                WHEN 'oracle.session.long_transaction' THEN
                    append_step(l_check_id, '长事务', 'db.transaction.long_running', 'LONG_TRANSACTION', FALSE);
                WHEN 'oracle.session.lock_wait' THEN
                    append_step(l_check_id, '锁等待', 'db.session.blocking_chain', 'LOCK_WAIT', FALSE);
                WHEN 'oracle.object.invalid' THEN
                    append_step(l_check_id, '无效对象', 'db.objects.invalid_summary', 'INVALID_OBJECT', TRUE);
                WHEN 'oracle.object.sparse_index' THEN
                    append_step(l_check_id, '稀疏索引', 'db.index.health', 'DB_INDEX_HEALTH', TRUE);
                WHEN 'oracle.log.archive_generation' THEN
                    append_step(l_check_id, '归档生成量', 'db.archive.status', 'ARCHIVE_HEADROOM', TRUE);
                WHEN 'oracle.performance.datafile_io' THEN
                    append_step(l_check_id, '数据文件 IO', 'db.instance.performance', 'DB_INSTANCE_PERFORMANCE', TRUE);
                WHEN 'oracle.performance.critical_waits' THEN
                    append_step(l_check_id, '关键等待', 'db.wait.class_summary', 'WAIT_CLASS', TRUE);
                WHEN 'oracle.performance.heavy_sort_sql' THEN
                    append_step(l_check_id, '严重排序 SQL', 'db.sql.top_current', 'TOP_SQL', TRUE);
                WHEN 'oracle.performance.core_parameters' THEN
                    append_step(l_check_id, '核心参数', 'db.instance.parameters', 'DB_INSTANCE_PARAMETERS', TRUE);
                WHEN 'oracle.exadata.health_report' THEN
                    NULL;
                ELSE
                    RAISE_APPLICATION_ERROR(
                        -20069,
                        '既有巡检计划包含未知或未开放检查项：' || l_check_id
                    );
            END CASE;
        END LOOP;

        IF l_steps.get_size = 0 THEN
            RAISE_APPLICATION_ERROR(
                -20070,
                '既有巡检计划没有可执行的数据库取证检查项'
            );
        END IF;

        l_definition.put('schema_version', 'INSPECTION_TEMPLATE.v1');
        l_definition.put('display_name', p_display_name);
        l_definition.put(
            'catalog_hash',
            'db4644a21de690a9902833c3a6c1f7db0f85a07bc9f3fc5a5a429f618c844860'
        );
        l_definition.put('selected_check_ids', l_selected);
        l_definition.put('evidence_steps', l_steps);
        RETURN l_definition.to_clob();
    END;

BEGIN
    SELECT COUNT(*) INTO l_selected_column_count
      FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_OPS_INSPECTION_PLAN'
       AND COLUMN_NAME = 'SELECTED_CHECKS_JSON';

    IF l_selected_column_count = 0 THEN
        RETURN;
    END IF;

    OPEN l_plan_cursor FOR
        'SELECT INSPECTION_PLAN_ID, DOMAIN_ID, DISPLAY_NAME, '
        || 'CREATED_BY, CREATED_AT, '
        || 'JSON_SERIALIZE(SELECTED_CHECKS_JSON RETURNING CLOB) '
        || 'FROM KBOT_OPS_INSPECTION_PLAN '
        || 'WHERE INSPECTION_TEMPLATE_ID IS NULL '
        || 'OR INSPECTION_TEMPLATE_VERSION_ID IS NULL '
        || 'FOR UPDATE';
    LOOP
        FETCH l_plan_cursor INTO
            l_plan_id,
            l_domain_id,
            l_plan_display_name,
            l_created_by,
            l_created_at,
            l_selected_json;
        EXIT WHEN l_plan_cursor%NOTFOUND;
        l_template_id := uuid7_raw();
        l_version_id := uuid7_raw();
        l_template_name := SUBSTR(
            l_plan_display_name || ' · 迁移巡检模板',
            1,
            256
        );
        l_definition := build_definition(
            l_template_name,
            l_selected_json
        );
        l_content_hash := clob_fingerprint(l_definition);

        INSERT INTO KBOT_OPS_INSPECTION_TEMPLATE (
            INSPECTION_TEMPLATE_ID,
            DOMAIN_ID,
            DISPLAY_NAME,
            STATUS,
            CURRENT_VERSION_ID,
            ROW_VERSION,
            CREATED_BY,
            UPDATED_BY,
            CREATED_AT,
            UPDATED_AT
        ) VALUES (
            l_template_id,
            l_domain_id,
            l_template_name,
            'ACTIVE',
            l_version_id,
            1,
            l_created_by,
            'system:schema-30-migration',
            l_created_at,
            SYSTIMESTAMP
        );

        INSERT INTO KBOT_OPS_INSPECTION_TEMPLATE_VER (
            INSPECTION_TEMPLATE_VERSION_ID,
            DOMAIN_ID,
            INSPECTION_TEMPLATE_ID,
            VERSION_NO,
            DEFINITION_JSON,
            CONTENT_HASH,
            CREATED_BY,
            CREATED_AT
        ) VALUES (
            l_version_id,
            l_domain_id,
            l_template_id,
            1,
            JSON(l_definition),
            l_content_hash,
            'system:schema-30-migration',
            SYSTIMESTAMP
        );

        UPDATE KBOT_OPS_INSPECTION_PLAN
           SET INSPECTION_TEMPLATE_ID = l_template_id,
               INSPECTION_TEMPLATE_VERSION_ID = l_version_id,
               UPDATED_BY = 'system:schema-30-migration',
               UPDATED_AT = SYSTIMESTAMP
         WHERE INSPECTION_PLAN_ID = l_plan_id;
    END LOOP;
    CLOSE l_plan_cursor;
END;
/

UPDATE KBOT_OPS_INSPECTION_FIRE fire
   SET (INSPECTION_TEMPLATE_ID, INSPECTION_TEMPLATE_VERSION_ID) = (
       SELECT plan.INSPECTION_TEMPLATE_ID,
              plan.INSPECTION_TEMPLATE_VERSION_ID
         FROM KBOT_OPS_INSPECTION_PLAN plan
        WHERE plan.INSPECTION_PLAN_ID = fire.INSPECTION_PLAN_ID
   )
 WHERE fire.INSPECTION_TEMPLATE_ID IS NULL
    OR fire.INSPECTION_TEMPLATE_VERSION_ID IS NULL;

DECLARE
    l_null_count PLS_INTEGER;
BEGIN
    SELECT
        (SELECT COUNT(*) FROM KBOT_OPS_INSPECTION_PLAN
          WHERE INSPECTION_TEMPLATE_ID IS NULL
             OR INSPECTION_TEMPLATE_VERSION_ID IS NULL)
        +
        (SELECT COUNT(*) FROM KBOT_OPS_INSPECTION_FIRE
          WHERE INSPECTION_TEMPLATE_ID IS NULL
             OR INSPECTION_TEMPLATE_VERSION_ID IS NULL)
      INTO l_null_count
      FROM DUAL;
    IF l_null_count > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20071,
            '巡检模板引用迁移不完整，拒绝收紧 NOT NULL'
        );
    END IF;
END;
/

ALTER TABLE KBOT_OPS_INSPECTION_PLAN MODIFY (
    INSPECTION_TEMPLATE_ID NOT NULL,
    INSPECTION_TEMPLATE_VERSION_ID NOT NULL
);
ALTER TABLE KBOT_OPS_INSPECTION_FIRE MODIFY (
    INSPECTION_TEMPLATE_ID NOT NULL,
    INSPECTION_TEMPLATE_VERSION_ID NOT NULL
);

DECLARE
    PROCEDURE add_constraint_if_missing(
        p_constraint_name VARCHAR2,
        p_statement VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*) INTO l_count FROM USER_CONSTRAINTS
         WHERE CONSTRAINT_NAME = p_constraint_name;
        IF l_count = 0 THEN
            EXECUTE IMMEDIATE p_statement;
        END IF;
    END;

    PROCEDURE create_index_if_missing(
        p_index_name VARCHAR2,
        p_statement VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*) INTO l_count FROM USER_INDEXES
         WHERE INDEX_NAME = p_index_name;
        IF l_count = 0 THEN
            EXECUTE IMMEDIATE p_statement;
        END IF;
    END;
BEGIN
    add_constraint_if_missing(
        'FK_OPS_INSP_PLAN_TEMPLATE',
        'ALTER TABLE KBOT_OPS_INSPECTION_PLAN '
        || 'ADD CONSTRAINT FK_OPS_INSP_PLAN_TEMPLATE '
        || 'FOREIGN KEY (INSPECTION_TEMPLATE_ID) '
        || 'REFERENCES KBOT_OPS_INSPECTION_TEMPLATE '
        || '(INSPECTION_TEMPLATE_ID)'
    );
    add_constraint_if_missing(
        'FK_OPS_INSP_PLAN_TEMPLATE_VER',
        'ALTER TABLE KBOT_OPS_INSPECTION_PLAN '
        || 'ADD CONSTRAINT FK_OPS_INSP_PLAN_TEMPLATE_VER '
        || 'FOREIGN KEY (INSPECTION_TEMPLATE_VERSION_ID) '
        || 'REFERENCES KBOT_OPS_INSPECTION_TEMPLATE_VER '
        || '(INSPECTION_TEMPLATE_VERSION_ID)'
    );
    add_constraint_if_missing(
        'FK_OPS_INSP_FIRE_TEMPLATE',
        'ALTER TABLE KBOT_OPS_INSPECTION_FIRE '
        || 'ADD CONSTRAINT FK_OPS_INSP_FIRE_TEMPLATE '
        || 'FOREIGN KEY (INSPECTION_TEMPLATE_ID) '
        || 'REFERENCES KBOT_OPS_INSPECTION_TEMPLATE '
        || '(INSPECTION_TEMPLATE_ID)'
    );
    add_constraint_if_missing(
        'FK_OPS_INSP_FIRE_TEMPLATE_VER',
        'ALTER TABLE KBOT_OPS_INSPECTION_FIRE '
        || 'ADD CONSTRAINT FK_OPS_INSP_FIRE_TEMPLATE_VER '
        || 'FOREIGN KEY (INSPECTION_TEMPLATE_VERSION_ID) '
        || 'REFERENCES KBOT_OPS_INSPECTION_TEMPLATE_VER '
        || '(INSPECTION_TEMPLATE_VERSION_ID)'
    );
    create_index_if_missing(
        'IX_OPS_INSP_PLAN_TEMPLATE',
        'CREATE INDEX IX_OPS_INSP_PLAN_TEMPLATE '
        || 'ON KBOT_OPS_INSPECTION_PLAN (INSPECTION_TEMPLATE_ID)'
    );
    create_index_if_missing(
        'IX_OPS_INSP_PLAN_TEMPLATE_VER',
        'CREATE INDEX IX_OPS_INSP_PLAN_TEMPLATE_VER '
        || 'ON KBOT_OPS_INSPECTION_PLAN (INSPECTION_TEMPLATE_VERSION_ID)'
    );
    create_index_if_missing(
        'IX_OPS_INSP_FIRE_TEMPLATE',
        'CREATE INDEX IX_OPS_INSP_FIRE_TEMPLATE '
        || 'ON KBOT_OPS_INSPECTION_FIRE (INSPECTION_TEMPLATE_ID)'
    );
    create_index_if_missing(
        'IX_OPS_INSP_FIRE_TEMPLATE_VER',
        'CREATE INDEX IX_OPS_INSP_FIRE_TEMPLATE_VER '
        || 'ON KBOT_OPS_INSPECTION_FIRE (INSPECTION_TEMPLATE_VERSION_ID)'
    );
END;
/

DECLARE
    PROCEDURE drop_constraint_if_present(p_constraint_name VARCHAR2) IS
        l_count PLS_INTEGER;
        l_table_name VARCHAR2(128 CHAR);
    BEGIN
        SELECT COUNT(*), MAX(TABLE_NAME)
          INTO l_count, l_table_name
          FROM USER_CONSTRAINTS
         WHERE CONSTRAINT_NAME = p_constraint_name;
        IF l_count > 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE '
                || DBMS_ASSERT.ENQUOTE_NAME(
                    l_table_name,
                    FALSE
                )
                || ' DROP CONSTRAINT '
                || DBMS_ASSERT.ENQUOTE_NAME(p_constraint_name, FALSE);
        END IF;
    END;

    PROCEDURE drop_column_if_present(
        p_table_name VARCHAR2,
        p_column_name VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*) INTO l_count FROM USER_TAB_COLUMNS
         WHERE TABLE_NAME = p_table_name AND COLUMN_NAME = p_column_name;
        IF l_count > 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE '
                || DBMS_ASSERT.ENQUOTE_NAME(p_table_name, FALSE)
                || ' DROP COLUMN '
                || DBMS_ASSERT.ENQUOTE_NAME(p_column_name, FALSE);
        END IF;
    END;
BEGIN
    drop_constraint_if_present('UK_OPS_INSP_FIRE');
    drop_constraint_if_present('UK_OPS_REPORT_VERSION');
    drop_constraint_if_present('UK_OPS_REPORT_TEMPLATE_NAME');
    drop_constraint_if_present('UK_OPS_REPORT_TEMPLATE_VER');
    drop_constraint_if_present('UK_OPS_REPORT_TEMPLATE_HASH');

    drop_column_if_present('KBOT_OPS_INSPECTION_PLAN', 'TEMPLATE_ID');
    drop_column_if_present('KBOT_OPS_INSPECTION_PLAN', 'TEMPLATE_VERSION');
    drop_column_if_present('KBOT_OPS_INSPECTION_PLAN', 'SELECTED_CHECKS_JSON');
    drop_column_if_present(
        'KBOT_OPS_INSPECTION_PLAN',
        'SCHEDULE_RESOLVER_VERSION'
    );
    drop_column_if_present('KBOT_OPS_INSPECTION_FIRE', 'TEMPLATE_ID');
    drop_column_if_present('KBOT_OPS_INSPECTION_FIRE', 'TEMPLATE_VERSION');
    drop_column_if_present('KBOT_OPS_INSPECTION_FIRE', 'SCHEDULED_FOR_UTC');
END;
/

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
    LOWER(
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_ID), 21, 12)
    ) AS INSPECTION_TEMPLATE_ID,
    LOWER(
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_VERSION_ID), 1, 8) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_VERSION_ID), 9, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_VERSION_ID), 13, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_VERSION_ID), 17, 4) || '-' ||
        SUBSTR(RAWTOHEX(p.INSPECTION_TEMPLATE_VERSION_ID), 21, 12)
    ) AS INSPECTION_TEMPLATE_VERSION_ID,
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
    30 AS SCHEMA_VERSION,
    'aiops-oracle-v20' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_template_table_count PLS_INTEGER;
    l_required_column_count PLS_INTEGER;
    l_legacy_column_count PLS_INTEGER;
    l_business_unique_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*) INTO l_template_table_count
      FROM USER_TABLES
     WHERE TABLE_NAME IN (
        'KBOT_OPS_INSPECTION_TEMPLATE',
        'KBOT_OPS_INSPECTION_TEMPLATE_VER'
     );

    SELECT COUNT(*) INTO l_required_column_count
      FROM USER_TAB_COLUMNS
     WHERE NULLABLE = 'N'
       AND (
            (TABLE_NAME = 'KBOT_OPS_INSPECTION_PLAN'
             AND COLUMN_NAME IN (
                'INSPECTION_TEMPLATE_ID',
                'INSPECTION_TEMPLATE_VERSION_ID'
             ))
         OR (TABLE_NAME = 'KBOT_OPS_INSPECTION_FIRE'
             AND COLUMN_NAME IN (
                'INSPECTION_TEMPLATE_ID',
                'INSPECTION_TEMPLATE_VERSION_ID'
             ))
       );

    SELECT COUNT(*) INTO l_legacy_column_count
      FROM USER_TAB_COLUMNS
     WHERE (
            TABLE_NAME = 'KBOT_OPS_INSPECTION_PLAN'
        AND COLUMN_NAME IN (
            'TEMPLATE_ID', 'TEMPLATE_VERSION',
            'SELECTED_CHECKS_JSON', 'SCHEDULE_RESOLVER_VERSION'
        )
     ) OR (
            TABLE_NAME = 'KBOT_OPS_INSPECTION_FIRE'
        AND COLUMN_NAME IN (
            'TEMPLATE_ID', 'TEMPLATE_VERSION', 'SCHEDULED_FOR_UTC'
        )
     );

    SELECT COUNT(*) INTO l_business_unique_count
      FROM USER_CONSTRAINTS
     WHERE CONSTRAINT_NAME IN (
        'UK_OPS_INSP_FIRE',
        'UK_OPS_REPORT_VERSION',
        'UK_OPS_REPORT_TEMPLATE_NAME',
        'UK_OPS_REPORT_TEMPLATE_VER',
        'UK_OPS_REPORT_TEMPLATE_HASH'
     );

    IF l_schema_version <> 30
       OR l_contract_version <> 'aiops-oracle-v20'
       OR l_template_table_count <> 2
       OR l_required_column_count <> 4
       OR l_legacy_column_count <> 0
       OR l_business_unique_count <> 0 THEN
        RAISE_APPLICATION_ERROR(
            -20072,
            'Schema 30 升级校验失败，请核对巡检模板、计划引用、旧列和业务唯一约束'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 30 / aiops-oracle-v20。'
    );
END;
/
