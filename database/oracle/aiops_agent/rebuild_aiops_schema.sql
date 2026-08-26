-- KBot 4.0 AIOps Schema 全量重建脚本。
-- 使用 KBot Schema 所有者在 SQLcl 或 SQL*Plus 中以脚本模式执行。
-- 本脚本永久删除当前 Schema 内全部 KBOT_OPS_% 表、KBOT_V_OPS_% 视图及其数据。
-- 执行前必须停止 AIOps API、Worker、Scheduler，并备份需要保留的数据。
-- 脚本拒绝错误的 PDB/Schema，也拒绝删除被非 AIOps 表引用的对象。
-- Oracle DDL 会自动提交；失败后应修复原因并重新执行本脚本。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

SET SERVEROUTPUT ON
SET VERIFY OFF
SET DEFINE ON

PROMPT === AIOps Schema 重建目标确认 ===

ACCEPT expected_pdb CHAR PROMPT '输入预期 PDB 名称： '
ACCEPT expected_schema CHAR PROMPT '输入预期 Schema 名称： '
ACCEPT services_stopped CHAR PROMPT '确认 AIOps API、Worker、Scheduler 已停止（输入 STOPPED）： '

DECLARE
    l_actual_pdb VARCHAR2(128);
    l_actual_schema VARCHAR2(128);
BEGIN
    SELECT
        SYS_CONTEXT('USERENV', 'CON_NAME'),
        SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
      INTO l_actual_pdb, l_actual_schema
      FROM dual;

    dbms_output.put_line('当前 PDB：' || l_actual_pdb);
    dbms_output.put_line('当前 Schema：' || l_actual_schema);

    IF UPPER(TRIM('&expected_pdb')) <> UPPER(l_actual_pdb) THEN
        raise_application_error(-20001, 'PDB 与预期不一致，拒绝重建。');
    END IF;
    IF UPPER(TRIM('&expected_schema')) <> UPPER(l_actual_schema) THEN
        raise_application_error(-20002, 'Schema 与预期不一致，拒绝重建。');
    END IF;
    IF UPPER(TRIM('&services_stopped')) <> 'STOPPED' THEN
        raise_application_error(-20003, '未确认 AIOps 服务已停止，拒绝重建。');
    END IF;
END;
/

PROMPT === AIOps Schema 重建预检 ===

DECLARE
    l_table_count PLS_INTEGER := 0;
    l_view_count PLS_INTEGER := 0;
    l_row_count NUMBER;
    l_external_fk_count PLS_INTEGER := 0;
BEGIN
    dbms_output.put_line('将永久删除以下 AIOps 表及其数据：');
    FOR table_row IN (
        SELECT table_name
        FROM user_tables
        WHERE table_name LIKE 'KBOT\_OPS\_%' ESCAPE '\'
        ORDER BY table_name
    ) LOOP
        EXECUTE IMMEDIATE
            'SELECT COUNT(*) FROM '
            || dbms_assert.enquote_name(table_row.table_name, FALSE)
            INTO l_row_count;
        l_table_count := l_table_count + 1;
        dbms_output.put_line('  ' || table_row.table_name || ': ' || l_row_count || ' 行');
    END LOOP;

    FOR view_row IN (
        SELECT view_name
        FROM user_views
        WHERE view_name LIKE 'KBOT\_V\_OPS\_%' ESCAPE '\'
        ORDER BY view_name
    ) LOOP
        l_view_count := l_view_count + 1;
        dbms_output.put_line('  视图 ' || view_row.view_name);
    END LOOP;

    SELECT COUNT(*)
      INTO l_external_fk_count
      FROM user_constraints child_constraint
      JOIN user_constraints parent_constraint
        ON parent_constraint.constraint_name = child_constraint.r_constraint_name
       AND parent_constraint.owner = child_constraint.r_owner
     WHERE child_constraint.constraint_type = 'R'
       AND parent_constraint.table_name LIKE 'KBOT\_OPS\_%' ESCAPE '\'
       AND child_constraint.table_name NOT LIKE 'KBOT\_OPS\_%' ESCAPE '\';

    IF l_external_fk_count > 0 THEN
        raise_application_error(
            -20004,
            '检测到 ' || l_external_fk_count
            || ' 个非 AIOps 表外键引用 KBOT_OPS_% 表，拒绝重建。'
        );
    END IF;

    dbms_output.put_line(
        '预检通过：将删除 ' || l_table_count || ' 张表、'
        || l_view_count || ' 个视图。'
    );
END;
/

ACCEPT rebuild_confirmation CHAR PROMPT '输入 REBUILD_AIOPS 以永久删除上述数据并部署 Schema Version 10： '

DECLARE
BEGIN
    IF UPPER(TRIM('&rebuild_confirmation')) <> 'REBUILD_AIOPS' THEN
        raise_application_error(-20005, '未收到 REBUILD_AIOPS 确认，已终止。');
    END IF;
END;
/

PROMPT === 正在删除旧 AIOps 视图和表 ===

DECLARE
BEGIN
    FOR view_row IN (
        SELECT view_name
        FROM user_views
        WHERE view_name LIKE 'KBOT\_V\_OPS\_%' ESCAPE '\'
        ORDER BY view_name
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP VIEW ' || dbms_assert.enquote_name(view_row.view_name, FALSE);
        dbms_output.put_line('已删除视图 ' || view_row.view_name);
    END LOOP;

    FOR table_row IN (
        SELECT table_name
        FROM user_tables
        WHERE table_name LIKE 'KBOT\_OPS\_%' ESCAPE '\'
        ORDER BY table_name
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP TABLE '
            || dbms_assert.enquote_name(table_row.table_name, FALSE)
            || ' CASCADE CONSTRAINTS PURGE';
        dbms_output.put_line('已删除表 ' || table_row.table_name);
    END LOOP;
END;
/

PROMPT === 正在执行当前规范 AIOps DDL ===

@@001_ops_roots.sql
@@002_ops_runtime.sql
@@003_ops_change.sql
@@004_ops_inspection.sql
@@005_ops_messaging.sql
@@006_ops_fks_views.sql
@@007_ops_agents.sql
@@008_ops_conversations_reports.sql

PROMPT === 正在验证 AIOps Schema ===

DECLARE
    l_table_count PLS_INTEGER;
    l_view_count PLS_INTEGER;
    l_invalid_count PLS_INTEGER;
    l_bad_constraint_count PLS_INTEGER;
    l_bad_index_count PLS_INTEGER;
    l_investigation_mode_count PLS_INTEGER;
    l_component VARCHAR2(32);
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64);
BEGIN
    SELECT COUNT(*)
      INTO l_table_count
      FROM user_tables
     WHERE table_name LIKE 'KBOT\_OPS\_%' ESCAPE '\';

    SELECT COUNT(*)
      INTO l_view_count
      FROM user_views
     WHERE view_name LIKE 'KBOT\_V\_OPS\_%' ESCAPE '\';

    SELECT COUNT(*)
      INTO l_invalid_count
      FROM user_objects
     WHERE (
            object_name LIKE 'KBOT\_OPS\_%' ESCAPE '\'
         OR object_name LIKE 'KBOT\_V\_OPS\_%' ESCAPE '\'
     )
       AND object_type IN ('TABLE', 'VIEW', 'INDEX')
       AND status <> 'VALID';

    SELECT COUNT(*)
      INTO l_bad_constraint_count
      FROM user_constraints
     WHERE table_name LIKE 'KBOT\_OPS\_%' ESCAPE '\'
       AND (status <> 'ENABLED' OR validated <> 'VALIDATED');

    SELECT COUNT(*)
      INTO l_bad_index_count
      FROM user_indexes
     WHERE table_name LIKE 'KBOT\_OPS\_%' ESCAPE '\'
       AND status <> 'VALID';

    SELECT COUNT(*)
      INTO l_investigation_mode_count
      FROM user_tab_columns
     WHERE table_name = 'KBOT_OPS_RUN'
       AND column_name = 'INVESTIGATION_MODE'
       AND nullable = 'N';

    SELECT component, schema_version, contract_version
      INTO l_component, l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION;

    IF l_table_count <> 34 OR l_view_count <> 10 THEN
        raise_application_error(
            -20006,
            'AIOps 对象数量错误：表=' || l_table_count || '，视图=' || l_view_count
        );
    END IF;
    IF l_invalid_count <> 0 THEN
        raise_application_error(-20007, '存在无效的 AIOps 对象。');
    END IF;
    IF l_bad_constraint_count <> 0 THEN
        raise_application_error(-20008, '存在未启用或未验证的 AIOps 约束。');
    END IF;
    IF l_bad_index_count <> 0 THEN
        raise_application_error(-20009, '存在无效的 AIOps 索引。');
    END IF;
    IF l_investigation_mode_count <> 1 THEN
        raise_application_error(-20010, 'KBOT_OPS_RUN.INVESTIGATION_MODE 缺失或允许为空。');
    END IF;
    IF l_component <> 'AIOPS'
       OR l_schema_version <> 10
       OR l_contract_version <> 'aiops-oracle-v2' THEN
        raise_application_error(
            -20011,
            'AIOps Schema 合同错误：'
            || l_component || '/' || l_schema_version || '/' || l_contract_version
        );
    END IF;

    dbms_output.put_line(
        '验证通过：34 张表、10 个视图，Schema Version 10，合同 aiops-oracle-v2。'
    );
END;
/

SELECT component, schema_version, contract_version
FROM KBOT_V_OPS_SCHEMA_VERSION;

PROMPT === AIOps Schema 重建完成；启动服务后检查 AIOps /ready ===
