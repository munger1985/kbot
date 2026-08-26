-- KBot 4.0 AIOps Schema 全量重建脚本。
-- 使用 KBot Schema 所有者在 SQL Developer 中以 Run Script（F5）执行。
-- 本脚本永久删除当前 Schema 内全部 KBOT_OPS_% 表、KBOT_V_OPS_% 视图及其数据。
-- 执行前必须停止 AIOps API、Worker、Scheduler，并备份需要保留的数据。
-- 平台用户、Domain、权限、角色以及 KC Collection 不在删除范围内。
-- Oracle DDL 会自动提交；失败后应修复原因并重新执行本脚本。

WHENEVER OSERROR EXIT FAILURE ROLLBACK
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

SET SERVEROUTPUT ON
SET VERIFY OFF

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
            -20001,
            'AIOps 对象数量错误：表=' || l_table_count || '，视图=' || l_view_count
        );
    END IF;
    IF l_invalid_count <> 0 THEN
        raise_application_error(-20002, '存在无效的 AIOps 对象。');
    END IF;
    IF l_bad_constraint_count <> 0 THEN
        raise_application_error(-20003, '存在未启用或未验证的 AIOps 约束。');
    END IF;
    IF l_bad_index_count <> 0 THEN
        raise_application_error(-20004, '存在无效的 AIOps 索引。');
    END IF;
    IF l_investigation_mode_count <> 1 THEN
        raise_application_error(-20005, 'KBOT_OPS_RUN.INVESTIGATION_MODE 缺失或允许为空。');
    END IF;
    IF l_component <> 'AIOPS'
       OR l_schema_version <> 10
       OR l_contract_version <> 'aiops-oracle-v2' THEN
        raise_application_error(
            -20006,
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
