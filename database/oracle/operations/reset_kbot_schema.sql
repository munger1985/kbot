-- KBot 4.0 开发环境 Schema 重置脚本。
-- 本脚本会永久删除当前用户下全部 KBOT_% 表、视图及其数据。
-- 仅在确认当前连接用户为 KBot 专用 Schema 后执行。
-- 失败后不能依赖事务回滚；应修复原因后再次清理，并从备份恢复数据或重新初始化空 Schema。

SET SERVEROUTPUT ON

DECLARE
    l_schema_name VARCHAR2(128);
    l_view_count PLS_INTEGER := 0;
    l_table_count PLS_INTEGER := 0;
BEGIN
    SELECT SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
      INTO l_schema_name
      FROM dual;

    dbms_output.put_line('正在清理 KBot Schema：' || l_schema_name);

    FOR view_row IN (
        SELECT view_name
        FROM user_views
        WHERE view_name LIKE 'KBOT\_%' ESCAPE '\'
        ORDER BY view_name
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP VIEW '
            || dbms_assert.enquote_name(view_row.view_name, FALSE);
        l_view_count := l_view_count + 1;
        dbms_output.put_line('已删除视图 ' || view_row.view_name);
    END LOOP;

    FOR table_row IN (
        SELECT table_name
        FROM user_tables
        WHERE table_name LIKE 'KBOT\_%' ESCAPE '\'
        ORDER BY table_name
    ) LOOP
        EXECUTE IMMEDIATE
            'DROP TABLE '
            || dbms_assert.enquote_name(table_row.table_name, FALSE)
            || ' CASCADE CONSTRAINTS PURGE';
        l_table_count := l_table_count + 1;
        dbms_output.put_line('已删除表 ' || table_row.table_name);
    END LOOP;

    dbms_output.put_line(
        'KBot Schema 清理完成：视图 ' || l_view_count
        || ' 个，表 ' || l_table_count || ' 张。'
    );
END;
/

-- 验证：执行后应返回 0 行。
SELECT object_name, object_type
FROM user_objects
WHERE object_name LIKE 'KBOT\_%' ESCAPE '\'
  AND object_type IN ('TABLE', 'VIEW')
ORDER BY object_type, object_name;
