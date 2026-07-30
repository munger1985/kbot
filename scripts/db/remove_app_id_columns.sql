-- KBot 4.0 一次性数据库修复：删除旧初始化遗留的 APP_ID 列。
-- 使用当前 KBot Schema 用户执行。本脚本只处理 KBOT_% 表，不触碰 APEX 或 KMCHAT 表。
-- DDL 会隐式提交；执行前请完成数据库备份。

SET SERVEROUTPUT ON

DECLARE
    l_foreign_key_count PLS_INTEGER := 0;
    l_constraint_count PLS_INTEGER := 0;
    l_removed_count PLS_INTEGER := 0;
BEGIN
    -- 先移除直接使用 APP_ID，或引用包含 APP_ID 的父键的外键。
    FOR constraint_row IN (
        SELECT DISTINCT constraint_name, table_name
        FROM user_constraints
        WHERE constraint_type = 'R'
          AND table_name LIKE 'KBOT\_%' ESCAPE '\'
          AND (
              constraint_name IN (
                  SELECT constraint_name
                  FROM user_cons_columns
                  WHERE column_name = 'APP_ID'
              )
              OR r_constraint_name IN (
                  SELECT constraint_name
                  FROM user_cons_columns
                  WHERE column_name = 'APP_ID'
              )
          )
        ORDER BY table_name, constraint_name
    ) LOOP
        EXECUTE IMMEDIATE
            'ALTER TABLE '
            || dbms_assert.enquote_name(constraint_row.table_name, FALSE)
            || ' DROP CONSTRAINT '
            || dbms_assert.enquote_name(
                constraint_row.constraint_name, FALSE
            );
        l_foreign_key_count := l_foreign_key_count + 1;
        dbms_output.put_line(
            '已删除外键 ' || constraint_row.table_name || '.'
            || constraint_row.constraint_name
        );
    END LOOP;

    -- 再移除 APP_ID 所在的主键、唯一键和检查约束，避免阻塞删列。
    FOR constraint_row IN (
        SELECT DISTINCT c.constraint_name, c.table_name
        FROM user_constraints c
        JOIN user_cons_columns cc
          ON cc.constraint_name = c.constraint_name
        WHERE c.constraint_type <> 'R'
          AND c.table_name LIKE 'KBOT\_%' ESCAPE '\'
          AND cc.column_name = 'APP_ID'
        ORDER BY c.table_name, c.constraint_name
    ) LOOP
        EXECUTE IMMEDIATE
            'ALTER TABLE '
            || dbms_assert.enquote_name(constraint_row.table_name, FALSE)
            || ' DROP CONSTRAINT '
            || dbms_assert.enquote_name(
                constraint_row.constraint_name, FALSE
            );
        l_constraint_count := l_constraint_count + 1;
        dbms_output.put_line(
            '已删除约束 ' || constraint_row.table_name || '.'
            || constraint_row.constraint_name
        );
    END LOOP;

    FOR column_row IN (
        SELECT columns.table_name
        FROM user_tab_columns columns
        JOIN user_tables tables
          ON tables.table_name = columns.table_name
        WHERE columns.column_name = 'APP_ID'
          AND columns.table_name LIKE 'KBOT\_%' ESCAPE '\'
        ORDER BY columns.table_name
    ) LOOP
        EXECUTE IMMEDIATE
            'ALTER TABLE '
            || dbms_assert.enquote_name(column_row.table_name, FALSE)
            || ' DROP COLUMN APP_ID';
        l_removed_count := l_removed_count + 1;
        dbms_output.put_line(
            '已删除 ' || column_row.table_name || '.APP_ID'
        );
    END LOOP;

    IF l_removed_count = 0 THEN
        dbms_output.put_line('未发现需要删除的 KBOT_%.APP_ID 列。');
    ELSE
        dbms_output.put_line(
            'APP_ID 清理完成：删除外键 ' || l_foreign_key_count
            || ' 个，删除其他约束 ' || l_constraint_count
            || ' 个，处理表 ' || l_removed_count || ' 张。'
        );
    END IF;
END;
/

-- 验证：执行后应返回 0 行。
SELECT table_name, column_name
FROM user_tab_columns columns
WHERE columns.column_name = 'APP_ID'
  AND columns.table_name LIKE 'KBOT\_%' ESCAPE '\'
  AND columns.table_name IN (SELECT table_name FROM user_tables)
ORDER BY columns.table_name;
