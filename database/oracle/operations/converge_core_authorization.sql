-- 既有 KBot 4.0 Schema 的一次性权限模型收敛脚本。
-- 本脚本只删除已废弃的人类 Agent Grant 表；权限目录和内置角色映射由
-- apply_oracle_schema.py --foundation-only 按核心规范精确收敛。
-- 前置条件：停止全部 KBot 写入进程并完成 Schema 级备份。
-- 敏感数据：脚本不导出、不打印任何身份、凭据或业务数据。
-- 恢复方式：如需回退，必须恢复执行前的 Schema 备份和同版本应用代码。

DECLARE
    PROCEDURE drop_legacy_table(table_name IN VARCHAR2) IS
        table_count NUMBER;
    BEGIN
        SELECT COUNT(*)
          INTO table_count
          FROM USER_TABLES
         WHERE TABLE_NAME = UPPER(table_name);

        IF table_count = 1 THEN
            EXECUTE IMMEDIATE
                'DROP TABLE ' || DBMS_ASSERT.SIMPLE_SQL_NAME(table_name)
                || ' CASCADE CONSTRAINTS PURGE';
        END IF;
    END;
BEGIN
    drop_legacy_table('KBOT_OPS_AGENT_GRANT');
    drop_legacy_table('KBOT_KM_AGENT_GRANT');
END;
/
