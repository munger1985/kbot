-- 修复已创建 Schema 中 Delegation 子运行唯一键对 NULL 的错误限制。
-- 使用当前 KBot Schema 用户执行；可重复执行，且不修改业务数据。

SET SERVEROUTPUT ON
WHENEVER SQLERROR EXIT SQL.SQLCODE

DECLARE
    l_constraint_count PLS_INTEGER := 0;
    l_index_count PLS_INTEGER := 0;
BEGIN
    SELECT COUNT(*)
      INTO l_constraint_count
      FROM user_constraints
     WHERE table_name = 'KBOT_AGENT_DELEGATION'
       AND constraint_name = 'UK_AGENT_DELEGATION_CHILD';

    IF l_constraint_count = 1 THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_AGENT_DELEGATION '
            || 'DROP CONSTRAINT UK_AGENT_DELEGATION_CHILD';
        dbms_output.put_line('已删除旧 Delegation 子运行唯一约束。');
    END IF;

    SELECT COUNT(*)
      INTO l_index_count
      FROM user_indexes
     WHERE table_name = 'KBOT_AGENT_DELEGATION'
       AND index_name = 'UX_AGENT_DELEGATION_CHILD';

    IF l_index_count = 0 THEN
        EXECUTE IMMEDIATE
            'CREATE UNIQUE INDEX UX_AGENT_DELEGATION_CHILD '
            || 'ON KBOT_AGENT_DELEGATION ('
            || 'CASE WHEN CHILD_RUN_ID IS NOT NULL '
            || 'THEN TARGET_SERVICE END, '
            || 'CASE WHEN CHILD_RUN_ID IS NOT NULL '
            || 'THEN CHILD_RUN_ID END)';
        dbms_output.put_line('已创建条件唯一索引。');
    END IF;
END;
/

-- 已返回子运行标识的记录不可重复；尚未提交的记录不参与该唯一性校验。
SELECT index_name, uniqueness
  FROM user_indexes
 WHERE table_name = 'KBOT_AGENT_DELEGATION'
   AND index_name = 'UX_AGENT_DELEGATION_CHILD';
