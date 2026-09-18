-- KBot 4.0 Data Query：将 Assistant 问数切换为 Agent 版本自动绑定与内部查询护栏。
--
-- 执行方式：使用 SQL Developer 的 Run Script（F5）执行本文件。
-- 执行前：停止 Data Query API、Data Query Worker 和 Main API，避免迁移期间产生新 Run。
-- 影响范围：
--   1. Agent Binding 不再强制依赖 Policy；
--   2. 同一 Agent 版本与语义模型只保留一条投影；
--   3. Run 的 POLICY_SNAPSHOT_JSON 改为 GUARDRAIL_SNAPSHOT_JSON；
--   4. 已有 Policy 快照中的 budget 会转换为直接的 Guardrail 快照。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

DECLARE
    V_COUNT NUMBER;
    V_DUPLICATE_COUNT NUMBER;

    PROCEDURE REQUIRE_TABLE(P_TABLE_NAME VARCHAR2) IS
    BEGIN
        SELECT COUNT(*)
          INTO V_COUNT
          FROM USER_TABLES
         WHERE TABLE_NAME = UPPER(P_TABLE_NAME);

        IF V_COUNT <> 1 THEN
            RAISE_APPLICATION_ERROR(
                -20001,
                '缺少必要表：' || UPPER(P_TABLE_NAME)
            );
        END IF;
    END;

    FUNCTION COLUMN_EXISTS(
        P_TABLE_NAME VARCHAR2,
        P_COLUMN_NAME VARCHAR2
    ) RETURN BOOLEAN IS
    BEGIN
        SELECT COUNT(*)
          INTO V_COUNT
          FROM USER_TAB_COLUMNS
         WHERE TABLE_NAME = UPPER(P_TABLE_NAME)
           AND COLUMN_NAME = UPPER(P_COLUMN_NAME);
        RETURN V_COUNT = 1;
    END;

    FUNCTION CONSTRAINT_EXISTS(P_CONSTRAINT_NAME VARCHAR2) RETURN BOOLEAN IS
    BEGIN
        SELECT COUNT(*)
          INTO V_COUNT
          FROM USER_CONSTRAINTS
         WHERE CONSTRAINT_NAME = UPPER(P_CONSTRAINT_NAME);
        RETURN V_COUNT = 1;
    END;
BEGIN
    REQUIRE_TABLE('KBOT_DQ_AGENT_BINDING');
    REQUIRE_TABLE('KBOT_DQ_RUN');

    IF CONSTRAINT_EXISTS('FK_DQ_BIND_POLICY') THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_DQ_AGENT_BINDING ' ||
            'DROP CONSTRAINT FK_DQ_BIND_POLICY';
        DBMS_OUTPUT.PUT_LINE('已删除 Agent Binding 到 Policy 的外键。');
    ELSE
        DBMS_OUTPUT.PUT_LINE('Agent Binding 到 Policy 的外键不存在，跳过。');
    END IF;

    IF NOT COLUMN_EXISTS('KBOT_DQ_AGENT_BINDING', 'POLICY_BINDING_ID') THEN
        RAISE_APPLICATION_ERROR(
            -20002,
            'KBOT_DQ_AGENT_BINDING 缺少 POLICY_BINDING_ID，当前结构不符合迁移前提'
        );
    END IF;

    EXECUTE IMMEDIATE
        'ALTER TABLE KBOT_DQ_AGENT_BINDING ' ||
        'MODIFY (POLICY_BINDING_ID NULL)';
    DBMS_OUTPUT.PUT_LINE('POLICY_BINDING_ID 已改为可空。');

    -- 旧设计允许同一版本与模型按不同 Policy 留下多条停用记录。
    -- 新设计只保留一条投影：优先保留 ACTIVE，其次保留最近更新的记录。
    DELETE FROM KBOT_DQ_AGENT_BINDING
     WHERE ROWID IN (
        SELECT BINDING_ROWID
          FROM (
            SELECT ROWID AS BINDING_ROWID,
                   ROW_NUMBER() OVER (
                       PARTITION BY DOMAIN_ID,
                                    CONSUMER_APP_ID,
                                    AGENT_ID,
                                    AGENT_VERSION_ID,
                                    SEMANTIC_MODEL_ID
                       ORDER BY CASE STATUS WHEN 'ACTIVE' THEN 0 ELSE 1 END,
                                UPDATED_AT DESC,
                                AGENT_BINDING_ID DESC
                   ) AS ROW_NO
              FROM KBOT_DQ_AGENT_BINDING
          )
         WHERE ROW_NO > 1
     );
    DBMS_OUTPUT.PUT_LINE('已清理重复 Agent Binding：' || SQL%ROWCOUNT || ' 条。');

    IF NOT CONSTRAINT_EXISTS('UK_DQ_BINDING_VERSION_MODEL') THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_DQ_AGENT_BINDING ADD CONSTRAINT ' ||
            'UK_DQ_BINDING_VERSION_MODEL UNIQUE (' ||
            'DOMAIN_ID, CONSUMER_APP_ID, AGENT_ID, AGENT_VERSION_ID, ' ||
            'SEMANTIC_MODEL_ID)';
        DBMS_OUTPUT.PUT_LINE('已创建 Agent 版本与语义模型唯一约束。');
    ELSE
        DBMS_OUTPUT.PUT_LINE('Agent Binding 唯一约束已存在，跳过。');
    END IF;

    IF COLUMN_EXISTS('KBOT_DQ_RUN', 'POLICY_SNAPSHOT_JSON')
       AND NOT COLUMN_EXISTS('KBOT_DQ_RUN', 'GUARDRAIL_SNAPSHOT_JSON') THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_DQ_RUN RENAME COLUMN ' ||
            'POLICY_SNAPSHOT_JSON TO GUARDRAIL_SNAPSHOT_JSON';
        DBMS_OUTPUT.PUT_LINE('已将 Run Policy 快照列重命名为 Guardrail 快照列。');
    ELSIF COLUMN_EXISTS('KBOT_DQ_RUN', 'POLICY_SNAPSHOT_JSON')
          AND COLUMN_EXISTS('KBOT_DQ_RUN', 'GUARDRAIL_SNAPSHOT_JSON') THEN
        RAISE_APPLICATION_ERROR(
            -20003,
            'KBOT_DQ_RUN 同时存在新旧快照列，请先人工确认数据后再迁移'
        );
    ELSIF NOT COLUMN_EXISTS('KBOT_DQ_RUN', 'GUARDRAIL_SNAPSHOT_JSON') THEN
        RAISE_APPLICATION_ERROR(
            -20004,
            'KBOT_DQ_RUN 缺少可迁移的 Policy/Guardrail 快照列'
        );
    ELSE
        DBMS_OUTPUT.PUT_LINE('Guardrail 快照列已存在，跳过重命名。');
    END IF;

    -- Oracle JSON 类型使用动态 SQL，确保脚本可在旧列名结构上完成编译。
    EXECUTE IMMEDIATE q'[
        UPDATE KBOT_DQ_RUN
           SET GUARDRAIL_SNAPSHOT_JSON = JSON_OBJECT(
                   'max_rows' VALUE COALESCE(
                       JSON_VALUE(
                           GUARDRAIL_SNAPSHOT_JSON,
                           '$.budget.max_rows' RETURNING NUMBER
                       ),
                       1000
                   ),
                   'max_result_bytes' VALUE COALESCE(
                       JSON_VALUE(
                           GUARDRAIL_SNAPSHOT_JSON,
                           '$.budget.max_result_bytes' RETURNING NUMBER
                       ),
                       1048576
                   ),
                   'statement_timeout_seconds' VALUE COALESCE(
                       JSON_VALUE(
                           GUARDRAIL_SNAPSHOT_JSON,
                           '$.budget.statement_timeout_seconds' RETURNING NUMBER
                       ),
                       30
                   ),
                   'max_concurrent_runs' VALUE COALESCE(
                       JSON_VALUE(
                           GUARDRAIL_SNAPSHOT_JSON,
                           '$.budget.max_concurrent_runs' RETURNING NUMBER
                       ),
                       4
                   )
                   RETURNING JSON
               )
         WHERE JSON_EXISTS(GUARDRAIL_SNAPSHOT_JSON, '$.budget')
    ]';
    DBMS_OUTPUT.PUT_LINE('已转换旧 Run 快照：' || SQL%ROWCOUNT || ' 条。');

    SELECT COUNT(*)
      INTO V_DUPLICATE_COUNT
      FROM (
        SELECT 1
          FROM KBOT_DQ_AGENT_BINDING
         GROUP BY DOMAIN_ID,
                  CONSUMER_APP_ID,
                  AGENT_ID,
                  AGENT_VERSION_ID,
                  SEMANTIC_MODEL_ID
        HAVING COUNT(*) > 1
      );

    IF V_DUPLICATE_COUNT <> 0 THEN
        RAISE_APPLICATION_ERROR(
            -20005,
            '迁移后仍存在重复 Agent Binding：' || V_DUPLICATE_COUNT || ' 组'
        );
    END IF;

    COMMIT;
    DBMS_OUTPUT.PUT_LINE('Data Query Agent Binding v2 表结构迁移完成。');
END;
/

PROMPT === 迁移结果检查 ===

SELECT COLUMN_NAME, NULLABLE, DATA_TYPE
  FROM USER_TAB_COLUMNS
 WHERE TABLE_NAME = 'KBOT_DQ_AGENT_BINDING'
   AND COLUMN_NAME = 'POLICY_BINDING_ID';

SELECT CONSTRAINT_NAME, CONSTRAINT_TYPE, STATUS
  FROM USER_CONSTRAINTS
 WHERE TABLE_NAME = 'KBOT_DQ_AGENT_BINDING'
   AND CONSTRAINT_NAME IN (
       'FK_DQ_BIND_POLICY',
       'UK_DQ_BINDING_VERSION_MODEL'
   )
 ORDER BY CONSTRAINT_NAME;

SELECT COLUMN_NAME, DATA_TYPE
  FROM USER_TAB_COLUMNS
 WHERE TABLE_NAME = 'KBOT_DQ_RUN'
   AND COLUMN_NAME IN (
       'POLICY_SNAPSHOT_JSON',
       'GUARDRAIL_SNAPSHOT_JSON'
   )
 ORDER BY COLUMN_NAME;

SELECT COUNT(*) AS DUPLICATE_BINDING_GROUPS
  FROM (
    SELECT 1
      FROM KBOT_DQ_AGENT_BINDING
     GROUP BY DOMAIN_ID,
              CONSUMER_APP_ID,
              AGENT_ID,
              AGENT_VERSION_ID,
              SEMANTIC_MODEL_ID
    HAVING COUNT(*) > 1
  );

PROMPT 预期：POLICY_BINDING_ID 的 NULLABLE=Y；仅存在 UK_DQ_BINDING_VERSION_MODEL；
PROMPT       KBOT_DQ_RUN 仅存在 GUARDRAIL_SNAPSHOT_JSON；重复绑定组数为 0。
