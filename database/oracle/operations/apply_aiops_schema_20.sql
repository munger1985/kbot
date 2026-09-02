-- AIOps Schema 19 -> 20 原地升级。
-- 影响范围：只为两张现有表补列、回填安全默认值并更新 Schema 版本视图。
-- 数据保护：不删除表、不删除业务行、不修改既有命令、参数和审计哈希。
-- 前置条件：停止 AIOps API、Worker 和 DB Executor，并确认没有执行中的受控动作。
-- 历史 Proposal：保留审计数据，但统一标记为仅人工处理，不能进入新版 Executor。
-- 历史 Agent–Target：保留绑定关系，但受控执行默认关闭，需在 Agent 页面重新明确授权。
-- 恢复方式：Oracle DDL 会隐式提交；执行前必须完成 Schema 备份，失败时按备份恢复。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_running_executions PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF l_schema_version <> 19
       OR l_contract_version <> 'aiops-oracle-v9' THEN
        RAISE_APPLICATION_ERROR(
            -20020,
            '只允许从 AIOPS Schema 19 / aiops-oracle-v9 升级'
        );
    END IF;

    SELECT COUNT(*)
      INTO l_running_executions
      FROM KBOT_OPS_EXECUTION
     WHERE STATUS IN ('CREATED', 'SUBMITTED', 'RUNNING');

    IF l_running_executions > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20021,
            '仍有未终结的受控动作 Execution，停止升级'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE('Schema 19 前置检查通过。');
END;
/

DECLARE
    PROCEDURE add_column_if_missing(
        p_table_name IN VARCHAR2,
        p_column_name IN VARCHAR2,
        p_definition IN VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*)
          INTO l_count
          FROM USER_TAB_COLUMNS
         WHERE TABLE_NAME = UPPER(p_table_name)
           AND COLUMN_NAME = UPPER(p_column_name);

        IF l_count = 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE ' || DBMS_ASSERT.SIMPLE_SQL_NAME(p_table_name) ||
                ' ADD (' || DBMS_ASSERT.SIMPLE_SQL_NAME(p_column_name) ||
                ' ' || p_definition || ')';
            DBMS_OUTPUT.PUT_LINE(
                p_table_name || '.' || p_column_name || ' 已新增。'
            );
        ELSE
            DBMS_OUTPUT.PUT_LINE(
                p_table_name || '.' || p_column_name || ' 已存在，跳过。'
            );
        END IF;
    END;
BEGIN
    add_column_if_missing(
        'KBOT_OPS_CHANGE_PROPOSAL',
        'ACTION_FAMILY',
        'VARCHAR2(64 CHAR)'
    );
    add_column_if_missing(
        'KBOT_OPS_CHANGE_PROPOSAL',
        'EFFECT_CLASS',
        'VARCHAR2(48 CHAR)'
    );
    add_column_if_missing(
        'KBOT_OPS_CHANGE_PROPOSAL',
        'EXECUTION_MODE',
        'VARCHAR2(40 CHAR)'
    );
    add_column_if_missing(
        'KBOT_OPS_CHANGE_PROPOSAL',
        'EXECUTOR_KIND',
        'VARCHAR2(16 CHAR)'
    );
    add_column_if_missing(
        'KBOT_OPS_CHANGE_PROPOSAL',
        'CANONICAL_OBJECT_REF_JSON',
        'JSON'
    );
    add_column_if_missing(
        'KBOT_OPS_CHANGE_PROPOSAL',
        'LOCK_IMPACT',
        'VARCHAR2(1000 CHAR)'
    );
    add_column_if_missing(
        'KBOT_OPS_CHANGE_PROPOSAL',
        'ESTIMATED_DURATION_SECONDS',
        'NUMBER(10)'
    );
    add_column_if_missing(
        'KBOT_OPS_AGENT_VERSION_TARGET',
        'CONTROLLED_ACTION_POLICY_JSON',
        'JSON'
    );
END;
/

-- Schema 19 只有旧版会话终止 Proposal。旧 Snapshot 不满足新版冻结合同，
-- 因此只保留审计语义，明确禁止审批后自动下发。
UPDATE KBOT_OPS_CHANGE_PROPOSAL
   SET ACTION_FAMILY = COALESCE(
           ACTION_FAMILY,
           CASE
               WHEN ACTION_TEMPLATE_ID = 'db.session.terminate'
                   THEN 'SESSION_TRANSACTION'
               ELSE 'LEGACY_CHANGE'
           END
       ),
       EFFECT_CLASS = COALESCE(
           EFFECT_CLASS,
           CASE
               WHEN ACTION_TEMPLATE_ID = 'db.session.terminate'
                   THEN 'SESSION_CONTROL'
               ELSE 'LEGACY_CHANGE'
           END
       ),
       EXECUTION_MODE = COALESCE(EXECUTION_MODE, 'MANUAL_ONLY'),
       EXECUTOR_KIND = COALESCE(EXECUTOR_KIND, 'NONE'),
       LOCK_IMPACT = COALESCE(
           LOCK_IMPACT,
           'Schema 20 前历史提案，仅保留审计，不允许自动执行'
       ),
       ESTIMATED_DURATION_SECONDS = COALESCE(
           ESTIMATED_DURATION_SECONDS,
           0
       )
 WHERE ACTION_FAMILY IS NULL
    OR EFFECT_CLASS IS NULL
    OR EXECUTION_MODE IS NULL
    OR EXECUTOR_KIND IS NULL
    OR LOCK_IMPACT IS NULL
    OR ESTIMATED_DURATION_SECONDS IS NULL;

-- 空策略在运行时解释为受控执行关闭；不继承旧 Target Binding 的修改权限。
UPDATE KBOT_OPS_AGENT_VERSION_TARGET
   SET CONTROLLED_ACTION_POLICY_JSON = JSON_OBJECT(
           'enabled' VALUE 'false' FORMAT JSON
           RETURNING JSON
       )
 WHERE CONTROLLED_ACTION_POLICY_JSON IS NULL;

COMMIT;

DECLARE
    l_invalid_proposals PLS_INTEGER;
    l_invalid_targets PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_invalid_proposals
      FROM KBOT_OPS_CHANGE_PROPOSAL
     WHERE ACTION_FAMILY IS NULL
        OR EFFECT_CLASS IS NULL
        OR EXECUTION_MODE IS NULL
        OR EXECUTOR_KIND IS NULL
        OR LOCK_IMPACT IS NULL
        OR ESTIMATED_DURATION_SECONDS IS NULL;

    SELECT COUNT(*)
      INTO l_invalid_targets
      FROM KBOT_OPS_AGENT_VERSION_TARGET
     WHERE CONTROLLED_ACTION_POLICY_JSON IS NULL;

    IF l_invalid_proposals > 0 OR l_invalid_targets > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20022,
            'Schema 20 新列回填不完整，停止添加 NOT NULL 约束'
        );
    END IF;
END;
/

ALTER TABLE KBOT_OPS_CHANGE_PROPOSAL MODIFY (
    ACTION_FAMILY NOT NULL,
    EFFECT_CLASS NOT NULL,
    EXECUTION_MODE NOT NULL,
    EXECUTOR_KIND NOT NULL,
    LOCK_IMPACT NOT NULL,
    ESTIMATED_DURATION_SECONDS NOT NULL
);

ALTER TABLE KBOT_OPS_AGENT_VERSION_TARGET MODIFY (
    CONTROLLED_ACTION_POLICY_JSON NOT NULL
);

DECLARE
    l_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_OPS_CHANGE_PROPOSAL'
       AND CONSTRAINT_NAME = 'CK_OPS_PROPOSAL_MODE';

    IF l_count = 0 THEN
        EXECUTE IMMEDIATE q'[
            ALTER TABLE KBOT_OPS_CHANGE_PROPOSAL
            ADD CONSTRAINT CK_OPS_PROPOSAL_MODE CHECK (
                EXECUTION_MODE IN (
                    'EXECUTABLE_AFTER_APPROVAL', 'MANUAL_ONLY'
                )
                AND EXECUTOR_KIND IN ('DATABASE', 'EXTERNAL', 'NONE')
                AND ESTIMATED_DURATION_SECONDS >= 0
            )
        ]';
    END IF;
END;
/

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    20 AS SCHEMA_VERSION,
    'aiops-oracle-v10' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_nullable_columns PLS_INTEGER;
    l_constraint_count PLS_INTEGER;
    l_invalid_objects PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*)
      INTO l_nullable_columns
      FROM USER_TAB_COLUMNS
     WHERE (
            TABLE_NAME = 'KBOT_OPS_CHANGE_PROPOSAL'
        AND COLUMN_NAME IN (
            'ACTION_FAMILY',
            'EFFECT_CLASS',
            'EXECUTION_MODE',
            'EXECUTOR_KIND',
            'LOCK_IMPACT',
            'ESTIMATED_DURATION_SECONDS'
        )
        AND NULLABLE <> 'N'
     ) OR (
            TABLE_NAME = 'KBOT_OPS_AGENT_VERSION_TARGET'
        AND COLUMN_NAME = 'CONTROLLED_ACTION_POLICY_JSON'
        AND NULLABLE <> 'N'
     );

    SELECT COUNT(*)
      INTO l_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_OPS_CHANGE_PROPOSAL'
       AND CONSTRAINT_NAME = 'CK_OPS_PROPOSAL_MODE'
       AND STATUS = 'ENABLED'
       AND VALIDATED = 'VALIDATED';

    SELECT COUNT(*)
      INTO l_invalid_objects
      FROM USER_OBJECTS
     WHERE OBJECT_NAME IN (
         'KBOT_OPS_CHANGE_PROPOSAL',
         'KBOT_OPS_AGENT_VERSION_TARGET',
         'KBOT_V_OPS_SCHEMA_VERSION'
     )
       AND STATUS <> 'VALID';

    IF l_schema_version <> 20
       OR l_contract_version <> 'aiops-oracle-v10'
       OR l_nullable_columns <> 0
       OR l_constraint_count <> 1
       OR l_invalid_objects <> 0 THEN
        RAISE_APPLICATION_ERROR(-20023, 'AIOps Schema 20 升级校验失败');
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已原地升级到 20；历史数据已保留，受控执行默认关闭。'
    );
END;
/
