-- AIOps Schema 27 -> 28 原地升级。
-- 影响范围：移除 AIOps 业务表的命名 CHECK 约束，业务规则回归应用层合同。
-- 数据保护：不删除表、不删除或改写任何业务行。
-- 前置条件：当前必须是 Schema 27 / aiops-oracle-v17，或已部分/全部升级到 Schema 28。
-- 恢复方式：Oracle DDL 会隐式提交；如需回退，应从变更前备份恢复旧约束和版本视图。
-- 并发要求：执行前停止 AIOps API、Worker、Scheduler 和 DB Executor；脚本允许中断后续跑。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
SET SQLBLANKLINES ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

ALTER SESSION SET TIME_ZONE = '+00:00';
ALTER SESSION SET DDL_LOCK_TIMEOUT = 60;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    IF NOT (
        (l_schema_version = 27
         AND l_contract_version = 'aiops-oracle-v17')
        OR
        (l_schema_version = 28
         AND l_contract_version = 'aiops-oracle-v18')
    ) THEN
        RAISE_APPLICATION_ERROR(
            -20062,
            '只允许从 AIOPS Schema 27 / aiops-oracle-v17 升级，'
            || '或续跑 Schema 28 / aiops-oracle-v18'
        );
    END IF;
END;
/

DECLARE
    l_locked_table VARCHAR2(128 CHAR);
    l_locked_table_count PLS_INTEGER := 0;
BEGIN
    FOR constraint_row IN (
        SELECT TABLE_NAME, CONSTRAINT_NAME
          FROM USER_CONSTRAINTS
         WHERE TABLE_NAME LIKE 'KBOT\_OPS\_%' ESCAPE '\'
           AND CONSTRAINT_TYPE = 'C'
           AND GENERATED = 'USER NAME'
         ORDER BY TABLE_NAME, CONSTRAINT_NAME
    ) LOOP
        IF l_locked_table = constraint_row.TABLE_NAME THEN
            CONTINUE;
        END IF;
        BEGIN
            EXECUTE IMMEDIATE
                'ALTER TABLE '
                || DBMS_ASSERT.ENQUOTE_NAME(
                    constraint_row.TABLE_NAME,
                    FALSE
                )
                || ' DROP CONSTRAINT '
                || DBMS_ASSERT.ENQUOTE_NAME(
                    constraint_row.CONSTRAINT_NAME,
                    FALSE
                );
            DBMS_OUTPUT.PUT_LINE(
                '已移除业务 CHECK：'
                || constraint_row.TABLE_NAME
                || '.'
                || constraint_row.CONSTRAINT_NAME
            );
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE = -54 THEN
                    l_locked_table := constraint_row.TABLE_NAME;
                    l_locked_table_count := l_locked_table_count + 1;
                    DBMS_OUTPUT.PUT_LINE(
                        '等待 60 秒后仍被 DML 锁定，已跳过本表：'
                        || constraint_row.TABLE_NAME
                    );
                ELSE
                    RAISE;
                END IF;
        END;
    END LOOP;

    IF l_locked_table_count > 0 THEN
        RAISE_APPLICATION_ERROR(
            -20064,
            '仍有 '
            || l_locked_table_count
            || ' 张 AIOps 表被活动事务锁定；停止相关服务后重新执行本脚本即可续跑'
        );
    END IF;
END;
/

CREATE OR REPLACE VIEW KBOT_V_OPS_SCHEMA_VERSION AS
SELECT
    'AIOPS' AS COMPONENT,
    28 AS SCHEMA_VERSION,
    'aiops-oracle-v18' AS CONTRACT_VERSION
FROM DUAL;

DECLARE
    l_schema_version NUMBER;
    l_contract_version VARCHAR2(64 CHAR);
    l_business_check_count PLS_INTEGER;
BEGIN
    SELECT SCHEMA_VERSION, CONTRACT_VERSION
      INTO l_schema_version, l_contract_version
      FROM KBOT_V_OPS_SCHEMA_VERSION
     WHERE COMPONENT = 'AIOPS';

    SELECT COUNT(*)
      INTO l_business_check_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME LIKE 'KBOT\_OPS\_%' ESCAPE '\'
       AND CONSTRAINT_TYPE = 'C'
       AND GENERATED = 'USER NAME';

    IF l_schema_version <> 28
       OR l_contract_version <> 'aiops-oracle-v18'
       OR l_business_check_count <> 0 THEN
        RAISE_APPLICATION_ERROR(
            -20063,
            'Schema 28 升级校验失败，请核对业务 CHECK 和版本视图'
        );
    END IF;

    DBMS_OUTPUT.PUT_LINE(
        'AIOps Schema 已升级到 28 / aiops-oracle-v18。'
    );
END;
/
