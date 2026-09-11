-- 为已有 KBot 4.0 Schema 增加 Model Serving 能力验收列。
-- 影响范围：仅新增 KBOT_AI_MODEL 的验收状态列和检查约束；不会修改模型配置或业务数据。
-- 执行前应停止 Model Serving 目录写入，避免与本次 DDL 并发。
-- 恢复方式：Oracle DDL 会隐式提交；请在执行前完成 Schema 备份。若需要回退，请由 DBA
-- 在确认没有依赖后显式删除本脚本新增列与约束。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

DECLARE
    PROCEDURE add_column_if_missing(
        p_column_name IN VARCHAR2,
        p_definition IN VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*)
          INTO l_count
          FROM USER_TAB_COLUMNS
         WHERE TABLE_NAME = 'KBOT_AI_MODEL'
           AND COLUMN_NAME = UPPER(p_column_name);

        IF l_count = 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE KBOT_AI_MODEL ADD (' ||
                DBMS_ASSERT.SIMPLE_SQL_NAME(p_column_name) || ' ' ||
                p_definition || ')';
            DBMS_OUTPUT.PUT_LINE(
                'KBOT_AI_MODEL.' || p_column_name || ' 已新增。'
            );
        ELSE
            DBMS_OUTPUT.PUT_LINE(
                'KBOT_AI_MODEL.' || p_column_name || ' 已存在，跳过。'
            );
        END IF;
    END;

    PROCEDURE add_constraint_if_missing(
        p_constraint_name IN VARCHAR2,
        p_definition IN VARCHAR2
    ) IS
        l_count PLS_INTEGER;
    BEGIN
        SELECT COUNT(*)
          INTO l_count
          FROM USER_CONSTRAINTS
         WHERE TABLE_NAME = 'KBOT_AI_MODEL'
           AND CONSTRAINT_NAME = UPPER(p_constraint_name);

        IF l_count = 0 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE KBOT_AI_MODEL ADD CONSTRAINT ' ||
                DBMS_ASSERT.SIMPLE_SQL_NAME(p_constraint_name) || ' ' ||
                p_definition;
            DBMS_OUTPUT.PUT_LINE(
                'KBOT_AI_MODEL.' || p_constraint_name || ' 已新增。'
            );
        ELSE
            DBMS_OUTPUT.PUT_LINE(
                'KBOT_AI_MODEL.' || p_constraint_name || ' 已存在，跳过。'
            );
        END IF;
    END;
BEGIN
    add_column_if_missing('SUPPORTS_X_SEARCH', 'NUMBER(1) DEFAULT 0 NOT NULL');
    add_column_if_missing(
        'SUPPORTS_IMAGE_GENERATION', 'NUMBER(1) DEFAULT 0 NOT NULL'
    );
    add_column_if_missing(
        'SUPPORTS_RESPONSES_STREAMING', 'NUMBER(1) DEFAULT 0 NOT NULL'
    );
    add_column_if_missing(
        'CAPABILITY_VERIFIED_AT', 'TIMESTAMP(6) WITH TIME ZONE'
    );
    add_constraint_if_missing(
        'CK_AI_MODEL_X_SEARCH', 'CHECK (SUPPORTS_X_SEARCH IN (0, 1))'
    );
    add_constraint_if_missing(
        'CK_AI_MODEL_IMAGE_GENERATION',
        'CHECK (SUPPORTS_IMAGE_GENERATION IN (0, 1))'
    );
    add_constraint_if_missing(
        'CK_AI_MODEL_RESPONSE_STREAM',
        'CHECK (SUPPORTS_RESPONSES_STREAMING IN (0, 1))'
    );
END;
/

COMMENT ON COLUMN KBOT_AI_MODEL.CAPABILITY_VERIFIED_AT IS
    '受控 Canary 最近一次完成能力验收的时间；未验收模型不得开放扩展入口';
