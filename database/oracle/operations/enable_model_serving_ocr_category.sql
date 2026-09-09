-- 为已有 KBot 4.0 Schema 启用 Model Serving OCR（类别 6）。
-- 影响范围：仅重建 KBOT_AI_MODEL 的类别检查约束；不修改任何模型或业务数据。
-- 执行前应停止 Model Serving 目录写入，避免与本次 DDL 并发。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK;

DECLARE
    l_constraint_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_AI_MODEL'
       AND CONSTRAINT_NAME = 'CK_AI_MODEL_CATEGORY'
       AND CONSTRAINT_TYPE = 'C';

    IF l_constraint_count = 0 THEN
        RAISE_APPLICATION_ERROR(
            -20060,
            '未找到 KBOT_AI_MODEL.CK_AI_MODEL_CATEGORY，拒绝执行升级'
        );
    END IF;
END;
/

ALTER TABLE KBOT_AI_MODEL DROP CONSTRAINT CK_AI_MODEL_CATEGORY;

ALTER TABLE KBOT_AI_MODEL
    ADD CONSTRAINT CK_AI_MODEL_CATEGORY CHECK (CATEGORY IN (1, 2, 3, 5, 6));

DECLARE
    l_constraint_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_AI_MODEL'
       AND CONSTRAINT_NAME = 'CK_AI_MODEL_CATEGORY'
       AND CONSTRAINT_TYPE = 'C'
       AND STATUS = 'ENABLED'
       AND VALIDATED = 'VALIDATED'
       AND SEARCH_CONDITION_VC LIKE '%6%';

    IF l_constraint_count <> 1 THEN
        RAISE_APPLICATION_ERROR(
            -20061,
            'KBOT_AI_MODEL OCR 类别约束校验失败'
        );
    END IF;
    DBMS_OUTPUT.PUT_LINE('Model Serving OCR 类别 6 已启用。');
END;
/
