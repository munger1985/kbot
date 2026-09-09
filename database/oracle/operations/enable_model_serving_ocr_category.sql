-- 为已有 KBot 4.0 Schema 启用 Model Serving OCR（类别 6）。
-- 影响范围：删除 KBOT_AI_MODEL 旧的类别检查约束；不修改任何模型或业务数据。
-- 执行前应停止 Model Serving 目录写入，避免与本次 DDL 并发。
-- 模型类别由 Model Serving 的应用契约和 Provider 校验管理，不在数据库层固化枚举。

SET DEFINE OFF;
SET SERVEROUTPUT ON;
WHENEVER SQLERROR EXIT SQL.SQLCODE;

DECLARE
    l_constraint_count PLS_INTEGER;
BEGIN
    SELECT COUNT(*)
      INTO l_constraint_count
      FROM USER_CONSTRAINTS
     WHERE TABLE_NAME = 'KBOT_AI_MODEL'
       AND CONSTRAINT_NAME = 'CK_AI_MODEL_CATEGORY'
       AND CONSTRAINT_TYPE = 'C';

    IF l_constraint_count = 1 THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_AI_MODEL DROP CONSTRAINT CK_AI_MODEL_CATEGORY';
        DBMS_OUTPUT.PUT_LINE('已删除 KBOT_AI_MODEL 的旧类别检查约束。');
    ELSE
        DBMS_OUTPUT.PUT_LINE('KBOT_AI_MODEL 未定义类别检查约束，无需变更。');
    END IF;
END;
/
