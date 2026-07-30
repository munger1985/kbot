-- 将历史模型目录的顶层 EMBEDDING_DIMENSION 迁移至 MODEL_PARAMS。
-- 仅适用于已按旧版 KBot 4.0 建表的 Oracle Schema；可重复执行。

DECLARE
    v_column_count NUMBER;
    v_constraint_count NUMBER;
BEGIN
    SELECT COUNT(*)
      INTO v_column_count
      FROM user_tab_columns
     WHERE table_name = 'KBOT_AI_MODEL'
       AND column_name = 'EMBEDDING_DIMENSION';

    IF v_column_count = 1 THEN
        UPDATE KBOT_AI_MODEL
           SET model_params = JSON_MERGEPATCH(
               COALESCE(model_params, JSON_OBJECT(RETURNING JSON)),
               JSON_OBJECT(
                   'embedding_dimension' VALUE embedding_dimension
                   RETURNING JSON
               )
           )
         WHERE embedding_dimension IS NOT NULL;

        SELECT COUNT(*)
          INTO v_constraint_count
          FROM user_constraints
         WHERE table_name = 'KBOT_AI_MODEL'
           AND constraint_name = 'CK_AI_MODEL_EMBED_DIM';

        IF v_constraint_count = 1 THEN
            EXECUTE IMMEDIATE
                'ALTER TABLE KBOT_AI_MODEL DROP CONSTRAINT CK_AI_MODEL_EMBED_DIM';
        END IF;

        EXECUTE IMMEDIATE
            'ALTER TABLE KBOT_AI_MODEL DROP COLUMN EMBEDDING_DIMENSION';
    END IF;
END;
/

COMMIT;

-- 应返回 0，确认旧的顶层字段已移除。
SELECT COUNT(*) AS remaining_embedding_dimension_columns
  FROM user_tab_columns
 WHERE table_name = 'KBOT_AI_MODEL'
   AND column_name = 'EMBEDDING_DIMENSION';
