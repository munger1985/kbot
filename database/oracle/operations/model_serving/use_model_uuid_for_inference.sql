-- 仅供现有本地库的维护窗口使用：执行前备份并停止使用此 Schema 的进程。
-- 先核验并转换 Agent 记忆索引画像，再删除旧调用名。不会删除模型行或向量。
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

DECLARE
    old_column_count NUMBER;
    new_column_count NUMBER;
    alias_column_count NUMBER;
    unmatched NUMBER;
BEGIN
    SELECT COUNT(*) INTO old_column_count FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_AGENT_MEMORY_INDEX_PROFILE'
       AND COLUMN_NAME = 'EMBEDDING_MODEL_NAME';
    SELECT COUNT(*) INTO new_column_count FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_AGENT_MEMORY_INDEX_PROFILE'
       AND COLUMN_NAME = 'EMBEDDING_MODEL_ID';
    SELECT COUNT(*) INTO alias_column_count FROM USER_TAB_COLUMNS
     WHERE TABLE_NAME = 'KBOT_AI_MODEL' AND COLUMN_NAME = 'SERVED_MODEL_NAME';
    IF old_column_count + new_column_count <> 1 THEN
        RAISE_APPLICATION_ERROR(-20001, '记忆索引画像表不符合切换前提');
    END IF;
    IF old_column_count = 1 THEN
        IF alias_column_count <> 1 THEN
            RAISE_APPLICATION_ERROR(-20002, '缺少旧模型映射，停止转换');
        END IF;
        -- 同一事务校验旧值及摘要。若已有部分数据转换，则只接受摘要匹配的 UUID。
        EXECUTE IMMEDIATE q'[
            SELECT COUNT(*) FROM KBOT_AGENT_MEMORY_INDEX_PROFILE p
             WHERE 1 <> (
                SELECT COUNT(*) FROM KBOT_AI_MODEL m
                 CROSS APPLY (
                    SELECT LOWER(REGEXP_REPLACE(RAWTOHEX(m.MODEL_ID),
                        '^(.{8})(.{4})(.{4})(.{4})(.{12})$', '\1-\2-\3-\4-\5')) AS UUID_TEXT
                    FROM DUAL
                 ) ids
                 WHERE m.CATEGORY = 2 AND p.NORMALIZATION = 'L2' AND (
                    (p.EMBEDDING_MODEL_NAME = m.SERVED_MODEL_NAME
                         AND p.CONFIG_SHA256 = LOWER(RAWTOHEX(STANDARD_HASH(
                           '{"embedding_dimension":' || TO_CHAR(p.EMBEDDING_DIMENSION, 'TM9') ||
                           ',"embedding_model_name":"' || m.SERVED_MODEL_NAME ||
                           '","normalization":"L2"}', 'SHA256'))))
                    OR (p.EMBEDDING_MODEL_NAME = ids.UUID_TEXT
                         AND p.CONFIG_SHA256 = LOWER(RAWTOHEX(STANDARD_HASH(
                           '{"embedding_dimension":' || TO_CHAR(p.EMBEDDING_DIMENSION, 'TM9') ||
                           ',"embedding_model_id":"' || ids.UUID_TEXT ||
                           '","normalization":"L2"}', 'SHA256'))))
                 )
             )
        ]' INTO unmatched;
        IF unmatched <> 0 THEN
            RAISE_APPLICATION_ERROR(-20003, '存在无唯一模型映射或摘要不匹配的记忆索引画像；未修改任何配置');
        END IF;
        EXECUTE IMMEDIATE q'[
            MERGE INTO KBOT_AGENT_MEMORY_INDEX_PROFILE p
            USING (
                SELECT src.INDEX_PROFILE_ID, LOWER(REGEXP_REPLACE(RAWTOHEX(model.MODEL_ID),
                    '^(.{8})(.{4})(.{4})(.{4})(.{12})$', '\1-\2-\3-\4-\5')) AS UUID_TEXT
                FROM KBOT_AGENT_MEMORY_INDEX_PROFILE src
                JOIN KBOT_AI_MODEL model ON src.EMBEDDING_MODEL_NAME = model.SERVED_MODEL_NAME
                WHERE model.CATEGORY = 2 AND src.NORMALIZATION = 'L2'
                  AND src.CONFIG_SHA256 = LOWER(RAWTOHEX(STANDARD_HASH(
                    '{"embedding_dimension":' || TO_CHAR(src.EMBEDDING_DIMENSION, 'TM9') ||
                    ',"embedding_model_name":"' || model.SERVED_MODEL_NAME ||
                    '","normalization":"L2"}', 'SHA256')))
            ) m
            ON (p.INDEX_PROFILE_ID = m.INDEX_PROFILE_ID)
            WHEN MATCHED THEN UPDATE SET
                p.EMBEDDING_MODEL_NAME = m.UUID_TEXT,
                p.CONFIG_SHA256 = LOWER(RAWTOHEX(STANDARD_HASH(
                    '{"embedding_dimension":' || TO_CHAR(p.EMBEDDING_DIMENSION, 'TM9') ||
                    ',"embedding_model_id":"' || m.UUID_TEXT ||
                    '","normalization":"L2"}', 'SHA256')))
        ]';
        EXECUTE IMMEDIATE 'ALTER TABLE KBOT_AGENT_MEMORY_INDEX_PROFILE RENAME COLUMN EMBEDDING_MODEL_NAME TO EMBEDDING_MODEL_ID';
    END IF;
    EXECUTE IMMEDIATE 'ALTER TABLE KBOT_AGENT_MEMORY_INDEX_PROFILE MODIFY EMBEDDING_MODEL_ID VARCHAR2(36 CHAR)';
    IF alias_column_count = 1 THEN
        EXECUTE IMMEDIATE 'ALTER TABLE KBOT_AI_MODEL DROP COLUMN SERVED_MODEL_NAME';
    END IF;
END;
/
