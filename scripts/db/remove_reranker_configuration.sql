SET SERVEROUTPUT ON

DECLARE
    l_collection_count PLS_INTEGER := 0;
    l_km_agent_count PLS_INTEGER := 0;
    l_kr_agent_count PLS_INTEGER := 0;
    l_ops_agent_count PLS_INTEGER := 0;
    l_model_count PLS_INTEGER := 0;
BEGIN
    UPDATE KBOT_KC_COLLECTION
       SET MODELS_JSON = JSON_TRANSFORM(
               MODELS_JSON,
               REMOVE '$.reranker'
           ),
           UPDATED_BY = 'DBA:remove-reranker',
           UPDATED_AT = SYSTIMESTAMP,
           ROW_VERSION = ROW_VERSION + 1
     WHERE JSON_EXISTS(MODELS_JSON, '$.reranker');

    l_collection_count := SQL%ROWCOUNT;

    UPDATE KBOT_KM_AGENT_VERSION
       SET MODELS_JSON = JSON_TRANSFORM(
               MODELS_JSON,
               REMOVE '$.reranker'
           )
     WHERE JSON_EXISTS(MODELS_JSON, '$.reranker');

    l_km_agent_count := SQL%ROWCOUNT;

    UPDATE KBOT_KR_AGENT_VERSION
       SET MODELS_JSON = JSON_TRANSFORM(
               MODELS_JSON,
               REMOVE '$.reranker'
           )
     WHERE JSON_EXISTS(MODELS_JSON, '$.reranker');

    l_kr_agent_count := SQL%ROWCOUNT;

    UPDATE KBOT_OPS_AGENT_VERSION
       SET MODELS_JSON = JSON_TRANSFORM(
               MODELS_JSON,
               REMOVE '$.reranker'
           )
     WHERE JSON_EXISTS(MODELS_JSON, '$.reranker');

    l_ops_agent_count := SQL%ROWCOUNT;

    DELETE FROM KBOT_AI_MODEL
     WHERE MODEL_ID = HEXTORAW('01A017F2A1CC70299D94E1C474B8AB75')
        OR SERVED_MODEL_NAME = 'bge-reranker-cpu';

    l_model_count := SQL%ROWCOUNT;
    COMMIT;

    DBMS_OUTPUT.PUT_LINE(
        '已移除 KC reranker 绑定：' || l_collection_count || ' 条'
    );
    DBMS_OUTPUT.PUT_LINE(
        '已清理 KM Agent reranker 绑定：' || l_km_agent_count || ' 条'
    );
    DBMS_OUTPUT.PUT_LINE(
        '已清理知识 Agent reranker 绑定：' || l_kr_agent_count || ' 条'
    );
    DBMS_OUTPUT.PUT_LINE(
        '已清理 AIOps Agent reranker 绑定：' || l_ops_agent_count || ' 条'
    );
    DBMS_OUTPUT.PUT_LINE(
        '已删除 Reranker 模型记录：' || l_model_count || ' 条'
    );
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
/

SELECT COLLECTION_ID, DISPLAY_NAME, MODELS_JSON
  FROM KBOT_KC_COLLECTION
 WHERE JSON_EXISTS(MODELS_JSON, '$.reranker');

SELECT MODEL_ID, SERVED_MODEL_NAME, DISPLAY_NAME
  FROM KBOT_AI_MODEL
 WHERE MODEL_ID = HEXTORAW('01A017F2A1CC70299D94E1C474B8AB75')
    OR SERVED_MODEL_NAME = 'bge-reranker-cpu';

SELECT 'KBOT_KM_AGENT_VERSION' AS SOURCE_TABLE, AGENT_VERSION_ID
  FROM KBOT_KM_AGENT_VERSION
 WHERE JSON_EXISTS(MODELS_JSON, '$.reranker')
UNION ALL
SELECT 'KBOT_KR_AGENT_VERSION', AGENT_VERSION_ID
  FROM KBOT_KR_AGENT_VERSION
 WHERE JSON_EXISTS(MODELS_JSON, '$.reranker')
UNION ALL
SELECT 'KBOT_OPS_AGENT_VERSION', AGENT_VERSION_ID
  FROM KBOT_OPS_AGENT_VERSION
 WHERE JSON_EXISTS(MODELS_JSON, '$.reranker');
