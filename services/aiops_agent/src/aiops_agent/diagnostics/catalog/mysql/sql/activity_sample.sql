SELECT
    t.PROCESSLIST_ID AS mysql_connection_number,
    t.THREAD_ID AS mysql_thread_number,
    es.DIGEST AS digest,
    es.CURRENT_SCHEMA AS schema_name,
    t.PROCESSLIST_USER AS username,
    t.PROCESSLIST_HOST AS client,
    t.PROCESSLIST_COMMAND AS command_name,
    t.PROCESSLIST_STATE AS session_state,
    st.EVENT_NAME AS stage_name,
    ew.EVENT_NAME AS wait_name,
    CASE WHEN trx.THREAD_ID IS NULL THEN 0 ELSE 1 END AS transaction_active,
    CASE WHEN dlw.REQUESTING_THREAD_ID IS NULL THEN 0 ELSE 1 END AS lock_waiting
FROM performance_schema.threads t
LEFT JOIN performance_schema.events_statements_current es
    ON es.THREAD_ID = t.THREAD_ID
LEFT JOIN performance_schema.events_stages_current st
    ON st.THREAD_ID = t.THREAD_ID
LEFT JOIN performance_schema.events_waits_current ew
    ON ew.THREAD_ID = t.THREAD_ID
LEFT JOIN performance_schema.events_transactions_current trx
    ON trx.THREAD_ID = t.THREAD_ID
LEFT JOIN performance_schema.data_lock_waits dlw
    ON dlw.REQUESTING_THREAD_ID = t.THREAD_ID
WHERE t.TYPE = 'FOREGROUND'
  AND t.PROCESSLIST_ID IS NOT NULL
  AND t.PROCESSLIST_COMMAND <> 'Sleep'
ORDER BY t.PROCESSLIST_TIME DESC, t.THREAD_ID
LIMIT :limit
