SELECT
    pid AS session_id,
    usename AS username,
    datname AS database_name,
    xact_start AS transaction_started_at,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - xact_start)) AS bigint) AS elapsed_seconds,
    state AS transaction_state,
    CAST(backend_xid AS text) AS backend_xid,
    CAST(backend_xmin AS text) AS backend_xmin,
    wait_event_type,
    wait_event,
    query_start AS query_started_at,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - query_start)) AS bigint) AS query_elapsed_seconds,
    query AS query_text
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
  AND EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - xact_start)) >= :min_seconds
ORDER BY elapsed_seconds DESC
