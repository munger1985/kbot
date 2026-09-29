SELECT
    pid AS session_id,
    datname AS database_name,
    usename AS username,
    application_name,
    CAST(client_addr AS text) AS client_address,
    state AS session_state,
    wait_event_type,
    wait_event,
    CAST(query_id AS text) AS query_id,
    query_start,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - query_start)) AS bigint)
        AS query_elapsed_seconds,
    LEFT(query, 4096) AS query_text
FROM pg_stat_activity
WHERE pid <> pg_backend_pid()
  AND state IS DISTINCT FROM 'idle'
ORDER BY query_start NULLS LAST, pid
LIMIT :limit
