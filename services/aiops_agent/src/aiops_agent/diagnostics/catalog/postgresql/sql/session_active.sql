SELECT
    pid AS session_id,
    usename AS username,
    CAST(client_addr AS text) AS client_host,
    datname AS database_name,
    state AS command_name,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - query_start)) AS bigint) AS active_seconds,
    wait_event_type || ':' || wait_event AS session_state
FROM pg_stat_activity
WHERE state = 'active' AND pid <> pg_backend_pid()
ORDER BY query_start
