SELECT
    pid AS session_id,
    usename AS username,
    CAST(client_addr AS text) AS client_host,
    datname AS database_name,
    state AS session_state,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - state_change)) AS bigint) AS idle_seconds,
    query AS query_text
FROM pg_stat_activity
WHERE state IN ('idle in transaction', 'idle in transaction (aborted)')
    AND pid <> pg_backend_pid()
ORDER BY state_change
