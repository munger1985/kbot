SELECT
    pid AS session_id,
    usename AS username,
    datname AS database_name,
    LEFT(query, 4096) AS sql_text,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - query_start)) AS bigint) AS elapsed_seconds,
    wait_event
FROM pg_stat_activity
WHERE datname = current_database()
  AND pid <> pg_backend_pid()
  AND state = 'active'
ORDER BY query_start NULLS LAST, pid
LIMIT :limit
