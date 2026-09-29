SELECT
    datid AS database_oid,
    datname AS database_name,
    usesysid AS user_oid,
    usename AS username,
    COUNT(*) AS session_count,
    MIN(query_start) AS oldest_query_started_at,
    MAX(CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - query_start)) AS bigint))
        AS maximum_query_seconds,
    wait_event_type,
    wait_event,
    LEFT(MIN(query), 4096) AS query_text
FROM pg_stat_activity
WHERE pid <> pg_backend_pid()
  AND state = 'active'
  AND query_start IS NOT NULL
GROUP BY datid, datname, usesysid, usename, wait_event_type, wait_event
ORDER BY maximum_query_seconds DESC, session_count DESC, database_oid
LIMIT :limit
