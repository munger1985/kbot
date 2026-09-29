SELECT
    pid AS session_id,
    backend_type,
    datid AS database_oid,
    datname AS database_name,
    usesysid AS user_oid,
    usename AS username,
    application_name,
    CAST(client_addr AS text) AS client_address,
    state AS session_state,
    wait_event_type,
    wait_event,
    CAST(backend_xid AS text) AS backend_xid,
    CAST(backend_xmin AS text) AS backend_xmin,
    xact_start AS transaction_started_at,
    query_start AS query_started_at,
    state_change,
    cardinality(pg_blocking_pids(pid)) > 0 AS lock_waiting
FROM pg_stat_activity
WHERE pid <> pg_backend_pid()
ORDER BY xact_start NULLS LAST, query_start NULLS LAST, pid
LIMIT :limit
