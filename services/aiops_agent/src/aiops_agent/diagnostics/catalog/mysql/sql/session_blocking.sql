SELECT
    waiting_pid AS waiting_session_id,
    waiting_query AS waiting_statement,
    blocking_pid AS blocking_session_id,
    blocking_query AS blocking_statement,
    wait_age_secs AS wait_seconds
FROM sys.innodb_lock_waits
ORDER BY wait_age_secs DESC
