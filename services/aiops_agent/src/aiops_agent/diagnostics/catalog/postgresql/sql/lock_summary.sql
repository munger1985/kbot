SELECT
    locktype AS lock_type,
    mode AS lock_mode,
    granted,
    COUNT(*) AS lock_count,
    COUNT(DISTINCT pid) AS session_count
FROM pg_locks
GROUP BY locktype, mode, granted
ORDER BY granted, lock_count DESC, locktype, mode
