SELECT
    datid AS database_oid,
    datname AS database_name,
    confl_tablespace AS tablespace_conflicts,
    confl_lock AS lock_conflicts,
    confl_snapshot AS snapshot_conflicts,
    confl_bufferpin AS buffer_pin_conflicts,
    confl_deadlock AS deadlock_conflicts
FROM pg_stat_database_conflicts
WHERE datname IS NOT NULL
ORDER BY datname
