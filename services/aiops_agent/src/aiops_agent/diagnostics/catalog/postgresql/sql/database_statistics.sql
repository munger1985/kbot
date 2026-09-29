SELECT
    datid AS database_oid,
    datname AS database_name,
    numbackends AS backends,
    xact_commit,
    xact_rollback,
    blks_read,
    blks_hit,
    tup_returned,
    tup_fetched,
    tup_inserted,
    tup_updated,
    tup_deleted,
    conflicts,
    temp_files,
    temp_bytes,
    deadlocks,
    checksum_failures,
    checksum_last_failure,
    stats_reset
FROM pg_stat_database
WHERE datname IS NOT NULL
ORDER BY datname
