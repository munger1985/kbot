SELECT
    slot_name,
    slot_type,
    database AS database_name,
    active,
    active_pid,
    CAST(restart_lsn AS text) AS restart_lsn,
    CAST(confirmed_flush_lsn AS text) AS confirmed_flush_lsn,
    CASE
        WHEN restart_lsn IS NULL THEN NULL
        ELSE CAST(pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) AS numeric)
    END AS retained_wal_bytes,
    CAST(xmin AS text) AS xmin,
    CAST(catalog_xmin AS text) AS catalog_xmin
FROM pg_replication_slots
ORDER BY retained_wal_bytes DESC NULLS LAST, slot_name
