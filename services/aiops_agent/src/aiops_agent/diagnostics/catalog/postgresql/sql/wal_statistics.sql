SELECT
    wal_records,
    wal_fpi,
    wal_bytes,
    wal_buffers_full,
    wal_write,
    wal_sync,
    wal_write_time AS wal_write_time_ms,
    wal_sync_time AS wal_sync_time_ms,
    stats_reset
FROM pg_stat_wal
