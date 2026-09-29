SELECT
    wal_records,
    wal_fpi,
    wal_bytes,
    wal_buffers_full,
    stats_reset
FROM pg_stat_wal
