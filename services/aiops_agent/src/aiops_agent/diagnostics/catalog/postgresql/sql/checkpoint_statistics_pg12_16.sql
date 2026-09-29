SELECT
    checkpoints_timed,
    checkpoints_req AS checkpoints_requested,
    checkpoint_write_time AS checkpoint_write_time_ms,
    checkpoint_sync_time AS checkpoint_sync_time_ms,
    buffers_checkpoint,
    buffers_clean,
    maxwritten_clean,
    buffers_backend,
    buffers_backend_fsync,
    buffers_alloc,
    stats_reset
FROM pg_stat_bgwriter
