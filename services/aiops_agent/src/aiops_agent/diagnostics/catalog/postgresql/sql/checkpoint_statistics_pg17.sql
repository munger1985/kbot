SELECT
    num_timed AS checkpoints_timed,
    num_requested AS checkpoints_requested,
    restartpoints_timed,
    restartpoints_req AS restartpoints_requested,
    restartpoints_done,
    write_time AS checkpoint_write_time_ms,
    sync_time AS checkpoint_sync_time_ms,
    buffers_written AS buffers_checkpoint,
    stats_reset
FROM pg_stat_checkpointer
