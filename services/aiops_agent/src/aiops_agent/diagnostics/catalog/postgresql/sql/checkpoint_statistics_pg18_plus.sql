SELECT
    num_timed AS checkpoints_timed,
    num_requested AS checkpoints_requested,
    num_done AS checkpoints_done,
    restartpoints_timed,
    restartpoints_req AS restartpoints_requested,
    restartpoints_done,
    write_time AS checkpoint_write_time_ms,
    sync_time AS checkpoint_sync_time_ms,
    buffers_written AS buffers_checkpoint,
    slru_written,
    stats_reset
FROM pg_stat_checkpointer
