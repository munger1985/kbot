SELECT
    application_name AS channel_name,
    state AS service_state,
    CAST(EXTRACT(EPOCH FROM replay_lag) AS bigint) AS lag_seconds
FROM pg_stat_replication
ORDER BY application_name
