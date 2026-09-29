SELECT
    application_name AS channel_name,
    state AS service_state,
    CAST(client_addr AS text) AS client_address,
    client_hostname,
    client_port,
    backend_start AS backend_started_at,
    sync_state,
    CAST(sent_lsn AS text) AS sent_lsn,
    CAST(write_lsn AS text) AS write_lsn,
    CAST(flush_lsn AS text) AS flush_lsn,
    CAST(replay_lsn AS text) AS replay_lsn,
    EXTRACT(EPOCH FROM write_lag) AS write_lag_seconds,
    EXTRACT(EPOCH FROM flush_lag) AS flush_lag_seconds,
    EXTRACT(EPOCH FROM replay_lag) AS replay_lag_seconds
FROM pg_stat_replication
ORDER BY application_name
