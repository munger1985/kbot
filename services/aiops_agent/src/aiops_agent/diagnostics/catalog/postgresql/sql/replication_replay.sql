SELECT
    CASE WHEN pg_is_in_recovery() THEN 'STANDBY' ELSE 'PRIMARY' END AS database_role,
    CASE WHEN pg_is_in_recovery() THEN pg_is_wal_replay_paused() END AS replay_paused,
    CASE WHEN pg_is_in_recovery() THEN CAST(pg_last_wal_receive_lsn() AS text) END AS received_lsn,
    CASE WHEN pg_is_in_recovery() THEN CAST(pg_last_wal_replay_lsn() AS text) END AS replay_lsn,
    CASE WHEN pg_is_in_recovery() THEN
        GREATEST(0, pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()))
    END AS receive_replay_gap_bytes,
    CASE WHEN pg_is_in_recovery() THEN pg_last_xact_replay_timestamp() END AS last_replayed_transaction_at,
    CASE WHEN pg_is_in_recovery() THEN
        EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - pg_last_xact_replay_timestamp())
    END AS last_replay_age_seconds
