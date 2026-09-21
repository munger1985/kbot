SELECT
    channel_name,
    CAST(
        TIMESTAMPDIFF(
            SECOND,
            last_processed_transaction_original_commit_timestamp,
            last_processed_transaction_immediate_commit_timestamp
        ) AS SIGNED
    ) AS lag_seconds
FROM performance_schema.replication_applier_status_by_coordinator
ORDER BY channel_name
