SELECT
    channel_name,
    MAX(CASE
        WHEN last_applied_transaction_original_commit_timestamp > '1970-01-01'
         AND last_applied_transaction_end_apply_timestamp >= last_applied_transaction_original_commit_timestamp
        THEN TIMESTAMPDIFF(MICROSECOND,
             last_applied_transaction_original_commit_timestamp,
             last_applied_transaction_end_apply_timestamp) / 1000000.0
    END) AS latest_applied_transaction_latency_seconds,
    MAX(CASE
        WHEN service_state = 'ON'
         AND applying_transaction <> ''
         AND applying_transaction_original_commit_timestamp > '1970-01-01'
        THEN GREATEST(0, TIMESTAMPDIFF(MICROSECOND,
             applying_transaction_original_commit_timestamp, NOW(6)) / 1000000.0)
    END) AS applying_transaction_age_seconds,
    SUM(CASE WHEN service_state = 'ON' AND applying_transaction <> '' THEN 1 ELSE 0 END) AS applying_worker_count,
    MAX(NULLIF(last_applied_transaction_end_apply_timestamp, '0000-00-00 00:00:00')) AS last_applied_at
FROM performance_schema.replication_applier_status_by_worker
GROUP BY channel_name
ORDER BY channel_name
