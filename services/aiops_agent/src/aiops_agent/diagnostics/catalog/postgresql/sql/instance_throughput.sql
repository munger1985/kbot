SELECT
    datname AS database_name,
    xact_commit,
    xact_rollback,
    CASE
        WHEN xact_commit + xact_rollback = 0 THEN 0
        ELSE ROUND(
            100.0 * xact_rollback / (xact_commit + xact_rollback),
            3
        )
    END AS rollback_percent,
    numbackends AS backends,
    CAST(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - stats_reset)) AS bigint) AS sample_seconds,
    ROUND(
        (xact_commit + xact_rollback)
        / NULLIF(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - stats_reset)), 0),
        3
    ) AS transactions_per_second
FROM pg_stat_database
WHERE datname IS NOT NULL
ORDER BY datname
