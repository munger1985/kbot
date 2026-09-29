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
    stats_reset
FROM pg_stat_database
WHERE datname IS NOT NULL
ORDER BY datname
