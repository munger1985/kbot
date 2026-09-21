WITH current_count AS (
    SELECT COUNT(*) AS current_sessions
    FROM pg_stat_activity
)
SELECT
    'connections' AS resource_name,
    CAST(current_sessions AS bigint) AS current_utilization,
    CAST(NULL AS bigint) AS max_utilization,
    current_setting('max_connections') AS limit_value,
    ROUND(
        100.0 * current_sessions
        / CAST(current_setting('max_connections') AS numeric),
        3
    ) AS utilization_percent
FROM current_count
