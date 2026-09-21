SELECT
    CAST(
        MAX(CASE WHEN variable_name = 'Questions' THEN variable_value END) AS SIGNED
    ) AS questions,
    CAST(
        MAX(CASE WHEN variable_name = 'Uptime' THEN variable_value END) AS SIGNED
    ) AS uptime_seconds,
    ROUND(
        CAST(MAX(CASE WHEN variable_name = 'Questions' THEN variable_value END) AS DECIMAL)
        / NULLIF(
            CAST(MAX(CASE WHEN variable_name = 'Uptime' THEN variable_value END) AS DECIMAL),
            0
        ),
        3
    ) AS questions_per_second,
    CAST(
        MAX(CASE WHEN variable_name = 'Threads_running' THEN variable_value END) AS SIGNED
    ) AS threads_running
FROM performance_schema.global_status
WHERE variable_name IN ('Questions', 'Uptime', 'Threads_running')
