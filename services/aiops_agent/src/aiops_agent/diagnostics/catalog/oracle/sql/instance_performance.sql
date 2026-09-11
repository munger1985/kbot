WITH connection_context AS (
    SELECT TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID')) AS container_number
    FROM dual
), visible_metrics AS (
    SELECT
        metric_name,
        value,
        metric_unit,
        intsize_csec
    FROM v$sysmetric
    CROSS JOIN connection_context
    WHERE connection_context.container_number = 1
    UNION ALL
    SELECT
        metric_name,
        value,
        metric_unit,
        intsize_csec
    FROM v$con_sysmetric
    CROSS JOIN connection_context
    WHERE connection_context.container_number <> 1
      AND con_id = connection_context.container_number
), ranked_metrics AS (
    SELECT
        metric_name,
        value,
        metric_unit,
        intsize_csec,
        ROW_NUMBER() OVER (
            PARTITION BY metric_name
            ORDER BY intsize_csec DESC
        ) AS window_rank
    FROM visible_metrics
    WHERE metric_name IN (
        'Host CPU Utilization (%)',
        'CPU Usage Per Sec',
        'Database CPU Time Ratio',
        'Database Time Per Sec',
        'Database Wait Time Ratio',
        'Physical Read Total Bytes Per Sec',
        'Physical Write Total Bytes Per Sec',
        'Physical Read Total IO Requests Per Sec',
        'Physical Write Total IO Requests Per Sec',
        'Average Active Sessions'
    )
)
SELECT
    metric_name,
    ROUND(value, 3) AS metric_value,
    metric_unit,
    ROUND(intsize_csec / 100, 0) AS window_seconds
FROM ranked_metrics
WHERE window_rank = 1
ORDER BY metric_name
