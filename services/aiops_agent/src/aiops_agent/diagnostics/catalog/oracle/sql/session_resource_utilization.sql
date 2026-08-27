SELECT
    resource_name,
    current_utilization,
    max_utilization,
    limit_value,
    CASE
        WHEN REGEXP_LIKE(limit_value, '^[0-9]+$')
             AND TO_NUMBER(limit_value) > 0
        THEN ROUND(100 * current_utilization / TO_NUMBER(limit_value), 3)
        ELSE NULL
    END AS utilization_percent
FROM v$resource_limit
WHERE resource_name IN ('sessions', 'processes')
ORDER BY resource_name
