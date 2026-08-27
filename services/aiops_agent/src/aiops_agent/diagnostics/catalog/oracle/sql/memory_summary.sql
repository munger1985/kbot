SELECT component, allocated_mb
FROM (
    SELECT
        'SGA' AS component,
        ROUND(SUM(value) / 1024 / 1024, 2) AS allocated_mb
    FROM v$sga
    UNION ALL
    SELECT
        'PGA_ALLOCATED' AS component,
        ROUND(value / 1024 / 1024, 2) AS allocated_mb
    FROM v$pgastat
    WHERE name = 'total PGA allocated'
)
ORDER BY component
