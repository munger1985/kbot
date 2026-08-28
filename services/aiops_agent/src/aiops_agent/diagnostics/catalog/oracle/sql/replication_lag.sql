SELECT
    name AS metric_name,
    value AS metric_value,
    unit AS metric_unit,
    time_computed,
    datum_time
FROM v$dataguard_stats
WHERE name IN (
    'transport lag',
    'apply lag',
    'apply finish time',
    'estimated startup time'
)
ORDER BY name
