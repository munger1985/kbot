SELECT
    table_schema AS database_name,
    ROUND(SUM(data_length + index_length) / 1048576, 2) AS allocated_mb,
    COUNT(*) AS table_count
FROM information_schema.tables
WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
GROUP BY table_schema
ORDER BY table_schema
