SELECT
    tables.table_schema,
    tables.table_name,
    tables.table_rows,
    tables.update_time,
    COUNT(statistics.index_name) AS index_count,
    SUM(CASE WHEN statistics.cardinality IS NULL THEN 1 ELSE 0 END) AS indexes_without_cardinality
FROM information_schema.tables tables
LEFT JOIN information_schema.statistics statistics
    ON statistics.table_schema = tables.table_schema
    AND statistics.table_name = tables.table_name
WHERE tables.table_schema NOT IN ('mysql', 'performance_schema', 'information_schema', 'sys')
GROUP BY tables.table_schema, tables.table_name, tables.table_rows, tables.update_time
ORDER BY indexes_without_cardinality DESC, tables.table_rows DESC
LIMIT :limit
