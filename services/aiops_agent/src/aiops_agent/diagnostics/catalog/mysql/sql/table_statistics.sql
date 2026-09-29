SELECT
    table_schema,
    table_name,
    engine,
    table_rows,
    data_length,
    index_length,
    data_free,
    update_time
FROM information_schema.tables
WHERE (:schema_name = '' OR table_schema = :schema_name)
  AND table_schema NOT IN ('mysql', 'performance_schema', 'information_schema', 'sys')
ORDER BY data_length + index_length DESC
LIMIT :limit
