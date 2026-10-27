SELECT tables.table_schema AS schema_name, tables.table_name,
       tables.table_name AS relation_name, indexes.index_name,
       indexes.seq_in_index AS column_position, indexes.column_name,
       NULL AS column_expression,
       indexes.index_type, indexes.is_visible
FROM information_schema.tables AS tables
LEFT JOIN information_schema.statistics AS indexes
  ON indexes.table_schema = tables.table_schema
 AND indexes.table_name = tables.table_name
WHERE tables.table_schema = :schema_name AND tables.table_name = :table_name
ORDER BY indexes.index_name, indexes.seq_in_index
LIMIT :limit
