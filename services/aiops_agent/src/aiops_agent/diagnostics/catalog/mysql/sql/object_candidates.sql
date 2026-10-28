SELECT tables.TABLE_SCHEMA AS schema_name,
       tables.TABLE_NAME AS table_name,
       tables.TABLE_TYPE AS relation_kind,
       CAST(NULL AS CHAR) AS native_object_id
FROM information_schema.TABLES tables
WHERE BINARY tables.TABLE_NAME = BINARY :table_name
  AND (:schema_name = '' OR BINARY tables.TABLE_SCHEMA = BINARY :schema_name)
ORDER BY BINARY tables.TABLE_SCHEMA, BINARY tables.TABLE_NAME
LIMIT :limit
