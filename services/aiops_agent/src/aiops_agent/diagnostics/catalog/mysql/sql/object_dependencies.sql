SELECT DISTINCT usage_rows.VIEW_SCHEMA AS schema_name,
       usage_rows.VIEW_NAME AS table_name,
       'VIEW' AS relation_kind,
       usage_rows.TABLE_SCHEMA AS dependency_schema_name,
       usage_rows.TABLE_NAME AS dependency_name,
       CASE WHEN tables.TABLE_TYPE = 'VIEW' THEN 'VIEW'
            WHEN tables.TABLE_TYPE = 'BASE TABLE' THEN 'TABLE' ELSE 'UNKNOWN' END AS dependency_kind,
       'LOCAL' AS reference_scope
FROM information_schema.VIEW_TABLE_USAGE usage_rows
LEFT JOIN information_schema.TABLES tables ON tables.TABLE_SCHEMA = usage_rows.TABLE_SCHEMA
     AND tables.TABLE_NAME = usage_rows.TABLE_NAME
WHERE usage_rows.VIEW_SCHEMA = :schema_name AND usage_rows.VIEW_NAME = :table_name
ORDER BY dependency_schema_name, dependency_name
LIMIT :limit
