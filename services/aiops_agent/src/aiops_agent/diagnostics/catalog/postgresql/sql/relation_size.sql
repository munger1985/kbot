SELECT
    classes.oid AS relation_oid,
    namespaces.nspname AS schema_name,
    classes.relname AS relation_name,
    classes.relkind AS relation_kind,
    pg_relation_size(classes.oid) AS main_bytes,
    pg_indexes_size(classes.oid) AS index_bytes,
    pg_total_relation_size(classes.oid) AS total_bytes
FROM pg_class classes
JOIN pg_namespace namespaces
  ON namespaces.oid = classes.relnamespace
WHERE classes.relkind IN ('r', 'm', 'p')
  AND namespaces.nspname NOT IN ('pg_catalog', 'information_schema')
  AND namespaces.nspname !~ '^pg_toast'
ORDER BY total_bytes DESC, namespaces.nspname, classes.relname
LIMIT :limit
