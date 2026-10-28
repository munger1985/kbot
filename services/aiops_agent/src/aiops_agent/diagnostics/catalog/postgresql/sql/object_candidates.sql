SELECT namespaces.nspname AS schema_name,
       relations.relname AS table_name,
       CASE relations.relkind WHEN 'r' THEN 'TABLE' WHEN 'p' THEN 'PARTITIONED TABLE'
            WHEN 'v' THEN 'VIEW' WHEN 'm' THEN 'MATERIALIZED VIEW'
            WHEN 'f' THEN 'FOREIGN TABLE' END AS relation_kind,
       CAST(relations.oid AS text) AS native_object_id
FROM pg_catalog.pg_class relations
JOIN pg_catalog.pg_namespace namespaces ON namespaces.oid = relations.relnamespace
WHERE relations.relname = :table_name
  AND (:schema_name = '' OR namespaces.nspname = :schema_name)
  AND relations.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND pg_catalog.has_schema_privilege(namespaces.oid, 'USAGE')
  AND pg_catalog.has_table_privilege(relations.oid, 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')
ORDER BY namespaces.nspname, relations.relname, relations.oid
LIMIT :limit
