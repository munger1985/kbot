SELECT DISTINCT source_namespace.nspname AS schema_name,
       source.relname AS table_name,
       CASE source.relkind WHEN 'v' THEN 'VIEW' WHEN 'm' THEN 'MATERIALIZED VIEW' END AS relation_kind,
       target_namespace.nspname AS dependency_schema_name,
       target.relname AS dependency_name,
       CASE target.relkind WHEN 'v' THEN 'VIEW' WHEN 'm' THEN 'MATERIALIZED VIEW'
            WHEN 'f' THEN 'FOREIGN TABLE' ELSE 'TABLE' END AS dependency_kind,
       'LOCAL' AS reference_scope
FROM pg_catalog.pg_class source
JOIN pg_catalog.pg_namespace source_namespace ON source_namespace.oid = source.relnamespace
JOIN pg_catalog.pg_rewrite rewrite ON rewrite.ev_class = source.oid AND rewrite.rulename = '_RETURN'
JOIN pg_catalog.pg_depend dependency ON dependency.objid = rewrite.oid
     AND dependency.classid = CAST('pg_catalog.pg_rewrite' AS regclass)
     AND dependency.refclassid = CAST('pg_catalog.pg_class' AS regclass)
JOIN pg_catalog.pg_class target ON target.oid = dependency.refobjid
JOIN pg_catalog.pg_namespace target_namespace ON target_namespace.oid = target.relnamespace
WHERE source_namespace.nspname = :schema_name AND source.relname = :table_name
  AND source.relkind IN ('v', 'm') AND target.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND source.oid <> target.oid
ORDER BY dependency_schema_name, dependency_name
LIMIT :limit
