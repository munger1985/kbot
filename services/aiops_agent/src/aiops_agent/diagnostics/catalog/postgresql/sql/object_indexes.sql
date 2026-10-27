WITH RECURSIVE relations AS (
    SELECT c.oid, c.relname, c.relnamespace, c.relkind
    FROM pg_catalog.pg_class AS c
    JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
    WHERE n.nspname = :schema_name AND c.relname = :table_name
      AND c.relkind IN ('r', 'p', 'm')
    UNION ALL
    SELECT child.oid, child.relname, child.relnamespace, child.relkind
    FROM relations AS parent
    JOIN pg_catalog.pg_inherits AS inheritance ON inheritance.inhparent = parent.oid
    JOIN pg_catalog.pg_class AS child ON child.oid = inheritance.inhrelid
)
SELECT n.nspname AS schema_name, :table_name AS table_name,
       r.relname AS relation_name, CAST(r.relkind AS text) AS relation_kind,
       idx.relname AS index_name, meta.indisvalid AS is_valid,
       pg_get_indexdef(idx.oid) AS index_definition
FROM relations AS r
JOIN pg_catalog.pg_namespace AS n ON n.oid = r.relnamespace
LEFT JOIN pg_catalog.pg_index AS meta ON meta.indrelid = r.oid
LEFT JOIN pg_catalog.pg_class AS idx ON idx.oid = meta.indexrelid
ORDER BY n.nspname, r.relname, idx.relname
LIMIT :limit
