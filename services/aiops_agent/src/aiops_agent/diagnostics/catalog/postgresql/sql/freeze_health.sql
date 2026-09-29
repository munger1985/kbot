SELECT
    classes.oid AS relation_oid,
    namespaces.nspname AS schema_name,
    classes.relname AS table_name,
    age(classes.relfrozenxid) AS xid_age,
    mxid_age(classes.relminmxid) AS multixact_age,
    CAST(current_setting('autovacuum_freeze_max_age') AS bigint)
        AS xid_stop_limit,
    CAST(current_setting('autovacuum_multixact_freeze_max_age') AS bigint)
        AS multixact_stop_limit,
    CAST(current_setting('autovacuum_freeze_max_age') AS bigint)
        - age(classes.relfrozenxid) AS xid_headroom,
    CAST(current_setting('autovacuum_multixact_freeze_max_age') AS bigint)
        - mxid_age(classes.relminmxid) AS multixact_headroom
FROM pg_class classes
JOIN pg_namespace namespaces
  ON namespaces.oid = classes.relnamespace
WHERE classes.relkind IN ('r', 'm', 'p')
  AND namespaces.nspname NOT IN ('pg_catalog', 'information_schema')
  AND namespaces.nspname !~ '^pg_toast'
ORDER BY xid_headroom, multixact_headroom, namespaces.nspname, classes.relname
LIMIT :limit
