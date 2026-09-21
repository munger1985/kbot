SELECT
    nsp.nspname AS schema_name,
    cls.relname AS table_name,
    age(cls.relfrozenxid) AS frozen_xid_age
FROM pg_class cls
INNER JOIN pg_namespace nsp
    ON nsp.oid = cls.relnamespace
WHERE cls.relkind = 'r'
    AND nsp.nspname NOT IN ('pg_catalog', 'information_schema')
    AND age(cls.relfrozenxid) >= 150000000
ORDER BY age(cls.relfrozenxid) DESC
