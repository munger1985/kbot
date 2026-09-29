SELECT
    tablespaces.oid AS tablespace_oid,
    tablespaces.spcname AS tablespace_name,
    pg_tablespace_size(tablespaces.oid) AS allocated_bytes
FROM pg_tablespace tablespaces
ORDER BY allocated_bytes DESC, tablespaces.spcname
