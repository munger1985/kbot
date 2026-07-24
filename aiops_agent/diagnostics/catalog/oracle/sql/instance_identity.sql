SELECT
    'Oracle Database' AS product,
    i.version_full AS version,
    i.instance_name AS instance_name,
    d.database_role AS database_role,
    SYSTIMESTAMP AS server_time
FROM v$instance i
CROSS JOIN v$database d
