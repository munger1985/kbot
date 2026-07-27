SELECT
    'MySQL' AS product,
    VERSION() AS version,
    @@hostname AS instance_name,
    CASE WHEN @@global.read_only = 1 THEN 'READ_ONLY' ELSE 'PRIMARY' END AS database_role,
    UTC_TIMESTAMP() AS server_time
