SELECT
    'PostgreSQL' AS product,
    current_setting('server_version') AS version,
    current_database() AS instance_name,
    CASE WHEN pg_is_in_recovery() THEN 'STANDBY' ELSE 'PRIMARY' END AS database_role,
    CURRENT_TIMESTAMP AT TIME ZONE 'UTC' AS server_time
