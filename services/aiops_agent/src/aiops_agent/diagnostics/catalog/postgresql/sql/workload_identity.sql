SELECT
    CAST(system_identifier AS text) AS system_identifier,
    pg_postmaster_start_time() AS instance_started_at,
    current_database() AS database_name,
    (SELECT CAST(oid AS text) FROM pg_database WHERE datname = current_database()) AS database_oid
FROM pg_control_system()
