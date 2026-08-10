SELECT
    application_name AS channel_name,
    state AS service_state,
    CAST(client_addr AS text) AS source_uuid,
    CAST(0 AS bigint) AS last_error_number
FROM pg_stat_replication
ORDER BY application_name
