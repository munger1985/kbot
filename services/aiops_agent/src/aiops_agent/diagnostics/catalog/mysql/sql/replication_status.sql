SELECT
    channel_name,
    service_state,
    source_uuid,
    last_error_number
FROM performance_schema.replication_connection_status
ORDER BY channel_name
