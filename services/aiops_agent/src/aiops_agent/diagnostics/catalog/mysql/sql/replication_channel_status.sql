SELECT
    connection.channel_name,
    connection.service_state AS receiver_state,
    connection.source_uuid,
    connection.received_transaction_set,
    connection.last_error_number AS receiver_error_number,
    coordinator.service_state AS applier_state,
    coordinator.last_error_number AS applier_error_number,
    coordinator.last_processed_transaction
FROM performance_schema.replication_connection_status connection
LEFT JOIN performance_schema.replication_applier_status_by_coordinator coordinator
    ON coordinator.channel_name = connection.channel_name
ORDER BY connection.channel_name
