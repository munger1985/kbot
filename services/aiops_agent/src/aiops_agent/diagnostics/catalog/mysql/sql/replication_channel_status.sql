SELECT
    connection.channel_name,
    connection.service_state AS receiver_state,
    connection.source_uuid,
    connection.received_transaction_set,
    connection.last_error_number AS receiver_error_number,
    applier.service_state AS applier_state,
    coordinator.service_state AS coordinator_state,
    coordinator.last_error_number AS applier_error_number,
    coordinator.last_processed_transaction,
    workers.worker_count,
    workers.running_worker_count,
    workers.worker_error_count,
    workers.worker_last_error_number
FROM performance_schema.replication_connection_status connection
LEFT JOIN performance_schema.replication_applier_status applier
    ON applier.channel_name = connection.channel_name
LEFT JOIN performance_schema.replication_applier_status_by_coordinator coordinator
    ON coordinator.channel_name = connection.channel_name
LEFT JOIN (
    SELECT channel_name,
           COUNT(*) AS worker_count,
           SUM(CASE WHEN service_state = 'ON' THEN 1 ELSE 0 END) AS running_worker_count,
           SUM(CASE WHEN last_error_number <> 0 THEN 1 ELSE 0 END) AS worker_error_count,
           MAX(last_error_number) AS worker_last_error_number
    FROM performance_schema.replication_applier_status_by_worker
    GROUP BY channel_name
) workers ON workers.channel_name = connection.channel_name
ORDER BY connection.channel_name
