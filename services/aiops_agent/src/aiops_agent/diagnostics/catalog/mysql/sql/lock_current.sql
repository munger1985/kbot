SELECT
    waits.requesting_thread_id,
    waits.blocking_thread_id,
    requested.object_schema,
    requested.object_name,
    requested.index_name,
    requested.lock_type,
    requested.lock_mode,
    requested.lock_status
FROM performance_schema.data_lock_waits waits
INNER JOIN performance_schema.data_locks requested
    ON requested.engine_lock_id = waits.requesting_engine_lock_id
ORDER BY waits.requesting_thread_id
LIMIT :limit
