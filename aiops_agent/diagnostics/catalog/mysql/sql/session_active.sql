SELECT
    id AS session_id,
    user AS username,
    host AS client_host,
    db AS database_name,
    command AS command_name,
    time AS active_seconds,
    state AS session_state
FROM information_schema.processlist
WHERE command <> 'Sleep'
ORDER BY time DESC
