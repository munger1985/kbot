SELECT
    event_schema,
    event_name,
    status,
    last_executed,
    interval_value,
    interval_field,
    starts,
    ends
FROM information_schema.events
ORDER BY event_schema, event_name
LIMIT :limit
