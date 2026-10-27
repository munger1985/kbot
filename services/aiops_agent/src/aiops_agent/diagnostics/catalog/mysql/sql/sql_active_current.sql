SELECT
    p.id AS session_id,
    p.user AS username,
    p.db AS database_name,
    LEFT(p.info, 4096) AS sql_text,
    p.time AS elapsed_seconds,
    p.state AS wait_event
FROM information_schema.processlist p
WHERE p.command = 'Query'
  AND p.info IS NOT NULL
  AND p.id <> CONNECTION_ID()
  AND p.db = DATABASE()
ORDER BY p.time DESC, p.id
LIMIT :limit
