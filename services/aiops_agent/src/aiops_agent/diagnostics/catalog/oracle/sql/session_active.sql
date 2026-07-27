SELECT
    s.inst_id AS instance_id,
    s.sid AS session_id,
    s.serial# AS serial_number,
    s.username AS username,
    s.status AS status,
    s.event AS wait_event,
    s.seconds_in_wait AS wait_seconds,
    s.machine AS client_host
FROM gv$session s
WHERE s.type = 'USER'
  AND s.status = 'ACTIVE'
  AND s.username IS NOT NULL
ORDER BY s.seconds_in_wait DESC
