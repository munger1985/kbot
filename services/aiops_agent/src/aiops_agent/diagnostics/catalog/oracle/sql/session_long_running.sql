SELECT *
FROM (
    SELECT
        s.inst_id AS instance_id,
        s.sid AS session_id,
        s.serial# AS serial_number,
        s.username AS username,
        s.status AS status,
        s.logon_time AS logon_time,
        s.last_call_et AS runtime_seconds,
        s.event AS wait_event,
        s.seconds_in_wait AS wait_seconds,
        s.machine AS client_host,
        s.sql_id AS sql_id
    FROM gv$session s
    WHERE s.type = 'USER'
      AND s.status = 'ACTIVE'
      AND s.username IS NOT NULL
      AND s.last_call_et >= :min_seconds
    ORDER BY s.last_call_et DESC, s.inst_id, s.sid
)
WHERE ROWNUM <= :limit
