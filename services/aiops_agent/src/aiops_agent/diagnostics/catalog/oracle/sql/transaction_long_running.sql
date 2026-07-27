SELECT
    s.inst_id AS instance_id,
    s.sid AS session_id,
    s.username AS username,
    t.start_date AS transaction_started_at,
    ROUND((SYSDATE - t.start_date) * 86400) AS elapsed_seconds,
    t.used_ublk AS undo_blocks,
    t.used_urec AS undo_records
FROM gv$transaction t
JOIN gv$session s
  ON s.inst_id = t.inst_id
 AND s.saddr = t.ses_addr
WHERE (SYSDATE - t.start_date) * 86400 >= :min_seconds
ORDER BY elapsed_seconds DESC
