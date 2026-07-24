SELECT
    waiter.inst_id AS waiting_instance_id,
    waiter.sid AS waiting_session_id,
    waiter.username AS waiting_username,
    blocker.inst_id AS blocking_instance_id,
    blocker.sid AS blocking_session_id,
    blocker.username AS blocking_username,
    waiter.event AS wait_event,
    waiter.seconds_in_wait AS wait_seconds
FROM gv$session waiter
JOIN gv$session blocker
  ON blocker.inst_id = waiter.blocking_instance
 AND blocker.sid = waiter.blocking_session
WHERE waiter.blocking_session IS NOT NULL
ORDER BY waiter.seconds_in_wait DESC
