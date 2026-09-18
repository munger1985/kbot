SELECT
    waiter.inst_id AS waiting_instance_id,
    waiter.sid AS waiting_session_id,
    waiter.serial# AS waiting_serial_number,
    waiter.username AS waiting_username,
    waiter.sql_id AS waiting_sql_id,
    waiter.prev_sql_id AS waiting_prev_sql_id,
    waiter.status AS waiting_status,
    blocker.inst_id AS blocking_instance_id,
    blocker.sid AS blocking_session_id,
    blocker.serial# AS blocking_serial_number,
    blocker.username AS blocking_username,
    blocker.sql_id AS blocking_sql_id,
    blocker.prev_sql_id AS blocking_prev_sql_id,
    blocker.status AS blocking_status,
    waiter.event AS wait_event,
    waiter.seconds_in_wait AS wait_seconds,
    COALESCE(holder_lock.type, waiter_lock.type) AS lock_type,
    COALESCE(holder_lock.lmode, waiter_lock.lmode) AS lock_mode,
    COALESCE(holder_lock.ctime, waiter_lock.ctime) AS lock_ctime_seconds,
    CASE WHEN blocker.blocking_session IS NULL THEN 1 ELSE 2 END AS chain_depth,
    CASE WHEN blocker.blocking_session IS NULL THEN 1 ELSE 0 END AS is_holder
FROM gv$session waiter
JOIN gv$session blocker
  ON blocker.inst_id = waiter.blocking_instance
 AND blocker.sid = waiter.blocking_session
LEFT JOIN "GV$LOCK" waiter_lock
  ON waiter_lock.inst_id = waiter.inst_id
 AND waiter_lock.sid = waiter.sid
 AND waiter_lock.request > 0
LEFT JOIN "GV$LOCK" holder_lock
  ON holder_lock.inst_id = blocker.inst_id
 AND holder_lock.sid = blocker.sid
 AND holder_lock.request = 0
 AND holder_lock.lmode > 0
 AND holder_lock.type = waiter_lock.type
 AND holder_lock.id1 = waiter_lock.id1
 AND holder_lock.id2 = waiter_lock.id2
WHERE waiter.blocking_session IS NOT NULL
ORDER BY waiter.seconds_in_wait DESC
