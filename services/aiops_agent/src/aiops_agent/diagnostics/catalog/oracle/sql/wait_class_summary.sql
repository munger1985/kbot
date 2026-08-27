SELECT
    wait_class,
    total_waits,
    ROUND(time_waited / 100, 3) AS time_waited_seconds,
    total_waits_fg,
    ROUND(time_waited_fg / 100, 3) AS foreground_waited_seconds
FROM v$system_wait_class
WHERE wait_class <> 'Idle'
ORDER BY time_waited_fg DESC, wait_class
