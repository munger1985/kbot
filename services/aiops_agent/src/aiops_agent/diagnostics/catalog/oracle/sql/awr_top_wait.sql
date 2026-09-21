WITH intervals AS (
    SELECT
        curr.snap_id AS snapshot_id,
        curr.instance_number AS instance_number,
        prev.snap_id AS begin_snapshot_id,
        curr.snap_id AS end_snapshot_id,
        prev.end_interval_time AS begin_time,
        curr.end_interval_time AS end_time,
        ROUND(
            (
                CAST(curr.end_interval_time AS DATE)
                - CAST(prev.end_interval_time AS DATE)
            ) * 86400,
            0
        ) AS elapsed_seconds
    FROM dba_hist_snapshot curr
    JOIN dba_hist_snapshot prev
      ON prev.dbid = curr.dbid
     AND prev.instance_number = curr.instance_number
     AND prev.startup_time = curr.startup_time
     AND prev.snap_id = (
            SELECT MAX(earlier.snap_id)
              FROM dba_hist_snapshot earlier
             WHERE earlier.dbid = curr.dbid
               AND earlier.instance_number = curr.instance_number
               AND earlier.startup_time = curr.startup_time
               AND earlier.snap_id < curr.snap_id
         )
    WHERE curr.dbid = (SELECT dbid FROM v$database)
      AND curr.instance_number = (SELECT instance_number FROM v$instance)
      AND curr.snap_id > :begin_snapshot_id
      AND curr.snap_id <= :end_snapshot_id
      AND (
            CAST(curr.end_interval_time AS DATE)
            - CAST(prev.end_interval_time AS DATE)
          ) * 86400 > 0
),
deltas AS (
    SELECT
        intervals.snapshot_id,
        intervals.instance_number,
        intervals.begin_time,
        intervals.end_time,
        intervals.elapsed_seconds,
        ending.wait_class,
        ending.event_name,
        (ending.total_waits - NVL(beginning.total_waits, 0)) AS total_waits,
        ROUND(
            (ending.time_waited_micro - NVL(beginning.time_waited_micro, 0))
            / 1000000,
            3
        ) AS time_waited_seconds
    FROM intervals
    JOIN dba_hist_system_event ending
      ON ending.snap_id = intervals.end_snapshot_id
     AND ending.instance_number = intervals.instance_number
     AND ending.dbid = (SELECT dbid FROM v$database)
     AND ending.wait_class <> 'Idle'
    LEFT JOIN dba_hist_system_event beginning
      ON beginning.snap_id = intervals.begin_snapshot_id
     AND beginning.instance_number = ending.instance_number
     AND beginning.dbid = ending.dbid
     AND beginning.event_id = ending.event_id
),
ranked AS (
    SELECT
        snapshot_id,
        instance_number,
        begin_time,
        end_time,
        elapsed_seconds,
        wait_class,
        event_name,
        total_waits,
        time_waited_seconds,
        ROUND(
            time_waited_seconds * 1000 / NULLIF(total_waits, 0),
            3
        ) AS avg_wait_ms,
        ROW_NUMBER() OVER (
            PARTITION BY snapshot_id
            ORDER BY time_waited_seconds DESC, event_name
        ) AS wait_rank
    FROM deltas
    WHERE time_waited_seconds > 0
)
SELECT
    snapshot_id,
    instance_number,
    begin_time,
    end_time,
    elapsed_seconds,
    wait_class,
    event_name,
    total_waits,
    time_waited_seconds,
    avg_wait_ms,
    wait_rank
FROM ranked
WHERE wait_rank <= 15
ORDER BY snapshot_id, wait_rank
