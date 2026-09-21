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
metrics AS (
    SELECT
        intervals.snapshot_id,
        intervals.instance_number,
        intervals.begin_time,
        intervals.end_time,
        intervals.elapsed_seconds,
        ending.stat_name AS metric_name,
        (ending.value - NVL(beginning.value, 0)) AS total_value,
        ROUND(
            (ending.value - NVL(beginning.value, 0))
            / NULLIF(intervals.elapsed_seconds, 0),
            3
        ) AS per_second,
        CASE
            WHEN ending.stat_name = 'redo size' THEN 'BYTES'
            ELSE 'COUNT'
        END AS unit
    FROM intervals
    JOIN dba_hist_sysstat ending
      ON ending.snap_id = intervals.end_snapshot_id
     AND ending.instance_number = intervals.instance_number
     AND ending.dbid = (SELECT dbid FROM v$database)
     AND ending.stat_name IN (
            'redo size',
            'session logical reads',
            'db block changes',
            'physical reads',
            'physical writes',
            'user calls',
            'parse count (total)',
            'parse count (hard)',
            'execute count',
            'user commits',
            'user rollbacks',
            'logons cumulative'
         )
    LEFT JOIN dba_hist_sysstat beginning
      ON beginning.snap_id = intervals.begin_snapshot_id
     AND beginning.instance_number = ending.instance_number
     AND beginning.dbid = ending.dbid
     AND beginning.stat_id = ending.stat_id
    UNION ALL
    SELECT
        intervals.snapshot_id,
        intervals.instance_number,
        intervals.begin_time,
        intervals.end_time,
        intervals.elapsed_seconds,
        ending.stat_name AS metric_name,
        ROUND(
            (ending.value - NVL(beginning.value, 0)) / 1000000,
            3
        ) AS total_value,
        ROUND(
            ((ending.value - NVL(beginning.value, 0)) / 1000000)
            / NULLIF(intervals.elapsed_seconds, 0),
            3
        ) AS per_second,
        'SECONDS' AS unit
    FROM intervals
    JOIN dba_hist_sys_time_model ending
      ON ending.snap_id = intervals.end_snapshot_id
     AND ending.instance_number = intervals.instance_number
     AND ending.dbid = (SELECT dbid FROM v$database)
     AND ending.stat_name IN ('DB time', 'DB CPU')
    LEFT JOIN dba_hist_sys_time_model beginning
      ON beginning.snap_id = intervals.begin_snapshot_id
     AND beginning.instance_number = ending.instance_number
     AND beginning.dbid = ending.dbid
     AND beginning.stat_name = ending.stat_name
)
SELECT
    snapshot_id,
    instance_number,
    begin_time,
    end_time,
    elapsed_seconds,
    metric_name,
    total_value,
    per_second,
    unit
FROM metrics
ORDER BY snapshot_id, metric_name
