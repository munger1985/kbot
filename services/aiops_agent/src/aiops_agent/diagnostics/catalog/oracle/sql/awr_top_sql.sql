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
ranked AS (
    SELECT
        intervals.snapshot_id,
        intervals.instance_number,
        intervals.begin_time,
        intervals.end_time,
        sqlstat.sql_id,
        sqlstat.plan_hash_value,
        sqlstat.executions_delta AS executions,
        ROUND(sqlstat.elapsed_time_delta / 1000000, 3) AS elapsed_seconds,
        ROUND(sqlstat.cpu_time_delta / 1000000, 3) AS cpu_seconds,
        sqlstat.buffer_gets_delta AS buffer_gets,
        sqlstat.disk_reads_delta AS disk_reads,
        sqlstat.rows_processed_delta AS rows_processed,
        ROW_NUMBER() OVER (
            PARTITION BY intervals.snapshot_id
            ORDER BY sqlstat.elapsed_time_delta DESC, sqlstat.sql_id
        ) AS sql_rank
    FROM intervals
    JOIN dba_hist_sqlstat sqlstat
      ON sqlstat.snap_id = intervals.snapshot_id
     AND sqlstat.instance_number = intervals.instance_number
     AND sqlstat.dbid = (SELECT dbid FROM v$database)
    WHERE sqlstat.elapsed_time_delta > 0
      AND sqlstat.sql_id IS NOT NULL
)
SELECT
    snapshot_id,
    instance_number,
    begin_time,
    end_time,
    sql_id,
    plan_hash_value,
    executions,
    elapsed_seconds,
    cpu_seconds,
    buffer_gets,
    disk_reads,
    rows_processed,
    sql_rank
FROM ranked
WHERE sql_rank <= 15
ORDER BY snapshot_id, sql_rank
