SELECT
    snap_id AS snapshot_id,
    instance_number AS instance_number,
    begin_interval_time AS begin_time,
    end_interval_time AS end_time,
    startup_time AS startup_time
  FROM dba_hist_snapshot
 WHERE dbid = (SELECT dbid FROM v$database)
   AND instance_number = (SELECT instance_number FROM v$instance)
 ORDER BY snap_id DESC
