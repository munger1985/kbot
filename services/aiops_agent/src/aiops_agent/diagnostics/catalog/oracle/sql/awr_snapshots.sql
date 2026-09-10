SELECT
    snap_id AS snapshot_id,
    instance_number AS instance_number,
    begin_interval_time AS begin_time,
    end_interval_time AS end_time,
    startup_time AS startup_time
  FROM dba_hist_snapshot
 WHERE instance_number = :instance_number
 ORDER BY snap_id DESC
