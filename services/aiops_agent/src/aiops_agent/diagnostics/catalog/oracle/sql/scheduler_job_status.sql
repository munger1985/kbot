SELECT owner,
       job_name,
       enabled,
       state,
       last_start_date,
       last_run_duration,
       NVL(run_count, 0) AS run_count,
       NVL(failure_count, 0) AS failure_count
  FROM dba_scheduler_jobs
 WHERE owner = UPPER(:schema_name)
   AND job_name = UPPER(:job_name)
