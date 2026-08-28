SELECT
    owner,
    job_name,
    status,
    error# AS error_code,
    actual_start_date,
    run_duration,
    additional_info
FROM (
    SELECT
        owner,
        job_name,
        status,
        error#,
        actual_start_date,
        run_duration,
        additional_info
    FROM dba_scheduler_job_run_details
    WHERE actual_start_date >= (
        SYSTIMESTAMP - NUMTODSINTERVAL(:hours, 'HOUR')
    )
      AND status NOT IN ('SUCCEEDED', 'STOPPED')
    ORDER BY actual_start_date DESC
)
WHERE ROWNUM <= :limit
