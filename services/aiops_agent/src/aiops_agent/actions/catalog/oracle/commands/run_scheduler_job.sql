BEGIN DBMS_SCHEDULER.RUN_JOB({{job_ref}}, use_current_session => FALSE); END;
