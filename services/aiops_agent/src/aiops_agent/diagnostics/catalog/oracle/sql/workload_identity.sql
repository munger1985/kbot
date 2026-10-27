SELECT
    TO_CHAR(d.dbid) AS database_id,
    i.instance_number,
    i.instance_name,
    TO_CHAR(i.startup_time, 'YYYY-MM-DD"T"HH24:MI:SS') ||
        TO_CHAR(SYSTIMESTAMP, 'TZH:TZM') AS instance_started_at,
    SYS_CONTEXT('USERENV', 'CON_ID') AS container_id,
    SYS_CONTEXT('USERENV', 'CON_NAME') AS container_name
FROM v$database d
CROSS JOIN v$instance i
