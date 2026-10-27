SELECT
    ranked.instance_id,
    ranked.session_id,
    ranked.username,
    ranked.sql_id,
    ranked.sql_text,
    ranked.elapsed_seconds,
    ranked.wait_event
FROM (
    SELECT
        s.inst_id AS instance_id,
        s.sid AS session_id,
        s.username,
        s.sql_id,
        (SELECT MAX(q.sql_text) FROM gv$sql q
         WHERE q.inst_id = s.inst_id AND q.sql_id = s.sql_id
           AND q.con_id = s.con_id) AS sql_text,
        CASE WHEN s.sql_exec_start IS NOT NULL
             THEN GREATEST(0, ROUND((SYSDATE - s.sql_exec_start) * 86400))
        END AS elapsed_seconds,
        s.event AS wait_event
    FROM gv$session s
    WHERE s.type = 'USER'
      AND s.status = 'ACTIVE'
      AND s.username IS NOT NULL
      AND s.sql_id IS NOT NULL
      AND s.con_id = TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID'))
      AND NOT (s.inst_id = TO_NUMBER(SYS_CONTEXT('USERENV', 'INSTANCE'))
               AND s.sid = TO_NUMBER(SYS_CONTEXT('USERENV', 'SID')))
    ORDER BY s.sql_exec_start ASC NULLS LAST, s.sid
) ranked
WHERE ROWNUM <= :limit
