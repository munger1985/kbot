SELECT
    observed_at,
    message_type,
    message_level,
    problem_key,
    message_text
FROM (
    SELECT
        originating_timestamp AS observed_at,
        message_type,
        message_level,
        problem_key,
        message_text
    FROM v$diag_alert_ext
    WHERE originating_timestamp >= (
        SYSTIMESTAMP - NUMTODSINTERVAL(:hours, 'HOUR')
    )
      AND (
          INSTR(UPPER(message_text), 'ORA-') > 0
          OR message_level <= 8
      )
    ORDER BY originating_timestamp DESC
)
WHERE ROWNUM <= :limit
