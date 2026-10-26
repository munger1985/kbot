SELECT
    observed_at,
    record_id,
    message_type,
    message_level,
    problem_key,
    message_text
FROM (
    SELECT
        originating_timestamp AS observed_at,
        record_id,
        message_type,
        message_level,
        problem_key,
        message_text
    FROM v$diag_alert_ext
    WHERE originating_timestamp >= (
        SYSTIMESTAMP - NUMTODSINTERVAL(:hours, 'HOUR')
    )
      AND (
          :scope = 'ALL'
          OR (
              :scope = 'ERRORS'
              AND INSTR(UPPER(message_text), 'ORA-') > 0
          )
          OR (
              :scope = 'IMPORTANT'
              AND (
                  INSTR(UPPER(message_text), 'ORA-') > 0
                  OR message_level <= 8
              )
          )
      )
      AND (
          :keyword = ''
          OR INSTR(UPPER(message_text), UPPER(:keyword)) > 0
      )
    ORDER BY originating_timestamp DESC, record_id DESC
)
WHERE ROWNUM <= :limit
