SELECT
    :limit AS requested_limit,
    clock.sampled_at,
    statements.schema_name,
    statements.digest,
    LEFT(statements.query_sample_text, 4096) AS sql_text,
    LEFT(statements.digest_text, 4096) AS digest_text,
    statements.count_star AS executions,
    statements.sum_timer_wait AS elapsed_picoseconds,
    statements.sum_rows_examined AS rows_examined,
    statements.sum_rows_sent AS rows_sent,
    statements.first_seen,
    (SELECT COALESCE(SUM(overflow.count_star), 0)
       FROM performance_schema.events_statements_summary_by_digest overflow
      WHERE overflow.digest IS NULL) AS overflow_count,
    COALESCE((SELECT consumers.enabled
       FROM performance_schema.setup_consumers consumers
      WHERE consumers.name = 'statements_digest'), 'NO') AS digest_enabled
FROM (SELECT NOW(6) AS sampled_at) clock
LEFT JOIN performance_schema.events_statements_summary_by_digest statements
    ON statements.digest IS NOT NULL
    AND statements.digest_text REGEXP '^(SELECT|INSERT|UPDATE|DELETE|REPLACE|WITH|CALL)( |$)'
    AND (statements.schema_name IS NULL OR statements.schema_name NOT IN
        ('performance_schema', 'information_schema', 'mysql', 'sys'))
    AND LOCATE('`performance_schema` .', statements.digest_text) = 0
    AND LOCATE('`information_schema` .', statements.digest_text) = 0
    AND LOCATE('`mysql` .', statements.digest_text) = 0
    AND LOCATE('`sys` .', statements.digest_text) = 0
ORDER BY statements.schema_name, statements.digest
LIMIT :sample_limit
