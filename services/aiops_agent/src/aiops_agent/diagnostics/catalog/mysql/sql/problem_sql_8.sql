WITH metrics AS (
SELECT CONCAT(COALESCE(schema_name, ''), ':', digest) AS sql_key,
       schema_name, COALESCE(NULLIF(query_sample_text, ''), digest_text) AS sql_text,
       CASE WHEN NULLIF(query_sample_text, '') IS NOT NULL THEN 'SAMPLE' ELSE 'NORMALIZED' END AS text_kind,
       count_star AS execution_count, sum_timer_wait / 1000000000.0 AS total_elapsed_ms,
       avg_timer_wait / 1000000000.0 AS average_elapsed_ms,
       max_timer_wait / 1000000000.0 AS maximum_elapsed_ms,
       NULL AS average_cpu_ms, NULL AS average_io_ms,
       sum_lock_time / 1000000000.0 / NULLIF(count_star, 0) AS average_lock_ms,
       NULL AS average_reads, sum_rows_examined / NULLIF(count_star, 0) AS average_rows_examined,
       sum_created_tmp_disk_tables / NULLIF(count_star, 0) AS average_spills
FROM performance_schema.events_statements_summary_by_digest
WHERE digest IS NOT NULL AND count_star > 0
), qualified AS (
    SELECT * FROM metrics
    WHERE (average_elapsed_ms >= :average_elapsed_ms_limit)
       OR (average_cpu_ms >= :average_cpu_ms_limit)
       OR (average_io_ms >= :average_io_ms_limit)
       OR (average_lock_ms >= :average_lock_ms_limit)
       OR (average_reads >= :average_reads_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
       OR (average_rows_examined >= :average_rows_examined_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
       OR (average_spills >= :average_spills_limit AND average_elapsed_ms >= :resource_minimum_elapsed_ms)
), collection AS (
SELECT CASE WHEN (SELECT COUNT(*) FROM performance_schema.setup_consumers
                  WHERE name IN ('global_instrumentation', 'thread_instrumentation', 'statements_digest')
                    AND enabled = 'YES') = 3
                 AND EXISTS (SELECT 1 FROM performance_schema.setup_instruments
                             WHERE name LIKE 'statement/sql/%' AND enabled = 'YES')
            THEN 'AVAILABLE' ELSE 'DISABLED' END AS collection_status
)
SELECT qualified.*, collection.collection_status
FROM collection LEFT JOIN qualified ON 1 = 1
ORDER BY qualified.sql_key
