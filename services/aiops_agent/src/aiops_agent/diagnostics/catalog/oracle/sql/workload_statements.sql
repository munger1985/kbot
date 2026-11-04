WITH tracked AS (
    SELECT identity_key FROM JSON_TABLE(:tracked_keys, '$[*]' COLUMNS (identity_key VARCHAR(512) PATH '$'))
), excluded AS (
    SELECT identity_key FROM JSON_TABLE(:excluded_keys, '$[*]' COLUMNS (identity_key VARCHAR(512) PATH '$'))
)
SELECT
    ranked.sql_id,
    ranked.sql_text,
    ranked.executions,
    ranked.elapsed_time,
    ranked.rows_processed,
    ranked.plan_hash_value,
    ranked.cpu_time,
    ranked.buffer_gets,
    ranked.disk_reads
FROM (
    SELECT candidates.*
    FROM (
    SELECT
        sql_id, sql_text, executions, elapsed_time, rows_processed,
        plan_hash_value, cpu_time, buffer_gets, disk_reads,
        ROW_NUMBER() OVER (ORDER BY elapsed_time DESC, sql_id) AS elapsed_rank,
        ROW_NUMBER() OVER (ORDER BY cpu_time DESC, sql_id) AS cpu_rank,
        ROW_NUMBER() OVER (ORDER BY buffer_gets DESC, sql_id) AS logical_rank,
        ROW_NUMBER() OVER (ORDER BY disk_reads DESC, sql_id) AS read_rank,
        ROW_NUMBER() OVER (ORDER BY executions DESC, sql_id) AS count_rank,
        ROW_NUMBER() OVER (ORDER BY elapsed_time / NULLIF(executions, 0) DESC NULLS LAST, sql_id) AS mean_rank,
        ROW_NUMBER() OVER (ORDER BY cpu_time / NULLIF(executions, 0) DESC NULLS LAST, sql_id) AS mean_cpu_rank,
        ROW_NUMBER() OVER (ORDER BY buffer_gets / NULLIF(executions, 0) DESC NULLS LAST, sql_id) AS mean_logical_rank
    FROM v$sqlstats
    WHERE con_id = TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID'))
      AND sql_id IS NOT NULL
      AND elapsed_time >= 0
      AND (sql_id IN (SELECT identity_key FROM tracked)
           OR sql_id NOT IN (SELECT identity_key FROM excluded))
    ) candidates
    ORDER BY CASE WHEN candidates.sql_id IN (SELECT identity_key FROM tracked) THEN 0 ELSE 1 END,
             LEAST(candidates.elapsed_rank, candidates.cpu_rank, candidates.logical_rank,
                   candidates.read_rank, candidates.count_rank, candidates.mean_rank,
                   candidates.mean_cpu_rank, candidates.mean_logical_rank),
             candidates.elapsed_rank, candidates.sql_id
) ranked
WHERE ROWNUM <= :limit
