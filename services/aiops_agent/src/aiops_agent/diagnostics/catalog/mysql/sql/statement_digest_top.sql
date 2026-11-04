WITH tracked AS (
    SELECT identity_key FROM JSON_TABLE(:tracked_keys, '$[*]' COLUMNS (identity_key VARCHAR(512) PATH '$')) AS keys_table
), excluded AS (
    SELECT identity_key FROM JSON_TABLE(:excluded_keys, '$[*]' COLUMNS (identity_key VARCHAR(512) PATH '$')) AS keys_table
)
SELECT
    schema_name,
    digest,
    digest_text,
    LEFT(query_sample_text, 4096) AS query_sample_text,
    count_star AS execution_count,
    sum_timer_wait AS total_latency_picoseconds,
    avg_timer_wait AS average_latency_picoseconds,
    max_timer_wait AS maximum_latency_picoseconds,
    sum_lock_time AS lock_time_picoseconds,
    sum_rows_examined AS rows_examined,
    sum_rows_sent AS rows_sent,
    sum_created_tmp_disk_tables AS temp_disk_tables
FROM performance_schema.events_statements_summary_by_digest
WHERE digest IS NOT NULL
  AND (CONCAT(UPPER(digest), ':', CASE WHEN schema_name IS NULL THEN 'N' ELSE CONCAT('V', schema_name) END) IN (SELECT identity_key FROM tracked)
       OR CONCAT(UPPER(digest), ':', CASE WHEN schema_name IS NULL THEN 'N' ELSE CONCAT('V', schema_name) END) NOT IN (SELECT identity_key FROM excluded))
ORDER BY CASE WHEN CONCAT(UPPER(digest), ':', CASE WHEN schema_name IS NULL THEN 'N' ELSE CONCAT('V', schema_name) END) IN (SELECT identity_key FROM tracked) THEN 0 ELSE 1 END,
LEAST(
    ROW_NUMBER() OVER (ORDER BY sum_timer_wait DESC, schema_name, digest),
    ROW_NUMBER() OVER (ORDER BY sum_timer_wait / NULLIF(count_star, 0) DESC, schema_name, digest),
    ROW_NUMBER() OVER (ORDER BY count_star DESC, schema_name, digest),
    ROW_NUMBER() OVER (ORDER BY sum_rows_examined DESC, schema_name, digest),
    ROW_NUMBER() OVER (ORDER BY sum_rows_sent DESC, schema_name, digest),
    ROW_NUMBER() OVER (ORDER BY sum_created_tmp_disk_tables DESC, schema_name, digest),
    ROW_NUMBER() OVER (ORDER BY sum_lock_time DESC, schema_name, digest)
), sum_timer_wait DESC, schema_name, digest
LIMIT :limit
