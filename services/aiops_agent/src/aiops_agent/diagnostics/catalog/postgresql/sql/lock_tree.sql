WITH RECURSIVE lock_edges AS (
    SELECT
        waiter.pid AS waiting_session_id,
        blocker.blocking_session_id
    FROM pg_stat_activity waiter
    CROSS JOIN LATERAL unnest(pg_blocking_pids(waiter.pid))
        AS blocker(blocking_session_id)
), lock_tree AS (
    SELECT
        edge.waiting_session_id,
        edge.blocking_session_id,
        1 AS chain_depth,
        ARRAY[edge.waiting_session_id, edge.blocking_session_id] AS path
    FROM lock_edges edge
    UNION ALL
    SELECT
        tree.waiting_session_id,
        edge.blocking_session_id,
        tree.chain_depth + 1,
        tree.path || edge.blocking_session_id
    FROM lock_tree tree
    JOIN lock_edges edge
      ON edge.waiting_session_id = tree.blocking_session_id
    WHERE tree.chain_depth < 16
      AND NOT edge.blocking_session_id = ANY(tree.path)
)
SELECT
    tree.waiting_session_id,
    waiter.usename AS waiting_username,
    waiter.datname AS waiting_database,
    waiter.wait_event_type,
    waiter.wait_event,
    waiter.xact_start AS waiting_transaction_started_at,
    waiter.query_start AS waiting_query_started_at,
    CAST(waiter.query_id AS text) AS waiting_query_id,
    LEFT(waiter.query, 2048) AS waiting_query_text,
    tree.blocking_session_id,
    blocker.usename AS blocking_username,
    blocker.datname AS blocking_database,
    blocker.state AS blocking_state,
    blocker.xact_start AS blocking_transaction_started_at,
    blocker.query_start AS blocking_query_started_at,
    CAST(blocker.query_id AS text) AS blocking_query_id,
    LEFT(blocker.query, 2048) AS blocking_query_text,
    tree.chain_depth
FROM lock_tree tree
LEFT JOIN pg_stat_activity waiter
  ON waiter.pid = tree.waiting_session_id
LEFT JOIN pg_stat_activity blocker
  ON blocker.pid = tree.blocking_session_id
ORDER BY tree.waiting_session_id, tree.chain_depth, tree.blocking_session_id
LIMIT :limit
