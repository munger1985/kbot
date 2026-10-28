-- 仅初始化缺失的基础设施策略，不覆盖已有额度，不初始化管理员或业务数据。
MERGE INTO KBOT_OPS_RESOURCE_POOL pool
USING (SELECT 'READONLY' AS pool_key FROM DUAL) seed
ON (pool.pool_key = seed.pool_key)
WHEN NOT MATCHED THEN INSERT
    (pool_key, global_limit, target_limit, source_limit, cleanup_grace_seconds)
VALUES (seed.pool_key, 8, 2, 4, 30);
