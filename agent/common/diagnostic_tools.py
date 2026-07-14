# agent/common/diagnostic_tools.py
"""
数据库深度诊断专家工具箱 (Database Diagnostic Tools)。

提供 16 个核心诊断工具，覆盖数据库故障根因分析 (RCA) 的五大场景。
LLM 通过 Function Calling / 单选题机制从工具箱中选择最合适的工具，
底层根据 db_type 自动路由到 Oracle / PostgreSQL / MySQL 的专家 SQL。

专家 SQL 来源: DBA 沉淀的黄金诊断脚本，针对各数据库最新版本优化。
  - Oracle: 基于 19c/21c/26ai，兼容 CDB/PDB 多租户架构
  - PostgreSQL: 基于 PG 18，需加载 pg_stat_statements 扩展
  - MySQL: 基于 8.0/8.4 LTS，依赖 performance_schema + sys 库

扩展方式:
  - 新增数据库: 在 sql_registry 中添加对应的 db_type 条目即可
  - 新增工具: 添加方法 + SQL 注册表条目
"""

from typing import Any, Dict, List

from loguru import logger


class DatabaseDiagnosticTools:
    """
    数据库根因诊断专家工具箱 (16 大核心金刚)。

    使用方式:
        tools = DatabaseDiagnosticTools(db_type="oracle", db_executor=executor)
        result = await tools.db_lock_chains()
    """

    def __init__(self, db_type: str, db_executor: Any, instance_id: str = ""):
        self.db_type = db_type.lower()
        self.executor = db_executor
        self.instance_id = instance_id

    # ========================================================================
    # 场景一：并发与卡死诊断（锁与事务冲突）
    # ========================================================================

    async def db_lock_chains(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'排查锁冲突'、'查找阻塞源头'、'分析死锁'、
        '查看行级锁/表级锁阻塞关系'、'事务卡死'、'大面积锁死'、'揪出源头会话'时调用。

        功能: 层级化展现阻塞关系，通过树形结构直接定位处于根部的源头会话及其 SQL。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    LEVEL AS tree_level,
                    LPAD(' ', (LEVEL - 1) * 2) || s.sid || ',' || s.serial# AS session_hierarchy,
                    s.username,
                    s.osuser,
                    s.machine,
                    s.blocking_session AS blocker,
                    s.final_blocking_session AS final_blocker,
                    s.status,
                    s.last_call_et AS active_seconds,
                    q.sql_text AS current_sql
                FROM v$session s
                LEFT JOIN v$sql q
                  ON s.sql_id = q.sql_id AND s.sql_child_number = q.child_number
                WHERE s.sid IN (SELECT sid FROM v$session WHERE blocking_session IS NOT NULL)
                   OR s.sid IN (SELECT blocking_session FROM v$session WHERE blocking_session IS NOT NULL)
                CONNECT BY PRIOR s.sid = s.blocking_session
                START WITH s.blocking_session IS NULL
            """,
            "postgresql": """
                WITH RECURSIVE lock_tree AS (
                    SELECT
                        1 AS level,
                        pid AS root_pid,
                        pid,
                        blocking_pids,
                        query,
                        state,
                        backend_type
                    FROM pg_stat_activity
                    WHERE cardinality(blocking_pids) = 0
                      AND pid IN (SELECT unnest(blocking_pids) FROM pg_stat_activity)
                    UNION ALL
                    SELECT
                        t.level + 1,
                        t.root_pid,
                        a.pid,
                        a.blocking_pids,
                        a.query,
                        a.state,
                        a.backend_type
                    FROM pg_stat_activity a
                    JOIN lock_tree t ON t.pid = ANY(a.blocking_pids)
                )
                SELECT
                    level,
                    root_pid AS source_pid,
                    repeat('  ', level - 1) || pid AS blocked_hierarchy,
                    query AS current_sql,
                    state,
                    backend_type
                FROM lock_tree
                ORDER BY root_pid, level
            """,
            "mysql": """
                SELECT
                    r.trx_mysql_thread_id AS waiting_pid,
                    r.trx_query AS waiting_query,
                    b.trx_mysql_thread_id AS blocking_pid,
                    b.trx_query AS blocking_query,
                    sys.format_statement(p.current_statement) AS blocking_current_sql
                FROM information_schema.innodb_trx r
                JOIN performance_schema.data_lock_waits w
                  ON r.trx_id = w.requesting_engine_transaction_id
                JOIN information_schema.innodb_trx b
                  ON w.blocking_engine_transaction_id = b.trx_id
                JOIN sys.processlist p
                  ON b.trx_mysql_thread_id = p.thd_id
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_lock_chains 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_lock_chains → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_lock_matrix(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'分析锁类型'、'查看锁模式'、'判断行锁还是表锁'、
        'DDL元数据锁'、'意向锁冲突'、'长事务未提交引发锁持有'时调用。

        功能: 看清具体是哪张表被锁，以及锁的持有模式和申请模式。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    lo.session_id AS sid,
                    s.serial#,
                    s.username,
                    do.object_name,
                    do.object_type,
                    DECODE(lo.locked_mode,
                           0, 'None', 1, 'Null', 2, 'Row-S (SS)', 3, 'Row-X (SX)',
                           4, 'Share (S)', 5, 'S/Row-X (SSX)', 6, 'Exclusive (X)', 'Unknown') AS lock_mode,
                    s.program,
                    s.machine
                FROM v$locked_object lo
                JOIN dba_objects do ON lo.object_id = do.object_id
                JOIN v$session s    ON lo.session_id = s.sid
                ORDER BY s.sid
            """,
            "postgresql": """
                SELECT
                    a.pid,
                    a.usename,
                    a.client_addr,
                    l.locktype,
                    c.relname AS relation_name,
                    l.mode AS lock_mode,
                    l.granted,
                    a.query AS current_sql,
                    age(clock_timestamp(), a.query_start) AS lock_duration
                FROM pg_locks l
                JOIN pg_stat_activity a ON l.pid = a.pid
                LEFT JOIN pg_class c ON l.relation = c.oid
                WHERE l.pid <> pg_backend_pid()
                  AND (c.relname IS NULL OR c.relname NOT LIKE 'pg_%')
                ORDER BY lock_duration DESC
            """,
            "mysql": """
                SELECT
                    engine_transaction_id AS trx_id,
                    thread_id,
                    object_schema,
                    object_name,
                    index_name,
                    lock_type,
                    lock_mode,
                    lock_status,
                    lock_data
                FROM performance_schema.data_locks
                WHERE lock_type = 'RECORD'
                ORDER BY thread_id
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_lock_matrix 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_lock_matrix → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_distributed_tx(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'排查分布式事务'、'跨库XA事务卡死'、
        '两阶段提交悬挂'、'分布式死锁'、'微服务跨库调用异常'时调用。

        功能: 抓取处于 PREPARED 状态的悬挂 XA 事务。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    local_tran_id,
                    global_tran_id,
                    state,
                    mixed,
                    advice,
                    comment$,
                    fail_time
                FROM dba_2pc_pending
                WHERE state = 'PREPARED'
                ORDER BY fail_time DESC
            """,
            "postgresql": """
                SELECT
                    gid AS global_transaction_id,
                    prepared AS prepared_time,
                    owner,
                    database,
                    age(clock_timestamp(), prepared) AS hanging_duration
                FROM pg_prepared_xacts
                ORDER BY prepared_time ASC
            """,
            "mysql": """
                SELECT
                    format_id,
                    gtrid_length,
                    bqual_length,
                    data AS xa_xid_info
                FROM information_schema.xa_prepared_transactions
                ORDER BY format_id
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_distributed_tx 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_distributed_tx → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    # ========================================================================
    # 场景二：资源暴满诊断（CPU / 内存 / 存储）
    # ========================================================================

    async def db_top_cpu_sql(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'查找消耗CPU最高的SQL'、'查看Top SQL'、
        '定位慢查询'、'排查烂SQL'、'抓取当前高负载语句'、'揪出CPU暴满的罪魁祸首'时调用。

        功能: 按总 CPU 时间降序，抓出消耗资源的 Top N SQL 文本及执行统计。
        """
        sql_registry = {
            "oracle": f"""
                SELECT * FROM (
                    SELECT
                        sql_id,
                        child_number,
                        executions,
                        ROUND(cpu_time / 1000000, 2) AS total_cpu_secs,
                        ROUND(elapsed_time / 1000000, 2) AS total_elapsed_secs,
                        ROUND((cpu_time / DECODE(executions, 0, 1, executions)) / 1000000, 4) AS avg_cpu_secs,
                        disk_reads AS total_physical_reads,
                        buffer_gets AS total_logical_reads,
                        sql_text
                    FROM v$sql
                    ORDER BY cpu_time DESC
                ) WHERE ROWNUM <= {top_n}
            """,
            "postgresql": f"""
                SELECT
                    r.rolname AS user_name,
                    d.datname AS db_name,
                    ROUND(s.total_exec_time::numeric, 2) AS total_time_ms,
                    s.calls AS execution_counts,
                    ROUND((s.total_exec_time / s.calls)::numeric, 4) AS avg_time_ms,
                    s.shared_blks_read AS disk_physical_reads,
                    s.shared_blks_hit AS memory_buffer_hits,
                    s.query AS sql_text
                FROM pg_stat_statements s
                JOIN pg_roles r ON s.userid = r.oid
                JOIN pg_database d ON s.dbid = d.oid
                ORDER BY s.total_exec_time DESC
                LIMIT {top_n}
            """,
            "mysql": f"""
                SELECT
                    query,
                    db,
                    full_scan,
                    exec_count,
                    sys.format_time(total_latency) AS total_latency,
                    sys.format_time(max_latency) AS max_latency,
                    rows_examined,
                    rows_sent
                FROM sys.statement_analysis
                ORDER BY total_latency DESC
                LIMIT {top_n}
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_top_cpu_sql 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_top_cpu_sql (top_n={top_n}) → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_session_memory(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'会话内存泄漏'、'PGA内存爆满'、'查找内存占用最高的会话'、
        '哪个连接吃光了服务器内存'、'排查OOM根因'时调用。

        功能: 揪出哪些会话大量侵占了私有内存（Oracle PGA / PG work_mem / MySQL 会话缓存）。
        """
        sql_registry = {
            "oracle": """
                SELECT * FROM (
                    SELECT
                        s.sid,
                        s.serial#,
                        s.username,
                        s.program,
                        s.machine,
                        ROUND(p.pga_used_mem / 1024 / 1024, 2) AS pga_used_mb,
                        ROUND(p.pga_alloc_mem / 1024 / 1024, 2) AS pga_allocated_mb,
                        ROUND(p.pga_max_mem / 1024 / 1024, 2) AS pga_max_mb
                    FROM v$process p
                    JOIN v$session s ON p.addr = s.paddr
                    ORDER BY p.pga_alloc_mem DESC
                ) WHERE ROWNUM <= 5
            """,
            "postgresql": """
                SELECT
                    pid,
                    usename,
                    client_addr,
                    application_name,
                    age(clock_timestamp(), backend_start) AS session_age,
                    query
                FROM pg_stat_activity
                WHERE backend_type = 'client backend'
                ORDER BY age(clock_timestamp(), query_start) DESC
                LIMIT 5
            """,
            "mysql": """
                SELECT
                    thd_id AS thread_id,
                    conn_id AS connection_id,
                    user,
                    current_allocated,
                    total_allocated
                FROM sys.memory_by_thread_by_current_bytes
                WHERE user IS NOT NULL
                ORDER BY current_allocated_bytes DESC
                LIMIT 5
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_session_memory 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_session_memory → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_tablespace_top_segments(self, tablespace_name: str = "") -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'分析空间由谁占用'、'查找大表/大索引'、
        '看表空间段分布'、'定位大Segment'、'找出疯狂膨胀的对象'、
        '查谁占满了表空间'时调用。

        功能: 返回占用空间最大的 Top 10 段（表/索引），可选按表空间过滤。
        """
        oracle_filter = f"WHERE tablespace_name = '{tablespace_name}'" if tablespace_name else ""
        sql_registry = {
            "oracle": f"""
                SELECT * FROM (
                    SELECT
                        tablespace_name,
                        owner,
                        segment_name,
                        segment_type,
                        ROUND(bytes / 1024 / 1024 / 1024, 2) AS size_gb,
                        extents
                    FROM dba_segments
                    {oracle_filter}
                    ORDER BY bytes DESC
                ) WHERE ROWNUM <= 10
            """,
            "postgresql": f"""
                SELECT
                    c.relname AS object_name,
                    c.relkind AS object_type,
                    n.nspname AS schema_name,
                    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
                  AND c.relkind IN ('r', 'i')
                ORDER BY pg_total_relation_size(c.oid) DESC
                LIMIT 10
            """,
            "mysql": """
                SELECT
                    table_schema,
                    table_name,
                    sys.format_bytes(data_length + index_length) AS total_size,
                    sys.format_bytes(data_length) AS data_size,
                    sys.format_bytes(index_length) AS index_size,
                    table_rows
                FROM information_schema.tables
                WHERE table_schema NOT IN ('sys', 'information_schema', 'mysql', 'performance_schema')
                ORDER BY (data_length + index_length) DESC
                LIMIT 10
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_tablespace_top_segments 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_tablespace_top_segments (ts={tablespace_name or 'ALL'}) → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_tablespace_datafiles(self, tablespace_name: str = "") -> List[Dict[str, Any]]:
        """
        适用场景: 表空间扩容/收缩、文件路径查询。当需要知道表空间的实际数据文件路径、
        当前大小、最大大小、是否自动扩展时调用。

        功能: 查询 DBA_DATA_FILES 获取表空间的数据文件详细信息（含真实 FILE_NAME）。
        """
        sql_registry = {
            "oracle": (
                "SELECT TABLESPACE_NAME, FILE_NAME, BYTES/1024/1024 AS SIZE_MB, "
                "MAXBYTES/1024/1024 AS MAXSIZE_MB, AUTOEXTENSIBLE "
                "FROM DBA_DATA_FILES"
                + (f" WHERE TABLESPACE_NAME = UPPER('{tablespace_name}')" if tablespace_name else "")
                + " ORDER BY TABLESPACE_NAME"
            ),
            "mysql": (
                "SELECT FILE_ID, FILE_NAME, TABLESPACE_NAME, "
                "TOTAL_EXTENTS*EXTENT_SIZE/1024/1024 AS SIZE_MB, "
                "MAXIMUM_SIZE/1024/1024 AS MAXSIZE_MB "
                "FROM INFORMATION_SCHEMA.FILES WHERE FILE_TYPE = 'DATAFILE'"
            ),
            "postgresql": (
                "SELECT spcname AS TABLESPACE_NAME, "
                "pg_tablespace_location(oid) AS FILE_NAME, "
                "0 AS SIZE_MB, 0 AS MAXSIZE_MB, 'NO' AS AUTOEXTENSIBLE "
                "FROM pg_tablespace"
            ),
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_tablespace_datafiles 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_tablespace_datafiles (ts={tablespace_name or 'ALL'}) → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_temp_segments_usage(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'临时表空间爆满'、'大排序撑爆临时空间'、
        '查找临时段消耗者'、'谁在做大排序/大Hash Join'、'Temp使用率告警'时调用。

        功能: 抓出当前正在消耗临时表空间的会话及 SQL。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    s.sid,
                    s.serial#,
                    s.username,
                    s.program,
                    su.blocks * 8192 / 1024 / 1024 AS temp_used_mb,
                    su.contents,
                    su.segtype,
                    q.sql_id,
                    q.sql_text
                FROM v$sort_usage su
                JOIN v$session s ON su.session_addr = s.saddr
                LEFT JOIN v$sql q ON s.sql_id = q.sql_id AND s.sql_child_number = q.child_number
                ORDER BY temp_used_mb DESC
            """,
            "postgresql": """
                SELECT
                    pid,
                    usename,
                    query,
                    age(clock_timestamp(), query_start) AS active_duration
                FROM pg_stat_activity
                WHERE state = 'active'
                ORDER BY active_duration DESC
                LIMIT 10
            """,
            "mysql": """
                SELECT
                    thd_id AS thread_id,
                    conn_id AS connection_id,
                    user,
                    current_statement,
                    statement_latency,
                    disk_tables,
                    memory_tables
                FROM sys.session
                WHERE disk_tables > 0 OR memory_tables > 0
                ORDER BY disk_tables DESC, statement_latency DESC
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_temp_segments_usage 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_temp_segments_usage → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    # ========================================================================
    # 场景三：性能瓶颈诊断（等待事件与线程现场）
    # ========================================================================

    async def db_active_session_wait(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'查看当前等待事件'、'活跃会话在等什么'、
        '数据库慢但CPU空闲'、'分析等待瓶颈'、'会话现场快照'、'定位数据库卡顿本质'时调用。

        功能: 实时现场切片。展示当前活跃会话正在等待什么事件及具体的 p1/p2/p3 参数。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    s.sid,
                    s.serial#,
                    s.username,
                    s.status,
                    s.event,
                    s.p1, s.p2, s.p3,
                    s.seconds_in_wait AS curr_wait_secs,
                    s.wait_class,
                    q.sql_id,
                    SUBSTR(q.sql_text, 1, 100) AS sql_snippet
                FROM v$session s
                LEFT JOIN v$sql q ON s.sql_id = q.sql_id AND s.sql_child_number = q.child_number
                WHERE s.status = 'ACTIVE'
                  AND s.wait_class <> 'Idle'
            """,
            "postgresql": """
                SELECT
                    pid,
                    usename,
                    state,
                    wait_event_type,
                    wait_event,
                    query,
                    age(clock_timestamp(), query_start) AS query_duration
                FROM pg_stat_activity
                WHERE state = 'active'
                  AND wait_event IS NOT NULL
                ORDER BY query_duration DESC
            """,
            "mysql": """
                SELECT
                    processlist_id AS conn_id,
                    s.thread_id,
                    processlist_user AS user,
                    processlist_command AS command,
                    processlist_state AS state,
                    current_connection_stage,
                    last_wait_event,
                    sys.format_time(last_wait_latency) AS wait_latency
                FROM sys.session s
                JOIN performance_schema.threads t ON s.thd_id = t.thread_id
                LEFT JOIN performance_schema.events_waits_current w ON t.thread_id = w.thread_id
                WHERE processlist_state IS NOT NULL
                  AND processlist_command <> 'Sleep'
                ORDER BY last_wait_latency DESC
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_active_session_wait 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_active_session_wait → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_historical_session_history(self, minutes_ago: int = 60) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'回溯历史等待事件'、'查看ASH/AWR采样'、
        '复现半夜故障瞬间现场'、'历史活动会话分析'时调用。

        功能: 基于历史采样数据复盘过去一段时间的等待事件分布和引发源。
        """
        sql_registry = {
            "oracle": f"""
                SELECT * FROM (
                    SELECT
                        h.sample_time,
                        h.session_id AS sid,
                        h.user_id,
                        u.username,
                        h.sql_id,
                        h.event,
                        h.wait_class,
                        COUNT(*) AS sample_count_weight
                    FROM v$active_session_history h
                    LEFT JOIN dba_users u ON h.user_id = u.user_id
                    WHERE h.sample_time > SYSDATE - {minutes_ago}/1440
                      AND h.wait_class <> 'Idle'
                    GROUP BY h.sample_time, h.session_id, h.user_id, u.username, h.sql_id, h.event, h.wait_class
                    ORDER BY sample_count_weight DESC
                ) WHERE ROWNUM <= 10
            """,
            "postgresql": """
                SELECT
                    pid,
                    event_type AS wait_event_type,
                    event AS wait_event,
                    count AS sample_count_weight
                FROM pg_wait_sampling_profile
                ORDER BY count DESC
                LIMIT 10
            """,
            "mysql": """
                SELECT
                    event_name,
                    count_star AS trigger_count,
                    sys.format_time(sum_timer_wait) AS total_time,
                    sys.format_time(avg_timer_wait) AS avg_time
                FROM performance_schema.events_waits_summary_global_by_event_name
                WHERE count_star > 0
                  AND event_name NOT LIKE 'wait/synch/mutex/sql/%'
                ORDER BY sum_timer_wait DESC
                LIMIT 10
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_historical_session_history 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_historical_session_history (mins={minutes_ago}) → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_undo_segments_usage(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'UNDO爆满'、'回滚段压力'、'大事务长时间未提交'、
        '排查ORA-01555快照过旧'、'UNDO使用率告警'时调用。

        功能: 揪出产生海量 Undo / 回滚记录的大事务。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    s.sid,
                    s.serial#,
                    s.username,
                    t.start_time,
                    t.used_urec AS undo_records,
                    ROUND(t.used_ublk * 8192 / 1024 / 1024, 2) AS undo_size_mb,
                    q.sql_id,
                    q.sql_text
                FROM v$transaction t
                JOIN v$session s ON t.ses_addr = s.saddr
                LEFT JOIN v$sql q ON s.sql_id = q.sql_id AND s.sql_child_number = q.child_number
                ORDER BY undo_size_mb DESC
            """,
            "postgresql": """
                SELECT
                    pid,
                    usename,
                    backend_xid AS current_tx_id,
                    state,
                    age(clock_timestamp(), xact_start) AS transaction_duration,
                    query AS tx_last_sql
                FROM pg_stat_activity
                WHERE state IN ('active', 'idle in transaction')
                  AND backend_xid IS NOT NULL
                ORDER BY transaction_duration DESC
                LIMIT 5
            """,
            "mysql": """
                SELECT
                    t.trx_mysql_thread_id AS conn_id,
                    t.trx_id,
                    t.trx_state AS state,
                    t.trx_started AS start_time,
                    t.trx_rows_modified AS modified_rows,
                    sys.format_statement(p.current_statement) AS current_sql
                FROM information_schema.innodb_trx t
                JOIN sys.processlist p ON t.trx_mysql_thread_id = p.thd_id
                ORDER BY t.trx_rows_modified DESC, t.trx_started ASC
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_undo_segments_usage 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_undo_segments_usage → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    # ========================================================================
    # 场景四：变更与审计诊断（怀疑有人动了系统）
    # ========================================================================

    async def db_recent_ddl_changes(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'查看最近DDL变更'、'排查谁改过表结构'、
        '审计索引删除'、'近期ALTER/DROP/CREATE操作'、'故障由变更引起'时调用。

        功能: 抓取过去 N 小时内被修改过的数据库对象结构。
        """
        sql_registry = {
            "oracle": f"""
                SELECT
                    owner,
                    object_name,
                    object_type,
                    created,
                    last_ddl_time,
                    timestamp,
                    status
                FROM dba_objects
                WHERE last_ddl_time > SYSDATE - {hours}/24
                  AND owner NOT IN ('SYS', 'SYSTEM', 'AUDSYS', 'MDSYS')
                ORDER BY last_ddl_time DESC
            """,
            "postgresql": """
                SELECT
                    c.relname AS object_name,
                    n.nspname AS schema_name,
                    CASE c.relkind
                        WHEN 'r' THEN 'table' WHEN 'i' THEN 'index' WHEN 'v' THEN 'view'
                    END AS object_type,
                    pg_xact_commit_timestamp(c.xmin) AS approximate_modification_time
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                ORDER BY approximate_modification_time DESC NULLS LAST
                LIMIT 10
            """,
            "mysql": """
                SELECT
                    event_id,
                    argument AS ddl_sql_text,
                    sys.format_time(timer_wait) AS duration,
                    user_name,
                    host_name
                FROM performance_schema.events_statements_history
                WHERE (argument LIKE 'ALTER%' OR argument LIKE 'DROP%' OR argument LIKE 'CREATE%')
                  AND argument NOT LIKE '%performance_schema%'
                ORDER BY event_id DESC
                LIMIT 10
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_recent_ddl_changes 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_recent_ddl_changes (hours={hours}) → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_invalid_objects(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'查找失效对象'、'排查INVALID状态'、
        '存储过程/触发器/视图失效'、'索引不可用'、'DDL变更导致级联失效'时调用。

        功能: 排查全库处于 INVALID / 不可用状态的对象。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    owner,
                    object_name,
                    object_type,
                    status,
                    created,
                    last_ddl_time
                FROM dba_objects
                WHERE status = 'INVALID'
                  AND owner NOT IN ('SYS', 'SYSTEM')
                ORDER BY owner, object_type
            """,
            "postgresql": """
                SELECT
                    n.nspname AS schema_name,
                    c.relname AS table_name,
                    i.relname AS invalid_index_name,
                    idx.indisvalid
                FROM pg_index idx
                JOIN pg_class i ON i.oid = idx.indexrelid
                JOIN pg_class c ON c.oid = idx.indrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE idx.indisvalid = false
                ORDER BY schema_name, table_name
            """,
            "mysql": """
                SELECT
                    table_schema AS view_schema,
                    table_name AS view_name
                FROM information_schema.views v
                WHERE NOT EXISTS (
                    SELECT 1 FROM information_schema.tables t
                    WHERE t.table_schema = v.table_schema AND t.table_name = v.table_name
                )
                ORDER BY view_schema, view_name
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_invalid_objects 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_invalid_objects → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_non_default_parameters(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'核对初始化参数基线'、'查看非默认参数'、
        '配置审计'、'排查参数被动态修改'、'比对参数变动'时调用。

        功能: 列出所有被修改过的非默认初始化参数，含修改来源。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    name,
                    type,
                    value,
                    display_value,
                    isdefault,
                    ismodified,
                    description
                FROM v$parameter
                WHERE isdefault = 'FALSE'
                ORDER BY name
            """,
            "postgresql": """
                SELECT
                    name,
                    setting,
                    unit,
                    category,
                    source,
                    short_desc
                FROM pg_settings
                WHERE source NOT IN ('default', 'override')
                ORDER BY category, name
            """,
            "mysql": """
                SELECT
                    variable_name AS parameter_name,
                    variable_value AS parameter_value,
                    variable_source AS source
                FROM performance_schema.variables_info
                WHERE variable_source NOT IN ('COMPILED', 'DEFAULT')
                ORDER BY parameter_name
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_non_default_parameters 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_non_default_parameters → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    # ========================================================================
    # 场景五：集群与高可用状态诊断（主备/分布式节点）
    # ========================================================================

    async def db_replication_lag_status(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'主备延迟'、'复制链路异常'、'备库日志重做卡住'、
        'DataGuard状态检查'、'流复制延迟'、'主备同步中断'时调用。

        功能: 查明主备复制的传输延迟和应用延迟。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    name,
                    value,
                    datum_time,
                    time_computed
                FROM v$dataguard_stats
                WHERE name IN ('transport lag', 'apply lag', 'finish time')
            """,
            "postgresql": """
                SELECT
                    application_name,
                    client_addr AS standby_ip,
                    state,
                    sync_state,
                    age(clock_timestamp(), reply_time) AS replay_lag_time,
                    pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn) AS sent_lag_bytes,
                    pg_wal_lsn_diff(sent_lsn, write_lsn) AS write_lag_bytes,
                    pg_wal_lsn_diff(write_lsn, flush_lsn) AS flush_lag_bytes,
                    pg_wal_lsn_diff(flush_lsn, replay_lsn) AS replay_lag_bytes
                FROM pg_stat_replication
            """,
            "mysql": """
                SELECT
                    channel_name,
                    service_state AS connection_status,
                    remaining_delay AS scheduled_delay,
                    count_received_heartbeats AS heartbeats
                FROM performance_schema.replication_connection_status
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_replication_lag_status 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_replication_lag_status → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_cluster_interconnect_wait(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'RAC集群节点间竞争'、'gc buffer busy'、
        '私网交换延迟'、'集群内耗'、'cache fusion问题'、
        'MGR组复制流控'、'Citus/Patroni分布式等待'时调用。

        功能: 分析集群节点间缓存同步与网络竞争。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    inst_id,
                    event,
                    total_waits,
                    total_timeouts,
                    time_waited_micro / 1000000 AS time_waited_secs,
                    average_wait / 100 AS avg_wait_ms
                FROM gv$system_event
                WHERE event LIKE 'gc%'
                  AND wait_class = 'Cluster'
                ORDER BY time_waited_micro DESC
            """,
            "postgresql": """
                SELECT
                    pid,
                    state,
                    wait_event_type,
                    wait_event,
                    query
                FROM pg_stat_activity
                WHERE wait_event_type IN ('Extension', 'Client')
                  AND state = 'active'
            """,
            "mysql": """
                SELECT
                    member_id,
                    member_host,
                    member_state,
                    COUNT_TRANSACTIONS_IN_QUEUE AS queue_depth,
                    COUNT_TRANSACTIONS_LOCAL_ROLLBACK AS local_rollbacks
                FROM performance_schema.replication_group_member_stats
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_cluster_interconnect_wait 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_cluster_interconnect_wait → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    async def db_archivelog_dest_status(self) -> List[Dict[str, Any]]:
        """
        适用场景: 当手册提到'归档日志写满'、'归档目录空间不足'、
        '归档切换频率异常'、'ARCH进程卡住'、'数据库写入挂起风险'、
        'WAL归档失败'、'Binlog磁盘爆满'时调用。

        功能: 检查归档/Binlog/WAL 的状态与空间预警。
        """
        sql_registry = {
            "oracle": """
                SELECT
                    dest_id,
                    dest_name,
                    status,
                    binding,
                    target,
                    destination,
                    error
                FROM v$archive_dest
                WHERE status <> 'INACTIVE'
            """,
            "postgresql": """
                SELECT
                    archived_count AS success_archive_count,
                    last_archived_wal,
                    last_archived_time,
                    failed_count AS failed_archive_count,
                    last_failed_wal,
                    last_failed_time,
                    stats_reset AS counters_reset_time
                FROM pg_stat_archiver
            """,
            "mysql": """
                SELECT
                    file_name,
                    file_size,
                    sys.format_bytes(file_size) AS formatted_size
                FROM performance_schema.file_instances
                WHERE file_name LIKE '%binlog%' OR file_name LIKE '%mysql-bin%'
                ORDER BY file_size DESC
            """
        }
        sql = sql_registry.get(self.db_type)
        if not sql:
            raise NotImplementedError(f"db_archivelog_dest_status 暂不支持: {self.db_type}")
        logger.info(f"[DiagnosticTools] db_archivelog_dest_status → {self.db_type}")
        return await self.executor.execute_readonly_ops_sql(self.instance_id, sql)

    # ========================================================================
    # 工具清单辅助方法 (供 LLM Prompt 注入)
    # ========================================================================

    @classmethod
    def get_tool_manifest(cls) -> str:
        """
        返回所有 17 个诊断工具的清单文本，供 LLM Prompt 注入。
        LLM 通过此清单做"单选题"，选择最匹配的工具。
        """
        return """
【可用的数据库深度诊断工具箱 (Database Diagnostic Tools)】

当监控指标发现异常，需要深入数据库内部查证根因时，你只能从以下 17 个工具中选择。
请根据参考手册中的排查动作，输出精确的 tool_name：

场景一：并发与卡死诊断（锁与事务冲突）
  1. db_lock_chains()
     适用: 排查锁冲突、查找阻塞源头、分析死锁、大面积锁死、揪出源头会话
  2. db_lock_matrix()
     适用: 分析锁类型/锁模式、判断行锁还是表锁、DDL元数据锁、长事务未提交引发锁持有
  3. db_distributed_tx()
     适用: 排查分布式事务、跨库XA事务卡死、两阶段提交悬挂、分布式死锁

场景二：资源暴满诊断（CPU/内存/存储）
  4. db_top_cpu_sql(top_n=5)
     适用: 查找消耗CPU最高的SQL、查看Top SQL、定位慢查询、排查烂SQL、揪出CPU暴满的罪魁祸首
  5. db_session_memory()
     适用: 会话内存泄漏、PGA内存爆满、查找内存占用最高的会话、排查OOM根因
  6. db_tablespace_top_segments(tablespace_name="")
     适用: 分析空间由谁占用、查找大表/大索引、看表空间段分布、定位疯狂膨胀的对象
  7. db_tablespace_datafiles(tablespace_name="")
     适用: 查询表空间数据文件真实路径、当前大小、最大大小、是否自动扩展。表空间扩容/收缩前必须调用此工具获取真实 FILE_NAME
  8. db_temp_segments_usage()
     适用: 临时表空间爆满、大排序撑爆临时空间、查找Temp消耗者

场景三：性能瓶颈诊断（等待事件与线程现场）
  9. db_active_session_wait()
     适用: 查看当前等待事件、活跃会话在等什么、数据库慢但CPU空闲、定位卡顿本质
  10. db_historical_session_history(minutes_ago=60)
     适用: 回溯历史等待事件、查看ASH/AWR采样、复现半夜故障瞬间现场
  11. db_undo_segments_usage()
      适用: UNDO爆满、回滚段压力、大事务长时间未提交、排查ORA-01555快照过旧

场景四：变更与审计诊断（怀疑有人动了系统）
  12. db_recent_ddl_changes(hours=24)
      适用: 查看最近DDL变更、排查谁改过表结构、审计索引删除、查找引发性能暴跌的变更
  13. db_invalid_objects()
      适用: 查找失效对象、排查INVALID状态、存储过程/触发器/视图/索引失效
  14. db_non_default_parameters()
      适用: 核对初始化参数基线、查看非默认参数、配置审计、排查参数被动态修改

场景五：集群与高可用状态诊断（主备/分布式节点）
  15. db_replication_lag_status()
      适用: 主备延迟、复制链路异常、备库重做卡住、DataGuard/流复制/主从状态检查
  16. db_cluster_interconnect_wait()
      适用: RAC集群节点间竞争、gc buffer busy、私网交换延迟、MGR组复制流控
  16. db_archivelog_dest_status()
      适用: 归档日志写满、归档目录空间不足、ARCH进程卡住、WAL归档失败、Binlog磁盘爆满
"""
