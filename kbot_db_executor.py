"""数据库执行微服务应用程序。"""

import os
import time
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from loguru import logger

from core.config.settings import get_app_config, get_executor_config
from core.logger import LogConfig, LogManager
from core.middleware.log_middleware import log_requests

# 导入内部逻辑
from microservices.db_executor.factory import DriverFactory
from microservices.db_executor.security.sql_validator import SQLValidator
from microservices.db_executor.schemas.executor import ExecuteRequest, OpsExecuteRequest

# 加载环境变量
load_dotenv(Path(__file__).parent / ".env")

# --- 配置获取 ---
executor_config = get_executor_config()
SERVICE_NAME = executor_config.service_name
SERVICE_VERSION = executor_config.service_version
SERVICE_HOST = executor_config.service_host
SERVICE_PORT = executor_config.service_port

# 日志与调试配置
app_config = get_app_config()
DEBUG: bool = app_config.debug
LOG_DIR: str = app_config.log.dir
LOG_LEVEL: str = app_config.log.level
LOG_ROTATION: str = app_config.log.rotation
LOG_RETENTION: str = app_config.log.retention

# OPS token 已合并至统一的 create_internal_auth_middleware（KBOT_INTERNAL_SERVICE_TOKEN）

@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理服务生命周期。"""
    
    app.state.service_name = SERVICE_NAME

    # 1. 初始化日志系统
    log_conf = LogConfig(
        service_name=SERVICE_NAME, 
        log_dir=LOG_DIR, 
        level=LOG_LEVEL, 
        rotation=LOG_ROTATION, 
        retention=LOG_RETENTION
    )
    LogManager(log_conf).setup()
    
    start_time = time.time()
    logger.info(f"正在启动 [{SERVICE_NAME}] | PID: {os.getpid()} | 时间: {datetime.now()}")

    yield  # 运行阶段
    
    # 2. 清理阶段
    logger.info("应用正在关闭，执行清理任务...")
    # 由于 db_executor 采用短连接策略，目前无全局长连接池需要释放
    logger.success(f"[{SERVICE_NAME}] 已安全关闭")

# 创建应用实例
app = FastAPIOffline(
    title=f"{SERVICE_NAME} API",
    description="NexusCube 数据库执行微服务：支持多数据库动态连接与 SQL 安全审计。",
    version=SERVICE_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
app.middleware("http")(log_requests)

# 内部服务认证中间件 (保护轨道A, 轨道B有独立OpsToken保护故放行)
from microservices.common.security import create_internal_auth_middleware, PUBLIC_PATHS
DB_EXECUTOR_PUBLIC_PATHS = PUBLIC_PATHS
app.middleware("http")(create_internal_auth_middleware(public_paths=DB_EXECUTOR_PUBLIC_PATHS))

# =====================================================================
# 轨道 A 接口：原有业务 Text-to-SQL 查询入口 (保持完全兼容，只读)
# =====================================================================

@app.post("/api/v1/execute", tags=["Database"], summary="执行 SQL 查询")
async def execute(request: ExecuteRequest) -> dict[str, Any]:
    """
    接收连接配置和 SQL，安全验证后执行查询。
    """
    # 1. 安全校验
    is_safe, error_msg = SQLValidator.validate(request.sql)
    if not is_safe:
        logger.warning(f"检测到非法 SQL 请求 | KB_ID: {request.kb_id} | 错误: {error_msg}")
        raise HTTPException(status_code=403, detail=error_msg)

    # 2. 注入物理限流
    final_sql = SQLValidator.inject_limit(request.sql, request.db_type, request.limit or 100)

    # 3. 动态调度执行
    try:
        driver = DriverFactory.get_driver(request.db_type, request.connection_config)
        
        # 建立连接
        await driver.connect()
        try:
            logger.info(f"正在执行查询 | 类型: {request.db_type} | KB_ID: {request.kb_id}")
            df = await driver.execute_query(final_sql)
            
            return {
                "kb_id": request.kb_id,
                "status": "success",
                "row_count": len(df),
                "data": driver.format_results(df),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # 确保释放连接
            await driver.close()
            
    except Exception as e:
        logger.error(f"SQL 执行失败: {e}")
        # raise HTTPException(status_code=500, detail=f"Database Execution Error: {str(e)}")
        return {
            "kb_id": request.kb_id,
            "status": "error",
            "error_message": f"数据库执行错误: {e}",
            "timestamp": datetime.now().isoformat()
        }

# =====================================================================
# 轨道 B 接口：全新智能运维专职内核执行入口 (支持只读探测与控制面变更)
# =====================================================================
@app.post("/api/v1/ops/execute", tags=["Ops-Track"], summary="[运维线] 执行内核指标探测与高危变更")
async def ops_execute(request: OpsExecuteRequest) -> dict[str, Any]:
    """
    专门承接自愈系统、DBARootAgent 的内核调用。
    支持 READ_ONLY 黄金指标库查询与 MUTATION 会话强杀等高危变更动作。
    """
    logger.warning(
        f"[OpsService] 收到运维核心调用 | 实例: {request.instance_id} "
        f"| 模式: {request.run_mode} | 环境: {request.environment}"
    )
    logger.debug(
        f"[OpsService] connection_config: host={request.connection_config.get('host')}, "
        f"port={request.connection_config.get('port')}, "
        f"service_name={request.connection_config.get('service_name')}, "
        f"dsn={request.connection_config.get('dsn')}, "
        f"user={request.connection_config.get('user')}"
    )

    # 1. 运维线专属的白名单语义审计（不使用业务层盲防注入的 SQLValidator）
    if request.run_mode == "mutation":
        is_valid_ops_cmd = False
        upper_sql = request.sql.upper().strip()

        if request.db_type == "oracle":
            _allowed_oracle = (
                "ALTER SYSTEM KILL SESSION", "DISCONNECT SESSION",
                "ALTER TABLESPACE", "ALTER DATABASE DATAFILE",
                "ALTER SYSTEM SWITCH LOGFILE", "ALTER SYSTEM CHECKPOINT",
                "ALTER SYSTEM SET ", "ALTER SESSION ",
                "PURGE ", "TRUNCATE TABLE ",
                "BEGIN ", "EXEC ", "EXECUTE ", "CALL ",
                "DBMS_SPM", "DBMS_STATS", "DBMS_SPACE", "DBMS_SPACE_ADMIN",
                "DBMS_SCHEDULER", "DBMS_AUTO_TASK_ADMIN", "DBMS_AUTO_INDEX",
                "DBMS_SQLTUNE", "DBMS_ADVISOR",
                "DBMS_SPM_INTERNAL", "DBMS_AUTO_INDEX_INTERNAL",
            )
            is_valid_ops_cmd = any(p in upper_sql for p in _allowed_oracle)
        elif request.db_type == "mysql":
            _allowed_mysql = (
                "KILL ", "ALTER TABLE", "ALTER DATABASE",
                "SET GLOBAL ", "FLUSH ", "OPTIMIZE TABLE",
                "ANALYZE TABLE", "TRUNCATE TABLE",
                "PURGE BINARY LOGS",
            )
            is_valid_ops_cmd = any(upper_sql.startswith(p) or p in upper_sql for p in _allowed_mysql)
        elif request.db_type in ("postgresql", "postgres"):
            _allowed_pg = (
                "ALTER TABLESPACE", "ALTER DATABASE", "ALTER SYSTEM",
                "ALTER TABLE", "SELECT pg_terminate_backend",
                "SELECT pg_cancel_backend", "VACUUM", "ANALYZE",
                "REINDEX", "CHECKPOINT",
            )
            is_valid_ops_cmd = any(p in upper_sql for p in _allowed_pg)

        if not is_valid_ops_cmd:
            logger.critical(f"安全熔断：检测到非法的生产运维变更指令！| 实例: {request.instance_id} | SQL: {request.sql}")
            raise HTTPException(status_code=400, detail="拒绝执行：该命令未在 DBA 专家自愈变更安全白名单中放行。")
    else:
        # 如果是只读探测类（read_only），为防大表扫描，强行注入较小的物理限制
        request.sql = SQLValidator.inject_limit(request.sql, request.db_type, request.limit or 50)

    # 2. 动态调度执行
    try:
        # 清洗连接配置：过滤掉 None/空值，避免 oracledb 因无效 DSN 报 DPY-4021
        clean_config = {k: v for k, v in request.connection_config.items() if v is not None and v != ""}
        logger.debug(f"[OpsService] 清洗后 connection_config keys: {list(clean_config.keys())}")
        driver = DriverFactory.get_driver(request.db_type, clean_config)
        await driver.connect()
        try:
            logger.info(f"数据库驱动正在向实例 [{request.instance_id}] 投递指令...")

            # 🎯 参数化查询：若携带 params 则使用驱动级防注入参数绑定
            effective_sql = request.sql
            if request.params and request.run_mode == "read_only":
                effective_sql, bound_params = _convert_named_params(
                    request.sql, request.params, request.db_type
                )
            else:
                bound_params = None

            # 运维通常涉及无结果集返回的管理指令，或者是指标结果集
            if request.run_mode == "mutation":
                # 调用驱动底层的非查询管理命令执行器
                if bound_params:
                    await driver.execute_non_query(effective_sql, bound_params)
                else:
                    await driver.execute_non_query(request.sql)

                return {
                    "instance_id": request.instance_id,
                    "status": "success",
                    "message": "运维变更指令已成功在数据库内核级安全落盘执行。",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # 正常指标读取 — 优先使用参数化查询
                if bound_params:
                    df = await driver.execute_query(effective_sql, bound_params)
                else:
                    df = await driver.execute_query(request.sql)
                return {
                    "instance_id": request.instance_id,
                    "status": "success",
                    "row_count": len(df),
                    "data": driver.format_results(df),
                    "timestamp": datetime.now().isoformat()
                }
        finally:
            await driver.close()
            
    except Exception as e:
        logger.error(f"运维指令在实例 [{request.instance_id}] 上执行彻底崩溃: {e}")
        return {
            "instance_id": request.instance_id,
            "status": "error",
            "error_message": f"物理内核执行异常: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
    

@app.get("/health", tags=["System"], summary="健康检查")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "timestamp": datetime.now().isoformat()
    }

import re

def _convert_named_params(sql: str, params: dict[str, Any], db_type: str) -> tuple[str, dict | list]:
    """
    将命名占位符 SQL（如 WHERE sid = :sid）转换为各数据库驱动的原生参数化格式。
    返回 (converted_sql, ordered_params_tuple)
    """
    # 提取 SQL 中所有的 :param_name 占位符（按出现顺序）
    placeholders = re.findall(r":(\w+)", sql)
    ordered_values = []
    converted_sql = sql

    if db_type == "postgresql":
        # PostgreSQL asyncpg: $1, $2, ...
        for i, name in enumerate(placeholders):
            if name in params:
                converted_sql = converted_sql.replace(f":{name}", f"${i + 1}", 1)
                ordered_values.append(params[name])
    elif db_type == "mysql":
        # MySQL aiomysql: %s
        for name in placeholders:
            if name in params:
                converted_sql = converted_sql.replace(f":{name}", "%s", 1)
                ordered_values.append(params[name])
    else:
        # Oracle oracledb: 原生支持 :name 命名占位符，直接透传 dict
        for name in placeholders:
            if name in params:
                ordered_values.append(params[name])
        # Oracle 驱动收 dict，沿用原始 SQL
        return sql, params

    return converted_sql, ordered_values


# --- 启动逻辑 ---

if __name__ == "__main__":
    logger.info(f"服务启动中 -> {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(
        app, 
        host=SERVICE_HOST, 
        port=SERVICE_PORT,
        log_config=None,
        loop="asyncio" 
    )