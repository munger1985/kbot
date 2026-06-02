import json
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import AsyncIterator
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from core.config.settings import get_settings
from core.exceptions import DataNotFoundException, NotFoundError

# ==============================================================================
# 1. 基础配置加载与数据库 URL 构建
# ==============================================================================
db_config = get_settings()
username = db_config.oracle.username
password = db_config.oracle.password
host = db_config.oracle.host
port = db_config.oracle.port
service_name = db_config.oracle.service_name

# 使用 thin 模式的异步 oracledb 驱动连接串
url = f"oracle+oracledb://{username}:{password}@{host}:{port}/?service_name={service_name}"

# 连接池与 SQLAlchemy 行为微调参数
echo = db_config.sqlalchemy.echo
pool_size = db_config.sqlalchemy.pool_size
pool_timeout = db_config.sqlalchemy.pool_timeout
max_overflow = db_config.sqlalchemy.max_overflow
pool_pre_ping = db_config.sqlalchemy.pool_pre_ping
pool_recycle = db_config.sqlalchemy.pool_recycle
pool_use_lifo = db_config.sqlalchemy.pool_use_lifo


# ==============================================================================
# 2. 声明增强型 JSON 序列化/反序列化处理器（处理 Oracle 23ai OSON/JSON 映射）
# ==============================================================================
def extended_json_dumps(obj, **kwargs) -> str:
    """增强型 JSON 编码器：无损兼容 Oracle 变长数值中的 Decimal 类型，并完美防原生中文转义。"""
    def default_encoder(item):
        if isinstance(item, Decimal):
            # 将高精度 Decimal 安全转换为 float（若业务要求绝对精度，可改为 str(item)）
            return float(item)
        raise TypeError(f"Object of type {item.__class__.__name__} is not JSON serializable")
    
    kwargs.setdefault('ensure_ascii', False)
    return json.dumps(obj, default=default_encoder, **kwargs)


def flexible_json_loads(value):
    """自适应 JSON 解码器：兼容驱动层已自动反序列化的复合对象或原生字节/字符串。"""
    if value is None:
        return None
    # 若驱动层（oracledb）已提前将数据解构成结构化字典/列表，直接放行
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


# ==============================================================================
# 3. 异步引擎 (Engine) 实例化与单例静态补丁注入
# ==============================================================================
try:
    async_engine = create_async_engine(
        url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,      # 🚀 探针防御：每次借出连接前主动检测可用性，预防 ORA-03113
        pool_recycle=pool_recycle,        # 🚀 定时回收：防止 Oracle 侧因 idle_time 强行切断导致死连
        pool_timeout=pool_timeout,
        pool_use_lifo=pool_use_lifo,      # 🚀 后进先出：优先复用活跃连接，最大化共享连接利用率
        future=True
    )
    
    # 🎯【最佳实践核心修复】
    # 严禁将处理器覆盖写在 @event.listens_for("connect") 事件中！
    # 应该在 Engine 刚刚初始化完成后的单线程安全期，直接静态注入到全局全局单例 dialect 中，
    # 从物理层面上彻底根除多协程高并发竞争（asyncio.gather）时参数编译命名空间踩踏 Bug。
    async_engine.dialect._json_serializer = extended_json_dumps    # type: ignore
    async_engine.dialect._json_deserializer = flexible_json_loads  # type: ignore
    
    logger.info("Async database engine initialized and JSON dialect patched successfully (Thread-Safe)")

except Exception as e:
    logger.critical(f"Failed to create async database engine: {str(e)}", exc_info=True)
    raise RuntimeError(f"Failed to create async database engine: {str(e)}") from e


# ==============================================================================
# 4. 异步会话工厂 (Session Factory) 绑定
# ==============================================================================
async_session = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False,  # RAG 场景高频只读/检索，关闭过期机制可大幅提升内存对象访问性能
    class_=AsyncSession,
    autoflush=False
)


# ==============================================================================
# 5. 上下文管理器与系统级生命周期管控
# ==============================================================================
@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    异步数据库会话上下文管理器。
    支持自动事务提交、一键防爆回滚与彻底的物理连接释放防泄漏卡点。
    
    Yields:
        AsyncSession: 隔离的 SQLAlchemy 异步会话对象
    """
    session: AsyncSession = async_session()
    
    # 🚀 拦截当前 session 的 execute 核心入口
    original_execute = session.execute
    
    async def monitored_execute(statement, params=None, *args, **kwargs):
        # 🎯 物理防线：如果发现上游任何拦截器作妖，强行过滤掉带单引号或双引号的变异畸形键
        if params and isinstance(params, dict):
            purified_params = {}
            for k, v in params.items():
                k_str = str(k).strip()
                # 🛡️ 只要键名两端被包了引号，说明被上游拦截器搞脏了，直接剥离两端的引号恢复原样！
                if (k_str.startswith("'") and k_str.endswith("'")) or (k_str.startswith('"') and k_str.endswith('"')):
                    clean_key = k_str[1:-1] # 剥去引号
                    # 如果剥离后的干净键已经在字典里了，就不用重复赋值，避免重复污染
                    if clean_key not in purified_params:
                        purified_params[clean_key] = v
                    logger.debug(f"[Engine强力净化] 成功拦截变异键 {k_str!r}，已无损将其剥离还原为 {clean_key!r}")
                else:
                    purified_params[k] = v
            
            # 将绝对无污染的纯净字典回传给 SQLAlchemy 编译器
            params = purified_params

        # 回归并执行原本的 SQLAlchemy 逻辑
        return await original_execute(statement, params, *args, **kwargs)
    
    # 挂载底层净化盾牌
    session.execute = monitored_execute  # type: ignore

    try:
        yield session
        await session.commit()
    except DataNotFoundException as e:
        # 业务级别预期内的未命中，执行轻量警告，无需回滚基础设施
        logger.warning(f"Data status alert: {str(e)}")
    except NotFoundError as e:
        # 流程级常规降级（例如未找到配置），不标记为系统错误
        logger.warning(f"Resource status alert: {str(e)}")
    except Exception as e:
        # 🚨 第一线真实物理崩溃卡点（网络断连、SQL语法错误、Oracle ORA等异常）
        try:
            await session.rollback()
            logger.warning("[Database] Rollback executed successfully due to query exception.")
        except Exception as rollback_err:
            logger.critical(f"[Critical] Rollback failed! Resource might be locked: {rollback_err}")
            
        # 打印纯粹干净的异常堆栈，绝不受变异参数拦截器的干扰
        logger.error(
            "Database operation unexpected crashed - "
            "Type: {}, Message: {}, Module: {}",
            type(e).__name__,
            repr(str(e)),
            type(e).__module__,
            exc_info=True
        )
        raise RuntimeError(f"Database operation failed: {str(e)}") from e
    finally:
        # 🛡️ 铁壁防御：强行卡死连接释放。无论 commit/rollback 是否暴毙，连接必须退回连接池！
        await session.close()


async def close_engine() -> None:
    """
    安全释放异步数据库引擎。
    通常在 FastAPI/Sanic 等 Web 容器的 @app.on_event("shutdown") 阶段调用。
    """
    try:
        await async_engine.dispose()
        logger.info("Async database engine pool disposed successfully")
    except Exception as e:
        logger.error(f"Failed to close async database engine pool: {str(e)}")
        raise RuntimeError(f"Failed to close async database engine: {str(e)}") from e


async def test_connection() -> bool:
    """
    高可用数据库连接池健康度探针测试。
    
    Returns:
        bool: 连接池可用返回 True，发生不可逆故障返回 False
    """
    try:
        async with get_session() as session:
            result = await session.execute(text("SELECT 1 FROM DUAL"))
            test_result = result.scalar()
            if test_result == 1:
                logger.info("[HealthCheck] Oracle 23ai DB Connection verified successfully.")
                return True
            else:
                logger.error("[HealthCheck] Unexpected scalar result returned from DUAL.")
                return False
    except Exception as e:
        logger.error(f"[HealthCheck] Database connection test failed: {str(e)}")
        return False