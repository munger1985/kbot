from fastapi import Request
from loguru import logger
from core.config.settings import get_app_config
from pathlib import Path
import time

# 记录已初始化的服务 API 日志
_initialized_services = set()

def setup_api_access_logger(service_name: str, log_dir: str, rotation: str = "100 MB", retention: str = "10 days"):
    """为指定服务初始化 API 访问日志的独立文件处理器。

    Args:
        service_name: 服务名称（如 main, llm, vlm, embedding, reranker, parser）
        log_dir: 日志目录
        rotation: 日志轮转策略
        retention: 日志保留时间
    """
    global _initialized_services

    # 为每个服务创建独立的日志文件
    log_filename = f"api_access_{service_name}.log"
    log_path = Path(log_dir) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 检查是否已初始化
    if service_name in _initialized_services:
        return

    # 手动添加文件处理器（不调用 setup()，避免影响主 logger）
    logger.add(
        str(log_path),
        rotation=rotation,
        retention=retention,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        enqueue=True,
        filter=lambda r: r["extra"].get("service_name") == f"api_access_{service_name}",
        serialize=False
    )

    _initialized_services.add(service_name)

    _api_access_logger_initialized = True


def get_status_color(status_code: int) -> str:
    """根据 HTTP 状态码返回对应的颜色标记。

    Args:
        status_code: HTTP 状态码

    Returns:
        Loguru 颜色标记字符串
    """
    if 200 <= status_code < 300:
        # 2xx 成功 - 绿色
        return "<green>"
    elif 300 <= status_code < 400:
        # 3xx 重定向 - 蓝色
        return "<blue>"
    elif 400 <= status_code < 500:
        # 4xx 客户端错误 - 黄色
        return "<yellow>"
    elif 500 <= status_code < 600:
        # 5xx 服务器错误 - 红色
        return "<red>"
    else:
        # 其他状态码 - 默认白色
        return ""


async def log_requests(request: Request, call_next):
    """
    记录所有 HTTP 请求和响应的中间件。
    根据配置文件中的 log.api_log_enabled 来决定是否启用 API 日志记录。
    响应日志会根据 HTTP 状态码显示不同颜色。

    日志输出位置：
    1. 控制台：所有 API 请求和响应（带颜色）
    2. 文件：logs/api_access_{service_name}.log（不带颜色，纯净格式）

    Args:
        request: FastAPI 请求对象
        call_next: 下一个中间件或路由处理函数

    Returns:
        Response: 处理后的响应对象
    """
    # 获取配置
    settings = get_app_config()
    api_log_enabled = settings.log.api_log_enabled

    # 从 app 中获取当前服务名称（需要在应用启动时设置）
    app = request.app
    current_service_name = getattr(app.state, 'service_name', 'main')

    # 初始化当前服务的 API 访问日志文件
    setup_api_access_logger(
        current_service_name,
        settings.log.dir,
        settings.log.rotation,
        settings.log.retention
    )

    start_time = time.time()

    # 记录请求信息
    method = request.method
    url = str(request.url)
    client_host = request.client.host if request.client else "unknown"
    path = request.url.path

    # 只在启用API日志且非文档页面时记录请求
    if api_log_enabled and path not in ["/docs", "/redoc", "/openapi.json"]:
        # 控制台输出（带颜色）
        logger.info(f"API请求 | {method} {url} | 客户端: {client_host}")
        # 文件输出（纯净格式，独立文件）
        logger.bind(service_name=f"api_access_{current_service_name}").info(
            f"API请求 | {method} {url} | 客户端: {client_host}"
        )

    try:
        response = await call_next(request)

        # 计算处理时间
        process_time = (time.time() - start_time) * 1000  # 转换为毫秒

        # 只在启用API日志且非文档页面时记录响应
        if api_log_enabled and path not in ["/docs", "/redoc", "/openapi.json"]:
            status_code = response.status_code
            color = get_status_color(status_code)
            log_message = f"API响应 | {method} {url} | 状态码: {status_code} | 耗时: {process_time:.2f}ms"

            # 控制台输出（带颜色）
            logger.opt(ansi=True).info(f"API响应 | {method} {url} | 状态码: {color}{status_code}</> | 耗时: {process_time:.2f}ms")
            # 文件输出（纯净格式，独立文件）
            logger.bind(service_name=f"api_access_{current_service_name}").info(log_message)

        # 添加处理时间到响应头（始终添加）
        response.headers["X-Process-Time"] = str(process_time)

        return response

    except Exception as e:
        # 记录异常（始终记录异常，不受配置影响）
        process_time = (time.time() - start_time) * 1000
        if path not in ["/docs", "/redoc", "/openapi.json"]:
            logger.error(f"API异常 | {method} {url} | 错误: {str(e)} | 耗时: {process_time:.2f}ms")
        raise
