from fastapi import Request
from loguru import logger
from platform_core.config.settings import get_app_config
from pathlib import Path
import time

# Record initialized service API logs
_initialized_services = set()

def setup_api_access_logger(service_name: str, log_dir: str, rotation: str = "100 MB", retention: str = "10 days"):
    """Initialize independent file handler for API access logs of the specified service.

    Args:
        service_name: Service name (e.g., main, llm, vlm, embedding, reranker, parser)
        log_dir: Log directory
        rotation: Log rotation strategy
        retention: Log retention period
    """
    global _initialized_services

    # Create independent log file for each service
    log_filename = f"api_access_{service_name}.log"
    log_path = Path(log_dir) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already initialized
    if service_name in _initialized_services:
        return

    # Manually add file handler (do not call setup() to avoid affecting main logger)
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
    """Return corresponding color marker based on HTTP status code.

    Args:
        status_code: HTTP status code

    Returns:
        Loguru color marker string
    """
    if 200 <= status_code < 300:
        # 2xx Success - Green
        return "<green>"
    elif 300 <= status_code < 400:
        # 3xx Redirection - Blue
        return "<blue>"
    elif 400 <= status_code < 500:
        # 4xx Client Error - Yellow
        return "<yellow>"
    elif 500 <= status_code < 600:
        # 5xx Server Error - Red
        return "<red>"
    else:
        # Other status codes - Default white
        return ""


async def log_requests(request: Request, call_next):
    """
    Middleware to record all HTTP requests and responses.
    Determines whether to enable API logging based on log.api_log_enabled in the configuration file.
    Response logs display different colors according to HTTP status codes.

    Log output locations:
    1. Console: All API requests and responses (with colors)
    2. File: logs/api_access_{service_name}.log (without colors, clean format)

    Args:
        request: FastAPI request object
        call_next: Next middleware or route handler function

    Returns:
        Response: Processed response object
    """
    # Get configuration
    settings = get_app_config()
    api_log_enabled = settings.log.api_log_enabled

    # Get current service name from app (needs to be set during app startup)
    app = request.app
    current_service_name = getattr(app.state, 'service_name', 'main')

    # Initialize API access log file for current service
    setup_api_access_logger(
        current_service_name,
        settings.log.dir,
        settings.log.rotation,
        settings.log.retention
    )

    start_time = time.time()

    # Record request information
    method = request.method
    url = str(request.url)
    client_host = request.client.host if request.client else "unknown"
    path = request.url.path

    # Only record requests when API logging is enabled and not a documentation page
    if api_log_enabled and path not in ["/docs", "/redoc", "/openapi.json"]:
        # Console output (with colors)
        logger.info(f"API Request | {method} {url} | Client: {client_host}")
        # File output (clean format, independent file)
        logger.bind(service_name=f"api_access_{current_service_name}").info(
            f"API Request | {method} {url} | Client: {client_host}"
        )

    try:
        response = await call_next(request)

        # Calculate processing time
        process_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Only record responses when API logging is enabled and not a documentation page
        if api_log_enabled and path not in ["/docs", "/redoc", "/openapi.json"]:
            status_code = response.status_code
            color = get_status_color(status_code)
            log_message = f"API Response | {method} {url} | Status Code: {status_code} | Processing Time: {process_time:.2f}ms"

            # Console output (with colors)
            logger.opt(ansi=True).info(f"API Response | {method} {url} | Status Code: {color}{status_code}</> | Processing Time: {process_time:.2f}ms")
            # File output (clean format, independent file)
            logger.bind(service_name=f"api_access_{current_service_name}").info(log_message)

        # Add processing time to response header (always add)
        response.headers["X-Process-Time"] = str(process_time)

        return response

    except Exception as e:
        # Record exception (always record exceptions, regardless of configuration)
        process_time = (time.time() - start_time) * 1000
        if path not in ["/docs", "/redoc", "/openapi.json"]:
            logger.error(f"API Exception | {method} {url} | Error: {str(e)} | Processing Time: {process_time:.2f}ms")
        raise