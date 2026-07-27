"""端口可用性检查工具。

在所有微服务 uvicorn.run() 之前调用，避免因端口被占用导致
服务静默启动失败（EADDRINUSE 错误被 stderr 吞掉的问题）。
"""

import socket
import errno
import sys

from loguru import logger


def check_port_available(host: str, port: int, service_name: str = "") -> bool:
    """检查端口是否可用，不可用时输出明确错误。

    同时使用 print(stderr) 和 loguru 输出，确保在 loguru 尚未初始化时
    （port_check 在 lifespan 前执行）也能看到错误信息。

    Args:
        host: 监听地址 (如 "0.0.0.0")
        port: 监听端口
        service_name: 服务名称，用于错误信息

    Returns:
        True 如果端口可用，False 如果已被占用
    """
    tag = f"[{service_name}] " if service_name else ""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            msg = (
                f"{tag}端口 {host}:{port} 已被占用，服务无法启动。\n"
                f"  → 查找占用进程: ss -tlnp | grep {port}\n"
                f"  → 停止旧进程:    ./stop_kbot.sh  或  kill <PID>"
            )
            logger.error(msg)
            print(msg, file=sys.stderr)
            return False
        msg = f"{tag}检查端口 {host}:{port} 时发生意外错误: {e}"
        logger.error(msg)
        print(msg, file=sys.stderr)
        return False
