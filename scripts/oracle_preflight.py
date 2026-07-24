"""Oracle 验收脚本共享的有界 TCP 前置检查。"""

from __future__ import annotations

import socket


def require_oracle_listener(
    *, host: str, port: int, timeout_seconds: float = 3
) -> None:
    """在创建驱动连接池前确认 Listener 可达，避免无界建连等待。"""
    try:
        with socket.create_connection(
            (host, port), timeout=timeout_seconds
        ):
            return
    except OSError as exc:
        raise RuntimeError(
            f"Oracle Listener 不可达：{host}:{port}，"
            f"TCP Preflight 超时上限 {timeout_seconds:g} 秒"
        ) from exc
