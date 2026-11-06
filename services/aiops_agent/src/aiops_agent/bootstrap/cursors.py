"""为 API 和 Worker 创建相同签名规则的查询游标。"""

import hashlib
import os

from loguru import logger

from aiops_agent.application.configuration.common import SignedCursorCodec
from aiops_agent.config import AIOpsSettings


def create_cursor_codec(settings: AIOpsSettings) -> SignedCursorCodec:
    """复用已有配置；开发派生密钥固定使用 API 身份，不随进程或副本变化。"""
    secret = os.getenv(settings.management.cursor_secret_env)
    if not secret:
        if settings.is_production():
            raise RuntimeError("生产环境缺少 AIOps Cursor 签名密钥")
        secret = hashlib.sha256(
            f"{settings.api.service_name}:development:cursor".encode("utf-8")
        ).hexdigest()
        logger.warning("开发环境使用临时派生的 AIOps Cursor 签名密钥")
    return SignedCursorCodec(
        secret=secret,
        ttl_seconds=settings.management.cursor_ttl_seconds,
    )
