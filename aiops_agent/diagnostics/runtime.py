"""从 AIOps 配置创建诊断目录和 Grant 编解码器。"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from aiops_agent.config import AIOpsSettings

from .grants import DiagnosticGrantCodec
from .registry import DiagnosticRegistry


_DEV_GRANT_SECRET = "kbot-development-diagnostic-grant-secret-change-me"


def create_diagnostic_registry(
    settings: AIOpsSettings,
) -> DiagnosticRegistry:
    configured = settings.executor.diagnostic_catalog_path
    return DiagnosticRegistry.load(Path(configured) if configured else None)


def create_diagnostic_grant_codec(
    settings: AIOpsSettings,
) -> DiagnosticGrantCodec:
    config = settings.executor
    secret = os.getenv(config.grant_secret_env)
    if not secret:
        if settings.is_production():
            raise RuntimeError(
                f"生产环境必须设置 {config.grant_secret_env}"
            )
        logger.warning(
            "当前使用默认开发诊断 Grant 密钥，生产环境必须通过环境变量注入"
        )
        secret = _DEV_GRANT_SECRET
    return DiagnosticGrantCodec(
        secret=secret,
        issuer=config.grant_issuer,
        audience=config.service_name,
        clock_skew_seconds=5,
    )
