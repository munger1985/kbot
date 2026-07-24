"""创建 Mutation Grant 编解码器。"""

from __future__ import annotations

import os

from loguru import logger

from aiops_agent.config import AIOpsSettings

from .grants import MutationGrantCodec


_DEV_SECRET = "kbot-development-mutation-grant-secret-change-me"


def create_mutation_grant_codec(
    settings: AIOpsSettings,
) -> MutationGrantCodec:
    config = settings.executor
    secret = os.getenv(config.mutation_grant_secret_env)
    if not secret:
        if settings.is_production():
            raise RuntimeError(
                f"生产环境必须设置 {config.mutation_grant_secret_env}"
            )
        logger.warning(
            "当前使用默认开发 Mutation Grant 密钥，生产环境必须通过环境变量注入"
        )
        secret = _DEV_SECRET
    return MutationGrantCodec(
        secret=secret,
        issuer=config.mutation_grant_issuer,
        audience=config.service_name,
        clock_skew_seconds=5,
    )
