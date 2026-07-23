"""从平台配置构造认证运行时对象。"""

from __future__ import annotations

import os
from functools import lru_cache
from uuid import uuid4

from loguru import logger

from platform_core.config.settings import get_security_config, get_settings
from platform_core.contracts import AuthContext, PrincipalKind

from .api_key import PortalApiKeyRecord, PortalApiKeyVerifier
from .auth_context import AUTH_CONTEXT_HEADER, AuthContextJWTCodec


INTERNAL_TOKEN_HEADER = "X-KBot-Internal-Token"
DEFAULT_DEV_SERVICE_TOKEN = "kbot_internal_service_token"
DEFAULT_DEV_JWT_SECRET = "kbot-development-auth-context-secret-change-me"
DEFAULT_DEV_API_KEY_PEPPER = "kbot-development-api-key-pepper-change-me"


@lru_cache(maxsize=1)
def get_internal_service_token() -> str:
    """读取内部服务凭证；生产环境禁止使用开发默认值。"""
    config = get_security_config()
    token = os.getenv(config.internal_service_token_env)
    if token:
        return token
    if get_settings().is_production():
        raise RuntimeError(
            f"生产环境必须设置 {config.internal_service_token_env}"
        )
    logger.warning(
        "当前使用默认开发内部服务凭证，生产环境必须通过环境变量注入"
    )
    return DEFAULT_DEV_SERVICE_TOKEN


@lru_cache(maxsize=1)
def create_auth_context_codec() -> AuthContextJWTCodec:
    """根据平台配置创建内部 JWT 编解码器。"""
    config = get_security_config()
    secret = os.getenv(config.internal_jwt_secret_env)
    if not secret:
        if get_settings().is_production():
            raise RuntimeError(
                f"生产环境必须设置 {config.internal_jwt_secret_env}"
            )
        logger.warning(
            "当前使用默认开发内部 JWT 密钥，生产环境必须通过环境变量注入"
        )
        secret = DEFAULT_DEV_JWT_SECRET
    return AuthContextJWTCodec(
        secret=secret,
        issuer=config.internal_jwt_issuer,
        ttl_seconds=config.internal_jwt_ttl_seconds,
        clock_skew_seconds=config.internal_jwt_clock_skew_seconds,
    )


@lru_cache(maxsize=1)
def create_portal_api_key_verifier() -> PortalApiKeyVerifier:
    """从配置中的摘要注册表创建 Portal API Key 校验器。"""
    config = get_security_config()
    pepper = os.getenv(config.api_key_pepper_env)
    if not pepper:
        if get_settings().is_production():
            raise RuntimeError(
                f"生产环境必须设置 {config.api_key_pepper_env}"
            )
        logger.warning(
            "当前使用默认开发 API Key Pepper，生产环境必须通过环境变量注入"
        )
        pepper = DEFAULT_DEV_API_KEY_PEPPER
    records = [
        PortalApiKeyRecord(
            key_id=item.key_id,
            client_id=item.client_id,
            key_digest=item.key_digest,
            enabled=item.enabled,
            expires_at=item.expires_at,
        )
        for item in config.portal_api_keys
    ]
    return PortalApiKeyVerifier(records=records, pepper=pepper)


def create_service_auth_context(
    *,
    caller_service: str,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> AuthContext:
    """为后台 Worker 或内部 Client 创建服务身份上下文。"""
    resolved_request_id = request_id or str(uuid4())
    return AuthContext(
        principal_kind=PrincipalKind.SERVICE,
        client_id=caller_service,
        calling_service=caller_service,
        request_id=resolved_request_id,
        trace_id=trace_id or resolved_request_id,
    )


def build_internal_auth_headers(
    *,
    audience: str,
    caller_service: str,
    context: AuthContext | None = None,
    codec: AuthContextJWTCodec | None = None,
    service_token: str | None = None,
) -> dict[str, str]:
    """为一次内部 HTTP 请求创建服务凭证和短期身份 JWT。"""
    resolved_codec = codec or create_auth_context_codec()
    resolved_context = context or create_service_auth_context(
        caller_service=caller_service
    )
    return {
        INTERNAL_TOKEN_HEADER: service_token or get_internal_service_token(),
        AUTH_CONTEXT_HEADER: resolved_codec.issue(
            resolved_context,
            audience=audience,
            caller_service=caller_service,
        ),
        "X-Request-ID": resolved_context.request_id,
    }
