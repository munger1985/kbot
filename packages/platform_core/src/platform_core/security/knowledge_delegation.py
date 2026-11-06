"""知识委托签名；只传最小冻结上下文，服务身份和用户上下文仍单独校验。"""

import time
from uuid import uuid4

import jwt


class KnowledgeDelegationCodec:
    def __init__(self, secret: bytes, *, max_ttl_seconds=300, clock=time.time):
        if not isinstance(secret, bytes) or len(secret) < 32 or not 1 <= max_ttl_seconds <= 3600:
            raise ValueError("知识委托签名配置无效")
        self.secret, self.max_ttl, self.clock = secret, max_ttl_seconds, clock

    def encode(self, claims: dict) -> str:
        now = int(self.clock())
        expires = claims.get("expires_at")
        if type(expires) is not int or not now < expires <= now + self.max_ttl:
            raise ValueError("知识委托期限无效")
        return jwt.encode({"iss": "knowledge-delegation/v1", "aud": claims["audience"],
            "sub": claims["subject_ref"], "iat": now, "exp": expires, "jti": uuid4().hex,
            "data": claims}, self.secret, algorithm="HS256")

    def decode(self, token: str, *, audience: str) -> dict:
        if type(token) is not str or len(token) > 16384:
            raise ValueError("知识委托令牌无效")
        try:
            value = jwt.decode(token, self.secret, algorithms=["HS256"],
                audience=audience, issuer="knowledge-delegation/v1",
                options={"require": ["exp", "iat", "jti", "sub"], "verify_exp": False, "verify_iat": False})
            now, claims = int(self.clock()), value["data"]
            if (type(value["exp"]) is not int or type(value["iat"]) is not int
                    or not value["iat"] <= now < value["exp"] <= value["iat"] + self.max_ttl
                    or claims["expires_at"] != value["exp"] or claims["subject_ref"] != value["sub"]
                    or claims["audience"] != audience):
                raise ValueError("知识委托上下文或期限无效")
            return claims
        except (jwt.PyJWTError, KeyError, TypeError) as exc:
            raise ValueError("知识委托验证失败") from exc

def knowledge_signing_key(purpose):
    """从现有内部认证密钥作用途隔离派生；生产环境不允许开发默认密钥。"""
    import hmac
    import os
    from platform_core.config.settings import get_security_config, get_settings
    from .runtime import DEFAULT_DEV_JWT_SECRET
    config = get_security_config()
    secret = os.getenv(config.internal_jwt_secret_env)
    if not secret:
        if get_settings().is_production():
            raise RuntimeError("知识委托必须配置内部认证密钥")
        secret = DEFAULT_DEV_JWT_SECRET
    if len(secret.encode("utf-8")) < 32 or purpose not in {"delegation", "cursor"}:
        raise ValueError("知识签名配置无效")
    return hmac.digest(secret.encode("utf-8"), ("app-knowledge/v1/" + purpose).encode("ascii"), "sha256")


def create_knowledge_delegation_codec():
    return KnowledgeDelegationCodec(knowledge_signing_key("delegation"))
