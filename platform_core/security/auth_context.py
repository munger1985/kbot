"""短期内部 AuthContext JWT 的签发与验证。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError

from platform_core.contracts import AuthContext


AUTH_CONTEXT_HEADER = "X-KBot-Auth-Context"
AUTH_CONTEXT_TOKEN_TYPE = "kbot-auth-context"
AUTH_CONTEXT_ALGORITHM = "HS256"


class AuthContextTokenError(ValueError):
    """内部 AuthContext JWT 无效。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class AuthContextJWTCodec:
    """使用限定 audience 的短期 JWT 传播可信身份上下文。"""

    def __init__(
        self,
        *,
        secret: str,
        issuer: str,
        ttl_seconds: int = 60,
        clock_skew_seconds: int = 5,
    ):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("内部 JWT 密钥至少需要 32 字节")
        if not issuer:
            raise ValueError("内部 JWT issuer 不能为空")
        self._secret = secret
        self._issuer = issuer
        self._ttl_seconds = ttl_seconds
        self._clock_skew_seconds = clock_skew_seconds

    def issue(
        self,
        context: AuthContext,
        *,
        audience: str,
        caller_service: str | None = None,
        now: datetime | None = None,
    ) -> str:
        """为一个明确的下游 audience 签发短期令牌。"""
        if not audience:
            raise ValueError("内部 JWT audience 不能为空")
        issued_at = now or datetime.now(timezone.utc)
        issued_timestamp = int(issued_at.timestamp())
        resolved_caller = caller_service or context.calling_service or context.client_id
        propagated_context = context.model_copy(
            update={"calling_service": resolved_caller}
        )
        claims = {
            "iss": self._issuer,
            "aud": audience,
            "sub": context.client_id,
            "svc": resolved_caller,
            "iat": issued_timestamp,
            "nbf": issued_timestamp,
            "exp": issued_timestamp + self._ttl_seconds,
            "jti": str(uuid4()),
            "typ": AUTH_CONTEXT_TOKEN_TYPE,
            "ctx": propagated_context.model_dump(mode="json"),
        }
        return jwt.encode(claims, self._secret, algorithm=AUTH_CONTEXT_ALGORITHM)

    def verify(
        self,
        token: str,
        *,
        audience: str,
    ) -> AuthContext:
        """验证签名、时效、issuer、audience 和上下文 Schema。"""
        if not token:
            raise AuthContextTokenError(
                "AUTH_CONTEXT_REQUIRED",
                "缺少内部 AuthContext JWT",
            )
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[AUTH_CONTEXT_ALGORITHM],
                audience=audience,
                issuer=self._issuer,
                options={
                    "require_aud": True,
                    "require_exp": True,
                    "require_iat": True,
                    "require_iss": True,
                    "require_jti": True,
                    "require_nbf": True,
                    "require_sub": True,
                    "leeway": self._clock_skew_seconds,
                },
            )
            if claims.get("typ") != AUTH_CONTEXT_TOKEN_TYPE:
                raise AuthContextTokenError(
                    "INVALID_AUTH_CONTEXT",
                    "内部令牌类型无效",
                )
            context = AuthContext.model_validate(claims.get("ctx"))
            if claims.get("sub") != context.client_id:
                raise AuthContextTokenError(
                    "INVALID_AUTH_CONTEXT",
                    "内部令牌主体与上下文不一致",
                )
            if claims.get("svc") != context.calling_service:
                raise AuthContextTokenError(
                    "INVALID_AUTH_CONTEXT",
                    "内部调用服务与上下文不一致",
                )
            return context
        except AuthContextTokenError:
            raise
        except ExpiredSignatureError as exc:
            raise AuthContextTokenError(
                "AUTH_CONTEXT_EXPIRED",
                "内部 AuthContext JWT 已过期",
            ) from exc
        except (JWTClaimsError, JWTError, ValueError) as exc:
            raise AuthContextTokenError(
                "INVALID_AUTH_CONTEXT",
                "内部 AuthContext JWT 无效",
            ) from exc
