"""短期 Service Identity JWT 的签发与验证。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError

from platform_core.contracts import ServiceIdentity


SERVICE_IDENTITY_HEADER = "X-KBot-Service-Identity"
SERVICE_IDENTITY_TOKEN_TYPE = "kbot-service-identity"
SERVICE_IDENTITY_ALGORITHM = "HS256"


class ServiceIdentityTokenError(ValueError):
    """Service Identity JWT 无效。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ServiceIdentityJWTCodec:
    """签发并验证限定调用方、audience 和 scopes 的短期令牌。"""

    def __init__(
        self,
        *,
        secret: str,
        issuer: str,
        ttl_seconds: int = 60,
        clock_skew_seconds: int = 5,
    ):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("Service Identity JWT 密钥至少需要 32 字节")
        if not issuer:
            raise ValueError("Service Identity issuer 不能为空")
        self._secret = secret
        self._issuer = issuer
        self._ttl_seconds = ttl_seconds
        self._clock_skew_seconds = clock_skew_seconds

    def issue(
        self,
        *,
        subject: str,
        audience: str,
        scopes: tuple[str, ...],
        now: datetime | None = None,
    ) -> str:
        if not subject or not audience or not scopes:
            raise ValueError("Service Identity 必须包含主体、audience 和 scope")
        issued_at = now or datetime.now(UTC)
        expires_at = issued_at + timedelta(seconds=self._ttl_seconds)
        token_id = uuid4()
        claims = {
            "iss": self._issuer,
            "sub": subject,
            "aud": audience,
            "scope": " ".join(scopes),
            "iat": int(issued_at.timestamp()),
            "nbf": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
            "jti": str(token_id),
            "typ": SERVICE_IDENTITY_TOKEN_TYPE,
        }
        return jwt.encode(
            claims,
            self._secret,
            algorithm=SERVICE_IDENTITY_ALGORITHM,
        )

    def verify(
        self,
        token: str,
        *,
        audience: str,
    ) -> ServiceIdentity:
        if not token:
            raise ServiceIdentityTokenError(
                "SERVICE_IDENTITY_REQUIRED",
                "缺少 Service Identity JWT",
            )
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[SERVICE_IDENTITY_ALGORITHM],
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
            if claims.get("typ") != SERVICE_IDENTITY_TOKEN_TYPE:
                raise ServiceIdentityTokenError(
                    "INVALID_SERVICE_IDENTITY",
                    "Service Identity 令牌类型无效",
                )
            scopes = tuple(
                scope
                for scope in str(claims.get("scope", "")).split()
                if scope
            )
            return ServiceIdentity(
                issuer=str(claims["iss"]),
                subject=str(claims["sub"]),
                audience=audience,
                scopes=scopes,
                issued_at=datetime.fromtimestamp(claims["iat"], tz=UTC),
                expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
                token_id=claims["jti"],
            )
        except ServiceIdentityTokenError:
            raise
        except ExpiredSignatureError as exc:
            raise ServiceIdentityTokenError(
                "SERVICE_IDENTITY_EXPIRED",
                "Service Identity JWT 已过期",
            ) from exc
        except (JWTClaimsError, JWTError, ValueError, KeyError) as exc:
            raise ServiceIdentityTokenError(
                "INVALID_SERVICE_IDENTITY",
                "Service Identity JWT 无效",
            ) from exc
