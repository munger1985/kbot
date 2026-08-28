"""短期、受 audience 约束的诊断执行 Grant。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError

from platform_core.contracts.aiops.executor import (
    DiagnosticExecutionGrant,
    DynamicDiagnosticExecutionGrant,
)


GRANT_ALGORITHM = "HS256"
GRANT_TOKEN_TYPE = "kbot-diagnostic-grant"
DYNAMIC_GRANT_TOKEN_TYPE = "kbot-dynamic-diagnostic-grant"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DiagnosticGrantError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class DiagnosticGrantCodec:
    def __init__(
        self,
        *,
        secret: str,
        issuer: str,
        audience: str,
        clock_skew_seconds: int = 5,
    ):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("诊断 Grant 密钥至少需要 32 字节")
        self._secret = secret
        self._issuer = issuer
        self._audience = audience
        self._clock_skew = clock_skew_seconds

    def issue(self, grant: DiagnosticExecutionGrant) -> str:
        return self._issue(grant, token_type=GRANT_TOKEN_TYPE)

    def issue_dynamic(self, grant: DynamicDiagnosticExecutionGrant) -> str:
        """签发与固定目录查询隔离的动态查询 Grant。"""
        return self._issue(grant, token_type=DYNAMIC_GRANT_TOKEN_TYPE)

    def _issue(self, grant, *, token_type: str) -> str:
        if grant.issuer != self._issuer or grant.audience != self._audience:
            raise ValueError("诊断 Grant issuer 或 audience 与签发器不匹配")
        claims = grant.model_dump(mode="json")
        claims.update(
            {
                "iss": grant.issuer,
                "aud": grant.audience,
                "iat": int(grant.issued_at.timestamp()),
                "nbf": int(grant.issued_at.timestamp()),
                "exp": int(grant.expires_at.timestamp()),
                "jti": str(grant.grant_id),
                "typ": token_type,
            }
        )
        return jwt.encode(claims, self._secret, algorithm=GRANT_ALGORITHM)

    def verify(
        self, token: str, *, now: datetime | None = None
    ) -> DiagnosticExecutionGrant:
        return self._verify(
            token,
            contract=DiagnosticExecutionGrant,
            token_type=GRANT_TOKEN_TYPE,
            now=now,
        )

    def verify_dynamic(
        self, token: str, *, now: datetime | None = None
    ) -> DynamicDiagnosticExecutionGrant:
        """验证动态查询 Grant，并拒绝固定目录 Grant 混用。"""
        return self._verify(
            token,
            contract=DynamicDiagnosticExecutionGrant,
            token_type=DYNAMIC_GRANT_TOKEN_TYPE,
            now=now,
        )

    def _verify(self, token: str, *, contract, token_type: str, now):
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[GRANT_ALGORITHM],
                audience=self._audience,
                issuer=self._issuer,
                options={
                    "require_aud": True,
                    "require_exp": True,
                    "require_iat": True,
                    "require_iss": True,
                    "require_jti": True,
                    "require_nbf": True,
                    "leeway": self._clock_skew,
                },
            )
            if claims.get("typ") != token_type:
                raise DiagnosticGrantError(
                    "GRANT_INVALID", "诊断 Grant 类型无效"
                )
            payload = dict(claims)
            for key in ("iss", "aud", "iat", "nbf", "exp", "jti", "typ"):
                payload.pop(key, None)
            grant = contract.model_validate(payload)
            current = now or datetime.now(UTC)
            if current >= grant.expires_at:
                raise DiagnosticGrantError(
                    "GRANT_EXPIRED", "诊断 Grant 已过期"
                )
            return grant
        except DiagnosticGrantError:
            raise
        except ExpiredSignatureError as exc:
            raise DiagnosticGrantError(
                "GRANT_EXPIRED", "诊断 Grant 已过期"
            ) from exc
        except (JWTClaimsError, JWTError, ValueError, KeyError) as exc:
            raise DiagnosticGrantError(
                "GRANT_INVALID", "诊断 Grant 无效"
            ) from exc
