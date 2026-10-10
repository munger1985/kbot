"""短期、不透明且绑定 Domain/Source/用途的实例发现引用。"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from platform_core.managed_credentials import (
    ManagedCredentialCipher,
    ManagedCredentialPayload,
)

from aiops_agent.application.errors import AIOpsApplicationError


DiscoveryRefPurpose = Literal["candidate", "cursor"]


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class InstanceCandidateRefCodec:
    _NAMESPACE = "aiops-instance-discovery"
    _TTL_SECONDS = 600

    def __init__(self, *, cipher: ManagedCredentialCipher):
        self._cipher = cipher

    def encode(
        self,
        *,
        domain_id: int,
        source_id: UUID,
        actor_id: str,
        purpose: DiscoveryRefPurpose,
        value: dict[str, Any],
        now: datetime | None = None,
    ) -> str:
        issued_at = (now or datetime.now(UTC)).astimezone(UTC)
        encrypted = self._cipher.encrypt(
            {
                "v": 1,
                "purpose": purpose,
                "source_id": str(source_id),
                "actor_id": str(actor_id),
                "expires_at": int(
                    (issued_at + timedelta(seconds=self._TTL_SECONDS)).timestamp()
                ),
                "value": value,
            },
            domain_id=domain_id,
            namespace=self._NAMESPACE,
            credential_kind=purpose,
            credential_id=source_id,
        )
        envelope = {
            "v": 1,
            "k": encrypted.key_version,
            "n": _encode(encrypted.nonce),
            "c": _encode(encrypted.ciphertext),
        }
        return _encode(
            json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        )

    def decode(
        self,
        token: str,
        *,
        domain_id: int,
        source_id: UUID,
        actor_id: str,
        purpose: DiscoveryRefPurpose,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        try:
            envelope = json.loads(_decode(token).decode("utf-8"))
            if not isinstance(envelope, dict) or envelope.get("v") != 1:
                raise ValueError("引用版本无效")
            payload = self._cipher.decrypt(
                ManagedCredentialPayload(
                    ciphertext=_decode(str(envelope["c"])),
                    nonce=_decode(str(envelope["n"])),
                    key_version=str(envelope["k"]),
                ),
                    domain_id=domain_id,
                namespace=self._NAMESPACE,
                credential_kind=purpose,
                credential_id=source_id,
            )
            current = int((now or datetime.now(UTC)).timestamp())
            if (
                payload.get("v") != 1
                or payload.get("purpose") != purpose
                or payload.get("source_id") != str(source_id)
                or payload.get("actor_id") != str(actor_id)
                or int(payload.get("expires_at", 0)) < current
                or not isinstance(payload.get("value"), dict)
            ):
                raise ValueError("引用上下文不匹配或已过期")
            return dict(payload["value"])
        except Exception as exc:
            raise AIOpsApplicationError(
                code="INSTANCE_CANDIDATE_REF_INVALID",
                message="实例候选引用无效、已过期或不属于当前上下文",
                status_code=400,
            ) from exc


__all__ = ["InstanceCandidateRefCodec"]
