"""Knowledge Answering 与 Knowledge Core 的跨服务稳定契约。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


KNOWLEDGE_EVIDENCE_CANDIDATE_LIMIT = 128
KNOWLEDGE_DISCOVERY_REVISION_SCOPE_LIMIT = 1000


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _canonical_datetime(value: Any) -> Any:
    """统一等价 ISO-8601 UTC 表示，保证跨服务序列化后哈希稳定。"""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    else:
        return value
    return parsed.isoformat().replace("+00:00", "Z")


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KnowledgeBusinessResourceRef(_Contract):
    """业务 App 拥有的知识资源定位，不替代 KC Revision 身份。"""

    kind: str = Field(min_length=1, max_length=128)
    id: UUID


class KnowledgeScopeSnapshotV1(_Contract):
    """由业务 App 服务端签发并冻结到单次 Run 的不可变知识范围。"""

    schema_version: Literal["1.0"] = "1.0"
    owner_app_id: Literal["knowledge_retrieval", "aiops"]
    domain_id: int = Field(ge=1)
    consumer_agent_id: UUID
    consumer_agent_version_id: UUID
    collection_ids: tuple[UUID, ...] = Field(min_length=1, max_length=128)
    bundle_revision_ids: tuple[UUID, ...] = Field(min_length=1)
    security_level: int = Field(ge=0, le=999)
    business_context: dict[str, Any] = Field(default_factory=dict)
    business_resource_refs: tuple[KnowledgeBusinessResourceRef, ...] = ()
    policy_version: str = Field(min_length=1, max_length=128)
    issued_at: datetime
    scope_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @classmethod
    def calculate_hash(cls, value: dict[str, Any]) -> str:
        payload = dict(value)
        payload.pop("scope_hash", None)
        if "issued_at" in payload:
            payload["issued_at"] = _canonical_datetime(
                payload["issued_at"]
            )
        digest = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        return f"sha256:{digest}"

    @model_validator(mode="before")
    @classmethod
    def verify_scope_hash(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        supplied = str(value.get("scope_hash") or "")
        expected = cls.calculate_hash(value)
        if supplied != expected:
            raise ValueError("knowledge_scope 的 scope_hash 无效")
        return value

    @model_validator(mode="after")
    def verify_identity_and_sets(self) -> "KnowledgeScopeSnapshotV1":
        if len(set(self.collection_ids)) != len(self.collection_ids):
            raise ValueError("knowledge_scope 不能包含重复 Collection")
        if len(set(self.bundle_revision_ids)) != len(self.bundle_revision_ids):
            raise ValueError("knowledge_scope 不能包含重复 Bundle Revision")
        resource_keys = {(item.kind, item.id) for item in self.business_resource_refs}
        if len(resource_keys) != len(self.business_resource_refs):
            raise ValueError("knowledge_scope 不能包含重复业务资源")
        return self


class KnowledgeReferenceV1(_Contract):
    """跨 App 使用的引用事实；业务展示字段由 Presenter 追加。"""

    schema_version: Literal["1.0"] = "1.0"
    label: str = Field(pattern=r"^C[1-9][0-9]*$")
    citation_id: UUID
    collection_id: UUID
    bundle_id: UUID
    bundle_revision_id: UUID
    document_id: UUID
    document_version_id: UUID
    evidence_ids: tuple[UUID, ...] = Field(min_length=1)
    title: str = Field(min_length=1, max_length=1000)
    locator: dict[str, Any] = Field(default_factory=dict)
    locator_schema_version: str = Field(min_length=1, max_length=128)
    quote: str = Field(min_length=1)
    business_resource_ref: KnowledgeBusinessResourceRef | None = None


class KnowledgeAnswerV1(_Contract):
    schema_version: Literal["1.0"] = "1.0"
    answer_markdown: str
    status: Literal[
        "READY", "PARTIAL", "INSUFFICIENT_EVIDENCE", "CLARIFICATION_REQUIRED"
    ]
    references: tuple[KnowledgeReferenceV1, ...] = ()
    coverage: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
    warnings: tuple[str, ...] = ()
