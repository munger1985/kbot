"""知识摄取的 App 领域合同；不携带数据库会话或路由目标。"""

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import AppBackendBinding
from knowledge_algorithms.parsing.contracts import ParseProfile, SourceDescriptor
from knowledge_algorithms.parsing.events import ParseRunIdentity


OWNER = "knowledge_retrieval"
BINDING = AppBackendBinding(OWNER, OWNER)


@dataclass(frozen=True, slots=True)
class IngestionScope:
    domain_id: int | UUID
    tenant_id: UUID | None
    subject_ref: str

    def __post_init__(self) -> None:
        require(bool(self.subject_ref) and self.domain_id is not None, "摄取范围缺少可信主体或 Domain")


@dataclass(frozen=True, slots=True)
class JobLease:
    scope: IngestionScope
    document_version_id: UUID
    identity: ParseRunIdentity
    profile: ParseProfile
    descriptor: SourceDescriptor
    lease_owner: str
    lease_until: datetime
    sequence: int
    last_event_digest: str | None
    status: str
    compute_ref: str | None = None

    def __post_init__(self) -> None:
        BINDING.check_owner(self.identity.owner_binding_ref)
        BINDING.check_owner(self.descriptor.owner_binding_ref)
        require(str(self.document_version_id) == self.descriptor.document_version_ref, "源版本与任务不匹配")

    def require_live(self, now: datetime) -> None:
        require(now.tzinfo is not None and self.lease_until > now and self.status == "PARSING",
                "摄取租约过期、被取消或不处于接收状态")


@dataclass(frozen=True, slots=True)
class BatchReceipt:
    sequence: int
    event_digest: str
    payload_ref: str
    batch_key: str


@dataclass(frozen=True, slots=True)
class ObjectReceipt:
    handle_ref: str
    sha256: str
    byte_count: int
    mime_type: str


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
