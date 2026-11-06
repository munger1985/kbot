"""App 知识检索领域端口；范围只能由可信业务授权器产生。"""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from knowledge_algorithms.contracts.errors import require
from .knowledge_ingestion import IngestionScope


@dataclass(frozen=True, slots=True)
class KnowledgeGrant:
    revision: str
    version_ids: tuple[UUID, ...]
    operations: frozenset[str] = frozenset({"READ"})

    def __post_init__(self):
        require(bool(self.revision) and len(set(self.version_ids)) == len(self.version_ids)
                and all(isinstance(value, UUID) for value in self.version_ids), "知识授权清单无效")
        require(self.operations <= {"READ", "PUBLISH", "WITHDRAW"}, "知识授权操作无效")


class KnowledgeAuthorization(Protocol):
    async def authorize(self, access: IngestionScope, resource_ref: str) -> KnowledgeGrant: ...


@dataclass(frozen=True, slots=True)
class RetrievalAccess(IngestionScope):
    run_ref: str | None = None
    spec_hash: str | None = None
    delegated_scope_hash: str | None = None
    historical_read: bool = False

    def __post_init__(self):
        IngestionScope.__post_init__(self)
        require((self.run_ref is None) == (self.spec_hash is None), "知识访问的 Run/Spec 必须同时存在")



@dataclass(frozen=True, slots=True)
class RetrievalLimits:
    max_versions: int = 256
    max_units: int = 20000
    max_text_chars: int = 16000000
    max_vector_elements: int = 16000000
    max_relations: int = 100000
    max_object_bytes: int = 8388608
    max_total_object_bytes: int = 134217728
    scope_ttl_seconds: int = 900
    max_read_items: int = 128
    max_payload_bytes: int = 33554432

    def __post_init__(self):
        require(all(type(getattr(self, name)) is int and getattr(self, name) > 0
                    for name in self.__dataclass_fields__), "检索服务预算无效")
        require(self.max_versions <= 500, "授权范围超过当前固定数据库批次预算")
