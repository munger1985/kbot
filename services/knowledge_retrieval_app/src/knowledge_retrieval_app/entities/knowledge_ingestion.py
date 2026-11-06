"""本 App 知识摄取实体的静态注册，不按请求创建表。"""

from .knowledge_document import KnowledgeDocumentEntity
from .knowledge_document_version import KnowledgeDocumentVersionEntity
from .knowledge_parse_run import KnowledgeParseRunEntity
from .knowledge_parse_batch import KnowledgeParseBatchEntity
from .knowledge_node import KnowledgeNodeEntity
from .knowledge_visual_asset import KnowledgeVisualAssetEntity
from .knowledge_reference_occurrence import KnowledgeReferenceOccurrenceEntity
from .knowledge_relation import KnowledgeRelationEntity
from .knowledge_search_unit import KnowledgeSearchUnitEntity
from .knowledge_unit_source import KnowledgeUnitSourceEntity
from .knowledge_representation import KnowledgeRepresentationEntity
from .knowledge_index_generation import KnowledgeIndexGenerationEntity
from .knowledge_ingestion_job import KnowledgeIngestionJobEntity
from .knowledge_outbox import KnowledgeOutboxEntity
from .knowledge_retrieval_receipt import KnowledgeRetrievalReceiptEntity
from .knowledge_evidence_pin import KnowledgeEvidencePinEntity
from .knowledge_scope_snapshot import KnowledgeScopeSnapshotEntity
from .knowledge_scope_member import KnowledgeScopeMemberEntity

__all__ = [
    "KnowledgeDocumentEntity",
    "KnowledgeDocumentVersionEntity",
    "KnowledgeParseRunEntity",
    "KnowledgeParseBatchEntity",
    "KnowledgeNodeEntity",
    "KnowledgeVisualAssetEntity",
    "KnowledgeReferenceOccurrenceEntity",
    "KnowledgeRelationEntity",
    "KnowledgeSearchUnitEntity",
    "KnowledgeUnitSourceEntity",
    "KnowledgeRepresentationEntity",
    "KnowledgeIndexGenerationEntity",
    "KnowledgeIngestionJobEntity",
    "KnowledgeOutboxEntity",
    "KnowledgeRetrievalReceiptEntity",
    "KnowledgeEvidencePinEntity",
    "KnowledgeScopeSnapshotEntity",
    "KnowledgeScopeMemberEntity"
]
