"""本 App knowledge_document_version 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_document_version import KnowledgeDocumentVersionEntity
from .knowledge_base import BaseRepository


class KnowledgeDocumentVersionRepository(BaseRepository):
    entity = KnowledgeDocumentVersionEntity
    primary_key = "document_version_id"
