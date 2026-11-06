"""本 App knowledge_document 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_document import KnowledgeDocumentEntity
from .knowledge_base import BaseRepository


class KnowledgeDocumentRepository(BaseRepository):
    entity = KnowledgeDocumentEntity
    primary_key = "document_id"
