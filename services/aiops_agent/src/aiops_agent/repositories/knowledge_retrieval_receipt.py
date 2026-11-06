"""本 App knowledge_retrieval_receipt 的独立仓储。"""

from aiops_agent.entities.knowledge_retrieval_receipt import KnowledgeRetrievalReceiptEntity
from .knowledge_base import BaseRepository


class KnowledgeRetrievalReceiptRepository(BaseRepository):
    entity = KnowledgeRetrievalReceiptEntity
    primary_key = "retrieval_receipt_id"
