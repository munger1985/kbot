"""本 App knowledge_evidence_pin 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_evidence_pin import KnowledgeEvidencePinEntity
from .knowledge_base import BaseRepository


class KnowledgeEvidencePinRepository(BaseRepository):
    entity = KnowledgeEvidencePinEntity
    primary_key = "evidence_pin_id"
