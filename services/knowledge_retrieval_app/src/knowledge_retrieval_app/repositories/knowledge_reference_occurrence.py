"""本 App knowledge_reference_occurrence 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_reference_occurrence import KnowledgeReferenceOccurrenceEntity
from .knowledge_base import BaseRepository


class KnowledgeReferenceOccurrenceRepository(BaseRepository):
    entity = KnowledgeReferenceOccurrenceEntity
    primary_key = "reference_occurrence_id"
