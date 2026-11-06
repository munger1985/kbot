"""本 App knowledge_unit_source 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_unit_source import KnowledgeUnitSourceEntity
from .knowledge_base import BaseRepository


class KnowledgeUnitSourceRepository(BaseRepository):
    entity = KnowledgeUnitSourceEntity
    primary_key = "unit_source_id"
