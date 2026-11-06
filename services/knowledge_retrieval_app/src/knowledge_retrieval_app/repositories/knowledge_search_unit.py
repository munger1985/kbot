"""本 App knowledge_search_unit 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_search_unit import KnowledgeSearchUnitEntity
from .knowledge_base import BaseRepository


class KnowledgeSearchUnitRepository(BaseRepository):
    entity = KnowledgeSearchUnitEntity
    primary_key = "search_unit_id"
