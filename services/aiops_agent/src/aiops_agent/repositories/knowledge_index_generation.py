"""本 App knowledge_index_generation 的独立仓储。"""

from aiops_agent.entities.knowledge_index_generation import KnowledgeIndexGenerationEntity
from .knowledge_base import BaseRepository


class KnowledgeIndexGenerationRepository(BaseRepository):
    entity = KnowledgeIndexGenerationEntity
    primary_key = "index_generation_id"
