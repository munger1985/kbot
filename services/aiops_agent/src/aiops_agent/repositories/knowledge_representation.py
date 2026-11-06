"""本 App knowledge_representation 的独立仓储。"""

from aiops_agent.entities.knowledge_representation import KnowledgeRepresentationEntity
from .knowledge_base import BaseRepository


class KnowledgeRepresentationRepository(BaseRepository):
    entity = KnowledgeRepresentationEntity
    primary_key = "representation_id"
