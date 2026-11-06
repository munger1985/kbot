"""本 App knowledge_relation 的独立仓储。"""

from aiops_agent.entities.knowledge_relation import KnowledgeRelationEntity
from .knowledge_base import BaseRepository


class KnowledgeRelationRepository(BaseRepository):
    entity = KnowledgeRelationEntity
    primary_key = "relation_id"
