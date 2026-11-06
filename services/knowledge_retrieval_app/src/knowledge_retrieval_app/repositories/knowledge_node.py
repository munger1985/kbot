"""本 App knowledge_node 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_node import KnowledgeNodeEntity
from .knowledge_base import BaseRepository


class KnowledgeNodeRepository(BaseRepository):
    entity = KnowledgeNodeEntity
    primary_key = "node_id"
