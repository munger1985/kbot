"""本 App knowledge_outbox 的独立仓储。"""

from aiops_agent.entities.knowledge_outbox import KnowledgeOutboxEntity
from .knowledge_base import BaseRepository


class KnowledgeOutboxRepository(BaseRepository):
    entity = KnowledgeOutboxEntity
    primary_key = "outbox_id"
