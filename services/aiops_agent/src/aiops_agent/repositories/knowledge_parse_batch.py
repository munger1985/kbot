"""本 App knowledge_parse_batch 的独立仓储。"""

from aiops_agent.entities.knowledge_parse_batch import KnowledgeParseBatchEntity
from .knowledge_base import BaseRepository


class KnowledgeParseBatchRepository(BaseRepository):
    entity = KnowledgeParseBatchEntity
    primary_key = "parse_batch_id"
