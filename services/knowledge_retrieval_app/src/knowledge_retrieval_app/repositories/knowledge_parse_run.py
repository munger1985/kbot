"""本 App knowledge_parse_run 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_parse_run import KnowledgeParseRunEntity
from .knowledge_base import BaseRepository


class KnowledgeParseRunRepository(BaseRepository):
    entity = KnowledgeParseRunEntity
    primary_key = "parse_run_id"
