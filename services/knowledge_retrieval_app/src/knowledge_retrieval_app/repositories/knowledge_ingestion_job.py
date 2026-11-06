"""本 App knowledge_ingestion_job 的独立仓储。"""

from knowledge_retrieval_app.entities.knowledge_ingestion_job import KnowledgeIngestionJobEntity
from .knowledge_base import BaseRepository


class KnowledgeIngestionJobRepository(BaseRepository):
    entity = KnowledgeIngestionJobEntity
    primary_key = "ingestion_job_id"
