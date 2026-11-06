"""本 App knowledge_scope_snapshot 的独立仓储。"""

from aiops_agent.entities.knowledge_scope_snapshot import KnowledgeScopeSnapshotEntity
from .knowledge_base import BaseRepository


class KnowledgeScopeSnapshotRepository(BaseRepository):
    entity = KnowledgeScopeSnapshotEntity
    primary_key = "scope_snapshot_id"
