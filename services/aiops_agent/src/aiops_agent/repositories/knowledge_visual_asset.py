"""本 App knowledge_visual_asset 的独立仓储。"""

from aiops_agent.entities.knowledge_visual_asset import KnowledgeVisualAssetEntity
from .knowledge_base import BaseRepository


class KnowledgeVisualAssetRepository(BaseRepository):
    entity = KnowledgeVisualAssetEntity
    primary_key = "visual_asset_id"
