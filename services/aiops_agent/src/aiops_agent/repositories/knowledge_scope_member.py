"""本 App knowledge_scope_member 的独立仓储。"""

from aiops_agent.entities.knowledge_scope_member import KnowledgeScopeMemberEntity
from .knowledge_base import BaseRepository


class KnowledgeScopeMemberRepository(BaseRepository):
    entity = KnowledgeScopeMemberEntity
    primary_key = "scope_member_id"
