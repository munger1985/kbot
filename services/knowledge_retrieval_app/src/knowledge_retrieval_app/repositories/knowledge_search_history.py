"""直接检索历史：按主体与数据库可信范围读写。"""

from sqlalchemy import delete, select

from ..entities.knowledge_search_history import KnowledgeSearchHistoryEntity
from .knowledge_base import BaseRepository


class KnowledgeSearchHistoryRepository(BaseRepository):
    entity = KnowledgeSearchHistoryEntity
    primary_key = "history_id"

    def owner_statement(self, now):
        table = self.entity
        return self.statement().where(table.subject_ref == self.scope.subject_ref, table.expires_at > now)

    async def page(self, *, now, limit, offset):
        table = self.entity
        rows = (await self._session.scalars(self.owner_statement(now)
            .order_by(table.created_at.desc(), table.history_id.desc()).offset(offset).limit(limit + 1))).all()
        return [self.view(row) for row in rows[:limit]], len(rows) > limit

    async def owned(self, identity, *, now, lock=False):
        statement = self.owner_statement(now).where(self.entity.history_id == identity)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.scalars(statement)).one_or_none()

    async def remove(self, identity, *, now):
        row = await self.owned(identity, now=now, lock=True)
        if row is not None:
            await self._session.delete(row)

    @staticmethod
    def view(row):
        return dict(history_id=row.history_id, library_id=row.library_id, created_at=row.created_at,
                    conditions=row.conditions_json, result_count=row.result_count)

    @classmethod
    async def purge_expired(cls, session, *, now):
        # 仅后台维护使用，没有跨主体读取；每次提交有界，避免清理影响摄取。
        table = cls.entity
        expired = select(table.history_id).where(table.expires_at <= now).order_by(
            table.expires_at, table.history_id).limit(500)
        return (await session.execute(delete(table).where(table.history_id.in_(expired))
            .execution_options(synchronize_session=False))).rowcount
