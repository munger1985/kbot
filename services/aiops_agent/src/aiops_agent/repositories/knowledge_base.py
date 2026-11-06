"""固定实体和可信行范围的仓储基类；不接受调用方指定表。"""

from sqlalchemy import select


class BaseRepository:
    entity = None
    primary_key = ""

    def __init__(self, session, scope):
        self._session, self.scope = session, scope

    def statement(self):
        table = self.entity
        statement = select(table).where(table.domain_id == self.scope.domain_id)
        if hasattr(table, "tenant_id"):
            statement = statement.where(table.tenant_id == self.scope.tenant_id)
        return statement

    async def get(self, identity, *, lock=False):
        statement = self.statement().where(getattr(self.entity, self.primary_key) == identity)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, values):
        fields = dict(values, domain_id=self.scope.domain_id)
        if hasattr(self.entity, "tenant_id"):
            fields["tenant_id"] = self.scope.tenant_id
        row = self.entity(**fields)
        self._session.add(row)
        await self._session.flush()
        return row
