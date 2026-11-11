"""Exadata 专用目录与报告仓储，事务归 UoW。"""

from sqlalchemy import select
from aiops_agent.entities.exadata import ExadataInventoryEntity, ExadataReportEntity
from ._base import AIOpsRepository


class ExadataRepository(AIOpsRepository):
    async def inventory(self, binding_id):
        self._check_active()
        return (await self._session.execute(select(ExadataInventoryEntity).where(
            ExadataInventoryEntity.binding_id == binding_id))).scalar_one_or_none()

    async def save_inventory(self, binding_id, digest, members, actor_id, now):
        entity = await self.inventory(binding_id)
        if entity is None:
            entity = ExadataInventoryEntity(binding_id=binding_id)
            self._session.add(entity)
        entity.inventory_sha256, entity.members_json = digest, members
        entity.confirmed_by, entity.confirmed_at = actor_id, now
        await self._session.flush()

    async def add_report(self, entity):
        return await self._add(entity)

    async def report(self, report_id, resource_id, binding_id):
        self._check_active()
        return (await self._session.execute(select(ExadataReportEntity).where(
            ExadataReportEntity.report_id == report_id, ExadataReportEntity.resource_id == resource_id,
            ExadataReportEntity.binding_id == binding_id))).scalar_one_or_none()
