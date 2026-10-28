"""配额短事务串行准入；执行期间不持有数据库行锁。"""

from datetime import UTC

from sqlalchemy import case, func, literal_column, select

from aiops_agent.entities.resource_governance import ResourceLeaseEntity, ResourcePoolEntity
from aiops_agent.entities import OpsArtifactEntity
from aiops_agent.repositories._base import AIOpsRepository


class ResourceGovernanceRepository(AIOpsRepository):
    async def lock_pool(self, pool_key):
        self._check_active()
        return (await self._session.execute(
            select(ResourcePoolEntity).where(ResourcePoolEntity.pool_key == pool_key)
            .with_for_update(skip_locked=True)
        )).scalar_one_or_none()

    async def database_now(self):
        self._check_active()
        value = (await self._session.execute(select(literal_column("CURRENT_TIMESTAMP")))).scalar_one()
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    async def active_counts(self, *, pool_key, target_id, source_kind, source_id, now):
        self._check_active()
        lease = ResourceLeaseEntity
        row = (await self._session.execute(select(
            func.count(),
            func.coalesce(func.sum(case((lease.target_id == target_id, 1), else_=0)), 0),
            func.coalesce(func.sum(case(((lease.source_kind == source_kind) & (lease.source_id == source_id), 1), else_=0)), 0),
        ).where(lease.pool_key == pool_key, lease.released_at.is_(None), lease.held_until > now))).one()
        return tuple(int(value) for value in row)

    async def get_lease(self, lease_id, *, lock=False):
        self._check_active()
        statement = select(ResourceLeaseEntity).where(ResourceLeaseEntity.lease_id == lease_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add_lease(self, entity):
        return await self._add(entity)

    async def list_database_lifecycle(self, run_id, *, limit):
        """同 Run 有界读取；原始秘密 token 及数据库连接信息不离开仓储。"""
        self._check_active()
        lease, artifact = ResourceLeaseEntity, OpsArtifactEntity
        statement = select(lease).where(lease.run_id == run_id, lease.source_kind == "DATABASE").order_by(
            lease.acquired_at.desc(), lease.lease_id.desc()).limit(min(max(limit, 1), 51))
        leases = list((await self._session.execute(statement)).scalars())
        if not leases:
            return []
        # 不把 Oracle RAW UUID 强制转成字符串参与连接；两平台都用正式 UUID 转换生成键。
        keys = [f"readonly-termination:{item.lease_id}" for item in leases]
        artifacts = (await self._session.execute(select(artifact).where(
            artifact.ops_run_id == run_id, artifact.artifact_key.in_(keys)))).scalars()
        by_key = {item.artifact_key: item for item in artifacts}
        return [(item, by_key.get(key)) for item, key in zip(leases, keys, strict=True)]
