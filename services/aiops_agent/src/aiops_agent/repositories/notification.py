"""Target 主动分享订阅 Repository。"""

from uuid import UUID

from sqlalchemy import select

from aiops_agent.entities import NotificationSubscriptionEntity
from aiops_agent.repositories._base import AIOpsRepository


_SEVERITY_RANK = {"INFO": 0, "WARNING": 1, "HIGH": 2, "CRITICAL": 3}


class NotificationSubscriptionRepository(AIOpsRepository):
    async def add(
        self, entity: NotificationSubscriptionEntity
    ) -> NotificationSubscriptionEntity:
        return await self._add(entity)

    async def get_for_actor(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        actor_id: str,
        lock: bool = False,
    ) -> NotificationSubscriptionEntity | None:
        self._check_active()
        statement = select(NotificationSubscriptionEntity).where(
            NotificationSubscriptionEntity.domain_id == domain_id,
            NotificationSubscriptionEntity.target_id == target_id,
            NotificationSubscriptionEntity.recipient_actor_id == actor_id,
            NotificationSubscriptionEntity.channel == "IN_APP",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_for_actor(
        self, *, domain_id: int, actor_id: str
    ) -> list[NotificationSubscriptionEntity]:
        self._check_active()
        rows = await self._session.scalars(
            select(NotificationSubscriptionEntity)
            .where(
                NotificationSubscriptionEntity.domain_id == domain_id,
                NotificationSubscriptionEntity.recipient_actor_id == actor_id,
                NotificationSubscriptionEntity.channel == "IN_APP",
            )
            .order_by(
                NotificationSubscriptionEntity.updated_at.desc(),
                NotificationSubscriptionEntity.subscription_id,
            )
        )
        return list(rows)

    async def recipient_actor_ids(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        stage: str,
        severity: str,
    ) -> tuple[str, ...]:
        self._check_active()
        rows = await self._session.scalars(
            select(NotificationSubscriptionEntity).where(
                NotificationSubscriptionEntity.domain_id == domain_id,
                NotificationSubscriptionEntity.target_id == target_id,
                NotificationSubscriptionEntity.channel == "IN_APP",
                NotificationSubscriptionEntity.status == "ACTIVE",
            )
        )
        actual_rank = _SEVERITY_RANK.get(severity, 0)
        return tuple(
            sorted(
                {
                    row.recipient_actor_id
                    for row in rows
                    if stage in set(row.stages_json or ())
                    and actual_rank
                    >= _SEVERITY_RANK.get(row.minimum_severity, 2)
                }
            )
        )
