"""Repositories for Knowledge Core root aggregates.

Repositories intentionally never commit or open sessions. The Knowledge Core
application service will own the Unit of Work and transaction boundary.
"""
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_core.entities import KcCollectionBindingEntity, KcCollectionEntity, KcIngestionReceiptEntity


class CollectionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_scope_key(
        self, *, app_id: int, domain_id: int, collection_key: str, lock: bool = False
    ) -> KcCollectionEntity | None:
        statement: Select = select(KcCollectionEntity).where(
            KcCollectionEntity.app_id == app_id,
            KcCollectionEntity.domain_id == domain_id,
            KcCollectionEntity.collection_key == collection_key,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def list_by_scope(self, *, app_id: int, domain_id: int) -> list[KcCollectionEntity]:
        statement = select(KcCollectionEntity).where(
            KcCollectionEntity.app_id == app_id,
            KcCollectionEntity.domain_id == domain_id,
        ).order_by(KcCollectionEntity.display_name, KcCollectionEntity.collection_id)
        return list((await self.session.execute(statement)).scalars())

    async def add(self, collection: KcCollectionEntity) -> KcCollectionEntity:
        self.session.add(collection)
        await self.session.flush()
        return collection

    async def get_by_id(self, *, collection_id: int, lock: bool = False) -> KcCollectionEntity | None:
        statement: Select = select(KcCollectionEntity).where(
            KcCollectionEntity.collection_id == collection_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def get_by_id_scope(
        self, *, app_id: int, domain_id: int, collection_id: int
    ) -> KcCollectionEntity | None:
        statement: Select = select(KcCollectionEntity).where(
            KcCollectionEntity.collection_id == collection_id,
            KcCollectionEntity.app_id == app_id,
            KcCollectionEntity.domain_id == domain_id,
        )
        return (await self.session.execute(statement)).scalar_one_or_none()


class CollectionBindingRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def has_active_binding(self, *, collection_id: int) -> bool:
        statement: Select = select(KcCollectionBindingEntity.binding_id).where(
            KcCollectionBindingEntity.collection_id == collection_id,
            KcCollectionBindingEntity.status == "ACTIVE",
        ).limit(1)
        return (await self.session.execute(statement)).scalar_one_or_none() is not None

    async def get_by_consumer_collection(
        self, *, consumer_type: str, consumer_id: str, collection_id: int
    ) -> KcCollectionBindingEntity | None:
        statement: Select = select(KcCollectionBindingEntity).where(
            KcCollectionBindingEntity.consumer_type == consumer_type,
            KcCollectionBindingEntity.consumer_id == consumer_id,
            KcCollectionBindingEntity.collection_id == collection_id,
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def list_by_collection(self, *, collection_id: int) -> list[KcCollectionBindingEntity]:
        statement = select(KcCollectionBindingEntity).where(
            KcCollectionBindingEntity.collection_id == collection_id,
        ).order_by(KcCollectionBindingEntity.consumer_type, KcCollectionBindingEntity.consumer_id)
        return list((await self.session.execute(statement)).scalars())

    async def list_by_consumer(self, *, consumer_type: str, consumer_id: str) -> list[KcCollectionBindingEntity]:
        statement = select(KcCollectionBindingEntity).where(
            KcCollectionBindingEntity.consumer_type == consumer_type,
            KcCollectionBindingEntity.consumer_id == consumer_id,
            KcCollectionBindingEntity.status == "ACTIVE",
        ).order_by(KcCollectionBindingEntity.collection_id)
        return list((await self.session.execute(statement)).scalars())

    async def add(self, binding: KcCollectionBindingEntity) -> KcCollectionBindingEntity:
        self.session.add(binding)
        await self.session.flush()
        return binding


class IngestionReceiptRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_idempotency_key(
        self, *, collection_id: int, actor_id: str, idempotency_key: str
    ) -> KcIngestionReceiptEntity | None:
        statement: Select = select(KcIngestionReceiptEntity).where(
            KcIngestionReceiptEntity.collection_id == collection_id,
            KcIngestionReceiptEntity.actor_id == actor_id,
            KcIngestionReceiptEntity.idempotency_key == idempotency_key,
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def add(self, receipt: KcIngestionReceiptEntity) -> KcIngestionReceiptEntity:
        self.session.add(receipt)
        await self.session.flush()
        return receipt
