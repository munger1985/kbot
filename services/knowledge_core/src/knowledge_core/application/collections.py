"""Collection root use cases."""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import UUID

from knowledge_core.entities import KcCollectionBindingEntity, KcCollectionEntity, KcIngestionJobEntity
from knowledge_core.domain.model_bindings import (
    KC_IMMUTABLE_MODEL_ROLES,
    normalize_collection_models,
)
from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class CollectionNotFoundError(Exception):
    """The requested Collection is not in the authenticated App/Domain scope."""


class CollectionInUseError(Exception):
    """A Collection cannot enter purge while an active consumer binding exists."""


class CollectionDeletionStateError(Exception):
    """The Collection is already being deleted or has an invalid lifecycle state."""


@dataclass(frozen=True)
class CollectionSnapshot:
    """脱离数据库会话后仍可安全返回的 Collection 快照。"""

    collection_id: UUID
    domain_id: int
    display_name: str
    description: str | None
    models_json: dict[str, str]
    status: str
    default_security_level: int
    metadata_json: dict[str, Any]


@dataclass(frozen=True)
class CollectionBindingSnapshot:
    """脱离数据库会话后仍可安全返回的绑定快照。"""

    binding_id: UUID
    collection_id: UUID
    consumer_type: str
    consumer_id: UUID
    status: str
    note: str | None


def _collection_snapshot(entity: KcCollectionEntity) -> CollectionSnapshot:
    return CollectionSnapshot(
        collection_id=entity.collection_id,
        domain_id=int(entity.domain_id),
        display_name=entity.display_name,
        description=entity.description,
        models_json=dict(entity.models_json or {}),
        status=entity.status,
        default_security_level=int(entity.default_security_level),
        metadata_json=dict(entity.metadata_json or {}),
    )


def _binding_snapshot(
    entity: KcCollectionBindingEntity,
) -> CollectionBindingSnapshot:
    return CollectionBindingSnapshot(
        binding_id=entity.binding_id,
        collection_id=entity.collection_id,
        consumer_type=entity.consumer_type,
        consumer_id=entity.consumer_id,
        status=entity.status,
        note=entity.note,
    )


@dataclass(frozen=True)
class CreateCollectionCommand:
    domain_id: int
    display_name: str
    models: dict[str, UUID | str]
    default_security_level: int = 1
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    actor_id: str = "svc:knowledge-core"


@dataclass(frozen=True)
class BindAgentCollectionCommand:
    domain_id: int
    collection_id: UUID
    agent_id: UUID
    actor_id: str = "svc:knowledge-core"
    note: str | None = None


@dataclass(frozen=True)
class ChangeCollectionStatusCommand:
    domain_id: int
    collection_id: UUID
    status: str
    actor_id: str = "svc:knowledge-core"


@dataclass(frozen=True)
class UpdateCollectionModelsCommand:
    domain_id: int
    collection_id: UUID
    models: dict[str, UUID | str]
    actor_id: str = "svc:knowledge-core"


class KnowledgeCoreCollectionService:
    """Application service for Collection root creation.

    Domain 由可信调用上下文确定，不作为请求字段传入。
    Domain authorization is completed by the API layer before this use case.
    """

    def __init__(self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._uow_factory = uow_factory

    async def create(self, command: CreateCollectionCommand) -> KcCollectionEntity:
        display_name = command.display_name.strip()
        if command.domain_id <= 0:
            raise ValueError("domain_id must be positive")
        if not display_name:
            raise ValueError("display_name is required")
        models = normalize_collection_models(command.models)
        if command.default_security_level < 0:
            raise ValueError("default_security_level must be non-negative")

        async with self._uow_factory() as uow:
            if uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = KcCollectionEntity(
                domain_id=command.domain_id,
                display_name=display_name,
                description=command.description,
                models_json=models,
                status="ACTIVE",
                default_security_level=command.default_security_level,
                metadata_json=command.metadata,
                created_by=command.actor_id,
                updated_by=command.actor_id,
            )
            collection = await uow.collections.add(collection)
            await uow.commit()
            return collection

    async def list(self, *, domain_id: int) -> list[CollectionSnapshot]:
        async with self._uow_factory() as uow:
            if uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            entities = await uow.collections.list_by_scope(
                domain_id=domain_id,
            )
            return [_collection_snapshot(entity) for entity in entities]

    async def get(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
    ) -> CollectionSnapshot:
        async with self._uow_factory() as uow:
            if uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_id_scope(
                domain_id=domain_id,
                collection_id=collection_id,
            )
            if collection is None:
                raise CollectionNotFoundError("Collection not found")
            return _collection_snapshot(collection)

    async def change_status(self, command: ChangeCollectionStatusCommand) -> KcCollectionEntity:
        if command.status not in {"ACTIVE", "DISABLED"}:
            raise ValueError("status must be ACTIVE or DISABLED")
        async with self._uow_factory() as uow:
            if uow.collections is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_id_scope(
                domain_id=command.domain_id,
                collection_id=command.collection_id,
                lock=True,
            )
            if collection is None:
                raise CollectionNotFoundError("Collection not found")
            if collection.status in {"DELETING", "DELETION_FAILED"}:
                raise CollectionDeletionStateError("Collection is in deletion lifecycle")
            collection.status = command.status
            collection.updated_by = command.actor_id
            await uow.session.flush()
            await uow.commit()
            return collection

    async def update_models(
        self, command: UpdateCollectionModelsCommand
    ) -> KcCollectionEntity:
        """原子更新角色映射，并保护已经设定的 Embedding 身份。"""
        models = normalize_collection_models(command.models)
        async with self._uow_factory() as uow:
            collection = await uow.collections.get_by_id_scope(
                domain_id=command.domain_id,
                collection_id=command.collection_id,
                lock=True,
            )
            if collection is None:
                raise CollectionNotFoundError("Collection not found")
            current = dict(collection.models_json or {})
            for role in KC_IMMUTABLE_MODEL_ROLES:
                existing = current.get(role)
                requested = models.get(role)
                if existing is not None and requested != existing:
                    raise ValueError(
                        f"{role} 模型一经设定禁止更换或移除"
                    )
            collection.models_json = models
            collection.updated_by = command.actor_id
            await uow.session.flush()
            await uow.commit()
            return collection

    async def request_delete(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        actor_id: str,
    ) -> UUID:
        """Mark an unused Collection for asynchronous, all-descendant purge."""
        async with self._uow_factory() as uow:
            if uow.collections is None or uow.bindings is None or uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_id_scope(
                domain_id=domain_id,
                collection_id=collection_id,
            )
            if collection is None:
                raise CollectionNotFoundError("Collection not found")
            if await uow.bindings.has_active_binding(
                collection_id=collection.collection_id,
            ):
                raise CollectionInUseError("Collection has active Agent bindings")
            if collection.status == "DELETING":
                raise CollectionDeletionStateError("Collection deletion is already queued")
            collection.status = "DELETING"
            collection.updated_by = actor_id
            fingerprint = sha256(f"collection:{collection.collection_id}:purge".encode()).hexdigest()
            job = KcIngestionJobEntity(
                collection_id=collection.collection_id, bundle_revision_id=None,
                document_version_id=None, parse_view_id=None,
                job_type="COLLECTION_PURGE", idempotency_key=f"collection-purge:{collection.collection_id}",
                input_fingerprint=fingerprint,
                payload_json={"collection_id": str(collection.collection_id)},
                job_status="PENDING", priority=100, available_at=datetime.now(timezone.utc),
                attempt_count=0, max_attempts=3, created_by=actor_id, updated_by=actor_id,
            )
            await uow.jobs.add(job)
            await uow.session.flush()
            await uow.commit()
            return job.ingestion_job_id


class KnowledgeCoreBindingService:
    """Application service for the first generic Collection consumer: AGENT."""

    def __init__(self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._uow_factory = uow_factory

    async def bind_agent(self, command: BindAgentCollectionCommand) -> KcCollectionBindingEntity:
        agent_id = command.agent_id
        if command.domain_id <= 0:
            raise ValueError("domain_id must be positive")

        async with self._uow_factory() as uow:
            if uow.collections is None or uow.bindings is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_id_scope(
                domain_id=command.domain_id,
                collection_id=command.collection_id,
            )
            if collection is None:
                raise CollectionNotFoundError(
                    f"Collection not found: domain={command.domain_id}, id={command.collection_id}"
                )

            binding = await uow.bindings.get_by_consumer_collection(
                consumer_type="AGENT",
                consumer_id=agent_id,
                collection_id=collection.collection_id,
            )
            if binding is None:
                binding = KcCollectionBindingEntity(
                    collection_id=collection.collection_id,
                    consumer_type="AGENT",
                    consumer_id=agent_id,
                    status="ACTIVE",
                    note=command.note,
                    created_by=command.actor_id,
                    updated_by=command.actor_id,
                )
                binding = await uow.bindings.add(binding)
            elif binding.status != "ACTIVE":
                binding.status = "ACTIVE"
                binding.note = command.note
                binding.updated_by = command.actor_id
                if uow.session is None:
                    raise RuntimeError("Knowledge Core Unit of Work session is not initialized")
                await uow.session.flush()

            await uow.commit()
            return binding

    async def unbind_agent(self, command: BindAgentCollectionCommand) -> None:
        async with self._uow_factory() as uow:
            if uow.collections is None or uow.bindings is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_id_scope(
                domain_id=command.domain_id,
                collection_id=command.collection_id,
            )
            if collection is None:
                raise CollectionNotFoundError("Collection not found")
            binding = await uow.bindings.get_by_consumer_collection(
                consumer_type="AGENT",
                consumer_id=command.agent_id,
                collection_id=collection.collection_id,
            )
            if binding is not None and binding.status == "ACTIVE":
                binding.status = "REVOKED"
                binding.updated_by = command.actor_id
                await uow.session.flush()
            await uow.commit()

    async def list_agent(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
    ) -> list[CollectionBindingSnapshot]:
        async with self._uow_factory() as uow:
            if uow.bindings is None or uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            bindings = await uow.bindings.list_by_consumer(
                consumer_type="AGENT",
                consumer_id=agent_id,
            )
            result = []
            for binding in bindings:
                collection = await uow.collections.get_by_id_scope(
                    domain_id=domain_id,
                    collection_id=binding.collection_id,
                )
                if collection is not None:
                    result.append(_binding_snapshot(binding))
            return result
