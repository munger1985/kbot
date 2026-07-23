"""Collection root use cases."""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import UUID

from knowledge_core.entities import KcCollectionBindingEntity, KcCollectionEntity, KcIngestionJobEntity
from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class CollectionAlreadyExistsError(Exception):
    """The immutable Collection key already exists in the App/Domain scope."""


class CollectionNotFoundError(Exception):
    """The requested Collection is not in the authenticated App/Domain scope."""


class CollectionInUseError(Exception):
    """A Collection cannot enter purge while an active consumer binding exists."""


class CollectionDeletionStateError(Exception):
    """The Collection is already being deleted or has an invalid lifecycle state."""


@dataclass(frozen=True)
class CreateCollectionCommand:
    domain_id: int
    collection_key: str
    display_name: str
    embedding_model_id: int
    default_security_level: int = 1
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    actor_id: str = "svc:knowledge-core"


@dataclass(frozen=True)
class BindAgentCollectionCommand:
    domain_id: int
    collection_key: str
    agent_id: UUID
    actor_id: str = "svc:knowledge-core"
    note: str | None = None


@dataclass(frozen=True)
class ChangeCollectionStatusCommand:
    domain_id: int
    collection_key: str
    status: str
    actor_id: str = "svc:knowledge-core"


class KnowledgeCoreCollectionService:
    """Application service for Collection root creation.

    ``app_id`` is constructor configuration, never a command/request field.
    Domain authorization is completed by the API layer before this use case.
    """

    def __init__(self, *, app_id: int, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._app_id = app_id
        self._uow_factory = uow_factory

    async def create(self, command: CreateCollectionCommand) -> KcCollectionEntity:
        collection_key = command.collection_key.strip()
        display_name = command.display_name.strip()
        if command.domain_id <= 0:
            raise ValueError("domain_id must be positive")
        if not collection_key:
            raise ValueError("collection_key is required")
        if not display_name:
            raise ValueError("display_name is required")
        if command.embedding_model_id <= 0:
            raise ValueError("embedding_model_id must be positive")
        if command.default_security_level < 0:
            raise ValueError("default_security_level must be non-negative")

        async with self._uow_factory() as uow:
            if uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            existing = await uow.collections.get_by_scope_key(
                app_id=self._app_id,
                domain_id=command.domain_id,
                collection_key=collection_key,
            )
            if existing is not None:
                raise CollectionAlreadyExistsError(
                    f"Collection key already exists: domain={command.domain_id}, key={collection_key}"
                )

            collection = KcCollectionEntity(
                app_id=self._app_id,
                domain_id=command.domain_id,
                collection_key=collection_key,
                display_name=display_name,
                description=command.description,
                embedding_model_id=command.embedding_model_id,
                status="ACTIVE",
                default_security_level=command.default_security_level,
                metadata_json=command.metadata,
                created_by=command.actor_id,
                updated_by=command.actor_id,
            )
            collection = await uow.collections.add(collection)
            await uow.commit()
            return collection

    async def list(self, *, domain_id: int) -> list[KcCollectionEntity]:
        async with self._uow_factory() as uow:
            if uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            return await uow.collections.list_by_scope(app_id=self._app_id, domain_id=domain_id)

    async def get(self, *, domain_id: int, collection_key: str) -> KcCollectionEntity:
        async with self._uow_factory() as uow:
            if uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_scope_key(
                app_id=self._app_id, domain_id=domain_id, collection_key=collection_key,
            )
            if collection is None:
                raise CollectionNotFoundError("Collection not found")
            return collection

    async def change_status(self, command: ChangeCollectionStatusCommand) -> KcCollectionEntity:
        if command.status not in {"ACTIVE", "DISABLED"}:
            raise ValueError("status must be ACTIVE or DISABLED")
        async with self._uow_factory() as uow:
            if uow.collections is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_scope_key(
                app_id=self._app_id, domain_id=command.domain_id,
                collection_key=command.collection_key, lock=True,
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

    async def request_delete(
        self,
        *,
        domain_id: int,
        collection_key: str,
        actor_id: str,
    ) -> UUID:
        """Mark an unused Collection for asynchronous, all-descendant purge."""
        async with self._uow_factory() as uow:
            if uow.collections is None or uow.bindings is None or uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_scope_key(
                app_id=self._app_id, domain_id=domain_id, collection_key=collection_key, lock=True,
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

    def __init__(self, *, app_id: int, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._app_id = app_id
        self._uow_factory = uow_factory

    async def bind_agent(self, command: BindAgentCollectionCommand) -> KcCollectionBindingEntity:
        collection_key = command.collection_key.strip()
        agent_id = command.agent_id
        if command.domain_id <= 0 or not collection_key:
            raise ValueError("domain_id, collection_key and agent_id are required")

        async with self._uow_factory() as uow:
            if uow.collections is None or uow.bindings is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_scope_key(
                app_id=self._app_id,
                domain_id=command.domain_id,
                collection_key=collection_key,
            )
            if collection is None:
                raise CollectionNotFoundError(
                    f"Collection not found: domain={command.domain_id}, key={collection_key}"
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
            collection = await uow.collections.get_by_scope_key(
                app_id=self._app_id, domain_id=command.domain_id, collection_key=command.collection_key,
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
    ) -> list[KcCollectionBindingEntity]:
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
                    app_id=self._app_id, domain_id=domain_id, collection_id=binding.collection_id,
                )
                if collection is not None:
                    result.append(binding)
            return result
