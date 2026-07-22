"""The single text-embedding path for Knowledge Core retrieval.

Parsing produces deterministic text and provenance only.  This module is the
only application boundary allowed to turn that text into a retrieval vector.
It deliberately receives a model snapshot from the caller (the Collection
configuration service), so a worker cannot silently fall back to a process
default model.
"""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Protocol, Sequence

from knowledge_core.persistence import KnowledgeCoreUnitOfWork
from knowledge_core.domain.revision_status import reduce_revision_status
from knowledge_core.domain.parse_tasks import ParseLeaseError, ParseTaskClaim, claim_job, verify_lease


@dataclass(frozen=True)
class EmbeddingModelSnapshot:
    model_id: int
    model_key: str
    dimension: int
    config_fingerprint: str

    def validate(self) -> None:
        if self.model_id <= 0 or not self.model_key.strip():
            raise ValueError("embedding model identity is required")
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if len(self.config_fingerprint) != 64:
            raise ValueError("embedding config fingerprint must be SHA-256")


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    model_key: str
    dimension: int


@dataclass(frozen=True)
class ClaimedIndexTask:
    job_id: int
    worker_id: str
    input_fingerprint: str
    collection_id: int
    parse_view_id: int | None
    lease_until: datetime


class EmbeddingGateway(Protocol):
    async def embed_texts(
        self, *, model_key: str, texts: Sequence[str], is_query: bool = False,
    ) -> EmbeddingBatch:
        """Generate vectors using exactly ``model_key``."""


def retrieval_input_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def validate_embedding_batch(
    *, batch: EmbeddingBatch, model: EmbeddingModelSnapshot, expected_count: int,
) -> None:
    """Reject a provider response before any vector is persisted."""
    if batch.model_key != model.model_key:
        raise ValueError("embedding provider returned an unexpected model")
    if batch.dimension != model.dimension:
        raise ValueError("embedding provider returned an unexpected dimension")
    if len(batch.vectors) != expected_count:
        raise ValueError("embedding provider returned an unexpected item count")
    if any(len(vector) != model.dimension for vector in batch.vectors):
        raise ValueError("embedding vector length does not match the model dimension")


class KnowledgeCoreEvidenceIndexService:
    """Index ACTIVE Evidence for one Parse View in bounded batches.

    ``model_resolver`` must read the Collection-bound model and return its
    immutable runtime snapshot.  The service never accepts a model from a
    parser policy or request payload.
    """

    def __init__(
        self, *,
        uow_factory: Callable[[], KnowledgeCoreUnitOfWork],
        embedding_gateway: EmbeddingGateway,
        model_resolver: Callable[[int], Awaitable[EmbeddingModelSnapshot]],
    ):
        self._uow_factory = uow_factory
        self._embedding_gateway = embedding_gateway
        self._model_resolver = model_resolver

    async def index_parse_view(
        self, *, parse_view_id: int, collection_id: int, batch_size: int = 64,
        job_id: int | None = None,
    ) -> int:
        if batch_size < 1 or batch_size > 500:
            raise ValueError("batch_size must be between 1 and 500")
        model = await self._model_resolver(collection_id)
        model.validate()
        if job_id is not None:
            await self.snapshot_index_model(job_id=job_id, model=model)
        indexed = 0
        while True:
            async with self._uow_factory() as uow:
                if uow.collections is None or uow.evidence is None or uow.session is None:
                    raise RuntimeError("Knowledge Core Unit of Work is not initialized")
                collection = await uow.collections.get_by_id(collection_id=collection_id)
                if collection is None:
                    raise ValueError("collection not found")
                if int(collection.embedding_model_id) != model.model_id:
                    raise ValueError("model snapshot is not the Collection-bound model")
                pending = await uow.evidence.list_needing_index(
                    parse_view_id=parse_view_id, model_id=model.model_id,
                    model_key=model.model_key, limit=batch_size,
                )
                # A changed retrieval_text has a new input hash even when
                # the model identity is unchanged.  It is checked after the
                # bounded query so stale rows are also regenerated.
                if not pending:
                    stale = await uow.evidence.list_active(parse_view_id=parse_view_id, limit=batch_size)
                    pending = [
                        row for row in stale
                        if row.embedding is None
                        or row.embedding_input_hash != retrieval_input_hash(row.retrieval_text)
                    ]
                if not pending:
                    return indexed
                texts = [row.retrieval_text for row in pending]
                batch = await self._embedding_gateway.embed_texts(
                    model_key=model.model_key, texts=texts, is_query=False,
                )
                validate_embedding_batch(batch=batch, model=model, expected_count=len(pending))
                now = datetime.now(timezone.utc)
                for row, vector in zip(pending, batch.vectors):
                    row.embedding = vector
                    row.embedding_model_id = model.model_id
                    row.embedding_model_key = model.model_key
                    row.embedding_config_fingerprint = model.config_fingerprint
                    row.embedding_input_hash = retrieval_input_hash(row.retrieval_text)
                    row.indexed_at = now
                await uow.session.flush()
                await uow.commit()
                indexed += len(pending)

    async def snapshot_index_model(self, *, job_id: int, model: EmbeddingModelSnapshot) -> None:
        """Freeze the model identity on the INDEX job before provider I/O."""
        model.validate()
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "INDEX":
                raise ValueError("invalid INDEX job")
            payload = dict(job.payload_json or {})
            existing = payload.get("embedding_model_snapshot")
            snapshot = {
                "model_id": model.model_id,
                "model_key": model.model_key,
                "dimension": model.dimension,
                "config_fingerprint": model.config_fingerprint,
            }
            if existing is not None and existing != snapshot:
                raise ValueError("INDEX job model snapshot cannot be changed")
            payload["embedding_model_snapshot"] = snapshot
            job.payload_json = payload
            await uow.session.flush()
            await uow.commit()

    async def claim(self, claim: ParseTaskClaim) -> list[ClaimedIndexTask]:
        """Lease INDEX jobs independently from PARSE workers."""
        claim.validate()
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            jobs = await uow.jobs.claim_index_candidates(now=now, limit=claim.max_tasks)
            tasks: list[ClaimedIndexTask] = []
            for job in jobs:
                if job.parse_view_id is None and (job.payload_json or {}).get("target") != "DISCOVERY":
                    continue
                lease_until = claim_job(job, claim, now)
                tasks.append(ClaimedIndexTask(
                    job_id=job.ingestion_job_id, worker_id=claim.worker_id,
                    input_fingerprint=job.input_fingerprint,
                    collection_id=job.collection_id, parse_view_id=job.parse_view_id,
                    lease_until=lease_until,
                ))
            if tasks:
                await uow.session.flush()
                await uow.commit()
            return tasks

    async def run_job(self, *, job_id: int, worker_id: str, input_fingerprint: str, batch_size: int = 64) -> str:
        """Execute a leased INDEX job and promote its revision member."""
        async with self._uow_factory() as uow:
            if uow.jobs is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=False)
            if job is None or job.job_type != "INDEX":
                raise ValueError("invalid INDEX job")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint)
            collection_id = int(job.collection_id)
            parse_view_id = int(job.parse_view_id) if job.parse_view_id is not None else None
            target = (job.payload_json or {}).get("target")
        model = await self._model_resolver(collection_id)
        if target == "DISCOVERY":
            return await self._run_discovery_job(
                job_id=job_id, worker_id=worker_id, input_fingerprint=input_fingerprint,
                model=model,
            )
        if parse_view_id is None:
            raise ValueError("INDEX job has no Parse View")
        count = await self.index_parse_view(
            parse_view_id=parse_view_id, collection_id=collection_id,
            batch_size=batch_size, job_id=job_id,
        )
        return await self.finalize_index_job(job_id=job_id, indexed_count=count, model=model)

    async def _run_discovery_job(self, *, job_id: int, worker_id: str, input_fingerprint: str, model: EmbeddingModelSnapshot) -> str:
        model.validate()
        async with self._uow_factory() as uow:
            if not all((uow.jobs, uow.discovery, uow.collections, uow.revisions, uow.bundles, uow.session)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "INDEX" or job.bundle_revision_id is None:
                raise ValueError("invalid Discovery INDEX job")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint)
            revision = await uow.revisions.get_by_id(bundle_revision_id=job.bundle_revision_id, lock=True)
            if revision is None or int(revision.collection_id) != int(job.collection_id):
                raise ValueError("Discovery INDEX revision is stale")
            collection = await uow.collections.get_by_id(collection_id=job.collection_id)
            if collection is None or int(collection.embedding_model_id) != model.model_id:
                raise ValueError("Discovery INDEX model is not Collection-bound")
            payload = dict(job.payload_json or {})
            snapshot = payload.get("embedding_model_snapshot")
            expected_snapshot = {
                "model_id": model.model_id, "model_key": model.model_key,
                "dimension": model.dimension, "config_fingerprint": model.config_fingerprint,
            }
            if snapshot is not None and snapshot != expected_snapshot:
                raise ValueError("Discovery INDEX model snapshot cannot change")
            payload["embedding_model_snapshot"] = expected_snapshot
            job.payload_json = payload
            objects = await uow.discovery.list_staged(bundle_revision_id=revision.bundle_revision_id)
            pending = [obj for obj in objects if obj.embedding is None or obj.embedding_input_hash != retrieval_input_hash(obj.profile_text) or int(obj.embedding_model_id or 0) != model.model_id or obj.embedding_model_key != model.model_key]
            if pending:
                texts = [obj.profile_text for obj in pending]
                batch = await self._embedding_gateway.embed_texts(model_key=model.model_key, texts=texts, is_query=False)
                validate_embedding_batch(batch=batch, model=model, expected_count=len(pending))
                now = datetime.now(timezone.utc)
                for obj, vector in zip(pending, batch.vectors):
                    obj.embedding = vector
                    obj.embedding_model_id = model.model_id
                    obj.embedding_model_key = model.model_key
                    obj.embedding_config_fingerprint = model.config_fingerprint
                    obj.embedding_input_hash = retrieval_input_hash(obj.profile_text)
                    obj.indexed_at = now
            if any(obj.embedding is None or int(obj.embedding_model_id or 0) != model.model_id or obj.embedding_model_key != model.model_key for obj in objects):
                raise ValueError("Discovery INDEX has unindexed profiles")
            now = datetime.now(timezone.utc)
            for obj in objects:
                obj.discovery_status = "ACTIVE"
            await uow.discovery.retire_other_revisions(bundle_id=revision.bundle_id, except_revision_id=revision.bundle_revision_id)
            bundle = await uow.bundles.get_by_id(bundle_id=revision.bundle_id, lock=True)
            if bundle is not None:
                bundle.current_revision_id = revision.bundle_revision_id
                bundle.availability_status = revision.status
                bundle.row_version += 1
            job.job_status, job.completed_at = "SUCCEEDED", now
            job.result_json = {"profile_count": len(objects), "status": "ACTIVE"}
            job.lease_owner = job.lease_until = None
            await uow.session.flush()
            await uow.commit()
            return str(revision.status)

    async def heartbeat(
        self, *, job_id: int, worker_id: str, input_fingerprint: str, lease_seconds: int = 120,
    ) -> datetime:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "INDEX":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            job.heartbeat_at = now
            job.lease_until = now + timedelta(seconds=lease_seconds)
            job.row_version += 1
            await uow.session.flush()
            await uow.commit()
            return job.lease_until

    async def fail(
        self, *, job_id: int, worker_id: str, input_fingerprint: str,
        failure_class: str, failure_code: str, failure_message: str | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if not all((uow.jobs, uow.members, uow.revisions, uow.bundles)) or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "INDEX":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            retryable = failure_class == "TRANSIENT" and job.attempt_count < job.max_attempts
            job.failure_class, job.failure_code, job.failure_message = failure_class, failure_code, failure_message
            job.lease_owner = job.lease_until = None
            if retryable:
                job.job_status = "RETRY_WAIT"
                job.available_at = now + timedelta(seconds=min(300, 2 ** job.attempt_count * 5))
                result = job.job_status
            else:
                job.job_status, job.completed_at = "FAILED", now
                if uow.members is not None and job.bundle_revision_id is not None and job.document_version_id is not None:
                    member = await uow.members.get_by_version(
                        bundle_revision_id=job.bundle_revision_id,
                        document_version_id=job.document_version_id,
                        lock=True,
                    )
                    if member is not None:
                        member.member_status = "FAILED"
                        member.failure_stage, member.failure_code, member.failure_message = "INDEX", failure_code, failure_message
                if job.bundle_revision_id is not None:
                    revision = await uow.revisions.get_by_id(bundle_revision_id=job.bundle_revision_id, lock=True)
                    if revision is not None:
                        members = await uow.members.list_by_revision(bundle_revision_id=revision.bundle_revision_id)
                        revision.status = reduce_revision_status(members)
                        if revision.status in {"READY", "PARTIAL", "FAILED"}:
                            revision.completed_at = now
                result = job.job_status
            await uow.session.flush()
            await uow.commit()
            return result

    async def finalize_index_job(
        self, *, job_id: int, indexed_count: int, model: EmbeddingModelSnapshot,
    ) -> str:
        """Mark INDEX successful only after every active Evidence has a vector."""
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if not all((uow.jobs, uow.evidence, uow.members, uow.revisions, uow.bundles)) or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "INDEX" or job.parse_view_id is None:
                raise ValueError("invalid INDEX job")
            model.validate()
            snapshot = (job.payload_json or {}).get("embedding_model_snapshot")
            expected_snapshot = {
                "model_id": model.model_id,
                "model_key": model.model_key,
                "dimension": model.dimension,
                "config_fingerprint": model.config_fingerprint,
            }
            if snapshot != expected_snapshot:
                raise ValueError("INDEX job model snapshot does not match finalization model")
            offset = 0
            while True:
                rows = await uow.evidence.list_active(
                    parse_view_id=job.parse_view_id, limit=500, offset=offset,
                )
                if not rows:
                    break
                if any(
                    row.embedding is None
                    or int(row.embedding_model_id or 0) != model.model_id
                    or row.embedding_model_key != model.model_key
                    or row.embedding_input_hash != retrieval_input_hash(row.retrieval_text)
                    for row in rows
                ):
                    raise ValueError("INDEX job still has unindexed Evidence")
                offset += len(rows)
            if job.bundle_revision_id is None or job.document_version_id is None:
                raise ValueError("INDEX job has no revision member")
            member = await uow.members.get_by_version(
                bundle_revision_id=job.bundle_revision_id,
                document_version_id=job.document_version_id,
                lock=True,
            )
            if member is None:
                raise ValueError("INDEX job member not found")
            member.member_status, member.completed_at = "READY", now
            job.job_status, job.completed_at = "SUCCEEDED", now
            job.result_json = {"indexed_count": indexed_count}
            job.lease_owner = job.lease_until = None
            await uow.session.flush()
            revision = await uow.revisions.get_by_id(bundle_revision_id=job.bundle_revision_id, lock=True)
            if revision is None:
                raise ValueError("INDEX job revision not found")
            members = await uow.members.list_by_revision(bundle_revision_id=revision.bundle_revision_id)
            status = reduce_revision_status(members)
            revision.status = status
            if status in {"READY", "PARTIAL", "FAILED"}:
                revision.completed_at = now
            if status in {"READY", "PARTIAL"}:
                # Discovery/Profile is the final publication gate.  Do not
                # switch Bundle.current_revision_id merely because Evidence
                # vectors are ready.
                profile_fingerprint = sha256(
                    f"{revision.bundle_revision_id}:{revision.snapshot_fingerprint}".encode("utf-8")
                ).hexdigest()
                profile_key = f"PROFILE:{revision.bundle_revision_id}:{profile_fingerprint}"
                existing_profile = await uow.jobs.get_by_idempotency_key(
                    collection_id=revision.collection_id,
                    idempotency_key=profile_key,
                    input_fingerprint=profile_fingerprint,
                )
                if existing_profile is None:
                    await uow.jobs.add(KcIngestionJobEntity(
                        collection_id=revision.collection_id,
                        bundle_revision_id=revision.bundle_revision_id,
                        job_type="PROFILE", idempotency_key=profile_key,
                        input_fingerprint=profile_fingerprint,
                        payload_json={"profile_schema_version": "profile/v1"},
                        job_status="PENDING", priority=job.priority,
                        max_attempts=job.max_attempts, created_by=worker_id, updated_by=worker_id,
                    ))
            await uow.commit()
            return status
