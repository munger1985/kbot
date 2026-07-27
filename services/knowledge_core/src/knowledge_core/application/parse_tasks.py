"""解析 Worker 内部 HTTP 协议背后的应用用例。"""
from uuid import UUID
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any

from knowledge_core.entities import (
    KcEvidenceEntity,
    KcIngestionJobEntity,
    KcVisualAssetEntity,
)

from knowledge_core.domain.parse_tasks import ParseLeaseError, ParseTaskClaim, claim_job, verify_lease
from knowledge_core.domain.revision_status import reduce_revision_status
from knowledge_core.persistence import KnowledgeCoreUnitOfWork
from knowledge_core.parsing import (
    EVIDENCE_TYPES,
    build_evidence_key,
    build_output_fingerprint,
    evidence_fingerprint,
    validate_artifact_manifest,
    validate_locator,
    validate_quality_report,
    validate_source_spans,
)
from knowledge_core.ports.parser_artifact_store import ParserArtifactStore


@dataclass(frozen=True)
class ClaimedParseTask:
    job_id: UUID
    lease_owner: str
    lease_until: datetime
    input_fingerprint: str
    document_version_id: UUID
    parse_view_id: UUID
    detected_mime_type: str
    view_kind: str
    parse_config_fingerprint: str
    policy_snapshot: dict[str, Any]


@dataclass(frozen=True)
class EvidenceInput:
    evidence_key: str
    evidence_type: str
    ordinal: int
    fragment_index: int
    content_text: str
    source_spans: list[dict[str, Any]]
    locator_schema_version: str
    locator: dict[str, Any]
    provenance: dict[str, Any]
    parent_evidence_key: str | None = None
    source_item_ref: str | None = None
    heading_path: list[str] | None = None
    section_key: str | None = None
    hierarchy_depth: int | None = None
    heading_level: int | None = None
    payload_descriptor: dict[str, Any] | None = None
    page_start: int | None = None
    page_end: int | None = None
    language_code: str | None = None
    token_count: int | None = None
    quality_score: float | None = None


@dataclass(frozen=True)
class VisualAssetInput:
    asset_key: str
    asset_type: str
    page_no: int | None
    source_item_ref: str | None
    bbox: dict[str, Any] | None
    mime_type: str
    content_base64: str
    content_sha256: str
    description: str | None = None


class KnowledgeCoreParseTaskService:
    def __init__(
        self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork],
        artifact_store: ParserArtifactStore,
    ):
        self._uow_factory = uow_factory
        self._artifact_store = artifact_store

    async def claim(self, claim: ParseTaskClaim) -> list[ClaimedParseTask]:
        claim.validate()
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.versions is None or uow.parse_views is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            candidates = await uow.jobs.claim_candidates(now=now, limit=claim.max_tasks)
            tasks: list[ClaimedParseTask] = []
            for job in candidates:
                if job.document_version_id is None or job.parse_view_id is None:
                    continue
                version = await uow.versions.get_by_id(document_version_id=job.document_version_id)
                view = await uow.parse_views.get_by_id(parse_view_id=job.parse_view_id)
                if version is None or view is None or version.storage_state != "AVAILABLE":
                    continue
                lease_until = claim_job(job, claim, now)
                tasks.append(ClaimedParseTask(
                    job_id=job.ingestion_job_id, lease_owner=claim.worker_id, lease_until=lease_until,
                    input_fingerprint=job.input_fingerprint, document_version_id=version.document_version_id,
                    parse_view_id=job.parse_view_id,
                    detected_mime_type=version.detected_mime_type,
                    view_kind=view.view_kind, parse_config_fingerprint=view.parse_config_fingerprint,
                    policy_snapshot=view.parse_config_json or {},
                ))
            if tasks:
                if uow.session is None:
                    raise RuntimeError("Knowledge Core Unit of Work session is not initialized")
                await uow.session.flush()
                await uow.commit()
            return tasks

    async def source_descriptor(
        self, *, job_id: UUID, worker_id: str, input_fingerprint: str,
    ) -> tuple[str, str]:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.versions is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=False)
            if job is None or job.document_version_id is None:
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            version = await uow.versions.get_by_id(document_version_id=job.document_version_id)
            if version is None or version.storage_state != "AVAILABLE":
                raise ParseLeaseError("JOB_STALE")
            return version.storage_uri, version.detected_mime_type

    async def heartbeat(self, *, job_id: UUID, worker_id: str, input_fingerprint: str, lease_seconds: int = 120) -> datetime:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None:
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            job.heartbeat_at = now
            job.lease_until = now + timedelta(seconds=lease_seconds)
            job.row_version += 1
            await uow.session.flush()
            await uow.commit()
            return job.lease_until

    async def upload_artifact(
        self, *, job_id: UUID, worker_id: str, input_fingerprint: str,
        artifact_name: str, payload: Any, expected_sha256: str,
        schema: str, generator: str,
    ) -> dict[str, str]:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=False)
            if job is None:
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
        return await self._artifact_store.put_json(
            job_id=job_id, artifact_name=artifact_name, payload=payload,
            expected_sha256=expected_sha256, schema=schema, generator=generator,
        )

    async def submit_evidence(self, *, job_id: UUID, worker_id: str, input_fingerprint: str, items: list[EvidenceInput]) -> int:
        if not items or len(items) > 500:
            raise ValueError("evidence batch must contain between 1 and 500 items")
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if not all((uow.jobs, uow.versions, uow.members, uow.parse_views, uow.evidence)) or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None:
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            if job.parse_view_id is None or job.document_version_id is None or job.bundle_revision_id is None:
                raise ParseLeaseError("JOB_STALE")
            version = await uow.versions.get_by_id(document_version_id=job.document_version_id)
            view = await uow.parse_views.get_by_id(parse_view_id=job.parse_view_id, lock=True)
            member = await uow.members.get_by_version(bundle_revision_id=job.bundle_revision_id, document_version_id=job.document_version_id, lock=True)
            if version is None or view is None or member is None or view.document_version_id != version.document_version_id:
                raise ParseLeaseError("JOB_STALE")
            inserted = 0
            for item in items:
                content = item.content_text.strip()
                if not item.evidence_key.strip() or not content:
                    raise ValueError("evidence_key and content_text are required")
                if item.evidence_type not in EVIDENCE_TYPES:
                    raise ValueError(f"unsupported evidence_type: {item.evidence_type}")
                if item.fragment_index < 0:
                    raise ValueError("fragment_index must be non-negative")
                validate_source_spans(item.source_spans)
                expected_key = build_evidence_key(
                    parse_view_id=view.parse_view_id, source_spans=item.source_spans,
                    fragment_index=item.fragment_index, evidence_type=item.evidence_type,
                )
                if item.evidence_key != expected_key:
                    raise ValueError("evidence_key does not match its source spans")
                if not item.locator_schema_version.strip() or not item.locator:
                    raise ValueError("locator_schema_version and locator are required")
                validate_locator(item.locator_schema_version, item.locator)
                if not item.provenance:
                    raise ValueError("provenance is required")
                payload_uri = None
                if item.payload_descriptor is not None:
                    payload_uri = item.payload_descriptor.get("uri")
                    if not isinstance(payload_uri, str) or not payload_uri.strip():
                        raise ValueError("payload_descriptor.uri is required")
                content_hash = evidence_fingerprint(
                    content_text=content, source_spans=item.source_spans, locator=item.locator,
                )
                existing = await uow.evidence.get_by_key(parse_view_id=view.parse_view_id, evidence_key=item.evidence_key)
                if existing is not None:
                    if existing.content_hash != content_hash:
                        raise ParseLeaseError("EVIDENCE_KEY_CONFLICT")
                    continue
                retrieval_text = "\n".join([*(item.heading_path or []), content])
                await uow.evidence.add(KcEvidenceEntity(
                    collection_id=job.collection_id, bundle_revision_id=job.bundle_revision_id,
                    bundle_revision_document_id=member.bundle_revision_document_id,
                    document_id=version.document_id, document_version_id=version.document_version_id,
                    parse_view_id=view.parse_view_id, evidence_key=item.evidence_key,
                    evidence_type=item.evidence_type, ordinal=item.ordinal,
                    fragment_index=item.fragment_index,
                    parent_evidence_key=item.parent_evidence_key, source_item_ref=item.source_item_ref,
                    source_spans_json=item.source_spans, heading_path_json=item.heading_path,
                    section_key=item.section_key, hierarchy_depth=item.hierarchy_depth,
                    heading_level=item.heading_level, locator_schema_version=item.locator_schema_version,
                    locator_json=item.locator, payload_uri=payload_uri, provenance_json=item.provenance,
                    content_text=content, retrieval_text=retrieval_text, content_hash=content_hash,
                    page_start=item.page_start, page_end=item.page_end, language_code=item.language_code,
                    token_count=item.token_count, quality_score=item.quality_score, status="STAGED",
                    created_by=worker_id, updated_by=worker_id,
                ))
                inserted += 1
            view.view_status = "PARSING"
            member.member_status = "PARSING"
            await uow.session.flush()
            await uow.commit()
            return inserted

    async def submit_visual_assets(
        self,
        *,
        job_id: UUID,
        worker_id: str,
        input_fingerprint: str,
        items: list[VisualAssetInput],
    ) -> int:
        """保存 Parser 原始图片；此阶段禁止生成视觉向量。"""
        import base64

        if not items or len(items) > 500:
            raise ValueError("视觉资产批次必须包含 1 到 500 项")
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if not all(
                (
                    uow.jobs,
                    uow.versions,
                    uow.visual_assets,
                    uow.evidence,
                    uow.session,
                )
            ):
                raise RuntimeError("Knowledge Core Unit of Work 未初始化")
            job = await uow.jobs.get_by_id(
                ingestion_job_id=job_id, lock=True
            )
            if job is None:
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(
                job,
                worker_id=worker_id,
                input_fingerprint=input_fingerprint,
                now=now,
            )
            if (
                job.parse_view_id is None
                or job.document_version_id is None
                or job.bundle_revision_id is None
            ):
                raise ParseLeaseError("JOB_STALE")
            version = await uow.versions.get_by_id(
                document_version_id=job.document_version_id
            )
            if version is None:
                raise ParseLeaseError("JOB_STALE")
            inserted = 0
            for item in items:
                if item.asset_type not in {"PAGE", "FIGURE"}:
                    raise ValueError("不支持的视觉资产类型")
                existing = await uow.visual_assets.get_by_key(
                    parse_view_id=job.parse_view_id,
                    asset_key=item.asset_key,
                )
                if existing is not None:
                    if existing.content_hash != item.content_sha256:
                        raise ParseLeaseError("VISUAL_ASSET_KEY_CONFLICT")
                    continue
                content = base64.b64decode(
                    item.content_base64, validate=True
                )
                if len(content) > 16 * 1024 * 1024:
                    raise ValueError("单个视觉资产超过 16 MiB")
                descriptor = await self._artifact_store.put_bytes(
                    job_id=job_id,
                    asset_key=item.asset_key,
                    payload=content,
                    expected_sha256=item.content_sha256,
                    mime_type=item.mime_type,
                )
                evidence = (
                    await uow.evidence.get_by_source_ref(
                        parse_view_id=job.parse_view_id,
                        source_item_ref=item.source_item_ref,
                    )
                    if item.source_item_ref
                    else None
                )
                await uow.visual_assets.add(
                    KcVisualAssetEntity(
                        collection_id=job.collection_id,
                        bundle_revision_id=job.bundle_revision_id,
                        document_id=version.document_id,
                        document_version_id=version.document_version_id,
                        parse_view_id=job.parse_view_id,
                        evidence_id=(
                            evidence.evidence_id if evidence else None
                        ),
                        asset_key=item.asset_key,
                        asset_type=item.asset_type,
                        page_no=item.page_no,
                        source_item_ref=item.source_item_ref,
                        bbox_json=item.bbox,
                        mime_type=item.mime_type,
                        payload_uri=descriptor["uri"],
                        content_hash=item.content_sha256,
                        description_text=item.description,
                        status="STAGED",
                        created_by=worker_id,
                        updated_by=worker_id,
                    )
                )
                inserted += 1
            await uow.commit()
            return inserted

    async def complete(
        self, *, job_id: UUID, worker_id: str, input_fingerprint: str,
        artifact_manifest: dict[str, Any], output_fingerprint: str,
        quality_score: float | None, quality_report: dict[str, Any],
    ) -> int:
        old_artifact_manifests: list[dict[str, Any]] = []
        validate_artifact_manifest(artifact_manifest)
        validate_quality_report(quality_report)
        normalized_output_fingerprint = output_fingerprint.lower()
        if len(normalized_output_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_output_fingerprint
        ):
            raise ValueError("output_fingerprint must be lowercase SHA-256")
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if not all((uow.jobs, uow.members, uow.parse_views, uow.evidence, uow.revisions, uow.bundles)) or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None:
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            if job.parse_view_id is None or job.document_version_id is None or job.bundle_revision_id is None:
                raise ParseLeaseError("JOB_STALE")
            view = await uow.parse_views.get_by_id(parse_view_id=job.parse_view_id, lock=True)
            member = await uow.members.get_by_version(bundle_revision_id=job.bundle_revision_id, document_version_id=job.document_version_id, lock=True)
            count = await uow.evidence.count_staged(parse_view_id=job.parse_view_id)
            if view is None or member is None or count <= 0:
                raise ValueError("parse completion requires staged evidence")
            evidence_keys = await uow.evidence.list_staged_keys(parse_view_id=job.parse_view_id)
            expected_output_fingerprint = build_output_fingerprint(
                artifact_hashes={
                    name: descriptor["sha256"] for name, descriptor in artifact_manifest.items()
                },
                evidence_keys=evidence_keys,
            )
            if expected_output_fingerprint != normalized_output_fingerprint:
                raise ValueError("output_fingerprint does not match uploaded artifacts and Evidence")
            old_views = await uow.parse_views.list_active_others(
                document_version_id=view.document_version_id,
                view_kind=view.view_kind,
                except_parse_view_id=view.parse_view_id,
            )
            old_view_ids = [old_view.parse_view_id for old_view in old_views]
            old_artifact_manifests = [
                old_view.artifact_manifest_json for old_view in old_views
                if old_view.artifact_manifest_json
            ]
            if uow.visual_assets is not None:
                await uow.visual_assets.delete_by_view_ids(old_view_ids)
            await uow.evidence.delete_by_view_ids(old_view_ids)
            await uow.parse_views.delete_by_ids(old_view_ids)
            await uow.evidence.activate_staged(parse_view_id=view.parse_view_id)
            if uow.visual_assets is not None:
                await uow.visual_assets.activate_staged(
                    parse_view_id=view.parse_view_id
                )
            view.view_status, view.quality_score = "ACTIVE", quality_score
            view.quality_report_json, view.artifact_manifest_json, view.activated_at = quality_report, artifact_manifest, now
            # Parsing and indexing are separate durable stages.  The view can
            # be structurally ACTIVE while its Evidence remains unavailable
            # to retrieval until the Collection-bound INDEX job succeeds.
            member.member_status, member.completed_at = "INDEXING", None
            index_fingerprint = sha256(
                f"{view.parse_view_id}:{normalized_output_fingerprint}".encode("utf-8")
            ).hexdigest()
            index_key = f"INDEX:{view.parse_view_id}:{normalized_output_fingerprint}"
            index_job = await uow.jobs.get_by_idempotency_key(
                collection_id=job.collection_id,
                idempotency_key=index_key,
                input_fingerprint=index_fingerprint,
            )
            if index_job is None:
                index_job = await uow.jobs.add(KcIngestionJobEntity(
                    collection_id=job.collection_id,
                    bundle_revision_id=job.bundle_revision_id,
                    document_version_id=job.document_version_id,
                    parse_view_id=view.parse_view_id,
                    job_type="INDEX", idempotency_key=index_key,
                    input_fingerprint=index_fingerprint,
                    payload_json={"parse_output_fingerprint": normalized_output_fingerprint},
                    job_status="PENDING", priority=job.priority,
                    max_attempts=job.max_attempts, created_by=worker_id, updated_by=worker_id,
                ))
            job.job_status, job.completed_at, job.result_json = "SUCCEEDED", now, {
                "evidence_count": count,
                "output_fingerprint": normalized_output_fingerprint,
                "artifact_manifest": artifact_manifest,
                "index_job_id": index_job.ingestion_job_id,
            }
            job.lease_owner = job.lease_until = None
            await uow.session.flush()
            await self._reconcile_revision(uow, job.bundle_revision_id, now)
            await uow.commit()
        for old_manifest in old_artifact_manifests:
            try:
                await self._artifact_store.delete_manifest(old_manifest)
            except Exception:
                # Database correctness is already committed; orphan cleanup can be retried by storage maintenance.
                pass
        return count

    async def fail(
        self, *, job_id: UUID, worker_id: str, input_fingerprint: str,
        failure_class: str, failure_code: str, failure_message: str | None = None,
        artifact_manifest: dict[str, Any] | None = None,
    ) -> str:
        if artifact_manifest is not None:
            validate_artifact_manifest(artifact_manifest)
        now = datetime.now(timezone.utc)
        cleanup_manifest: dict[str, Any] | None = None
        async with self._uow_factory() as uow:
            if not all((uow.jobs, uow.members, uow.parse_views, uow.evidence, uow.revisions, uow.bundles)) or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None:
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            view = await uow.parse_views.get_by_id(parse_view_id=job.parse_view_id, lock=True)
            member = await uow.members.get_by_version(bundle_revision_id=job.bundle_revision_id, document_version_id=job.document_version_id, lock=True)
            if view is None or member is None:
                raise ParseLeaseError("JOB_STALE")
            retryable = failure_class == "TRANSIENT" and job.attempt_count < job.max_attempts
            job.failure_class, job.failure_code, job.failure_message = failure_class, failure_code, failure_message
            job.lease_owner = job.lease_until = None
            if retryable:
                job.job_status = "RETRY_WAIT"
                job.available_at = now + timedelta(seconds=min(300, 2 ** job.attempt_count * 5))
                view.view_status, member.member_status = "PENDING", "RECEIVED"
            else:
                job.job_status, job.completed_at = "FAILED", now
                member.member_status = "FAILED"
                member.failure_stage, member.failure_code, member.failure_message = "PARSE", failure_code, failure_message
                cleanup_manifest = artifact_manifest or view.artifact_manifest_json
                # 失败 Parse View 按设计不留存；先解除任务外键，再清理视图。
                job.parse_view_id = None
                await uow.session.flush()
                if uow.visual_assets is not None:
                    await uow.visual_assets.delete_by_view_ids(
                        [view.parse_view_id]
                    )
                await uow.evidence.delete_by_view_ids([view.parse_view_id])
                await uow.parse_views.delete_by_ids([view.parse_view_id])
            await uow.session.flush()
            if not retryable:
                await self._reconcile_revision(uow, job.bundle_revision_id, now)
            await uow.commit()
            result_status = job.job_status
        if cleanup_manifest:
            try:
                await self._artifact_store.delete_manifest(cleanup_manifest)
            except Exception:
                pass
        return result_status

    @staticmethod
    async def _reconcile_revision(uow, bundle_revision_id: UUID, now: datetime) -> str:
        revision = await uow.revisions.get_by_id(bundle_revision_id=bundle_revision_id, lock=True)
        if revision is None:
            raise ParseLeaseError("JOB_STALE")
        members = await uow.members.list_by_revision(bundle_revision_id=bundle_revision_id)
        status = reduce_revision_status(members)
        revision.status = status
        if status in {"READY", "PARTIAL", "FAILED"}:
            revision.completed_at = now
        if status in {"READY", "PARTIAL"}:
            bundle = await uow.bundles.get_by_id(bundle_id=revision.bundle_id, lock=True)
            if bundle is None:
                raise ParseLeaseError("JOB_STALE")
            bundle.current_revision_id = revision.bundle_revision_id
            bundle.availability_status = status
            bundle.row_version += 1
        elif status == "FAILED":
            bundle = await uow.bundles.get_by_id(bundle_id=revision.bundle_id, lock=True)
            if bundle is not None and bundle.current_revision_id is None:
                bundle.availability_status = "FAILED"
        return status
