"""Scope-safe status queries for intake and parsing progress."""
from uuid import UUID
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class KnowledgeObjectNotFoundError(Exception):
    """The object does not exist inside the authenticated App/Domain scope."""


@dataclass(frozen=True)
class MemberStatus:
    external_document_id: str
    declared_name: str | None
    document_role: str
    member_status: str
    failure_stage: str | None
    failure_code: str | None
    failure_message: str | None
    received_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class RevisionStatus:
    bundle_revision_id: UUID
    revision_no: int
    source_revision: str
    status: str
    approval_status: str
    reviewed_by: str | None
    reviewed_at: datetime | None
    review_comment: str | None
    accepted_at: datetime | None
    completed_at: datetime | None
    members: list[MemberStatus] | None = None


@dataclass(frozen=True)
class BundleStatus:
    bundle_id: UUID
    collection_id: UUID
    source_id: str
    availability_status: str
    current_revision_id: UUID | None
    revisions: list[RevisionStatus]


@dataclass(frozen=True)
class ProcessingJobStatus:
    job_type: str
    job_status: str
    document_version_id: UUID | None
    attempt_count: int
    started_at: datetime | None
    completed_at: datetime | None
    failure_code: str | None
    failure_message: str | None


@dataclass(frozen=True)
class ProcessingFileStatus:
    document_version_id: UUID | None
    name: str
    document_role: str
    status: str
    failure_stage: str | None
    failure_code: str | None
    failure_message: str | None
    jobs: list[ProcessingJobStatus]


@dataclass(frozen=True)
class ProcessingRevisionStatus:
    bundle_id: UUID
    bundle_revision_id: UUID
    collection_id: UUID
    title: str
    revision_no: int
    status: str
    current_stage: str
    progress_percent: int
    file_count: int
    ready_count: int
    failed_count: int
    reviewed_at: datetime | None
    completed_at: datetime | None
    jobs: list[ProcessingJobStatus]
    files: list[ProcessingFileStatus]


@dataclass(frozen=True)
class KnowledgeFileStatus:
    bundle_id: UUID
    bundle_revision_id: UUID
    collection_id: UUID
    bundle_title: str
    revision_no: int
    document_version_id: UUID | None
    file_name: str
    document_role: str
    status: str
    failure_stage: str | None
    failure_code: str | None
    failure_message: str | None
    mime_type: str | None
    byte_size: int | None
    received_at: datetime | None
    completed_at: datetime | None
    preview_available: bool


@dataclass(frozen=True, slots=True)
class VisualAssetContent:
    payload_uri: str
    mime_type: str


class KnowledgeCoreStatusService:
    def __init__(self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._uow_factory = uow_factory

    async def get_bundle(self, *, domain_id: int, bundle_id: UUID) -> BundleStatus:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.bundles, uow.revisions)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            bundle = await uow.bundles.get_by_id(bundle_id=bundle_id)
            if bundle is None:
                raise KnowledgeObjectNotFoundError("Bundle not found")
            collection = await uow.collections.get_by_id_scope(
                domain_id=domain_id,
                collection_id=bundle.collection_id,
            )
            if collection is None:
                raise KnowledgeObjectNotFoundError("Bundle not found")
            revisions = await uow.revisions.list_by_bundle(bundle_id=bundle_id)
            return BundleStatus(
                bundle_id=bundle.bundle_id,
                collection_id=bundle.collection_id,
                source_id=bundle.source_id,
                availability_status=bundle.availability_status,
                current_revision_id=bundle.current_revision_id,
                revisions=[self._revision(item) for item in revisions],
            )

    async def get_revision(
        self, *, domain_id: int, bundle_id: UUID, bundle_revision_id: UUID, include_members: bool = False
    ) -> RevisionStatus:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.bundles, uow.revisions, uow.members)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            bundle = await uow.bundles.get_by_id(bundle_id=bundle_id)
            if bundle is None or await uow.collections.get_by_id_scope(
                domain_id=domain_id,
                collection_id=bundle.collection_id,
            ) is None:
                raise KnowledgeObjectNotFoundError("Bundle not found")
            revision = await uow.revisions.get_by_id(bundle_revision_id=bundle_revision_id)
            if revision is None or revision.bundle_id != bundle_id:
                raise KnowledgeObjectNotFoundError("Revision not found")
            members = None
            if include_members:
                entities = await uow.members.list_by_revision(bundle_revision_id=bundle_revision_id)
                members = [MemberStatus(
                    external_document_id=item.external_document_id,
                    declared_name=item.declared_name,
                    document_role=item.document_role,
                    member_status=item.member_status,
                    failure_stage=item.failure_stage,
                    failure_code=item.failure_code,
                    failure_message=item.failure_message,
                    received_at=item.received_at,
                    completed_at=item.completed_at,
                ) for item in entities]
            result = self._revision(revision)
            return RevisionStatus(**{**result.__dict__, "members": members})

    async def list_processing(
        self, *, domain_id: int, collection_id: UUID,
        query: str | None, status: str | None, page: int, page_size: int,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.bundles, uow.revisions,
                        uow.members, uow.jobs)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            if await uow.collections.get_by_id_scope(
                domain_id=domain_id, collection_id=collection_id,
            ) is None:
                raise KnowledgeObjectNotFoundError("Collection not found")
            revisions, total = await uow.revisions.list_approved_page(
                collection_id=collection_id, query=query, status=status,
                offset=(page - 1) * page_size, limit=page_size,
            )
            revision_ids = [item.bundle_revision_id for item in revisions]
            bundles = {
                item.bundle_id: item
                for item in await uow.bundles.list_by_ids(
                    bundle_ids=[item.bundle_id for item in revisions]
                )
            }
            members = await uow.members.list_by_revisions(
                bundle_revision_ids=revision_ids
            )
            jobs = await uow.jobs.list_by_revisions(
                bundle_revision_ids=revision_ids
            )
            member_groups = {value: [] for value in revision_ids}
            job_groups = {value: [] for value in revision_ids}
            for member in members:
                member_groups.setdefault(member.bundle_revision_id, []).append(member)
            for job in jobs:
                if job.bundle_revision_id is not None:
                    job_groups.setdefault(job.bundle_revision_id, []).append(job)

            items: list[ProcessingRevisionStatus] = []
            for revision in revisions:
                if bundles.get(revision.bundle_id) is None:
                    continue
                revision_jobs = job_groups.get(revision.bundle_revision_id, [])
                files = [
                    ProcessingFileStatus(
                        document_version_id=member.document_version_id,
                        name=member.declared_name or member.external_document_id,
                        document_role=member.document_role,
                        status=member.member_status,
                        failure_stage=member.failure_stage,
                        failure_code=member.failure_code,
                        failure_message=member.failure_message,
                        jobs=[self._processing_job(job) for job in revision_jobs
                              if job.document_version_id
                              == member.document_version_id],
                    )
                    for member in member_groups.get(
                        revision.bundle_revision_id, []
                    )
                    if member.document_role != "MANIFEST"
                ]
                ready_count = sum(item.status == "READY" for item in files)
                failed_count = sum(
                    item.status in {"FAILED", "SOURCE_UNAVAILABLE"}
                    for item in files
                )
                items.append(ProcessingRevisionStatus(
                    bundle_id=revision.bundle_id,
                    bundle_revision_id=revision.bundle_revision_id,
                    collection_id=revision.collection_id,
                    title=revision.title,
                    revision_no=int(revision.revision_no),
                    status=revision.status,
                    current_stage=self._current_stage(
                        revision.status, revision_jobs
                    ),
                    progress_percent=(
                        round((ready_count + failed_count) * 100 / len(files))
                        if files else 0
                    ),
                    file_count=len(files), ready_count=ready_count,
                    failed_count=failed_count, reviewed_at=revision.reviewed_at,
                    completed_at=revision.completed_at,
                    jobs=[self._processing_job(job) for job in revision_jobs],
                    files=files,
                ))
            return {
                "items": [asdict(item) for item in items],
                "page": page, "page_size": page_size, "total": total,
            }

    async def list_library_files(
        self, *, domain_id: int, collection_id: UUID,
        query: str | None, status: str | None, page: int, page_size: int,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.members)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            if await uow.collections.get_by_id_scope(
                domain_id=domain_id, collection_id=collection_id,
            ) is None:
                raise KnowledgeObjectNotFoundError("Collection not found")
            rows, total = await uow.members.list_current_files_page(
                collection_id=collection_id, query=query, status=status,
                offset=(page - 1) * page_size, limit=page_size,
            )
            items = [KnowledgeFileStatus(
                bundle_id=row["bundle_id"],
                bundle_revision_id=row["bundle_revision_id"],
                collection_id=collection_id,
                bundle_title=row["bundle_title"],
                revision_no=int(row["revision_no"]),
                document_version_id=row["document_version_id"],
                file_name=row["declared_name"] or row["external_document_id"],
                document_role=row["document_role"], status=row["member_status"],
                failure_stage=row["failure_stage"],
                failure_code=row["failure_code"],
                failure_message=row["failure_message"],
                mime_type=row["detected_mime_type"],
                byte_size=(int(row["byte_size"])
                           if row["byte_size"] is not None else None),
                received_at=row["received_at"],
                completed_at=row["completed_at"],
                preview_available=(row["document_version_id"] is not None
                                   and row["storage_state"] == "AVAILABLE"),
            ) for row in rows]
            return {
                "items": [asdict(item) for item in items],
                "page": page, "page_size": page_size, "total": total,
            }

    async def list_file_evidence(
        self, *, domain_id: int, collection_id: UUID,
        document_version_id: UUID, query: str | None,
        evidence_type: str | None, page_no: int | None,
        page: int, page_size: int,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.members, uow.evidence, uow.jobs)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            if await uow.collections.get_by_id_scope(
                domain_id=domain_id, collection_id=collection_id,
            ) is None:
                raise KnowledgeObjectNotFoundError("Collection not found")
            file_row = await uow.members.get_current_file(
                collection_id=collection_id,
                document_version_id=document_version_id,
            )
            if file_row is None:
                raise KnowledgeObjectNotFoundError("Current knowledge file not found")
            items, total = await uow.evidence.list_active_document_page(
                document_version_id=document_version_id, query=query,
                evidence_type=evidence_type, page_no=page_no,
                offset=(page - 1) * page_size, limit=page_size,
            )
            jobs = await uow.jobs.list_by_document_version(
                document_version_id=document_version_id,
            )
            evidence_types = await uow.evidence.list_active_document_types(
                document_version_id=document_version_id,
            )
            return {
                "file": {
                    "document_version_id": document_version_id,
                    "bundle_id": file_row["bundle_id"],
                    "bundle_revision_id": file_row["bundle_revision_id"],
                    "bundle_title": file_row["bundle_title"],
                    "revision_no": int(file_row["revision_no"]),
                    "file_name": file_row["declared_name"]
                    or file_row["external_document_id"],
                    "document_role": file_row["document_role"],
                    "status": file_row["member_status"],
                    "mime_type": file_row["detected_mime_type"],
                    "byte_size": (int(file_row["byte_size"])
                                  if file_row["byte_size"] is not None
                                  else None),
                    "received_at": file_row["received_at"],
                    "completed_at": file_row["completed_at"],
                },
                "items": [{
                    "evidence_id": item.evidence_id,
                    "evidence_type": item.evidence_type,
                    "ordinal": int(item.ordinal),
                    "fragment_index": int(item.fragment_index),
                    "heading_path": item.heading_path_json or [],
                    "section_key": item.section_key,
                    "content_text": item.content_text,
                    "page_start": (int(item.page_start)
                                   if item.page_start is not None else None),
                    "page_end": (int(item.page_end)
                                 if item.page_end is not None else None),
                    "language_code": item.language_code,
                    "token_count": (int(item.token_count)
                                    if item.token_count is not None else None),
                    "quality_score": (float(item.quality_score)
                                      if item.quality_score is not None
                                      else None),
                    "locator": item.locator_json,
                    "source_spans": item.source_spans_json,
                    "created_at": item.created_at,
                } for item in items],
                "evidence_types": evidence_types,
                "jobs": [asdict(self._processing_job(job)) for job in jobs],
                "page": page, "page_size": page_size, "total": total,
            }

    async def list_file_visual_assets(
        self, *, domain_id: int, collection_id: UUID,
        document_version_id: UUID, asset_type: str | None,
        page_no: int | None, page: int, page_size: int,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.members, uow.visual_assets)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            if await uow.collections.get_by_id_scope(
                domain_id=domain_id, collection_id=collection_id,
            ) is None or await uow.members.get_current_file(
                collection_id=collection_id,
                document_version_id=document_version_id,
            ) is None:
                raise KnowledgeObjectNotFoundError("Current knowledge file not found")
            items, total = await uow.visual_assets.list_active_document_page(
                document_version_id=document_version_id,
                asset_type=asset_type, page_no=page_no,
                offset=(page - 1) * page_size, limit=page_size,
            )
            return {
                "items": [{
                    "visual_asset_id": item.visual_asset_id,
                    "asset_type": item.asset_type,
                    "page_no": (int(item.page_no)
                                if item.page_no is not None else None),
                    "bbox": item.bbox_json,
                    "mime_type": item.mime_type,
                    "description": item.description_text,
                    "evidence_id": item.evidence_id,
                    "visual_indexed": item.visual_embedding is not None,
                    "created_at": item.created_at,
                } for item in items],
                "page": page, "page_size": page_size, "total": total,
            }

    async def get_file_visual_asset_content(
        self, *, domain_id: int, collection_id: UUID,
        document_version_id: UUID, visual_asset_id: UUID,
    ) -> VisualAssetContent:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.members, uow.visual_assets)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            if await uow.collections.get_by_id_scope(
                domain_id=domain_id, collection_id=collection_id,
            ) is None or await uow.members.get_current_file(
                collection_id=collection_id,
                document_version_id=document_version_id,
            ) is None:
                raise KnowledgeObjectNotFoundError("Current knowledge file not found")
            asset = await uow.visual_assets.get_active_document_asset(
                document_version_id=document_version_id,
                visual_asset_id=visual_asset_id,
            )
            if asset is None:
                raise KnowledgeObjectNotFoundError("Visual asset not found")
            return VisualAssetContent(
                payload_uri=asset.payload_uri, mime_type=asset.mime_type
            )

    @staticmethod
    def _processing_job(entity) -> ProcessingJobStatus:
        return ProcessingJobStatus(
            job_type=entity.job_type, job_status=entity.job_status,
            document_version_id=entity.document_version_id,
            attempt_count=int(entity.attempt_count),
            started_at=entity.started_at, completed_at=entity.completed_at,
            failure_code=entity.failure_code,
            failure_message=entity.failure_message,
        )

    @staticmethod
    def _current_stage(status: str, jobs: list) -> str:
        active = [item for item in jobs
                  if item.job_status in {"PENDING", "RUNNING", "RETRY_WAIT"}]
        if any(item.job_type == "PROFILE" for item in active):
            return "PROFILING"
        if any(item.job_type == "INDEX" for item in active):
            return "INDEXING"
        if any(item.job_type == "PARSE" for item in active):
            return "PARSING"
        return {
            "READY": "AVAILABLE", "PARTIAL": "PARTIAL",
            "FAILED": "FAILED",
        }.get(status, "QUEUED")

    @staticmethod
    def _revision(entity) -> RevisionStatus:
        return RevisionStatus(
            bundle_revision_id=entity.bundle_revision_id,
            revision_no=entity.revision_no,
            source_revision=entity.source_revision,
            status=entity.status,
            approval_status=entity.approval_status,
            reviewed_by=entity.reviewed_by,
            reviewed_at=entity.reviewed_at,
            review_comment=entity.review_comment,
            accepted_at=entity.accepted_at,
            completed_at=entity.completed_at,
        )
