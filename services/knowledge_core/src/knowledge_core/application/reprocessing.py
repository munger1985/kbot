"""对终态解析任务执行人工触发的重新处理。"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from knowledge_core.application.parse_policy import (
    build_parse_plan,
    validate_parse_policy_overrides,
)
from knowledge_core.application.notifications import KnowledgeOutboxPublisher
from knowledge_core.entities import KcIngestionJobEntity, KcParseViewEntity
from knowledge_core.persistence import KnowledgeCoreUnitOfWork
from knowledge_core.ports.parser_artifact_store import ParserArtifactStore
from platform_core.identity import uuid7


class ReprocessingNotFoundError(LookupError):
    """所选 Revision 或文件不属于当前 Domain。"""


class ReprocessingConflictError(RuntimeError):
    """所选解析任务尚未进入终态。"""


@dataclass(frozen=True)
class ReprocessingResult:
    bundle_revision_id: UUID
    generation: UUID
    scheduled_file_count: int


class KnowledgeCoreReprocessingService:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], KnowledgeCoreUnitOfWork],
        artifact_store: ParserArtifactStore,
        parse_policy_overrides: dict[str, Any] | None = None,
    ):
        self._uow_factory = uow_factory
        self._artifact_store = artifact_store
        self._parse_policy_overrides = parse_policy_overrides or {}
        validate_parse_policy_overrides(self._parse_policy_overrides)

    async def reprocess(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        actor_id: str,
        document_version_id: UUID | None = None,
    ) -> ReprocessingResult:
        generation = uuid7()
        cleanup_manifests: list[dict[str, Any]] = []
        cleanup_asset_uris: list[str] = []
        async with self._uow_factory() as uow:
            if not all(
                (
                    uow.collections,
                    uow.bundles,
                    uow.revisions,
                    uow.members,
                    uow.versions,
                    uow.jobs,
                    uow.parse_views,
                    uow.evidence,
                    uow.visual_assets,
                    uow.discovery,
                )
            ):
                raise RuntimeError("Knowledge Core UoW 未初始化")
            collection = await uow.collections.get_by_id_scope(
                domain_id=domain_id, collection_id=collection_id
            )
            bundle = await uow.bundles.get_by_id(
                bundle_id=bundle_id, lock=True
            )
            revision = await uow.revisions.get_by_id(
                bundle_revision_id=bundle_revision_id, lock=True
            )
            if (
                collection is None
                or collection.status != "ACTIVE"
                or bundle is None
                or bundle.collection_id != collection_id
                or revision is None
                or revision.bundle_id != bundle_id
                or revision.collection_id != collection_id
            ):
                raise ReprocessingNotFoundError("解析对象不存在")
            if revision.approval_status not in {"APPROVED", "NOT_REQUIRED"}:
                raise ReprocessingConflictError(
                    "资料尚未通过审核，不能重新解析"
                )
            members = await uow.members.list_by_revision(
                bundle_revision_id=bundle_revision_id
            )
            if document_version_id is None:
                targets = [
                    item
                    for item in members
                    if item.document_version_id
                    and item.document_role != "MANIFEST"
                ]
            else:
                targets = [
                    item
                    for item in members
                    if item.document_version_id == document_version_id
                    and item.document_role != "MANIFEST"
                ]
            if not targets:
                raise ReprocessingNotFoundError("所选文件不属于该资料包")
            jobs = await uow.jobs.list_by_revisions(
                bundle_revision_ids=[bundle_revision_id]
            )
            latest: dict[UUID, Any] = {}
            for job in jobs:
                if job.job_type == "PARSE" and job.document_version_id:
                    latest[job.document_version_id] = job
            for member in targets:
                previous = latest.get(member.document_version_id)
                if previous is None or previous.job_status not in {
                    "SUCCEEDED",
                    "FAILED",
                }:
                    name = member.declared_name or member.external_document_id
                    raise ReprocessingConflictError(
                        f"文件“{name}”仍在处理中"
                    )
            await uow.discovery.delete_by_revision(
                bundle_revision_id=bundle_revision_id
            )
            for member in targets:
                version = await uow.versions.get_by_id(
                    document_version_id=member.document_version_id
                )
                name = member.declared_name or member.external_document_id
                if version is None or version.storage_state != "AVAILABLE":
                    raise ReprocessingConflictError(
                        f"文件“{name}”的源文件不可用"
                    )
                old_views = await uow.parse_views.list_by_document_version(
                    document_version_id=version.document_version_id,
                    lock=True,
                )
                old_view_ids = [item.parse_view_id for item in old_views]
                cleanup_manifests.extend(
                    item.artifact_manifest_json
                    for item in old_views
                    if item.artifact_manifest_json
                )
                cleanup_asset_uris.extend(
                    await uow.visual_assets.list_payload_uris_by_view_ids(
                        old_view_ids
                    )
                )
                await uow.jobs.detach_parse_views(
                    parse_view_ids=old_view_ids
                )
                await uow.visual_assets.delete_by_view_ids(old_view_ids)
                await uow.evidence.delete_by_view_ids(old_view_ids)
                await uow.parse_views.delete_by_ids(old_view_ids)
                await uow.flush()
                plan = build_parse_plan(
                    collection=collection,
                    version=version,
                    overrides=self._parse_policy_overrides,
                )
                view = await uow.parse_views.add(
                    KcParseViewEntity(
                        collection_id=collection_id,
                        document_version_id=version.document_version_id,
                        view_kind=plan.view_kind,
                        parser_name=plan.parser_name,
                        parse_config_fingerprint=plan.fingerprint,
                        parse_config_json=plan.policy,
                        view_status="PENDING",
                        created_by=actor_id,
                        updated_by=actor_id,
                    )
                )
                previous = latest[version.document_version_id]
                await uow.jobs.add(
                    KcIngestionJobEntity(
                        collection_id=collection_id,
                        bundle_revision_id=bundle_revision_id,
                        document_version_id=version.document_version_id,
                        parse_view_id=view.parse_view_id,
                        job_type="PARSE",
                        idempotency_key=(
                            f"reparse:{generation}:{version.document_version_id}"
                        ),
                        input_fingerprint=version.content_hash,
                        payload_json={
                            "document_version_id": str(
                                version.document_version_id
                            ),
                            "reprocess_generation": str(generation),
                            "notification_operation_id": str(generation),
                            "notification_actor_id": actor_id,
                            "previous_parse_job_id": str(
                                previous.ingestion_job_id
                            ),
                        },
                        job_status="PENDING",
                        priority=previous.priority,
                        max_attempts=previous.max_attempts,
                        created_by=actor_id,
                        updated_by=actor_id,
                    )
                )
                member.member_status = "RECEIVED"
                member.failure_stage = None
                member.failure_code = None
                member.failure_message = None
                member.completed_at = None
                member.updated_by = actor_id
                member.row_version = int(member.row_version) + 1
            revision.status = "PROCESSING"
            revision.completed_at = None
            revision.failure_code = None
            revision.failure_message = None
            revision.updated_by = actor_id
            if bundle.current_revision_id == bundle_revision_id:
                bundle.availability_status = "PROCESSING"
                bundle.updated_by = actor_id
                bundle.row_version = int(bundle.row_version) + 1
            await KnowledgeOutboxPublisher().publish(
                uow=uow,
                event_type="knowledge.ingestion.started",
                actor_id=actor_id,
                resource_id=str(collection.collection_id),
                payload={
                    "event_key": str(generation),
                    "operation_id": str(generation),
                    "correlation_id": str(bundle_revision_id),
                    "display_name": collection.display_name,
                    "progress_current": 0,
                    "progress_total": len(targets),
                },
            )
            await uow.flush()
            await uow.commit()
        for manifest in cleanup_manifests:
            try:
                await self._artifact_store.delete_manifest(manifest)
            except Exception:
                pass
        try:
            await self._artifact_store.delete_uris(cleanup_asset_uris)
        except Exception:
            pass
        return ReprocessingResult(
            bundle_revision_id=bundle_revision_id,
            generation=generation,
            scheduled_file_count=len(targets),
        )
