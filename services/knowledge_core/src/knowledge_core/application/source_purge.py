"""受控清理已拒绝或失败 Revision 的独占源对象。"""

from dataclasses import dataclass
from uuid import UUID

from loguru import logger

from knowledge_core.persistence import KnowledgeCoreUnitOfWork
from knowledge_core.ports.object_store import KnowledgeObjectStore


class RevisionSourcePurgeNotFoundError(LookupError):
    """Revision 不属于请求的 Domain/Collection。"""


class RevisionSourcePurgeConflictError(RuntimeError):
    """Revision 状态不允许清理源对象。"""


class RevisionSourcePurgeStorageError(RuntimeError):
    """至少一个物理对象未能完成清理，可安全重试。"""


@dataclass(frozen=True, slots=True)
class RevisionSourcePurgeResult:
    purged_count: int
    retained_count: int
    already_purged_count: int


class KnowledgeCoreSourcePurgeService:
    def __init__(self, *, uow_factory, object_store: KnowledgeObjectStore):
        self._uow_factory = uow_factory
        self._object_store = object_store

    async def purge_revision_sources(
        self, *, domain_id: int, collection_id: UUID,
        bundle_revision_id: UUID,
    ) -> RevisionSourcePurgeResult:
        """清理只被目标 Revision 引用的源对象，并保留完整审计关系。"""
        candidates: list[tuple[UUID, str]] = []
        retained_count = 0
        already_purged_count = 0
        async with self._uow_factory() as uow:
            self._require_uow(uow)
            revision = await uow.revisions.get_by_id(
                bundle_revision_id=bundle_revision_id,
            )
            if revision is None:
                raise RevisionSourcePurgeNotFoundError("Revision not found")
            collection = await uow.collections.get_by_id_scope(
                domain_id=domain_id,
                collection_id=revision.collection_id,
            )
            if collection is None or collection.collection_id != collection_id:
                raise RevisionSourcePurgeNotFoundError(
                    "Revision is outside the requested Collection"
                )
            bundle = await uow.bundles.get_by_id(
                bundle_id=revision.bundle_id, lock=True,
            )
            revision = await uow.revisions.get_by_id(
                bundle_revision_id=bundle_revision_id, lock=True,
            )
            if revision is None or bundle is None:
                raise RevisionSourcePurgeNotFoundError("Revision not found")
            if not (
                revision.approval_status == "REJECTED"
                or revision.status == "FAILED"
            ):
                raise RevisionSourcePurgeConflictError(
                    "Only rejected or failed revisions can purge source objects"
                )
            if bundle.current_revision_id == revision.bundle_revision_id:
                raise RevisionSourcePurgeConflictError(
                    "Current published revision source objects cannot be purged"
                )
            members = await uow.members.list_by_revision(
                bundle_revision_id=revision.bundle_revision_id,
            )
            seen_versions: set[UUID] = set()
            for member in members:
                version_id = member.document_version_id
                if version_id is None or version_id in seen_versions:
                    continue
                seen_versions.add(version_id)
                version = await uow.versions.get_by_id(
                    document_version_id=version_id, lock=True,
                )
                if version is None:
                    continue
                if version.storage_state == "PURGED":
                    already_purged_count += 1
                    continue
                references = await uow.members.count_other_references(
                    document_version_id=version_id,
                    excluding_revision_id=revision.bundle_revision_id,
                )
                storage_users = await uow.versions.count_other_storage_users(
                    storage_uri=str(version.storage_uri),
                    excluding_version_id=version_id,
                )
                if references or storage_users:
                    retained_count += 1
                    continue
                if version.storage_state not in {"AVAILABLE", "PURGE_PENDING"}:
                    retained_count += 1
                    continue
                version.storage_state = "PURGE_PENDING"
                candidates.append((version_id, str(version.storage_uri)))
            await uow.commit()

        failed = False
        purged_ids: list[UUID] = []
        for version_id, storage_uri in candidates:
            try:
                await self._object_store.delete(storage_uri)
                purged_ids.append(version_id)
            except Exception as exc:
                failed = True
                logger.bind(
                    bundle_revision_id=bundle_revision_id,
                    document_version_id=version_id,
                ).error(
                    "KC 源对象物理清理失败 | exception_type={}",
                    type(exc).__name__,
                )

        if purged_ids:
            async with self._uow_factory() as uow:
                self._require_uow(uow)
                for version_id in purged_ids:
                    version = await uow.versions.get_by_id(
                        document_version_id=version_id, lock=True,
                    )
                    if version is not None and version.storage_state == "PURGE_PENDING":
                        version.storage_state = "PURGED"
                await uow.commit()
        if failed:
            raise RevisionSourcePurgeStorageError(
                "One or more source objects could not be purged"
            )
        return RevisionSourcePurgeResult(
            purged_count=len(purged_ids),
            retained_count=retained_count,
            already_purged_count=already_purged_count,
        )

    @staticmethod
    def _require_uow(uow: KnowledgeCoreUnitOfWork) -> None:
        if not all((
            uow.collections, uow.bundles, uow.revisions,
            uow.members, uow.versions,
        )):
            raise RuntimeError("Knowledge Core Unit of Work is not initialized")
