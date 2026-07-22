"""Scope-safe status queries for intake and parsing progress."""
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class KnowledgeObjectNotFoundError(Exception):
    """The object does not exist inside the authenticated App/Domain scope."""


@dataclass(frozen=True)
class MemberStatus:
    external_document_id: str
    document_role: str
    member_status: str
    failure_stage: str | None
    failure_code: str | None


@dataclass(frozen=True)
class RevisionStatus:
    bundle_revision_id: int
    revision_no: int
    source_revision: str
    status: str
    accepted_at: datetime | None
    completed_at: datetime | None
    members: list[MemberStatus] | None = None


@dataclass(frozen=True)
class BundleStatus:
    bundle_id: int
    collection_id: int
    source_id: str
    availability_status: str
    current_revision_id: int | None
    revisions: list[RevisionStatus]


class KnowledgeCoreStatusService:
    def __init__(self, *, app_id: int, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._app_id = app_id
        self._uow_factory = uow_factory

    async def get_bundle(self, *, domain_id: int, bundle_id: int) -> BundleStatus:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.bundles, uow.revisions)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            bundle = await uow.bundles.get_by_id(bundle_id=bundle_id)
            if bundle is None:
                raise KnowledgeObjectNotFoundError("Bundle not found")
            collection = await uow.collections.get_by_id_scope(
                app_id=self._app_id, domain_id=domain_id, collection_id=bundle.collection_id,
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
        self, *, domain_id: int, bundle_id: int, bundle_revision_id: int, include_members: bool = False
    ) -> RevisionStatus:
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.bundles, uow.revisions, uow.members)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            bundle = await uow.bundles.get_by_id(bundle_id=bundle_id)
            if bundle is None or await uow.collections.get_by_id_scope(
                app_id=self._app_id, domain_id=domain_id, collection_id=bundle.collection_id,
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
                    document_role=item.document_role,
                    member_status=item.member_status,
                    failure_stage=item.failure_stage,
                    failure_code=item.failure_code,
                ) for item in entities]
            result = self._revision(revision)
            return RevisionStatus(**{**result.__dict__, "members": members})

    @staticmethod
    def _revision(entity) -> RevisionStatus:
        return RevisionStatus(
            bundle_revision_id=entity.bundle_revision_id,
            revision_no=entity.revision_no,
            source_revision=entity.source_revision,
            status=entity.status,
            accepted_at=entity.accepted_at,
            completed_at=entity.completed_at,
        )
