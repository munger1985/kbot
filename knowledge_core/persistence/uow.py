"""Explicit transaction boundary for Knowledge Core use cases."""
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_core.repositories import BundleRepository, BundleRevisionDocumentRepository, BundleRevisionRepository, CollectionBindingRepository, CollectionRepository, DiscoveryRepository, DocumentRepository, DocumentVersionRepository, EvidenceRepository, IngestionJobRepository, IngestionReceiptRepository, ParseViewRepository, RelationRepository


class KnowledgeCoreUnitOfWork:
    """A short-lived, explicit-commit Unit of Work.

    Unlike the V1 ``get_session`` helper, normal context-manager exit rolls
    back unless the application service explicitly committed the use case.
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.collections: CollectionRepository | None = None
        self.bindings: CollectionBindingRepository | None = None
        self.receipts: IngestionReceiptRepository | None = None
        self.bundles: BundleRepository | None = None
        self.revisions: BundleRevisionRepository | None = None
        self.documents: DocumentRepository | None = None
        self.versions: DocumentVersionRepository | None = None
        self.members: BundleRevisionDocumentRepository | None = None
        self.jobs: IngestionJobRepository | None = None
        self.parse_views: ParseViewRepository | None = None
        self.evidence: EvidenceRepository | None = None
        self.discovery: DiscoveryRepository | None = None
        self.relations: RelationRepository | None = None
        self._committed = False

    async def __aenter__(self) -> "KnowledgeCoreUnitOfWork":
        self.session = self._session_factory()
        self.collections = CollectionRepository(self.session)
        self.bindings = CollectionBindingRepository(self.session)
        self.receipts = IngestionReceiptRepository(self.session)
        self.bundles = BundleRepository(self.session)
        self.revisions = BundleRevisionRepository(self.session)
        self.documents = DocumentRepository(self.session)
        self.versions = DocumentVersionRepository(self.session)
        self.members = BundleRevisionDocumentRepository(self.session)
        self.jobs = IngestionJobRepository(self.session)
        self.parse_views = ParseViewRepository(self.session)
        self.evidence = EvidenceRepository(self.session)
        self.discovery = DiscoveryRepository(self.session)
        self.relations = RelationRepository(self.session)
        return self

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("KnowledgeCoreUnitOfWork has not been entered")
        await self.session.commit()
        self._committed = True

    async def rollback(self) -> None:
        if self.session is not None:
            await self.session.rollback()

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self.session is None:
            return
        try:
            if exc_type is not None or not self._committed:
                await self.session.rollback()
        finally:
            await self.session.close()
            self.session = None
            self.collections = None
            self.bindings = None
            self.receipts = None
            self.bundles = None
            self.revisions = None
            self.documents = None
            self.versions = None
            self.members = None
            self.jobs = None
            self.parse_views = None
            self.evidence = None
            self.discovery = None
            self.relations = None


def create_kc_uow(
    session_factory: async_sessionmaker[AsyncSession],
) -> KnowledgeCoreUnitOfWork:
    """Create a KC Unit of Work using an App-owned session factory."""
    return KnowledgeCoreUnitOfWork(session_factory)
