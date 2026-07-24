"""Final database transaction for an accepted KM Asset intake.

Object staging/publishing happens before this use case.  The caller supplies
only verified immutable objects; this transaction creates no externally visible
Revision until all KC facts and parse jobs can be committed together.
"""
from uuid import UUID
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from knowledge_core.entities import (
    KcBundleEntity, KcBundleRevisionDocumentEntity, KcBundleRevisionEntity,
    KcDocumentEntity, KcDocumentVersionEntity, KcIngestionJobEntity,
    KcIngestionReceiptEntity, KcParseViewEntity,
)
from knowledge_core.domain.intake import KmAssetIntakeManifest
from knowledge_core.domain.manifest import render_bundle_manifest
from knowledge_core.parsing import canonical_json_hash
from knowledge_core.persistence import KnowledgeCoreUnitOfWork


_FORBIDDEN_PARSE_POLICY_KEYS = frozenset({
    "embedding", "embedding_model", "embedding_model_id",
    "embedding_served_model_name",
    "txt_embed_model", "txt_embedding_model", "query_vector",
})


class IntakeConflictError(Exception):
    """A repeat request would change immutable source or receipt facts."""


class IntakeCollectionError(Exception):
    """Target Collection is absent or cannot receive new source snapshots."""


@dataclass(frozen=True)
class ReserveIntakeCommand:
    domain_id: int
    collection_key: str
    actor_id: str
    idempotency_key: str
    manifest: KmAssetIntakeManifest
    source_system: str = "metadb"
    source_type: str = "KM_ASSET"
    allowed_document_roles: tuple[str, ...] = ("ATTACHMENT",)


@dataclass(frozen=True)
class IntakeReservation:
    receipt_id: UUID
    receipt_status: str
    newly_created: bool
    bundle_id: UUID | None = None
    bundle_revision_id: UUID | None = None


@dataclass(frozen=True)
class PreparePublishCommand:
    domain_id: int
    collection_key: str
    actor_id: str
    idempotency_key: str
    manifest: KmAssetIntakeManifest
    receipt_id: UUID
    source_system: str = "metadb"
    source_type: str = "KM_ASSET"
    allowed_document_roles: tuple[str, ...] = ("ATTACHMENT",)


@dataclass(frozen=True)
class PublishPreparation:
    collection_id: UUID
    bundle_id: UUID
    document_ids: dict[str, UUID]


@dataclass(frozen=True)
class PublishedAttachment:
    external_document_id: str
    storage_uri: str
    detected_mime_type: str


@dataclass(frozen=True)
class PublishedManifest:
    storage_uri: str
    byte_size: int
    content_sha256: str
    detected_mime_type: str = "text/markdown"


@dataclass(frozen=True)
class AcceptKmAssetCommand:
    domain_id: int
    collection_key: str
    actor_id: str
    idempotency_key: str
    manifest: KmAssetIntakeManifest
    attachments: dict[str, PublishedAttachment]
    published_manifest: PublishedManifest | None
    source_system: str = "metadb"
    source_type: str = "KM_ASSET"
    generate_manifest: bool = True
    allowed_document_roles: tuple[str, ...] = ("ATTACHMENT",)
    approval_required: bool = False


@dataclass(frozen=True)
class IntakeAcceptance:
    bundle_id: UUID
    bundle_revision_id: UUID
    source_revision: str
    acceptance_status: str = "ACCEPTED"


@dataclass(frozen=True)
class ReviewIntakeCommand:
    domain_id: int
    collection_key: str
    bundle_revision_id: UUID
    decision: str
    actor_id: str
    comment: str | None = None


@dataclass(frozen=True)
class IntakeReview:
    bundle_id: UUID
    bundle_revision_id: UUID
    source_revision: str
    title: str
    approval_status: str
    revision_status: str
    document_names: tuple[str, ...]
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_comment: str | None = None


class IntakeReviewNotFoundError(LookupError):
    """待审批 Revision 不属于当前 Domain/Collection。"""


class IntakeReviewConflictError(RuntimeError):
    """审批决定与 Revision 当前状态冲突。"""


class KnowledgeCoreIntakeService:
    """写入不可变来源快照，并按来源信任策略决定审批或创建解析任务。"""

    def __init__(
        self, *, app_id: int, receipt_ttl_seconds: int,
        uow_factory: Callable[[], KnowledgeCoreUnitOfWork],
        parse_policy_overrides: dict | None = None,
    ):
        self._app_id = app_id
        self._receipt_ttl_seconds = receipt_ttl_seconds
        self._uow_factory = uow_factory
        self._parse_policy_overrides = parse_policy_overrides or {}
        forbidden_keys = _FORBIDDEN_PARSE_POLICY_KEYS.intersection(self._parse_policy_overrides)
        if forbidden_keys:
            raise ValueError(
                "Parser policy cannot select or generate retrieval embeddings: "
                + ", ".join(sorted(forbidden_keys))
            )

    async def reserve(self, command: ReserveIntakeCommand) -> IntakeReservation:
        """Commit a small Receipt transaction before any file byte is staged.

        A repeated request with the same semantic fingerprint receives the
        existing reservation.  The caller must not stage again unless the
        receipt is newly created and in ``RECEIVING`` state.
        """
        fingerprint = command.manifest.fingerprint()
        if not command.idempotency_key.strip():
            raise ValueError("idempotency_key is required")
        async with self._uow_factory() as uow:
            if uow.collections is None or uow.receipts is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_scope_key(
                app_id=self._app_id, domain_id=command.domain_id, collection_key=command.collection_key,
            )
            if collection is None or collection.status != "ACTIVE":
                raise IntakeCollectionError("Collection is not active in this Domain")
            existing = await uow.receipts.get_by_idempotency_key(
                collection_id=collection.collection_id, actor_id=command.actor_id, idempotency_key=command.idempotency_key,
            )
            if existing is not None:
                if existing.request_fingerprint != fingerprint:
                    raise IntakeConflictError("IDEMPOTENCY_KEY_REUSED")
                return IntakeReservation(
                    existing.ingestion_receipt_id, existing.receipt_status, False,
                    existing.bundle_id, existing.bundle_revision_id,
                )
            receipt = KcIngestionReceiptEntity(
                collection_id=collection.collection_id, actor_id=command.actor_id,
                idempotency_key=command.idempotency_key, request_fingerprint=fingerprint,
                receipt_status="RECEIVING",
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=self._receipt_ttl_seconds),
                created_by=command.actor_id, updated_by=command.actor_id,
            )
            receipt = await uow.receipts.add(receipt)
            await uow.commit()
            return IntakeReservation(receipt.ingestion_receipt_id, receipt.receipt_status, True)

    async def prepare_publish(self, command: PreparePublishCommand) -> PublishPreparation:
        """Allocate stable Bundle/Document identities after all bytes are staged.

        No Version, Member, Revision or Job is created here, so a later object
        publish failure cannot make a source snapshot queryable.
        """
        fingerprint = command.manifest.fingerprint()
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.receipts, uow.bundles, uow.documents)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_scope_key(app_id=self._app_id, domain_id=command.domain_id, collection_key=command.collection_key)
            if collection is None or collection.status != "ACTIVE":
                raise IntakeCollectionError("Collection is not active in this Domain")
            receipt = await uow.receipts.get_by_idempotency_key(collection_id=collection.collection_id, actor_id=command.actor_id, idempotency_key=command.idempotency_key)
            if receipt is None or receipt.ingestion_receipt_id != command.receipt_id:
                raise IntakeConflictError("INGESTION_RECEIPT_NOT_FOUND")
            if receipt.request_fingerprint != fingerprint:
                raise IntakeConflictError("IDEMPOTENCY_KEY_REUSED")
            if receipt.receipt_status == "ACCEPTED" and receipt.bundle_id is not None:
                return PublishPreparation(collection.collection_id, receipt.bundle_id, {})
            if receipt.receipt_status not in {"RECEIVING", "STAGED"}:
                raise IntakeConflictError("INGESTION_RECEIPT_NOT_STAGEABLE")
            bundle = await uow.bundles.get_by_source(
                collection_id=collection.collection_id, source_system=command.source_system, source_type=command.source_type,
                source_id=command.manifest.bundle.source_id, lock=True,
            )
            if bundle is None:
                bundle = await uow.bundles.add(KcBundleEntity(
                    collection_id=collection.collection_id, source_system=command.source_system, source_type=command.source_type,
                    source_id=command.manifest.bundle.source_id, availability_status="EMPTY",
                    created_by=command.actor_id, updated_by=command.actor_id,
                ))
            ids: dict[str, UUID] = {}
            for external_id in ["__manifest__", *[item.external_document_id for item in command.manifest.documents]]:
                document = await self._document(uow, bundle, external_id, command.actor_id)
                ids[external_id] = document.document_id
            receipt.receipt_status = "STAGED"
            receipt.staging_manifest_json = {
                "bundle_id": str(bundle.bundle_id),
                "document_ids": {
                    key: str(value) for key, value in ids.items()
                },
            }
            receipt.updated_by = command.actor_id
            if uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work session is not initialized")
            await uow.session.flush()
            await uow.commit()
            return PublishPreparation(collection.collection_id, bundle.bundle_id, ids)

    async def accept_published(self, command: AcceptKmAssetCommand) -> IntakeAcceptance:
        fingerprint = command.manifest.fingerprint()
        command.manifest.validate_declarations(
            set(command.attachments), allowed_roles=set(command.allowed_document_roles),
        )
        rendered_manifest = render_bundle_manifest(command.manifest.bundle)
        if command.generate_manifest and command.published_manifest is None:
            raise ValueError("published_manifest is required when generate_manifest is true")
        if command.generate_manifest and (
                command.published_manifest.content_sha256.lower() != rendered_manifest.content_sha256
                or command.published_manifest.byte_size != len(rendered_manifest.content)):
            raise ValueError("published_manifest does not match KC-rendered source metadata")
        if not command.idempotency_key.strip():
            raise ValueError("idempotency_key is required")

        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.receipts, uow.bundles, uow.revisions, uow.documents, uow.versions, uow.members, uow.jobs, uow.parse_views)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            collection = await uow.collections.get_by_scope_key(
                app_id=self._app_id, domain_id=command.domain_id, collection_key=command.collection_key,
            )
            if collection is None or collection.status != "ACTIVE":
                raise IntakeCollectionError("Collection is not active in this Domain")

            receipt = await uow.receipts.get_by_idempotency_key(
                collection_id=collection.collection_id, actor_id=command.actor_id, idempotency_key=command.idempotency_key,
            )
            if receipt is not None:
                if receipt.request_fingerprint != fingerprint:
                    raise IntakeConflictError("IDEMPOTENCY_KEY_REUSED")
                if receipt.bundle_id is not None and receipt.bundle_revision_id is not None:
                    return IntakeAcceptance(
                        receipt.bundle_id,
                        receipt.bundle_revision_id,
                        command.manifest.bundle.source_revision,
                        receipt.receipt_status,
                    )
                if receipt.receipt_status != "STAGED":
                    raise IntakeConflictError("INGESTION_IN_PROGRESS")

            bundle = await uow.bundles.get_by_source(
                collection_id=collection.collection_id, source_system=command.source_system, source_type=command.source_type,
                source_id=command.manifest.bundle.source_id, lock=True,
            )
            if bundle is None:
                bundle = await uow.bundles.add(KcBundleEntity(
                    collection_id=collection.collection_id,
                    source_system=command.source_system,
                    source_type=command.source_type,
                    source_id=command.manifest.bundle.source_id, availability_status="EMPTY",
                    created_by=command.actor_id, updated_by=command.actor_id,
                ))

            previous = await uow.revisions.get_by_source_revision(
                bundle_id=bundle.bundle_id, source_revision=command.manifest.bundle.source_revision,
            )
            if previous is not None:
                if previous.snapshot_fingerprint != fingerprint:
                    raise IntakeConflictError("SOURCE_REVISION_CONFLICT")
                acceptance_status = {
                    "PENDING": "PENDING_REVIEW",
                    "REJECTED": "REJECTED",
                }.get(previous.approval_status, "ACCEPTED")
                if receipt is not None:
                    receipt.receipt_status = acceptance_status
                    receipt.bundle_id = bundle.bundle_id
                    receipt.bundle_revision_id = (
                        previous.bundle_revision_id
                    )
                    receipt.updated_by = command.actor_id
                    if uow.session is None:
                        raise RuntimeError(
                            "Knowledge Core Unit of Work session is not initialized"
                        )
                    await uow.session.flush()
                    await uow.commit()
                return IntakeAcceptance(
                    bundle.bundle_id,
                    previous.bundle_revision_id,
                    previous.source_revision,
                    acceptance_status,
                )

            revision = await uow.revisions.add(KcBundleRevisionEntity(
                collection_id=collection.collection_id, bundle_id=bundle.bundle_id,
                revision_no=await uow.revisions.next_revision_no(bundle_id=bundle.bundle_id),
                source_revision=command.manifest.bundle.source_revision, snapshot_fingerprint=fingerprint,
                manifest_json=command.manifest.bundle.model_dump(mode="json"), title=command.manifest.bundle.title,
                canonical_url=command.manifest.bundle.canonical_url,
                security_level=command.manifest.bundle.security_level, facet_json=command.manifest.bundle.facet,
                status=(
                    "PENDING_REVIEW"
                    if command.approval_required
                    else "PROCESSING"
                ),
                approval_status=(
                    "PENDING"
                    if command.approval_required
                    else "NOT_REQUIRED"
                ),
                created_by=command.actor_id, updated_by=command.actor_id,
            ))
            if command.generate_manifest:
                manifest_document = await self._document(uow, bundle, "__manifest__", command.actor_id)
                manifest_version = await self._version_from_values(
                    uow, collection.collection_id, bundle.bundle_id, manifest_document,
                    content_hash=rendered_manifest.content_sha256, byte_size=command.published_manifest.byte_size,
                    storage_uri=command.published_manifest.storage_uri,
                    detected_mime_type=command.published_manifest.detected_mime_type,
                    security_level=command.manifest.bundle.security_level, actor_id=command.actor_id,
                )
                await uow.members.add(KcBundleRevisionDocumentEntity(
                    collection_id=collection.collection_id, bundle_revision_id=revision.bundle_revision_id,
                    document_id=manifest_document.document_id, document_version_id=manifest_version.document_version_id,
                    document_role="MANIFEST", ordinal=0, required_flag=1, external_document_id="__manifest__",
                    declared_name="manifest.md", declared_mime_type="text/markdown", member_status="RECEIVED",
                    created_by=command.actor_id, updated_by=command.actor_id,
                ))
                if not command.approval_required:
                    await self._enqueue_parse(
                        uow, collection.collection_id,
                        revision.bundle_revision_id, manifest_version,
                        command.actor_id,
                    )
            for declaration in command.manifest.documents:
                published = command.attachments[declaration.part_name]
                document = await self._document(uow, bundle, declaration.external_document_id, command.actor_id)
                version = await self._version(
                    uow, collection.collection_id, bundle.bundle_id, document, declaration, published,
                    command.manifest.bundle.security_level, command.actor_id,
                )
                await uow.members.add(KcBundleRevisionDocumentEntity(
                    collection_id=collection.collection_id, bundle_revision_id=revision.bundle_revision_id,
                    document_id=document.document_id, document_version_id=version.document_version_id,
                    document_role=declaration.role, ordinal=declaration.ordinal, required_flag=int(declaration.required_flag),
                    external_document_id=declaration.external_document_id, declared_name=declaration.declared_name,
                    declared_mime_type=declaration.declared_mime_type, source_url=declaration.source_url,
                    member_status="RECEIVED", created_by=command.actor_id, updated_by=command.actor_id,
                ))
                if not command.approval_required:
                    await self._enqueue_parse(
                        uow, collection.collection_id,
                        revision.bundle_revision_id, version,
                        command.actor_id,
                    )
            for failure in command.manifest.document_failures:
                document = await self._document(uow, bundle, failure.external_document_id, command.actor_id)
                await uow.members.add(KcBundleRevisionDocumentEntity(
                    collection_id=collection.collection_id, bundle_revision_id=revision.bundle_revision_id,
                    document_id=document.document_id, document_role="ATTACHMENT", ordinal=failure.ordinal,
                    required_flag=0, external_document_id=failure.external_document_id,
                    declared_name=failure.declared_name, source_url=failure.source_url,
                    member_status="SOURCE_UNAVAILABLE", failure_stage="SOURCE_DOWNLOAD",
                    failure_code=failure.failure_code, failure_message=failure.failure_message,
                    created_by=command.actor_id, updated_by=command.actor_id,
                ))
            if receipt is None:
                receipt = KcIngestionReceiptEntity(
                    collection_id=collection.collection_id, actor_id=command.actor_id, idempotency_key=command.idempotency_key,
                    request_fingerprint=fingerprint,
                    receipt_status=(
                        "PENDING_REVIEW"
                        if command.approval_required
                        else "ACCEPTED"
                    ),
                    bundle_id=bundle.bundle_id,
                    bundle_revision_id=revision.bundle_revision_id,
                    expires_at=datetime.now(timezone.utc) + timedelta(seconds=self._receipt_ttl_seconds),
                    created_by=command.actor_id, updated_by=command.actor_id,
                )
                await uow.receipts.add(receipt)
            else:
                receipt.receipt_status = (
                    "PENDING_REVIEW"
                    if command.approval_required
                    else "ACCEPTED"
                )
                receipt.bundle_id = bundle.bundle_id
                receipt.bundle_revision_id = revision.bundle_revision_id
                receipt.updated_by = command.actor_id
                if uow.session is None:
                    raise RuntimeError("Knowledge Core Unit of Work session is not initialized")
                await uow.session.flush()
            await uow.commit()
            return IntakeAcceptance(
                bundle.bundle_id,
                revision.bundle_revision_id,
                revision.source_revision,
                (
                    "PENDING_REVIEW"
                    if command.approval_required
                    else "ACCEPTED"
                ),
            )

    async def list_pending_reviews(
        self, *, domain_id: int, collection_key: str,
    ) -> list[IntakeReview]:
        """列出指定 Collection 中等待人工审批的用户上传 Revision。"""
        async with self._uow_factory() as uow:
            if not all((uow.collections, uow.revisions, uow.members)):
                raise RuntimeError(
                    "Knowledge Core Unit of Work is not initialized"
                )
            collection = await uow.collections.get_by_scope_key(
                app_id=self._app_id,
                domain_id=domain_id,
                collection_key=collection_key,
            )
            if collection is None:
                raise IntakeReviewNotFoundError("Collection not found")
            revisions = await uow.revisions.list_by_approval(
                collection_id=collection.collection_id,
                approval_status="PENDING",
            )
            result = []
            for revision in revisions:
                members = await uow.members.list_by_revision(
                    bundle_revision_id=revision.bundle_revision_id,
                )
                result.append(self._review_snapshot(revision, members))
            return result

    async def review(self, command: ReviewIntakeCommand) -> IntakeReview:
        """审批用户上传，并在批准事务中原子创建全部解析任务。"""
        decision = command.decision.strip().upper()
        if decision not in {"APPROVE", "REJECT"}:
            raise ValueError("decision must be APPROVE or REJECT")
        now = datetime.now(timezone.utc)
        target_status = "APPROVED" if decision == "APPROVE" else "REJECTED"
        async with self._uow_factory() as uow:
            if not all((
                uow.collections, uow.revisions, uow.members,
                uow.versions, uow.receipts, uow.session,
            )):
                raise RuntimeError(
                    "Knowledge Core Unit of Work is not initialized"
                )
            revision = await uow.revisions.get_by_id(
                bundle_revision_id=command.bundle_revision_id,
                lock=True,
            )
            if revision is None:
                raise IntakeReviewNotFoundError("Revision not found")
            collection = await uow.collections.get_by_id_scope(
                app_id=self._app_id,
                domain_id=command.domain_id,
                collection_id=revision.collection_id,
            )
            if (
                collection is None
                or collection.collection_key != command.collection_key
            ):
                raise IntakeReviewNotFoundError(
                    "Revision is outside the requested Collection"
                )
            if revision.approval_status != "PENDING":
                if revision.approval_status != target_status:
                    raise IntakeReviewConflictError(
                        "Revision already has a different review decision"
                    )
                members = await uow.members.list_by_revision(
                    bundle_revision_id=revision.bundle_revision_id,
                )
                return self._review_snapshot(revision, members)
            if revision.status != "PENDING_REVIEW":
                raise IntakeReviewConflictError(
                    "Revision is not waiting for review"
                )

            members = await uow.members.list_by_revision(
                bundle_revision_id=revision.bundle_revision_id,
            )
            if decision == "APPROVE":
                for member in members:
                    if (
                        member.member_status != "RECEIVED"
                        or member.document_version_id is None
                    ):
                        continue
                    version = await uow.versions.get_by_id(
                        document_version_id=member.document_version_id
                    )
                    if version is None:
                        raise IntakeReviewConflictError(
                            "Revision member version no longer exists"
                        )
                    await self._enqueue_parse(
                        uow,
                        revision.collection_id,
                        revision.bundle_revision_id,
                        version,
                        command.actor_id,
                    )
                revision.status = "PROCESSING"
            else:
                revision.status = "REJECTED"
                revision.completed_at = now

            revision.approval_status = target_status
            revision.reviewed_by = command.actor_id
            revision.reviewed_at = now
            revision.review_comment = (
                command.comment.strip() if command.comment else None
            )
            revision.updated_by = command.actor_id
            receipts = await uow.receipts.list_by_revision(
                bundle_revision_id=revision.bundle_revision_id,
                lock=True,
            )
            for receipt in receipts:
                receipt.receipt_status = (
                    "ACCEPTED" if decision == "APPROVE" else "REJECTED"
                )
                receipt.updated_by = command.actor_id
            await uow.session.flush()
            snapshot = self._review_snapshot(revision, members)
            await uow.commit()
            return snapshot

    @staticmethod
    def _review_snapshot(revision, members) -> IntakeReview:
        return IntakeReview(
            bundle_id=revision.bundle_id,
            bundle_revision_id=revision.bundle_revision_id,
            source_revision=revision.source_revision,
            title=revision.title,
            approval_status=revision.approval_status,
            revision_status=revision.status,
            document_names=tuple(
                member.declared_name or member.external_document_id
                for member in members
                if member.document_role != "MANIFEST"
            ),
            reviewed_by=revision.reviewed_by,
            reviewed_at=revision.reviewed_at,
            review_comment=revision.review_comment,
        )

    @staticmethod
    async def _document(uow, bundle, external_document_id: str, actor_id: str):
        document = await uow.documents.get_by_external_id(bundle_id=bundle.bundle_id, external_document_id=external_document_id)
        if document is None:
            document = await uow.documents.add(KcDocumentEntity(
                collection_id=bundle.collection_id, bundle_id=bundle.bundle_id, external_document_id=external_document_id,
                document_status="ACTIVE", created_by=actor_id, updated_by=actor_id,
            ))
        return document

    @staticmethod
    async def _version(uow, collection_id, bundle_id, document, declaration, published, security_level, actor_id):
        return await KnowledgeCoreIntakeService._version_from_values(
            uow, collection_id, bundle_id, document, content_hash=declaration.content_sha256,
            byte_size=declaration.byte_size, storage_uri=published.storage_uri,
            detected_mime_type=published.detected_mime_type, security_level=security_level, actor_id=actor_id,
        )

    @staticmethod
    async def _version_from_values(uow, collection_id, bundle_id, document, *, content_hash, byte_size, storage_uri, detected_mime_type, security_level, actor_id):
        version = await uow.versions.get_by_content_hash(document_id=document.document_id, content_hash=content_hash.lower())
        if version is None:
            version = await uow.versions.add(KcDocumentVersionEntity(
                collection_id=collection_id, bundle_id=bundle_id, document_id=document.document_id,
                version_no=await uow.versions.next_version_no(document_id=document.document_id),
                content_hash=content_hash.lower(), storage_uri=storage_uri,
                storage_state="AVAILABLE", byte_size=byte_size,
                detected_mime_type=detected_mime_type, security_level=security_level,
                content_metadata_json={}, created_by=actor_id, updated_by=actor_id,
            ))
        return version

    async def _enqueue_parse(self, uow, collection_id, bundle_revision_id, version, actor_id):
        collection = await uow.collections.get_by_id(
            collection_id=collection_id
        )
        if collection is None:
            raise IntakeCollectionError("Target Collection no longer exists")
        policy = {
            "pipeline": "kc-docling-structure/v1",
            "atom_ir_schema": "kc-atom/v1",
            "structure_ir_schema": "kc-structure/v1",
            "evidence_manifest_schema": "kc-evidence-manifest/v1",
            "quality_gate": "kc-structure-quality/v1",
            "do_ocr": True,
            "ocr_engine": "tesseract",
            "image_scale": 2.0,
            "parser_llm_model_id": str(
                collection.parser_llm_model_id
            ),
            "parser_vlm_model_id": (
                str(collection.parser_vlm_model_id)
                if collection.parser_vlm_model_id
                else None
            ),
            "retrieval_llm_model_id": str(
                collection.retrieval_llm_model_id
            ),
            "visual_embedding_model_id": (
                str(getattr(collection, "visual_embedding_model_id", None))
                if getattr(collection, "visual_embedding_model_id", None)
                else None
            ),
        }
        policy.update(self._parse_policy_overrides)
        if policy.get("ocr_model"):
            policy["do_ocr"] = False
            policy["ocr_provider"] = "DEEPSEEK_OCR"
        strategy = str(policy.get("parse_strategy", "AUTO")).upper()
        supports_page_visual = version.detected_mime_type in {
            "application/pdf",
            "image/png",
            "image/jpeg",
            "image/tiff",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        if (
            supports_page_visual
            and policy.get("parser_vlm_model_id")
            and strategy != "TEXT"
        ):
            view_kind = "VISUAL" if strategy == "VISUAL" else "HYBRID"
            parser_name = "kc-adaptive-visual-pipeline"
        elif policy.get("ocr_model"):
            view_kind = "TEXT"
            parser_name = "kc-deepseek-ocr-pipeline"
        else:
            view_kind = "TEXT"
            parser_name = "kc-docling-pipeline"
        fingerprint = canonical_json_hash(policy)
        view = await uow.parse_views.get_by_input(document_version_id=version.document_version_id, view_kind=view_kind, parse_config_fingerprint=fingerprint)
        if view is None:
            view = await uow.parse_views.add(KcParseViewEntity(
                collection_id=collection_id, document_version_id=version.document_version_id,
                view_kind=view_kind, parser_name=parser_name, parse_config_fingerprint=fingerprint,
                parse_config_json=policy, view_status="PENDING",
                created_by=actor_id, updated_by=actor_id,
            ))
        await uow.jobs.add(KcIngestionJobEntity(
            collection_id=collection_id, bundle_revision_id=bundle_revision_id,
            document_version_id=version.document_version_id, parse_view_id=view.parse_view_id, job_type="PARSE",
            idempotency_key=f"parse:{version.document_version_id}:{fingerprint}", input_fingerprint=version.content_hash,
            payload_json={"document_version_id": version.document_version_id}, job_status="PENDING",
            created_by=actor_id, updated_by=actor_id,
        ))
