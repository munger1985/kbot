"""在本地 Oracle 验收 KC S3 预览、模型引用与可恢复清理。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from sqlalchemy import delete, select

from knowledge_core.adapters.local_object_store import (
    LocalKnowledgeObjectStore,
)
from knowledge_core.application.collection_purge import (
    KnowledgeCoreCollectionPurgeService,
)
from knowledge_core.application.notifications import KnowledgeOutboxPublisher
from knowledge_core.application.model_references import (
    KnowledgeCoreModelReferenceService,
)
from knowledge_core.application.preview import KnowledgeCorePreviewService
from knowledge_core.entities import (
    KcBundleEntity,
    KcBundleRevisionDocumentEntity,
    KcBundleRevisionEntity,
    KcCollectionEntity,
    KcDocumentEntity,
    KcDocumentVersionEntity,
    KcIngestionJobEntity,
    KcParseViewEntity,
)
from knowledge_core.persistence import create_kc_uow
from main_api.entities import PlatformDomainEntity
from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7
from platform_core.notifications import NotificationOutboxEntity


async def smoke() -> None:
    runtime = create_database_runtime(get_settings())
    uow_factory = lambda: create_kc_uow(runtime.session_factory)
    domain_id = None
    collection_id = uuid7()
    bundle_id = uuid7()
    revision_id = uuid7()
    document_id = uuid7()
    version_id = uuid7()
    parse_view_id = uuid7()
    purge_job_id = uuid7()
    model_id = uuid7()
    fingerprint = "a" * 64
    with tempfile.TemporaryDirectory(prefix="kbot-kc-s3-") as directory:
        object_root = Path(directory) / "objects"
        source_path = (
            object_root
            / "kc"
            / str(collection_id)
            / str(document_id)
            / "source"
        )
        parser_path = (
            object_root
            / "kc-parser-artifacts"
            / str(purge_job_id)
            / "artifact.json"
        )
        source_path.parent.mkdir(parents=True)
        parser_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"0123456789")
        parser_path.write_bytes(b"{}")
        store = LocalKnowledgeObjectStore(object_root)
        try:
            async with runtime.session_factory() as session:
                domain = PlatformDomainEntity(
                    name=f"kc-s3-smoke-{uuid7()}",
                    status="ACTIVE",
                    created_by="kc-s3-smoke",
                    updated_by="kc-s3-smoke",
                )
                session.add(domain)
                await session.flush()
                domain_id = int(domain.domain_id)
                session.add(
                    KcCollectionEntity(
                        collection_id=collection_id,
                        domain_id=domain_id,
                        display_name="KC S3 Smoke",
                        models_json={"embedding": str(model_id)},
                        status="ACTIVE",
                        default_security_level=1,
                        metadata_json={},
                    )
                )
                await session.flush()
                bundle = KcBundleEntity(
                    bundle_id=bundle_id,
                    collection_id=collection_id,
                    source_system="smoke",
                    source_type="FILE",
                    source_id="source-1",
                    availability_status="READY",
                )
                session.add(bundle)
                await session.flush()
                session.add(
                    KcBundleRevisionEntity(
                        bundle_revision_id=revision_id,
                        collection_id=collection_id,
                        bundle_id=bundle_id,
                        revision_no=1,
                        source_revision="revision-1",
                        snapshot_fingerprint="b" * 64,
                        manifest_json={},
                        title="S3 预览验收",
                        security_level=1,
                        status="READY",
                        approval_status="NOT_REQUIRED",
                    )
                )
                await session.flush()
                bundle.current_revision_id = revision_id
                session.add(
                    KcDocumentEntity(
                        document_id=document_id,
                        collection_id=collection_id,
                        bundle_id=bundle_id,
                        external_document_id="source.txt",
                        document_status="ACTIVE",
                    )
                )
                await session.flush()
                session.add(
                    KcDocumentVersionEntity(
                        document_version_id=version_id,
                        collection_id=collection_id,
                        bundle_id=bundle_id,
                        document_id=document_id,
                        version_no=1,
                        content_hash="c" * 64,
                        storage_uri=str(source_path),
                        storage_state="AVAILABLE",
                        byte_size=10,
                        detected_mime_type="text/plain",
                        security_level=1,
                    )
                )
                await session.flush()
                session.add(
                    KcBundleRevisionDocumentEntity(
                        collection_id=collection_id,
                        bundle_revision_id=revision_id,
                        document_id=document_id,
                        document_version_id=version_id,
                        document_role="CONTENT",
                        ordinal=1,
                        required_flag=1,
                        external_document_id="source.txt",
                        declared_name="验收.txt",
                        declared_mime_type="text/plain",
                        member_status="READY",
                    )
                )
                session.add(
                    KcParseViewEntity(
                        parse_view_id=parse_view_id,
                        collection_id=collection_id,
                        document_version_id=version_id,
                        view_kind="TEXT",
                        parser_name="smoke-parser",
                        parser_version="1.0.0",
                        parse_config_fingerprint="d" * 64,
                        parse_config_json={},
                        view_status="ACTIVE",
                        artifact_manifest_json={
                            "raw_docling": {"uri": str(parser_path)}
                        },
                    )
                )
                await session.commit()

            preview_service = KnowledgeCorePreviewService(
                uow_factory=uow_factory
            )
            preview = await preview_service.get_source_file(
                domain_id=domain_id,
                collection_id=collection_id,
                bundle_id=bundle_id,
                bundle_revision_id=revision_id,
                document_version_id=version_id,
            )
            content = b"".join(
                [
                    chunk
                    async for chunk in store.stream(
                        preview.storage_uri,
                        offset=2,
                        length=4,
                        chunk_size=2,
                    )
                ]
            )
            assert content == b"2345"

            references = await KnowledgeCoreModelReferenceService(
                uow_factory=uow_factory
            ).list(model_id=model_id)
            assert any(
                item["resource_type"] == "collection"
                and item["binding_role"] == "embedding"
                for item in references
            )

            now = datetime.now(timezone.utc)
            async with uow_factory() as uow:
                collection = await uow.collections.get_by_id(
                    collection_id=collection_id, lock=True
                )
                collection.status = "DELETING"
                await uow.jobs.add(
                    KcIngestionJobEntity(
                        ingestion_job_id=purge_job_id,
                        collection_id=collection_id,
                        job_type="COLLECTION_PURGE",
                        idempotency_key=f"collection-purge:{collection_id}",
                        input_fingerprint=fingerprint,
                        payload_json={"collection_id": str(collection_id)},
                        job_status="RUNNING",
                        priority=100,
                        available_at=now,
                        attempt_count=1,
                        max_attempts=3,
                        lease_owner="kc-s3-smoke",
                        lease_until=now + timedelta(minutes=5),
                    )
                )
                await uow.commit()

            result = await KnowledgeCoreCollectionPurgeService(
                uow_factory=uow_factory,
                object_store=store,
                notification_publisher=KnowledgeOutboxPublisher(),
            ).run(
                job_id=purge_job_id,
                worker_id="kc-s3-smoke",
                input_fingerprint=fingerprint,
            )
            assert result["status"] == "SUCCEEDED"
            assert not source_path.exists()
            assert not parser_path.exists()
            async with runtime.session_factory() as session:
                remaining = (
                    await session.execute(
                        select(KcCollectionEntity).where(
                            KcCollectionEntity.collection_id == collection_id
                        )
                    )
                ).scalar_one_or_none()
                assert remaining is None
            print(
                "Knowledge Core S3 Oracle Smoke 通过："
                "范围预览、Range 流、模型引用和两阶段对象清理均正常"
            )
        finally:
            async with uow_factory() as uow:
                await uow.collection_purge.purge_descendants(
                    collection_id=collection_id,
                    purge_job_id=purge_job_id,
                )
                await uow.collection_purge.finalize(
                    collection_id=collection_id,
                    purge_job_id=purge_job_id,
                )
                await uow.commit()
            async with runtime.session_factory() as session:
                if domain_id is not None:
                    await session.execute(
                        delete(NotificationOutboxEntity).where(
                            NotificationOutboxEntity.domain_id == domain_id
                        )
                    )
                    await session.execute(
                        delete(PlatformDomainEntity).where(
                            PlatformDomainEntity.domain_id == domain_id
                        )
                    )
                await session.commit()
    await runtime.engine.dispose()


if __name__ == "__main__":
    asyncio.run(smoke())
