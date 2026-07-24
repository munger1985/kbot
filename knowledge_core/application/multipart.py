"""Five-phase multipart orchestration; HTTP and storage remain outside repositories."""
import tempfile
from dataclasses import dataclass
from pathlib import Path

from knowledge_core.application.intake import (
    AcceptKmAssetCommand, IntakeAcceptance, KnowledgeCoreIntakeService, PreparePublishCommand,
    PublishedAttachment, PublishedManifest, ReserveIntakeCommand,
)
from knowledge_core.domain.intake import KmAssetIntakeManifest
from knowledge_core.domain.manifest import render_bundle_manifest
from knowledge_core.ports.object_store import KnowledgeObjectStore, StoredObject


class IntakeInProgressError(RuntimeError):
    """The same idempotent request is already staging or publishing."""


@dataclass(frozen=True)
class MultipartIntakeCommand:
    domain_id: int
    collection_key: str
    actor_id: str
    idempotency_key: str
    manifest: KmAssetIntakeManifest
    file_paths: dict[str, Path]
    source_system: str = "metadb"
    source_type: str = "KM_ASSET"
    generate_manifest: bool = True
    allowed_document_roles: tuple[str, ...] = ("ATTACHMENT",)
    approval_required: bool = False


class KnowledgeCoreMultipartOrchestrator:
    def __init__(self, *, intake_service: KnowledgeCoreIntakeService, object_store: KnowledgeObjectStore):
        self._intake = intake_service
        self._objects = object_store

    async def accept(self, command: MultipartIntakeCommand) -> IntakeAcceptance:
        command.manifest.validate_declarations(
            set(command.file_paths), allowed_roles=set(command.allowed_document_roles),
        )
        reservation = await self._intake.reserve(ReserveIntakeCommand(
            command.domain_id, command.collection_key, command.actor_id,
            command.idempotency_key, command.manifest, command.source_system,
            command.source_type, command.allowed_document_roles,
        ))
        if not reservation.newly_created:
            if reservation.bundle_id is not None and reservation.bundle_revision_id is not None:
                return IntakeAcceptance(
                    reservation.bundle_id,
                    reservation.bundle_revision_id,
                    command.manifest.bundle.source_revision,
                    reservation.receipt_status,
                )
            raise IntakeInProgressError("INGESTION_IN_PROGRESS")

        staged: list[StoredObject] = []
        published: list[StoredObject] = []
        manifest_temp: Path | None = None
        try:
            receipt_key = str(reservation.receipt_id)
            staged_by_part: dict[str, StoredObject] = {}
            for declaration in command.manifest.documents:
                staged_by_part[declaration.part_name] = await self._objects.stage_file(
                    receipt_id=receipt_key, part_name=declaration.part_name,
                    source_path=command.file_paths[declaration.part_name],
                    expected_sha256=declaration.content_sha256, expected_size=declaration.byte_size,
                    detected_mime_type=declaration.declared_mime_type,
                )
                staged.append(staged_by_part[declaration.part_name])
            staged_manifest = None
            rendered_manifest = None
            if command.generate_manifest:
                rendered_manifest = render_bundle_manifest(command.manifest.bundle)
                with tempfile.NamedTemporaryFile(prefix="kc_manifest_", suffix=".md", delete=False) as stream:
                    stream.write(rendered_manifest.content)
                    manifest_temp = Path(stream.name)
                staged_manifest = await self._objects.stage_file(
                    receipt_id=receipt_key, part_name="__manifest__", source_path=manifest_temp,
                    expected_sha256=rendered_manifest.content_sha256, expected_size=len(rendered_manifest.content),
                    detected_mime_type=rendered_manifest.mime_type,
                )
                staged.append(staged_manifest)
            preparation = await self._intake.prepare_publish(PreparePublishCommand(
                command.domain_id, command.collection_key, command.actor_id, command.idempotency_key,
                command.manifest, reservation.receipt_id, command.source_system,
                command.source_type, command.allowed_document_roles,
            ))
            published_by_part: dict[str, StoredObject] = {}
            for declaration in command.manifest.documents:
                published_item = await self._objects.publish_staged(
                    staged=staged_by_part[declaration.part_name],
                    collection_id=preparation.collection_id,
                    document_id=preparation.document_ids[declaration.external_document_id],
                )
                published.append(published_item)
                published_by_part[declaration.part_name] = published_item
            published_manifest = None
            if command.generate_manifest and staged_manifest is not None:
                published_manifest = await self._objects.publish_staged(
                    staged=staged_manifest,
                    collection_id=preparation.collection_id,
                    document_id=preparation.document_ids["__manifest__"],
                )
                published.append(published_manifest)
            return await self._intake.accept_published(AcceptKmAssetCommand(
                command.domain_id, command.collection_key, command.actor_id, command.idempotency_key,
                command.manifest,
                {item.part_name: PublishedAttachment(item.external_document_id, published_by_part[item.part_name].uri, published_by_part[item.part_name].detected_mime_type) for item in command.manifest.documents},
                PublishedManifest(published_manifest.uri, len(rendered_manifest.content), rendered_manifest.content_sha256) if published_manifest is not None and rendered_manifest is not None else None,
                command.source_system, command.source_type, command.generate_manifest,
                command.allowed_document_roles, command.approval_required,
            ))
        except Exception:
            for item in [*published, *staged]:
                try:
                    await self._objects.delete(item.uri)
                except Exception:
                    pass
            raise
        finally:
            if manifest_temp is not None:
                manifest_temp.unlink(missing_ok=True)
