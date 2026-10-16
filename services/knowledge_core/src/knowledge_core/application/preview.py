"""按 Domain 校验 Bundle Revision 与源文件预览范围。"""

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class KnowledgePreviewNotFoundError(LookupError):
    """预览对象不存在或不属于受信 Domain 范围。"""


@dataclass(frozen=True, slots=True)
class BundleFilePreview:
    document_version_id: UUID | None
    external_document_id: str
    declared_name: str | None
    document_role: str
    ordinal: int
    member_status: str
    declared_mime_type: str | None
    detected_mime_type: str | None
    byte_size: int | None
    preview_available: bool


@dataclass(frozen=True, slots=True)
class BundleRevisionPreview:
    bundle_id: UUID
    bundle_revision_id: UUID
    collection_id: UUID
    title: str
    revision_no: int
    status: str
    approval_status: str
    is_current_revision: bool
    files: list[BundleFilePreview]


@dataclass(frozen=True, slots=True)
class SourceFilePreview:
    storage_uri: str
    file_name: str
    mime_type: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class ReviewTextExcerpt:
    document_version_id: UUID
    file_name: str
    content: str
    locator: dict


@dataclass(frozen=True, slots=True)
class ReviewTextProjection:
    bundle_revision_id: UUID
    excerpts: list[ReviewTextExcerpt]
    warnings: list[str]


class KnowledgeCorePreviewService:
    """仅通过 Repository 读取不可变入库事实，不向 Router 暴露对象 URI。"""

    def __init__(
        self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork],
        object_store=None, converter=None,
        max_chars_per_file: int = 24_000, max_total_chars: int = 96_000,
    ):
        self._uow_factory = uow_factory
        self._object_store = object_store
        self._converter = converter
        self._max_chars_per_file = max_chars_per_file
        self._max_total_chars = max_total_chars

    async def get_bundle_revision(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
    ) -> BundleRevisionPreview:
        async with self._uow_factory() as uow:
            self._require_repositories(uow)
            bundle, revision = await self._scoped_revision(
                uow,
                domain_id=domain_id,
                collection_id=collection_id,
                bundle_id=bundle_id,
                bundle_revision_id=bundle_revision_id,
            )
            members = await uow.members.list_by_revision(
                bundle_revision_id=bundle_revision_id
            )
            files: list[BundleFilePreview] = []
            for member in sorted(members, key=lambda item: int(item.ordinal)):
                if member.collection_id != collection_id:
                    continue
                version = (
                    await uow.versions.get_by_id(
                        document_version_id=member.document_version_id
                    )
                    if member.document_version_id is not None
                    else None
                )
                version_is_scoped = bool(
                    version is not None
                    and version.collection_id == bundle.collection_id
                    and version.bundle_id == bundle_id
                    and version.document_id == member.document_id
                    and version.storage_state == "AVAILABLE"
                )
                files.append(
                    BundleFilePreview(
                        document_version_id=(
                            member.document_version_id
                            if version_is_scoped
                            else None
                        ),
                        external_document_id=member.external_document_id,
                        declared_name=member.declared_name,
                        document_role=member.document_role,
                        ordinal=int(member.ordinal),
                        member_status=member.member_status,
                        declared_mime_type=member.declared_mime_type,
                        detected_mime_type=(
                            version.detected_mime_type
                            if version_is_scoped
                            else None
                        ),
                        byte_size=(
                            int(version.byte_size)
                            if version_is_scoped
                            else None
                        ),
                        preview_available=version_is_scoped,
                    )
                )
            return BundleRevisionPreview(
                bundle_id=bundle.bundle_id,
                bundle_revision_id=revision.bundle_revision_id,
                collection_id=bundle.collection_id,
                title=revision.title,
                revision_no=int(revision.revision_no),
                status=revision.status,
                approval_status=revision.approval_status,
                is_current_revision=(
                    bundle.current_revision_id == revision.bundle_revision_id
                ),
                files=files,
            )

    async def get_source_file(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        document_version_id: UUID,
    ) -> SourceFilePreview:
        async with self._uow_factory() as uow:
            self._require_repositories(uow)
            bundle, _ = await self._scoped_revision(
                uow,
                domain_id=domain_id,
                collection_id=collection_id,
                bundle_id=bundle_id,
                bundle_revision_id=bundle_revision_id,
            )
            member = await uow.members.get_by_version(
                bundle_revision_id=bundle_revision_id,
                document_version_id=document_version_id,
            )
            version = await uow.versions.get_by_id(
                document_version_id=document_version_id
            )
            if (
                member is None
                or version is None
                or member.collection_id != collection_id
                or version.collection_id != bundle.collection_id
                or version.bundle_id != bundle_id
                or version.document_id != member.document_id
                or version.storage_state != "AVAILABLE"
            ):
                raise KnowledgePreviewNotFoundError("源文件不存在")
            return SourceFilePreview(
                storage_uri=version.storage_uri,
                file_name=(
                    member.declared_name
                    or member.external_document_id
                    or "document"
                ),
                mime_type=(
                    version.detected_mime_type
                    or member.declared_mime_type
                    or "application/octet-stream"
                ),
                byte_size=int(version.byte_size),
            )

    async def extract_review_text(
        self, *, domain_id: int, collection_id: UUID, bundle_id: UUID,
        bundle_revision_id: UUID,
    ) -> ReviewTextProjection:
        """对待审批源文件生成有界文本，不创建 Parse View 或检索索引。"""
        if self._object_store is None or self._converter is None:
            raise RuntimeError("审批预分析组件未初始化")
        async with self._uow_factory() as uow:
            self._require_repositories(uow)
            bundle, revision = await self._scoped_revision(
                uow, domain_id=domain_id, collection_id=collection_id,
                bundle_id=bundle_id, bundle_revision_id=bundle_revision_id,
            )
            if revision.approval_status != "PENDING":
                raise ValueError("只有待审批 Revision 可以生成审批预分析文本")
            members = await uow.members.list_by_revision(bundle_revision_id=bundle_revision_id)
            sources: list[tuple[UUID, str, str]] = []
            for member in sorted(members, key=lambda item: int(item.ordinal)):
                if member.document_role == "MANIFEST" or member.document_version_id is None:
                    continue
                version = await uow.versions.get_by_id(document_version_id=member.document_version_id)
                if (
                    version is None
                    or version.collection_id != collection_id
                    or version.bundle_id != bundle.bundle_id
                    or version.storage_state != "AVAILABLE"
                ):
                    continue
                sources.append((
                    version.document_version_id,
                    member.declared_name or member.external_document_id or "document",
                    version.storage_uri,
                ))
        excerpts: list[ReviewTextExcerpt] = []
        warnings: list[str] = []
        remaining = self._max_total_chars
        with tempfile.TemporaryDirectory(prefix="kbot-kc-review-") as directory:
            root = Path(directory)
            for ordinal, (document_version_id, file_name, storage_uri) in enumerate(sources, start=1):
                if remaining <= 0:
                    warnings.append("审批预分析文本达到总量上限，后续文件未继续抽取")
                    break
                source_path = root / f"{ordinal:03d}{Path(file_name).suffix or '.bin'}"
                try:
                    with source_path.open("wb") as writer:
                        async for chunk in self._object_store.stream(storage_uri):
                            writer.write(chunk)
                    document = await self._converter.convert(
                        source_path=source_path, do_ocr=True, ocr_engine="tesseract",
                        generate_page_images=False, generate_picture_images=False,
                        generate_table_images=False,
                    )
                    content = document.export_to_markdown().strip()
                    if not content:
                        warnings.append(f"{file_name} 未抽取到可用于标签识别的文字")
                        continue
                    bounded = content[:min(self._max_chars_per_file, remaining)]
                    remaining -= len(bounded)
                    excerpts.append(ReviewTextExcerpt(
                        document_version_id=document_version_id,
                        file_name=file_name,
                        content=bounded,
                        locator={"ordinal": ordinal - 1, "projection": "DOCLING_MARKDOWN", "truncated": len(bounded) < len(content)},
                    ))
                except Exception as exc:
                    warnings.append(f"{file_name} 审批预分析失败：{type(exc).__name__}")
        return ReviewTextProjection(bundle_revision_id=bundle_revision_id, excerpts=excerpts, warnings=warnings)

    @staticmethod
    def _require_repositories(uow: KnowledgeCoreUnitOfWork) -> None:
        if not all(
            (
                uow.collections,
                uow.bundles,
                uow.revisions,
                uow.members,
                uow.versions,
            )
        ):
            raise RuntimeError("Knowledge Core Unit of Work 未初始化")

    @staticmethod
    async def _scoped_revision(
        uow: KnowledgeCoreUnitOfWork,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
    ):
        collection = await uow.collections.get_by_id_scope(
            domain_id=domain_id,
            collection_id=collection_id,
        )
        bundle = await uow.bundles.get_by_id(bundle_id=bundle_id)
        if (
            collection is None
            or bundle is None
            or bundle.collection_id != collection_id
        ):
            raise KnowledgePreviewNotFoundError("Bundle 不存在")
        revision = await uow.revisions.get_by_id(
            bundle_revision_id=bundle_revision_id
        )
        if (
            revision is None
            or revision.bundle_id != bundle_id
            or revision.collection_id != collection_id
        ):
            raise KnowledgePreviewNotFoundError("Bundle Revision 不存在")
        return bundle, revision
