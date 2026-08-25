"""按 Domain 校验 Bundle Revision 与源文件预览范围。"""

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from knowledge_core.persistence import KnowledgeCoreUnitOfWork


_KM_ASSET_METADATA_SCHEMA = "km_asset/v1"
_KM_ASSET_FIELD_ALIASES = {
    "asset_id": ("asset_id", "external_asset_id"),
    "asset_title": ("asset_title", "title"),
    "author_email": ("author_email", "author_mail", "author"),
    "briefing": (
        "briefing",
        "solution_briefing",
        "description",
        "asset_details",
    ),
    "publish_time": (
        "publish_time",
        "create_time",
        "publish_date",
        "asset_date",
    ),
    "last_update_time": (
        "last_update_time",
        "update_time",
        "updated_at",
    ),
    "product": ("asset_product", "product"),
    "solution": ("asset_solution", "solution"),
    "industry": ("industry_id", "industry"),
    "content_category": ("content_category", "category"),
}


def _clean_asset_field(value: object) -> str:
    """只允许可直接展示的单值进入 Asset 预览。"""
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _asset_fields(manifest_json: object, *, title: str) -> dict[str, str]:
    """从不可变 Bundle Manifest 投影 KM Asset 展示白名单。"""
    if not isinstance(manifest_json, dict):
        return {}
    metadata = manifest_json.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    normalized = {
        str(key).strip().lower(): value for key, value in metadata.items()
    }
    if normalized.get("metadata_schema") != _KM_ASSET_METADATA_SCHEMA:
        return {}

    fields: dict[str, str] = {}
    for target, aliases in _KM_ASSET_FIELD_ALIASES.items():
        value = next(
            (
                cleaned
                for alias in aliases
                if (cleaned := _clean_asset_field(normalized.get(alias)))
            ),
            "",
        )
        if value:
            fields[target] = value

    if "asset_id" not in fields:
        source_id = _clean_asset_field(manifest_json.get("source_id"))
        if source_id:
            fields["asset_id"] = source_id
    if "asset_title" not in fields:
        cleaned_title = _clean_asset_field(title)
        if cleaned_title:
            fields["asset_title"] = cleaned_title
    return fields


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
    asset_fields: dict[str, str]
    files: list[BundleFilePreview]


@dataclass(frozen=True, slots=True)
class SourceFilePreview:
    storage_uri: str
    file_name: str
    mime_type: str
    byte_size: int


class KnowledgeCorePreviewService:
    """仅通过 Repository 读取不可变入库事实，不向 Router 暴露对象 URI。"""

    def __init__(self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._uow_factory = uow_factory

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
                asset_fields=_asset_fields(
                    getattr(revision, "manifest_json", None),
                    title=revision.title,
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
