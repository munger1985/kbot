"""资料组织和业务治理读写；SQL 仅留在 App 仓储边界。"""

from uuid import UUID
from sqlalchemy import select
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import fingerprint
from .knowledge_base import BaseRepository
from ..entities.knowledge_library import KnowledgeLibraryEntity
from ..entities.knowledge_document import KnowledgeDocumentEntity as Document
from ..entities.knowledge_document_version import KnowledgeDocumentVersionEntity as Version
from ..entities.knowledge_ingestion_job import KnowledgeIngestionJobEntity as Job
from ..entities.knowledge_index_generation import KnowledgeIndexGenerationEntity as Generation


OPERATIONS = frozenset({"READ", "UPLOAD", "REVIEW", "MANAGE", "PUBLISH", "WITHDRAW"})


def library_view(row, subject):
    permissions = tuple(row.grants_json.get(subject, ()))
    return {"library_id": str(row.library_id), "display_name": row.display_name,
        "description": row.description, "status": row.status, "row_version": row.row_version,
        "permissions": permissions, "configuration": dict(row.configuration_json),
        "grants": dict(row.grants_json) if "MANAGE" in permissions else None}


class KnowledgeLibraryRepository(BaseRepository):
    entity, primary_key = KnowledgeLibraryEntity, "library_id"

    async def authorized(self, library_id, operation, *, lock=False):
        row = await self.get(UUID(str(library_id)), lock=lock)
        require(row is not None and operation in row.grants_json.get(self.scope.subject_ref, ()),
                "资料库不存在或当前主体无权执行该操作")
        require(row.status == "ACTIVE" or operation == "MANAGE", "资料库已停用")
        return row

    async def detail(self, library_id):
        return library_view(await self.authorized(library_id, "READ"), self.scope.subject_ref)

    async def upload_receipt(self, library_id, idempotency_key):
        row = (await self._session.execute(self._documents(library_id).add_columns(Version)
            .join(Version, Version.document_id == Document.document_id).where(
                Version.business_version_ref == idempotency_key))).one_or_none()
        if row is None:
            return None
        document, version = row
        job_id = await self._session.scalar(select(Job.ingestion_job_id).where(
            Job.document_version_id == version.document_version_id))
        return {"document_id": str(document.document_id), "version_id": str(version.document_version_id),
            "job_id": str(job_id), "name": document.title, "sha256": version.source_sha256,
            "mime_type": version.mime_type, "status": version.status}

    async def listing(self):
        rows = (await self._session.scalars(self.statement().order_by(
            self.entity.created_at.desc(), self.entity.library_id.desc()).limit(501))).all()
        require(len(rows) <= 500, "资料库目录超过当前分页预算")
        return [library_view(row, self.scope.subject_ref) for row in rows
                if "READ" in row.grants_json.get(self.scope.subject_ref, ())]

    async def create(self, *, library_id, name, description, configuration):
        row = await self.add(dict(library_id=library_id, display_name=name, description=description,
            status="ACTIVE", grants_json={self.scope.subject_ref: sorted(OPERATIONS)},
            configuration_json=configuration, created_by=self.scope.subject_ref))
        return library_view(row, self.scope.subject_ref)

    async def change(self, library_id, *, expected_revision, changes):
        row = await self.authorized(library_id, "MANAGE", lock=True)
        require(int(row.row_version) == expected_revision, "资料库已被其他请求修改")
        for field, value in changes.items():
            require(field in {"display_name", "description", "status", "grants", "configuration"},
                    "资料库更新字段无效")
            setattr(row, {"grants": "grants_json", "configuration": "configuration_json"}.get(field, field), value)
        require(row.status in {"ACTIVE", "DISABLED"}, "资料库状态无效")
        require(row.display_name and row.display_name.strip(), "资料库名称不能为空")
        require(isinstance(row.grants_json, dict) and row.grants_json and all(
                isinstance(subject, str) and 0 < len(subject) <= 256
                and isinstance(permissions, list) and permissions and "READ" in permissions
                and len(permissions) == len(set(permissions)) and set(permissions) <= OPERATIONS
                for subject, permissions in row.grants_json.items()), "资料库成员授权无效")
        require(any("MANAGE" in permissions for permissions in row.grants_json.values()),
                "资料库必须保留至少一个管理主体")
        await self._session.flush()
        return library_view(row, self.scope.subject_ref)

    def _documents(self, library_id):
        statement = select(Document).where(Document.domain_id == self.scope.domain_id,
            Document.business_asset_ref.startswith("library:" + str(library_id) + ":"))
        if hasattr(Document, "tenant_id"):
            statement = statement.where(Document.tenant_id == self.scope.tenant_id)
        return statement

    async def files(self, library_id, *, page=1, page_size=20, operation="READ"):
        library = await self.authorized(library_id, operation)
        require(page >= 1 and 1 <= page_size <= 100, "文档分页预算无效")
        permissions = set(library.grants_json[self.scope.subject_ref])
        rows = (await self._session.execute(self._documents(library_id).add_columns(Version)
            .join(Version, Version.document_id == Document.document_id).order_by(
                Document.created_at.desc(), Document.document_id.desc(), Version.created_at.desc())
            .limit(501))).all()
        require(len(rows) <= 500, "资料库目录超过当前版本预算")
        documents = {}
        for row, version in rows:
            governance = row.metadata_json.get("versions", {}).get(str(version.document_version_id), {})
            if (version.status != "PUBLISHED" and not permissions & {"REVIEW", "MANAGE"}
                    and governance.get("uploaded_by") != self.scope.subject_ref):
                continue
            item = documents.setdefault(str(row.document_id), {
                "document_id": str(row.document_id), "title": row.title, "status": row.status,
                "row_version": row.row_version, "versions": {}})
            item["versions"][str(version.document_version_id)] = dict(governance)
        items = list(documents.values())[(page - 1) * page_size:page * page_size]
        for item in items:
            for version_ref, governance in item["versions"].items():
                detail = await self.version(library_id, UUID(version_ref), operation=operation)
                governance.update(status=detail["status"], job_id=str(detail["job_id"]),
                    processing_status=detail["processing_status"], error_code=detail["error_code"],
                    generation_ready=detail["generation_id"] is not None)
        return {"items": items, "page": page, "has_more": len(documents) > page * page_size}

    async def govern(self, library_id, document_id, version_id, *, operation, changes):
        await self.authorized(library_id, operation, lock=True)
        row = (await self._session.scalars(self._documents(library_id).where(
            Document.document_id == document_id).with_for_update())).one_or_none()
        require(row is not None, "文档不属于当前资料库")
        version = await self._session.scalar(select(Version).where(
            Version.document_version_id == version_id, Version.document_id == row.document_id))
        require(version is not None, "文档版本不属于当前文档")
        metadata = dict(row.metadata_json)
        versions = dict(metadata.get("versions", {}))
        current = dict(versions.get(str(version_id), {}))
        current.update(changes)
        versions[str(version_id)] = current
        row.metadata_json = {**metadata, "versions": versions}
        await self._session.flush()
        return current

    async def version(self, library_id, version_id, *, operation):
        library = await self.authorized(library_id, operation)
        row = (await self._session.execute(self._documents(library_id).add_columns(Version)
            .join(Version, Version.document_id == Document.document_id)
            .where(Version.document_version_id == version_id))).one_or_none()
        require(row is not None, "版本不存在于当前资料库")
        document, version = row
        jobs = (await self._session.scalars(select(Job).where(
            Job.document_version_id == version.document_version_id))).all()
        require(len(jobs) == 1, "版本解析任务身份不唯一")
        generations = (await self._session.scalars(select(Generation).where(
            Generation.document_version_id == version_id, Generation.status == "READY").order_by(
                Generation.created_at.desc(), Generation.index_generation_id.desc()).limit(2))).all()
        governance = dict(document.metadata_json.get("versions", {}).get(str(version_id), {}))
        if operation == "READ":
            require(version.status == "PUBLISHED" or governance.get("uploaded_by") == self.scope.subject_ref
                or bool(set(library.grants_json[self.scope.subject_ref]) & {"REVIEW", "MANAGE"}),
                "当前主体无权读取未发布版本")
        return {"document_id": document.document_id, "version_id": version.document_version_id,
            "job_id": jobs[0].ingestion_job_id, "subject_ref": jobs[0].context_json["subject_ref"],
            "source_handle": version.source_object_ref, "mime_type": version.mime_type,
            "byte_count": int(version.byte_count), "title": document.title,
            "status": version.status, "revision": int(version.row_version),
            "processing_status": jobs[0].context_json.get("pipeline_status", jobs[0].status),
            "error_code": jobs[0].context_json.get("pipeline_error", jobs[0].error_code),
            "governance": governance, "generation_id": generations[0].index_generation_id if generations else None}

    async def grant(self, library_id, *, include_unpublished=False):
        library = await self.authorized(library_id, "READ")
        rows = (await self._session.execute(self._documents(library_id).add_columns(Version)
            .join(Version, Version.document_id == Document.document_id).limit(501))).all()
        require(len(rows) <= 500, "资料库版本范围超过当前固定预算")
        versions = []
        for document, version in rows:
            state = document.metadata_json.get("versions", {}).get(str(version.document_version_id), {})
            if state.get("review") == "APPROVED" and version.status == "PUBLISHED":
                versions.append(version.document_version_id)
            elif include_unpublished and state.get("review") == "APPROVED":
                versions.append(version.document_version_id)
        permissions = set(library.grants_json.get(self.scope.subject_ref, ()))
        return fingerprint({"library": str(library.library_id), "revision": int(library.row_version),
                            "subject": self.scope.subject_ref, "permissions": sorted(permissions)}), tuple(versions), permissions
