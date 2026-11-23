"""资料组织和业务治理读写；SQL 仅留在 App 仓储边界。"""

from datetime import datetime, timezone
from unicodedata import normalize
from uuid import UUID
from sqlalchemy import func, select, text
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import fingerprint
from platform_core.contracts.knowledge_parsing import LibraryIndexModels, LibraryParseModels
from .knowledge_base import BaseRepository
from ..domain.knowledge_library import (
    INDEX_ROLES, LibraryConflict, active_rebuild, index_batch_progress, index_rebuild_batches,
)
from ..entities.agent import KnowledgeRetrievalAgentVersionEntity as AgentVersion
from ..entities.knowledge_library import KnowledgeLibraryEntity
from ..entities.knowledge_document import KnowledgeDocumentEntity as Document
from ..entities.knowledge_document_version import KnowledgeDocumentVersionEntity as Version
from ..entities.knowledge_ingestion_job import KnowledgeIngestionJobEntity as Job
from ..entities.knowledge_index_generation import KnowledgeIndexGenerationEntity as Generation
from ..entities.knowledge_parse_run import KnowledgeParseRunEntity as ParseRun
from .knowledge_index_generation import KnowledgeIndexGenerationRepository
from .knowledge_outbox import KnowledgeOutboxRepository
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.projections.contracts import ProjectionProfile
from knowledge_algorithms.projections.splitting import REVISION as SPLIT_REVISION
from dataclasses import replace
from platform_core.identity import uuid7


OPERATIONS = frozenset({"READ", "UPLOAD", "REVIEW", "MANAGE", "WITHDRAW"})


def processing_status(job, parse_state):
    pipeline = job.context_json.get("pipeline_status")
    if job.status == "FAILED":
        return "INDEX_FAILED" if parse_state == "PARSED" and job.context_json.get("pipeline_error") else "FAILED"
    if job.status == "CANCELLED":
        return "CANCELLED"
    return pipeline or job.status


def processing_progress(job, stage, *, review, ready, outbox, now):
    """只采用已提交的解析事件、索引队列和索引就绪证据，不按耗时估算。"""
    if review == "REJECTED" or stage == "REJECTED":
        return "REJECTED"
    if stage == "CANCELLED":
        return "CANCELLED"
    if stage == "FAILED":
        return "PARSE_FAILED"
    if stage == "INDEX_FAILED":
        return "INDEX_FAILED"
    if ready:
        if outbox is not None and outbox.status == "RUNNING":
            return "INDEX_CONFIRMING" if outbox.lease_until and outbox.lease_until > now else "INDEX_RETRY"
        if outbox is not None and outbox.status == "PENDING" and outbox.error_code:
            return "INDEX_RETRY"
        return "READY"
    if job.status == "AWAITING_APPROVAL":
        return "AWAITING_APPROVAL"
    if job.status == "PARSED":
        if outbox is not None:
            if outbox.status == "FAILED":
                return "INDEX_FAILED"
            if outbox.status == "RUNNING" and outbox.lease_until and outbox.lease_until > now:
                return "INDEXING"
            if outbox.status == "DELIVERED":
                return "INDEX_CONFIRMING"
            if outbox.error_code or outbox.status == "RUNNING":
                return "INDEX_RETRY"
        return "INDEX_RETRY" if stage == "RETRY" else "INDEX_QUEUED"
    if stage == "RETRY" and job.retry_at and job.retry_at > now:
        return "PARSE_RETRY"
    if job.status == "PARSING":
        if job.lease_until <= now:
            return "PARSE_RETRY"
        # 首个事件确认开始识别；第二个非终态事件是提取后的结构化结果批次。
        return "ORGANIZING" if job.recovery_sequence > 1 else "EXTRACTING" if job.recovery_sequence else "PREPARING"
    return "PARSE_QUEUED"


def library_view(row, subject):
    permissions = tuple(permission for permission in row.grants_json.get(subject, ()) if permission in OPERATIONS)
    return {"library_id": str(row.library_id), "display_name": row.display_name,
        "description": row.description, "status": row.status, "row_version": row.row_version, "updated_at": row.updated_at.isoformat(),
        "permissions": permissions, "configuration": dict(row.configuration_json),
        "grants": {user: [permission for permission in grants if permission in OPERATIONS]
                   for user, grants in row.grants_json.items()} if "MANAGE" in permissions else None}


class KnowledgeLibraryRepository(BaseRepository):
    entity, primary_key = KnowledgeLibraryEntity, "library_id"

    async def _lock_catalog(self):
        # 只有库本身写入使用目录锁；上传、解析、读取仍按其自身行锁/快照执行。
        dialect = self._session.get_bind().dialect.name
        if dialect == "oracle":
            # 固定 App 自有表，无需 DBMS_LOCK 额外授权；所有库写入遵循相同锁顺序。
            await self._session.execute(text("LOCK TABLE KBOT_KR_KNOWLEDGE_LIBRARY IN SHARE ROW EXCLUSIVE MODE"))
        elif dialect != "sqlite":
            raise RuntimeError("未登记知识库名称并发控制方式")

    async def _name(self, value, *, excluding=None):
        value = normalize("NFC", value.strip())
        require(0 < len(value) <= 256, "资料库名称不能为空或超长")
        key = normalize("NFKC", value).casefold()
        statement = self.statement().with_only_columns(self.entity.display_name)
        if excluding is not None:
            statement = statement.where(self.entity.library_id != excluding)
        names = await self._session.scalars(statement)
        if any(normalize("NFKC", name.strip()).casefold() == key for name in names):
            raise LibraryConflict("KNOWLEDGE_LIBRARY_NAME_CONFLICT")
        return value

    async def authorized(self, library_id, operation, *, lock=False):
        row = await self.get(UUID(str(library_id)), lock=lock)
        require(row is not None and operation in OPERATIONS and operation in row.grants_json.get(self.scope.subject_ref, ()),
                "资料库不存在或当前主体无权执行该操作")
        require(row.status == "ACTIVE" or operation == "MANAGE", "资料库已停用")
        return row

    async def detail(self, library_id, *, management=False):
        row = await self.authorized(library_id, "MANAGE" if management else "READ")
        value = library_view(row, self.scope.subject_ref)
        if management:
            value["delete_block"] = await self._delete_block(row.library_id)
        else:
            value["grants"] = None
        return value

    async def published_members(self, library_id):
        rows = (await self._session.execute(self._documents(library_id).add_columns(Version)
            .join(Version, Version.document_id == Document.document_id).where(
                Document.status == "PUBLISHED", Version.status == "PUBLISHED")
            .order_by(Version.document_version_id).limit(501))).all()
        require(len(rows) <= 500, "已发布资料超过当前固定预算")
        return {str(version.document_version_id): (document.title, version.active_generation_id, version.published_parse_run_id)
                for document, version in rows if document.metadata_json.get("versions", {}).get(
                    str(version.document_version_id), {}).get("review") == "APPROVED"}

    @staticmethod
    def _store_rebuild(row, batch):
        config = dict(row.configuration_json)
        bulk, files = index_rebuild_batches(config)
        if batch.get("scope") == "FILE":
            files[next(iter(batch["members"]))] = batch
        else:
            bulk = batch
        if bulk is None:
            config.pop("index_rebuild", None)
        else:
            config["index_rebuild"] = bulk
        config["file_index_rebuilds"] = files
        row.configuration_json = config

    @staticmethod
    def _find_rebuild(configuration, batch_id):
        bulk, files = index_rebuild_batches(configuration)
        return next((batch for batch in (bulk, *files.values()) if batch and batch["id"] == str(batch_id)), None)

    async def rebuild_summary(self, row, members=None, *, version_id=None, batch_id=None):
        members = await self.published_members(row.library_id) if members is None else members
        configuration = row.configuration_json
        bulk, files = index_rebuild_batches(configuration)
        if batch_id is not None:
            batch = self._find_rebuild(configuration, batch_id)
        elif version_id is not None and not active_rebuild(bulk):
            batch = files.get(str(version_id))
        else:
            batch = bulk
        result = {"published_file_count": len(members), "library_revision": int(row.row_version),
                  "pending_roles": configuration.get("index_rebuild_pending", []),
                  "active_file_rebuild_count": sum(active_rebuild(entry) for entry in files.values()),
                  "batch": None}
        if not batch:
            return result
        outboxes = KnowledgeOutboxRepository(self._session, self.scope)
        generations = KnowledgeIndexGenerationRepository(self._session, self.scope)
        entries = []
        for version_id, item in batch["members"].items():
            outbox = await outboxes.get(UUID(item["outbox_id"]))
            candidate = await generations.get(UUID(item["candidate_id"]))
            require(outbox is not None and candidate is not None, "批量索引任务记录缺失")
            entries.append({"version_id": version_id, "title": item["title"],
                "status": "READY" if outbox.status == "DELIVERED" and candidate.status in {"READY", "CANDIDATE"} else outbox.status,
                "error_code": outbox.error_code,
                "visual_not_applicable": "visual_embedding" in batch["roles"] and
                    candidate.manifest_json.get("has_images") is False,
                "index_progress": index_batch_progress(candidate.manifest_json,
                    validated=candidate.status in {"READY", "CANDIDATE"})})
        state = batch.get("terminal_status")
        file_mode = batch.get("scope") == "FILE"
        source_published = batch.get("source_published", True)
        if state == "PUBLISHED" and file_mode and not source_published and row.status == "ACTIVE":
            ref, item = next(iter(batch["members"].items()))
            record = await self._version(row.library_id, UUID(ref))
            if record["status"] == "PARSED" and str(record["generation_id"]) == item["candidate_id"]:
                state = "READY" if batch["target_models"] == configuration.get("index_models", {}) else "DISCARDED"
        if not state:
            if file_mode:
                ref = next(iter(batch["members"]))
                record = (await self._version(row.library_id, UUID(ref))
                          if row.status == "ACTIVE" else None)
                expected_status = "PUBLISHED" if source_published else "PARSED"
                current = ({ref: str(record["generation_id"])} if record and record["status"] == expected_status
                           and record["job_status"] == "PARSED" and record["governance"].get("review") == "APPROVED" else {})
            else:
                current = {key: str(value[1]) for key, value in members.items()}
            if (configuration.get("index_models", {}) != batch["target_models"] or
                current != {key: item["source_id"] for key, item in batch["members"].items()}):
                state = "STALE"
            elif all(entry["status"] == "READY" for entry in entries):
                state = "READY"
            elif any(entry["status"] == "FAILED" for entry in entries):
                state = "FAILED"
            else:
                state = "RUNNING"
        result["batch"] = {"batch_id": batch["id"], "status": state, "roles": batch["roles"], "items": entries,
                           "scope": "FILE" if file_mode else "LIBRARY", "source_published": source_published}
        return result

    @staticmethod
    def assert_reprocess_available(row, version_id):
        bulk, files = index_rebuild_batches(row.configuration_json)
        if active_rebuild(bulk):
            raise LibraryConflict("KNOWLEDGE_INDEX_BATCH_ACTIVE")
        if active_rebuild(files.get(str(version_id))):
            raise LibraryConflict("KNOWLEDGE_FILE_INDEX_BATCH_ACTIVE")

    async def start_rebuild(self, row, *, target_profile, expected_revision, now, version_id=None,
                            expected_generation_id=None):
        if expected_revision is not None and int(row.row_version) != expected_revision:
            raise LibraryConflict("KNOWLEDGE_LIBRARY_REVISION_CONFLICT")
        require(row.status == "ACTIVE", "仅启用的知识库可重建索引")
        config = row.configuration_json
        single = version_id is not None
        bulk, files = index_rebuild_batches(config)
        if active_rebuild(bulk) or (not single and any(map(active_rebuild, files.values()))):
            raise LibraryConflict("KNOWLEDGE_INDEX_BATCH_ACTIVE")
        if single and active_rebuild(files.get(str(version_id))):
            raise LibraryConflict("KNOWLEDGE_FILE_INDEX_BATCH_ACTIVE")
        target_models = config.get("index_models", {})
        roles = sorted(target_models) if single else config.get("index_rebuild_pending", [])
        source_published = True
        generations = KnowledgeIndexGenerationRepository(self._session, self.scope)
        if single:
            record = await self.version(row.library_id, version_id, operation="MANAGE", require_initial_completion=True)
            if (record["generation_id"] is None or record["generation_id"] != expected_generation_id
                or record["job_status"] != "PARSED" or record["status"] not in {"PARSED", "PUBLISHED"}
                or record["replacement_pending"]
                or record["governance"].get("review") != "APPROVED"):
                raise LibraryConflict("KNOWLEDGE_INDEX_BATCH_CONFLICT")
            old = await generations.get(record["generation_id"])
            require(old is not None, "文件缺少可复用的索引来源")
            members = {str(version_id): (record["title"], record["generation_id"], old.parse_run_id)}
            source_published = record["status"] == "PUBLISHED"
        else:
            members = await self.published_members(row.library_id)
            latest = select(Job.ingestion_job_id).where(
                Job.document_version_id == Version.document_version_id
            ).order_by(Job.attempt.desc(), Job.ingestion_job_id.desc()).limit(1).correlate(Version).scalar_subquery()
            pending = self._documents(row.library_id).with_only_columns(Version.document_version_id).select_from(Document).join(
                Version, Version.document_id == Document.document_id).join(Job, Job.ingestion_job_id == latest).where(
                    Version.status == "PUBLISHED", Job.parse_run_id != Version.published_parse_run_id).limit(1)
            if await self._session.scalar(pending) is not None:
                raise LibraryConflict("KNOWLEDGE_INDEX_BATCH_CONFLICT")
        if not members:
            row.configuration_json = {**config, "index_rebuild_pending": []}
            await self._session.flush()
            return
        target = {spec.input_kind: spec for spec in target_profile.representations}
        require((single or roles) and all(INDEX_ROLES[role] in target
                    and target[INDEX_ROLES[role]].model_ref == target_models[role] for role in roles)
                and (not single or set(target) == {INDEX_ROLES[role] for role in roles}),
                "重建模型配置已改变")
        batch_id = str(uuid7())
        entries = {}
        outboxes = KnowledgeOutboxRepository(self._session, self.scope)
        for version_id, (title, old_id, parse_run_id) in members.items():
            old = await generations.get(old_id) if old_id is not None else None
            require(old is not None and old.status == "READY" and old.parse_run_id == parse_run_id
                    and old.document_version_id == UUID(version_id), "文件缺少有效的索引来源")
            manifest = old.manifest_json
            original = decode(ProjectionProfile, manifest["profile"])
            prior = {spec.input_kind: spec for spec in original.representations}
            # 只替换待重建通道；其余通道严格复用发布时的表示，不重新调用模型。
            specs = tuple(spec for role, kind in INDEX_ROLES.items()
                if role in target_models and (spec := target.get(kind) if role in roles else prior.get(kind)))
            profile = replace(target_profile, representations=specs)
            resplit = (manifest.get("split_revision") != SPLIT_REVISION
                       or original.max_tokens != profile.max_tokens
                       or original.batch_size != profile.batch_size
                       or original.representations != profile.representations)
            projection_ref = fingerprint([batch_id, old.projection_ref, encode(profile), SPLIT_REVISION])
            candidate = await generations.add(dict(document_version_id=UUID(version_id), parse_run_id=parse_run_id,
                projection_ref=projection_ref, manifest_json={"profile": encode(profile),
                    "allowed_view_hash": manifest["allowed_view_hash"], "rebuild_id": batch_id,
                    "source_generation_id": str(old_id), "projection_ref": projection_ref,
                    "split_revision": SPLIT_REVISION, "resplit": resplit,
                    **({} if resplit else {"planned_batch_count": manifest["batch_count"]}),
                    **({"file_candidate": True} if single else {})},
                channel_indexes_json={}, status="PREPARING", active_revision=0))
            outbox = await outboxes.add(dict(document_version_id=UUID(version_id), event_type="REBUILD_PROJECTION",
                idempotency_key=f"{parse_run_id}:rebuild:{batch_id}", payload_ref=str(candidate.index_generation_id),
                status="PENDING", attempt=0, retry_at=now))
            entries[version_id] = {"title": title, "source_id": str(old_id),
                                   "candidate_id": str(candidate.index_generation_id), "outbox_id": str(outbox.outbox_id)}
        self._store_rebuild(row, {"id": batch_id, "target_models": target_models, "roles": roles, "members": entries,
                                 "scope": "FILE" if single else "LIBRARY", "source_published": source_published})
        await self._session.flush()

    async def complete_rebuild(self, row, *, publish, now, batch_id=None, outbox_id=None, version_id=None):
        """只切换本次交付所属批次，不处理无关文件；恢复时逐项检查。"""
        if batch_id is None:
            bulk, files = index_rebuild_batches(row.configuration_json)
            for batch in (bulk, *files.values()):
                if batch:
                    await self.complete_rebuild(row, publish=publish, now=now, batch_id=batch["id"])
            return
        config = row.configuration_json
        batch = self._find_rebuild(config, batch_id)
        if (row.status != "ACTIVE" or not batch
                or batch["target_models"] != config.get("index_models", {})):
            return
        if outbox_id is not None and batch["members"].get(str(version_id), {}).get("outbox_id") != str(outbox_id):
            return
        if batch.get("terminal_status"):
            # 旧批次可能已采用候选，但当时尚未填写文件发布指针。只恢复该已校验候选。
            if (batch["terminal_status"] == "PUBLISHED" and batch.get("scope") == "FILE"
                    and batch.get("source_published") is False):
                ref, item = next(iter(batch["members"].items()))
                record = await self._version(row.library_id, UUID(ref))
                if (record["status"] == "PARSED" and record["job_status"] == "PARSED"
                        and record["governance"].get("review") == "APPROVED"
                        and str(record["generation_id"]) == item["candidate_id"]):
                    candidate = await KnowledgeIndexGenerationRepository(self._session, self.scope).get(
                        UUID(item["candidate_id"]), lock=True)
                    if candidate and candidate.manifest_json.get("adopted_from_batch") == batch["id"]:
                        await publish(self.scope, version_id=UUID(ref), generation_id=UUID(item["candidate_id"]),
                                      expected_generation_id=None, now=now)
            return
        await self._session.flush()
        outboxes = KnowledgeOutboxRepository(self._session, self.scope)
        task_ids = {UUID(item["outbox_id"]) for item in batch["members"].values()}
        delivered = await self._session.scalar(outboxes.statement().with_only_columns(func.count(outboxes.entity.outbox_id))
            .where(outboxes.entity.outbox_id.in_(task_ids), outboxes.entity.status == "DELIVERED"))
        if delivered != len(task_ids):
            return
        members = {} if batch.get("scope") == "FILE" else None
        if (await self.rebuild_summary(row, members=members, batch_id=batch_id))["batch"]["status"] != "READY":
            return
        for ref, item in sorted(batch["members"].items()):
            await publish(self.scope, version_id=UUID(ref), generation_id=UUID(item["candidate_id"]),
                expected_generation_id=UUID(item["source_id"]) if batch.get("source_published", True) else None,
                now=now, rebuild_id=batch["id"])
        if batch.get("scope") != "FILE":
            authorized = {**config.get("index_authorized_models", {}), **{
                role: [batch["target_models"][role]] for role in batch["roles"]}}
            row.configuration_json = {**config, "index_rebuild_pending": [], "index_authorized_models": authorized}
        self._store_rebuild(row, {**batch, "terminal_status": "PUBLISHED"})
        await self._session.flush()

    async def rebuild_action(self, row, *, batch_id, action, version_id, now):
        batch = self._find_rebuild(row.configuration_json, batch_id)
        if not active_rebuild(batch):
            raise LibraryConflict("KNOWLEDGE_INDEX_BATCH_CONFLICT")
        if action == "RETRY":
            require(row.status == "ACTIVE" and (await self.rebuild_summary(row, batch_id=batch_id))["batch"]["status"] != "STALE",
                    "启用的当前批次才允许重试")
            item = batch["members"].get(str(version_id))
            require(item is not None, "文件不属于当前索引批次")
            outbox = await KnowledgeOutboxRepository(self._session, self.scope).get(UUID(item["outbox_id"]), lock=True)
            require(outbox is not None and outbox.status == "FAILED", "只能重试失败文件")
            outbox.status, outbox.error_code, outbox.retry_at = "PENDING", None, now
            outbox.lease_until, outbox.lease_owner = None, None
        else:
            require(action == "DISCARD", "未登记索引批次操作")
            outboxes = KnowledgeOutboxRepository(self._session, self.scope)
            for item in batch["members"].values():
                outbox = await outboxes.get(UUID(item["outbox_id"]), lock=True)
                if outbox is not None and outbox.status in {"PENDING", "RUNNING", "FAILED"}:
                    outbox.status, outbox.lease_until = "CANCELLED", now
            self._store_rebuild(row, {**batch, "terminal_status": "DISCARDED"})
        await self._session.flush()

    async def _delete_block(self, library_id):
        if await self._session.scalar(self._documents(library_id).with_only_columns(Document.document_id).limit(1)):
            return "HAS_DOCUMENTS"
        if await self._session.scalar(select(AgentVersion.agent_version_id).where(
                AgentVersion.library_id == library_id).limit(1)):
            return "AGENT_REFERENCE"
        return None

    async def delete(self, library_id, *, expected_revision):
        await self._lock_catalog()
        row = await self.authorized(library_id, "MANAGE", lock=True)
        if int(row.row_version) != expected_revision:
            raise LibraryConflict("KNOWLEDGE_LIBRARY_REVISION_CONFLICT")
        blocked = await self._delete_block(row.library_id)
        if blocked:
            raise LibraryConflict("KNOWLEDGE_LIBRARY_" + blocked)
        await self._session.delete(row)

    async def upload_receipt(self, library_id, idempotency_key):
        row = (await self._session.execute(self._documents(library_id).add_columns(Version)
            .join(Version, Version.document_id == Document.document_id).where(
                Version.business_version_ref == idempotency_key))).one_or_none()
        if row is None:
            return None
        document, version = row
        job_id = await self._session.scalar(select(Job.ingestion_job_id).where(
            Job.document_version_id == version.document_version_id
        ).order_by(Job.attempt, Job.ingestion_job_id).limit(1))
        owner = document.metadata_json.get("versions", {}).get(str(version.document_version_id), {}).get("uploaded_by")
        require(owner == self.scope.subject_ref, "上传幂等键冲突")
        return {"document_id": str(document.document_id), "version_id": str(version.document_version_id),
            "job_id": str(job_id), "name": document.title, "sha256": version.source_sha256,
            "mime_type": version.mime_type, "status": "RECEIVED"}

    async def listing(self, *, management=False):
        rows = (await self._session.scalars(self.statement().order_by(
            self.entity.created_at.desc(), self.entity.library_id.desc()).limit(501))).all()
        require(len(rows) <= 500, "资料库目录超过当前分页预算")
        operation = "MANAGE" if management else "READ"
        return [{**library_view(row, self.scope.subject_ref), "grants":
            dict(row.grants_json) if management else None} for row in rows
            if operation in row.grants_json.get(self.scope.subject_ref, ()) and (management or row.status == "ACTIVE")]

    async def create(self, *, library_id, name, description, configuration):
        await self._lock_catalog()
        name = await self._name(name)
        row = await self.add(dict(library_id=library_id, display_name=name, description=description,
            status="ACTIVE", grants_json={self.scope.subject_ref: sorted(OPERATIONS)},
            configuration_json=configuration, created_by=self.scope.subject_ref, updated_at=datetime.now(timezone.utc)))
        return library_view(row, self.scope.subject_ref)

    async def change(self, library_id, *, expected_revision, changes):
        await self._lock_catalog()
        row = await self.authorized(library_id, "MANAGE", lock=True)
        if int(row.row_version) != expected_revision:
            raise LibraryConflict("KNOWLEDGE_LIBRARY_REVISION_CONFLICT")
        for field, value in changes.items():
            require(field in {"display_name", "description", "status", "grants", "configuration", "parse_models", "index_models"},
                    "资料库更新字段无效")
            if field in {"parse_models", "index_models"}:
                schema = LibraryParseModels if field == "parse_models" else LibraryIndexModels
                models = schema.model_validate(value or {}).model_dump(mode="json", exclude_none=True)
                configuration = row.configuration_json
                if field == "index_models":
                    before = configuration.get("index_models", {})
                    authorized = dict(configuration.get("index_authorized_models", {}))
                    for role in INDEX_ROLES:
                        if models.get(role) == before.get(role):
                            continue
                        authorized[role] = sorted(
                            {models[role]} | set(authorized.get(role, [before[role]] if role in before else []))
                        ) if role in models else []
                        require(len(authorized[role]) <= 64, "历次向量模型过多，请先完成索引切换")
                    pending = sorted(role for role in INDEX_ROLES if models.get(role)
                        and (role in configuration.get("index_rebuild_pending", []) or models.get(role) != before.get(role)))
                    if pending and not await self.published_members(row.library_id):
                        pending = []
                    configuration = {**configuration, "index_rebuild_pending": pending, "index_authorized_models": authorized}
                row.configuration_json = {**configuration, field: models}
                continue
            if field == "configuration":
                row.configuration_json = {**row.configuration_json, **value}
                continue
            if field == "display_name" and value != row.display_name:
                value = await self._name(value, excluding=row.library_id)
            setattr(row, {"grants": "grants_json", "configuration": "configuration_json"}.get(field, field), value)
        require(row.status in {"ACTIVE", "DISABLED"}, "资料库状态无效")
        require(row.display_name and row.display_name.strip(), "资料库名称不能为空")
        existing_operations = OPERATIONS if "grants" in changes else OPERATIONS | {"PUBLISH"}
        require(isinstance(row.grants_json, dict) and row.grants_json and all(
                isinstance(subject, str) and 0 < len(subject) <= 256
                and isinstance(permissions, list) and permissions and "READ" in permissions
                and len(permissions) == len(set(permissions)) and set(permissions) <= existing_operations
                for subject, permissions in row.grants_json.items()), "资料库成员授权无效")
        require(any("MANAGE" in permissions for permissions in row.grants_json.values()),
                "资料库必须保留至少一个管理主体")
        row.updated_at = datetime.now(timezone.utc)
        await self._session.flush()
        return library_view(row, self.scope.subject_ref)

    def _documents(self, library_id):
        statement = select(Document).where(Document.domain_id == self.scope.domain_id,
            Document.business_asset_ref.startswith("library:" + str(library_id) + ":"))
        if hasattr(Document, "tenant_id"):
            statement = statement.where(Document.tenant_id == self.scope.tenant_id)
        return statement

    async def lock_library_for_version(self, version_id):
        """后台从已持久化的版本取得归属库并加锁。"""
        query = (select(Document.business_asset_ref).join(Version, Version.document_id == Document.document_id)
                 .where(Version.document_version_id == version_id, Document.domain_id == self.scope.domain_id))
        if hasattr(Document, "tenant_id"):
            query = query.where(Document.tenant_id == self.scope.tenant_id)
        asset = await self._session.scalar(query)
        parts = (asset or "").split(":", 2)
        if len(parts) != 3 or parts[0] != "library":
            return None
        try:
            library_id = UUID(parts[1])
        except ValueError:
            return None
        return await self.get(library_id, lock=True)

    async def lock_approved_projection_publication(self, version_id, job_id):
        """以当前任务、已持久化审批和旧发布指针决定是否发布或暂缓切换。"""
        library = await self.lock_library_for_version(version_id)
        if library is None:
            return False, None, False
        latest = select(Job.ingestion_job_id).where(Job.document_version_id == Version.document_version_id).order_by(
            Job.attempt.desc(), Job.ingestion_job_id.desc()).limit(1).correlate(Version).scalar_subquery()
        query = self._documents(library.library_id).with_only_columns(Document.metadata_json, Version, Job).join(
            Version, Version.document_id == Document.document_id).where(
                Version.document_version_id == version_id, Document.status.not_in(("DELETED", "WITHDRAWN"))
            ).join(Job, Job.ingestion_job_id == latest).where(Job.ingestion_job_id == job_id, Job.status == "PARSED")
        entry = (await self._session.execute(query)).one_or_none()
        if entry is None:
            return False, None, False
        metadata, version, job = entry
        approval = (metadata or {}).get("versions", {}).get(str(version_id), {})
        if approval.get("review") != "APPROVED" or approval.get("uploaded_by") != self.scope.subject_ref:
            return False, None, False
        replaces = job.context_json.get("replaces_generation_id")
        if replaces is None:
            eligible = version.status == "PARSED" and version.active_generation_id is None
            return eligible and library.status == "ACTIVE", None, False
        base = version.active_generation_id
        eligible = (version.status == "PUBLISHED" and base is not None and str(base) == replaces
                    and version.published_parse_run_id != job.parse_run_id)
        return eligible and library.status == "ACTIVE", base, eligible and library.status != "ACTIVE"

    async def index_models_for_version(self, version_id):
        """读取已审批版本的库绑定，不跨 Tenant/Domain，也不重新采纳当前配置。"""
        statement = (select(Document.metadata_json, Document.business_asset_ref)
            .join(Version, Version.document_id == Document.document_id)
            .where(Version.document_version_id == version_id, Document.domain_id == self.scope.domain_id))
        if hasattr(Document, "tenant_id"):
            statement = statement.where(Document.tenant_id == self.scope.tenant_id)
        row = (await self._session.execute(statement)).one_or_none()
        require(row is not None and row.business_asset_ref.startswith("library:"), "索引任务不属于资料库")
        approval = (row.metadata_json or {}).get("versions", {}).get(str(version_id), {})
        return dict(approval.get("index_models") or {})

    async def files(self, library_id, *, page=1, page_size=20, operation="READ",
                    view="files", title="", state="", mine=False, version_id=None, document_id=None):
        library = await self.authorized(library_id, operation)
        require(page >= 1 and 1 <= page_size <= 100, "文档分页预算无效")
        permissions = set(library.grants_json[self.scope.subject_ref])
        if view == "reviews":
            require(bool(permissions & {"REVIEW", "WITHDRAW"}), "无权访问审核队列")
        active_job = select(Job.ingestion_job_id).where(
            Job.document_version_id == Version.document_version_id
        ).order_by(Job.attempt.desc(), Job.ingestion_job_id.desc()).limit(1).correlate(Version).scalar_subquery()
        generation = select(Generation.index_generation_id).where(
            Generation.document_version_id == Version.document_version_id,
            Generation.parse_run_id == Job.parse_run_id, Generation.status == "READY"
        ).order_by(Generation.created_at.desc(), Generation.index_generation_id.desc()
        ).limit(1).correlate(Version, Job).scalar_subquery()
        query = self._documents(library_id).add_columns(Version, Job, generation, ParseRun.status).join(
            Version, Version.document_id == Document.document_id).join(
            Job, Job.ingestion_job_id == active_job).join(ParseRun, ParseRun.parse_run_id == Job.parse_run_id)
        if version_id:
            query = query.where(Version.document_version_id == UUID(version_id))
        if document_id:
            query = query.where(Document.document_id == UUID(document_id))
        if title.strip():
            query = query.where(Document.title.icontains(title.strip(), autoescape=True))
        if state and view == "files":
            query = query.where(Version.status == state)
        rows = (await self._session.execute(query.order_by(
            Document.created_at.desc(), Document.document_id.desc(), Version.created_at.desc(),
            Version.document_version_id.desc()).limit(501))).all()
        require(len(rows) <= 500, "资料库目录超过当前版本预算")
        indexes = await self._initial_projection_outboxes(
            {str(job.parse_run_id) + ":projection": job.document_version_id for _, version, job, _, _ in rows
             if job.status == "PARSED" and version.status in {"PARSED", "PUBLISHED"}})
        pending_publications = self._pending_publications(indexes)
        bulk_rebuild, file_rebuilds = index_rebuild_batches(library.configuration_json)
        documents = {}
        visible_tasks = {}
        for row, version, job, generation_id, parse_state in rows:
            governance = row.metadata_json.get("versions", {}).get(str(version.document_version_id), {})
            own = governance.get("uploaded_by") == self.scope.subject_ref
            if (version.status != "PUBLISHED" and not permissions & {"REVIEW", "MANAGE"}
                    and not own):
                continue
            if view == "processing" and ((mine or not permissions & {"REVIEW", "MANAGE"}) and not own):
                continue
            stage = processing_status(job, parse_state)
            published = version.status == "PUBLISHED"
            replacement_pending = published and job.parse_run_id != version.published_parse_run_id
            if published:
                generation_id = None if view == "processing" and replacement_pending else version.active_generation_id
            review, ready = governance.get("review"), generation_id is not None
            auto_publish_pending = str(job.parse_run_id) + ":projection" in pending_publications
            if ready and stage in {"PARSED", "RETRY"}:
                stage = "READY"
            actions = []
            if not published and version.status != "WITHDRAWN" and own and "UPLOAD" in permissions and job.status in {
                    "AWAITING_APPROVAL", "RECEIVED", "PARSING"}:
                actions.append("cancel")
            if "REVIEW" in permissions and review == "PENDING" and version.status not in {"CANCELLED", "WITHDRAWN"}:
                actions.append("review")
            if "WITHDRAW" in permissions and version.status == "PUBLISHED":
                actions.append("withdraw")
            if (view == "processing" and "MANAGE" in permissions and review == "APPROVED" and ready and stage != "INDEX_FAILED"
                and not auto_publish_pending
                and job.status == "PARSED" and version.status in {"PARSED", "PUBLISHED"}):
                actions.append("rebuild-index")
            if (view == "processing" and library.status == "ACTIVE" and review == "APPROVED"
                    and version.status not in {"WITHDRAWN", "CANCELLED"}):
                parse_failed = stage == "FAILED" and parse_state != "PARSED"
                indexed_success = (not replacement_pending and ready and job.status == parse_state == "PARSED"
                                   and (published or version.status == "PARSED"
                                        and (index_task := indexes.get(str(job.parse_run_id) + ":projection")) is not None
                                        and index_task.status == "DELIVERED"))
                rebuild_active = active_rebuild(bulk_rebuild) or active_rebuild(file_rebuilds.get(str(version.document_version_id)))
                if "REVIEW" in permissions and not rebuild_active and (parse_failed or indexed_success):
                    actions.append("reprocess")
                if stage == "INDEX_FAILED" and "MANAGE" in permissions:
                    actions.append("retry-index")
            if view == "processing" and state and stage != state:
                continue
            if view == "reviews":
                if state == "PENDING" and "review" not in actions:
                    continue
                if state == "PUBLISHED" and version.status != "PUBLISHED":
                    continue
                if state == "REJECTED" and review != "REJECTED":
                    continue
            item = documents.setdefault(str(row.document_id), {
                "document_id": str(row.document_id), "title": row.title, "status": row.status,
                "row_version": row.row_version, "versions": {}})
            entry = item["versions"][str(version.document_version_id)] = {
                **{key: item for key, item in governance.items() if key != "index_models"},
                "status": version.status, "job_id": str(job.ingestion_job_id),
                "processing_status": stage, "parse_status": "PARSED" if stage == "INDEX_FAILED" else job.status,
                "error_code": job.error_code or job.context_json.get("pipeline_error"),
                "generation_ready": ready, "generation_id": str(generation_id) if ready else None,
                "created_at": version.created_at.isoformat(), "mime_type": version.mime_type,
                "byte_count": int(version.byte_count), "allowed_actions": actions}
            if view == "processing":
                visible_tasks[str(version.document_version_id)] = job
            if view == "processing" and version_id:
                history = (await self._session.scalars(select(Job).where(
                    Job.document_version_id == version.document_version_id,
                    Job.ingestion_job_id != job.ingestion_job_id
                ).order_by(Job.attempt.desc(), Job.ingestion_job_id.desc()).limit(25))).all()
                entry["parse_task_history"] = [{"job_id": str(task.ingestion_job_id), "status": task.status,
                    "error_code": task.error_code or task.context_json.get("pipeline_error"),
                    "created_at": task.created_at.isoformat()} for task in history]
        items = list(documents.values())[(page - 1) * page_size:page * page_size]
        if view == "processing":
            await self._add_processing_progress(items, visible_tasks, indexes)
            await self._add_rebuild_progress(items, library.configuration_json)
        return {"items": items, "page": page, "has_more": len(documents) > page * page_size}

    async def _initial_projection_outboxes(self, targets):
        if not targets:
            return {}
        repo = KnowledgeOutboxRepository(self._session, self.scope)
        outbox = repo.entity
        rows = await self._session.scalars(repo.statement().where(
            outbox.event_type == "BUILD_PROJECTION", outbox.idempotency_key.in_(targets),
            outbox.document_version_id.in_(set(targets.values()))))
        return {row.idempotency_key: row for row in rows
                if row.document_version_id == targets[row.idempotency_key]}

    @staticmethod
    def _pending_publications(outboxes):
        return {key for key, row in outboxes.items() if row.status != "DELIVERED"}

    async def _add_processing_progress(self, items, tasks, indexes):
        entries = [(tasks[version_id], entry)
                   for item in items for version_id, entry in item["versions"].items()]
        now = datetime.now(timezone.utc)
        ready_ids = {UUID(entry["generation_id"]) for _, entry in entries if entry["generation_id"]}
        preparing = {(job.parse_run_id, job.document_version_id) for job, entry in entries
                     if not entry["generation_ready"] and entry["parse_status"] == "PARSED"}
        generations, pending = {}, {}
        if ready_ids or preparing:
            repo = KnowledgeIndexGenerationRepository(self._session, self.scope)
            generation = repo.entity
            query = repo.statement().where(
                (generation.index_generation_id.in_(ready_ids)) |
                ((generation.status == "PREPARING") &
                 generation.parse_run_id.in_({run for run, _ in preparing}) &
                 generation.document_version_id.in_({version for _, version in preparing})))
            for record in await self._session.scalars(query):
                if record.index_generation_id in ready_ids:
                    generations[str(record.index_generation_id)] = record
                key = (record.parse_run_id, record.document_version_id)
                if key in preparing and "rebuild_id" not in record.manifest_json:
                    pending.setdefault(key, []).append(record)
        for job, entry in entries:
            stage = processing_progress(
                job, entry["processing_status"], review=entry.get("review"), ready=entry["generation_ready"],
                outbox=indexes.get(str(job.parse_run_id) + ":projection"), now=now)
            if stage in {"INDEX_QUEUED", "INDEXING", "INDEX_RETRY", "INDEX_CONFIRMING", "READY", "INDEX_FAILED"}:
                record = generations.get(entry["generation_id"])
                if record is None:
                    candidates = pending.get((job.parse_run_id, job.document_version_id), ())
                    record = candidates[0] if len(candidates) == 1 else None
                progress = index_batch_progress(record.manifest_json if record is not None else {},
                                                validated=stage == "READY" and record is not None and record.status == "READY")
                if stage == "INDEXING" and progress["total_batches"] == progress["committed_batches"]:
                    stage = "INDEX_CONFIRMING"
                entry["index_progress"] = progress
            entry["progress_stage"] = stage

    async def _add_rebuild_progress(self, items, configuration):
        """只读取当前页可见文件的候选索引；不替换正在服务的索引指针。"""
        bulk, files = index_rebuild_batches(configuration)
        models = configuration.get("index_models", {})
        entries = []
        for item in items:
            for version_id, entry in item["versions"].items():
                for batch in (files.get(version_id), bulk):
                    member = batch["members"].get(version_id) if batch else None
                    if not member:
                        continue
                    if (active_rebuild(batch) and batch["target_models"] == models
                            and entry["generation_id"] == member["source_id"]):
                        entries.append((version_id, entry, member, batch["id"]))
                        break
                    if (batch.get("terminal_status") == "PUBLISHED" and entry["status"] == "PUBLISHED"
                            and entry["generation_id"] == member["candidate_id"]):
                        entry["progress_stage"] = "REBUILD_COMPLETE"
        if not entries:
            return
        outboxes = KnowledgeOutboxRepository(self._session, self.scope)
        generations = KnowledgeIndexGenerationRepository(self._session, self.scope)
        outbox_ids = {UUID(member["outbox_id"]) for _, _, member, _ in entries}
        generation_ids = {UUID(member["candidate_id"]) for _, _, member, _ in entries}
        tasks = {str(record.outbox_id): record for record in await self._session.scalars(
            outboxes.statement().where(outboxes.entity.outbox_id.in_(outbox_ids),
                                      outboxes.entity.event_type == "REBUILD_PROJECTION"))}
        candidates = {str(record.index_generation_id): record for record in await self._session.scalars(
            generations.statement().where(generations.entity.index_generation_id.in_(generation_ids)))}
        for version_id, entry, member, batch_id in entries:
            task, candidate = tasks.get(member["outbox_id"]), candidates.get(member["candidate_id"])
            if (task is None or candidate is None or task.payload_ref != member["candidate_id"]
                    or str(task.document_version_id) != version_id or str(candidate.document_version_id) != version_id
                    or candidate.manifest_json.get("rebuild_id") != batch_id):
                continue
            ready = task.status == "DELIVERED" and candidate.status in {"READY", "CANDIDATE"}
            progress = index_batch_progress(candidate.manifest_json, validated=ready)
            if ready:
                stage = "REBUILD_READY"
            elif task.status == "FAILED":
                stage = "REBUILD_FAILED"
            elif task.status == "PENDING":
                stage = "REBUILD_RETRY" if task.error_code else "REBUILD_QUEUED"
            elif task.status in {"RUNNING", "DELIVERED"}:
                stage = ("REBUILD_CONFIRMING" if progress["total_batches"] == progress["committed_batches"]
                         else "REBUILDING")
            else:
                continue
            entry["progress_stage"], entry["index_progress"] = stage, progress

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

    async def version(self, library_id, version_id, *, operation, require_initial_completion=False):
        library = await self.authorized(library_id, operation)
        return await self._version(library_id, version_id, library=library, operation=operation,
                                   require_initial_completion=require_initial_completion)

    async def _version(self, library_id, version_id, *, library=None, operation=None, require_initial_completion=False):
        row = (await self._session.execute(self._documents(library_id).add_columns(Version)
            .join(Version, Version.document_id == Document.document_id)
            .where(Version.document_version_id == version_id))).one_or_none()
        require(row is not None, "版本不存在于当前资料库")
        document, version = row
        job = (await self._session.scalars(select(Job).where(
            Job.document_version_id == version.document_version_id
        ).order_by(Job.attempt.desc(), Job.ingestion_job_id.desc()).limit(1))).first()
        require(job is not None, "版本缺少解析任务")
        if require_initial_completion and version.status == "PARSED":
            outboxes = await self._initial_projection_outboxes({str(job.parse_run_id) + ":projection": version_id})
            require(not self._pending_publications(outboxes), "首次索引任务尚未完成，请等待自动发布或处理失败任务")
        generations = (await self._session.scalars(select(Generation).where(
            Generation.document_version_id == version_id, Generation.parse_run_id == job.parse_run_id,
            Generation.status == "READY").order_by(
                Generation.created_at.desc(), Generation.index_generation_id.desc()).limit(2))).all()
        governance = dict(document.metadata_json.get("versions", {}).get(str(version_id), {}))
        if operation == "READ":
            permissions = set(library.grants_json[self.scope.subject_ref])
            require(version.status == "PUBLISHED" or governance.get("uploaded_by") == self.scope.subject_ref
                or bool(permissions & {"REVIEW", "MANAGE"}),
                "当前主体无权读取未发布版本")
        return {"document_id": document.document_id, "version_id": version.document_version_id,
            "job_id": job.ingestion_job_id, "subject_ref": job.context_json["subject_ref"],
            "source_handle": version.source_object_ref, "mime_type": version.mime_type,
            "byte_count": int(version.byte_count), "title": document.title,
            "status": version.status, "revision": int(version.row_version),
            "job_status": job.status, "processing_status": job.context_json.get("pipeline_status", job.status),
            "replacement_pending": version.status == "PUBLISHED" and job.parse_run_id != version.published_parse_run_id,
            "error_code": job.error_code or job.context_json.get("pipeline_error"),
            "governance": governance, "generation_id": version.active_generation_id if version.status == "PUBLISHED"
                else generations[0].index_generation_id if generations else None}

    async def history_versions(self, library_id, version_ids):
        library = await self.authorized(library_id, "READ")
        if not version_ids:
            return library.display_name, set()
        statement = self._documents(library_id).add_columns(Version).join(
            Version, Version.document_id == Document.document_id).where(
            Version.document_version_id.in_(version_ids), Version.status == "PUBLISHED",
            Document.status == "PUBLISHED")
        rows = (await self._session.execute(statement)).all()
        return library.display_name, {version.document_version_id for document, version in rows
            if document.metadata_json.get("versions", {}).get(str(version.document_version_id), {}).get("review") == "APPROVED"}

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
