"""知识检索 App 的正式资料组织、上传与审核发布用例。"""

from contextlib import aclosing
from dataclasses import replace
from hashlib import sha256
from uuid import UUID
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.contracts import ParseModelBinding, ParseStrategy, SourceDescriptor
from knowledge_algorithms.parsing.formats import NATIVE_FORMATS
from platform_core.identity import uuid7
from platform_core.contracts.knowledge_parsing import LibraryIndexModels, LibraryParseModels
from platform_core.contracts.knowledge_parse_approval import available_parse_bindings, approval_parse_options, bindings_for_strategy
from ..domain.knowledge_ingestion import OWNER
from ..domain.knowledge_retrieval import KnowledgeGrant
from ..domain.knowledge_library import INDEX_ROLES, LibraryConflict
from .knowledge_retrieval import RetrievalReceipt
from .knowledge_search_history import KnowledgeSearchHistory


class LibraryAuthorization:
    def __init__(self, uow_factory):
        self.uow_factory = uow_factory

    async def authorize(self, access, resource_ref):
        require(resource_ref.startswith("library:"), "知识范围必须是本 App 资料库")
        library_id = UUID(resource_ref[8:])
        async with self.uow_factory() as uow:
            revision, versions, permissions = await uow.library(access).grant(library_id,
                include_unpublished=True)
        operations = {"READ"} | (permissions & {"WITHDRAW"})
        return KnowledgeGrant(revision, versions, frozenset(operations))

    async def index_policy(self, access, resource_ref):
        require(resource_ref.startswith("library:"), "知识范围必须是本 App 资料库")
        async with self.uow_factory() as uow:
            row = await uow.library(access).authorized(UUID(resource_ref[8:]), "READ")
            configuration = row.configuration_json
            authorized = configuration.get("index_authorized_models", {})
            return {kind: authorized.get(role) for role, kind in INDEX_ROLES.items()
                    if configuration.get("index_models", {}).get(role)}


class KnowledgeLibraryService:
    def __init__(self, *, uow_factory, ingestion, retrieval, parse_profile, configurations,
                 resolve_parse_model=None, model_executor_ready=False, resolve_index_profile=None):
        self.uow_factory, self.ingestion, self.retrieval = uow_factory, ingestion, retrieval
        self.search_history = KnowledgeSearchHistory(uow_factory, retrieval)
        self.parse_profile, self.configurations = parse_profile, dict(configurations)
        self.resolve_parse_model, self.model_executor_ready = resolve_parse_model, model_executor_ready
        self.resolve_index_profile = resolve_index_profile

    async def _parse_bindings(self, access, models):
        resolve = self.resolve_parse_model
        return await available_parse_bindings(models, (lambda role, model_id: resolve(access, role, model_id)) if resolve else None)

    async def _original_parse_available(self, access, profile):
        if profile.strategy is None:
            return False
        if profile.strategy == ParseStrategy.TEXT:
            return True
        if not self.model_executor_ready:
            return False
        frozen = {role: binding for role, binding in (("ocr", profile.ocr_model), ("vlm", profile.vlm_model))
                  if binding is not None}
        current = await self._parse_bindings(access, {role: item.model_id for role, item in frozen.items()})
        return bool(frozen) and all(role in current and ParseModelBinding(**current[role]) == item
                                    for role, item in frozen.items())

    async def list(self, access, *, management=False):
        async with self.uow_factory() as uow:
            values = await uow.library(access).listing(management=management)
        return [self.library_view(value) for value in values]

    async def create(self, access, *, display_name, description, profile_ref="default", parse_models=None, index_models=None):
        require(profile_ref in self.configurations, "资料库配置未在服务端登记")
        models = LibraryParseModels.model_validate(parse_models or {}).model_dump(mode="json", exclude_none=True)
        indexes = LibraryIndexModels.model_validate(index_models or {}).model_dump(mode="json", exclude_none=True)
        async with self.uow_factory() as uow:
            value = await uow.library(access).create(library_id=uuid7(), name=display_name,
                description=description, configuration={"profile_ref": profile_ref, "parse_models": models, "index_models": indexes})
            await uow.commit()
        return self.library_view(value)

    async def get(self, access, library_id, *, management=False):
        async with self.uow_factory() as uow:
            library = uow.library(access)
            value = await library.detail(library_id, management=management)
            if management:
                row = await library.authorized(library_id, "MANAGE")
                value["index_maintenance"] = await library.rebuild_summary(row)
        return self.library_view(value)

    def library_view(self, value):
        configuration = dict(value["configuration"])
        models = configuration.pop("parse_models", {})
        indexes = configuration.pop("index_models", {})
        configuration.pop("index_rebuild", None)
        configuration.pop("file_index_rebuilds", None)
        configuration.pop("index_rebuild_pending", None)
        configuration.pop("index_authorized_models", None)
        profile = self.retrieval.profiles[configuration["profile_ref"]]
        return {**value, "configuration": configuration,
            "parse_models": models if value.get("grants") is not None else None,
            "index_models": indexes if value.get("grants") is not None else None,
            "upload_max_bytes": min(self.parse_profile.max_input_bytes, 67108864),
            "upload_formats": NATIVE_FORMATS, "ocr_enabled": "OCR" in self.parse_profile.repair_strategies,
            "search_capabilities": [
            {"kind": channel.kind, "direction": channel.direction, "required": channel.required}
            for channel in profile.channels]}

    async def update(self, access, library_id, *, expected_revision, changes):
        changes = dict(changes)
        rebuild = changes.pop("rebuild_indexes", False)
        profile = None
        if rebuild:
            models = LibraryIndexModels.model_validate(changes.get("index_models") or {}).model_dump(
                mode="json", exclude_none=True)
            profile = await self._freeze_rebuild(access, library_id, expected_revision, proposed=models)
        if "configuration" in changes:
            require(set(changes["configuration"]) == {"profile_ref"}
                and changes["configuration"]["profile_ref"] in self.configurations, "资料库配置无效")
        async with self.uow_factory() as uow:
            library = uow.library(access)
            value = await library.change(library_id, expected_revision=expected_revision, changes=changes)
            if rebuild or changes.get("status") == "ACTIVE":
                row = await library.authorized(library_id, "MANAGE", lock=True)
                if rebuild:
                    await library.start_rebuild(row, target_profile=profile, expected_revision=None,
                                                now=self.ingestion.clock())
                else:
                    await library.complete_rebuild(row, publish=uow.knowledge_retrieval.publish,
                                                   now=self.ingestion.clock())
                value["index_maintenance"] = await library.rebuild_summary(row)
                value["row_version"] = row.row_version
            await uow.commit()
        return self.library_view(value)

    async def _freeze_rebuild(self, access, library_id, expected_revision, *, proposed=None, single_file=False):
        require(self.resolve_index_profile is not None, "未配置索引构建端")
        async with self.uow_factory() as uow:
            row = await uow.library(access).authorized(library_id, "MANAGE")
            if not single_file and int(row.row_version) != expected_revision:
                raise LibraryConflict("KNOWLEDGE_LIBRARY_REVISION_CONFLICT")
            config = row.configuration_json
            before, pending = config.get("index_models", {}), config.get("index_rebuild_pending", [])
            models = before if proposed is None else proposed
            selected = {role: model for role, model in models.items()
                        if single_file or role in pending or model != before.get(role)}
        require(single_file or selected, "没有需要重建的向量通道")
        return await self.resolve_index_profile(access, selected)

    async def index_rebuild(self, access, library_id, *, action=None, expected_revision=None, batch_id=None,
                            version_id=None, expected_generation_id=None):
        require(action in {None, "START", "RETRY", "DISCARD"}, "未登记索引批次操作")
        profile = (await self._freeze_rebuild(access, library_id, expected_revision, single_file=version_id is not None)
                   if action == "START" else None)
        async with self.uow_factory() as uow:
            library = uow.library(access)
            if action is None:
                row = await library.authorized(library_id, "READ")
                require("MANAGE" in row.grants_json[access.subject_ref],
                        "当前主体无权查看批量索引")
                if version_id is not None:
                    await library.version(library_id, version_id, operation="MANAGE")
            else:
                row = await library.authorized(library_id, "MANAGE", lock=True)
                now = self.ingestion.clock()
                if action == "START":
                    await library.start_rebuild(row, target_profile=profile,
                                                expected_revision=expected_revision if version_id is None else None, now=now,
                                                version_id=version_id, expected_generation_id=expected_generation_id)
                else:
                    await library.rebuild_action(row, batch_id=batch_id, action=action, version_id=version_id, now=now)
            view = await library.rebuild_summary(row, version_id=version_id, batch_id=batch_id)
            if action is not None:
                await uow.commit()
        return view

    async def files(self, access, library_id, **filters):
        pending, models, retry = [], {}, None
        async with self.uow_factory() as uow:
            library = uow.library(access)
            page = await library.files(library_id, **filters)
            if filters.get("view") == "reviews":
                pending = [version for item in page["items"] for version in item["versions"].values()
                           if version["parse_status"] == "AWAITING_APPROVAL" and "review" in version["allowed_actions"]]
                if pending:
                    row = await library.authorized(library_id, "REVIEW")
                    models = (row.configuration_json or {}).get("parse_models", {})
            if filters.get("view") == "processing" and filters.get("version_id"):
                retry = next((version for item in page["items"] for version in item["versions"].values()
                              if "reprocess" in version["allowed_actions"]), None)
                if retry is not None:
                    row = await library.authorized(library_id, "REVIEW")
                    models = (row.configuration_json or {}).get("parse_models", {})
                    record = await library.version(library_id, UUID(str(filters["version_id"])), operation="REVIEW")
                    lease = await uow.knowledge_ingestion.get_job(
                        replace(access, subject_ref=record["subject_ref"]), record["job_id"])
        if pending:
            bindings = await self._parse_bindings(access, models)
            options = approval_parse_options(bindings, executor_ready=self.model_executor_ready)
            for version in pending:
                version["parse_strategy_options"] = options
        if retry is not None:
            retry["original_parse_available"] = await self._original_parse_available(access, lease.profile)
            retry["parse_strategy_options"] = approval_parse_options(
                await self._parse_bindings(access, models), executor_ready=self.model_executor_ready)
        return page

    async def delete(self, access, library_id, *, expected_revision):
        async with self.uow_factory() as uow:
            await uow.library(access).delete(library_id, expected_revision=expected_revision)
            await uow.commit()

    async def upload(self, access, library_id, *, name, mime_type, raw, idempotency_key):
        require(name and len(name) <= 512 and "/" not in name and "\\" not in name
                and 0 < len(raw) <= self.parse_profile.max_input_bytes, "文件名或上传大小无效")
        require(0 < len(idempotency_key) <= 128, "上传缺少幂等键")
        require(mime_type in NATIVE_FORMATS, "当前解析端不支持该文件格式")
        upload_ref = "upload:" + fingerprint([access.subject_ref, idempotency_key])
        digest, document_id, version_id = sha256(raw).hexdigest(), uuid7(), uuid7()
        async with self.uow_factory() as uow:
            await uow.library(access).authorized(library_id, "UPLOAD")
        stored = await self.ingestion.store.put_bytes(raw, area="sources", max_bytes=self.parse_profile.max_input_bytes)
        source = SourceDescriptor(OWNER, str(version_id), digest, mime_type, len(raw), stored.handle_ref)
        async with self.uow_factory() as uow:
            library = uow.library(access)
            await library.authorized(library_id, "UPLOAD", lock=True)
            existing = await library.upload_receipt(library_id, upload_ref)
            if existing:
                require(existing["sha256"] == digest and existing["name"] == name and existing["mime_type"] == mime_type,
                        "相同上传幂等键对应不同文件")
                return existing
            job_id = await uow.knowledge_ingestion.register(access, document_id=document_id,
                document_version_id=version_id, source=source, profile=self.parse_profile,
                business_asset_ref="library:" + str(library_id) + ":" + str(document_id),
                business_version_ref=upload_ref, title=name, now=self.ingestion.clock(),
                awaiting_approval=True)
            await library.govern(library_id, document_id, version_id, operation="UPLOAD",
                changes={"review": "PENDING", "uploaded_by": access.subject_ref, "name": name})
            await uow.commit()
        return {"document_id": str(document_id), "version_id": str(version_id), "job_id": str(job_id),
                "sha256": digest, "name": name, "mime_type": mime_type, "status": "RECEIVED"}

    async def review(self, access, library_id, version_id, *, decision, comment, parse_strategy=None):
        require(decision in {"APPROVE", "REJECT"}, "审核决定无效")
        require(decision != "REJECT" or parse_strategy is None, "拒绝审核不选择解析方案")
        async with self.uow_factory() as uow:
            library = uow.library(access)
            library_row = await library.authorized(library_id, "REVIEW", lock=True)
            record = await library.version(library_id, version_id, operation="REVIEW")
            require(record["status"] not in {"CANCELLED", "WITHDRAWN"}, "版本当前不可审核")
            require(record["governance"].get("review") == "PENDING", "版本已审核，不能覆盖审核记录")
            changes = {"review": "APPROVED" if decision == "APPROVE" else "REJECTED",
                       "reviewed_by": access.subject_ref, "comment": comment}
            if record["job_status"] == "AWAITING_APPROVAL":
                profile = None
                if decision == "APPROVE":
                    require(parse_strategy in {strategy.value for strategy in ParseStrategy}, "请选择解析方案")
                    require(parse_strategy == "TEXT" or self.model_executor_ready, "当前运行环境尚未启用绑定模型的解析执行端")
                    models = (library_row.configuration_json or {}).get("parse_models", {})
                    bindings = await self._parse_bindings(access, models) if parse_strategy != "TEXT" else {}
                    try:
                        approved = bindings_for_strategy(parse_strategy, bindings)
                    except ValueError as exc:
                        require(False, str(exc))
                    profile = self.parse_profile.for_approval(ParseStrategy(parse_strategy),
                        **{role: ParseModelBinding(**binding) for role, binding in approved.items()})
                    changes["parse_strategy"] = parse_strategy
                    changes["index_models"] = (library_row.configuration_json or {}).get("index_models", {})
                await uow.knowledge_ingestion.decide_approval(
                    replace(access, subject_ref=record["subject_ref"]), record["job_id"],
                    document_version_id=version_id, profile=profile, now=self.ingestion.clock())
            else:
                require(parse_strategy is None, "已登记的历史任务不能更换解析方案")
            await library.govern(library_id, record["document_id"], version_id, operation="REVIEW",
                changes=changes)
            await uow.commit()
        return {"status": "APPROVED" if decision == "APPROVE" else "REJECTED"}

    async def reprocess(self, access, library_id, version_id, *, expected_job_id, mode, parse_strategy=None):
        async with self.uow_factory() as uow:
            library = uow.library(access)
            row = await library.authorized(library_id, "REVIEW", lock=True)
            record = await library.version(library_id, version_id, operation="REVIEW")
            if record["job_id"] != expected_job_id:
                raise LibraryConflict("KNOWLEDGE_TASK_REVISION_CONFLICT")
            require(row.status == "ACTIVE" and record["governance"].get("review") == "APPROVED",
                    "仅可重新解析启用库内经审核通过的文件")
            library.assert_reprocess_available(row, version_id)
            scope = replace(access, subject_ref=record["subject_ref"])
            lease = await uow.knowledge_ingestion.get_job(scope, expected_job_id)
            if mode == "ORIGINAL":
                require(parse_strategy is None and await self._original_parse_available(access, lease.profile),
                        "原解析方案或冻结模型已不可用，请选择当前库模型")
                profile = lease.profile
            else:
                require(mode == "CURRENT" and parse_strategy in {strategy.value for strategy in ParseStrategy},
                        "请选择有效的解析方案")
                require(parse_strategy == "TEXT" or self.model_executor_ready, "解析执行端不可用")
                configured = (row.configuration_json or {}).get("parse_models", {})
                bindings = await self._parse_bindings(access, configured) if parse_strategy != "TEXT" else {}
                try:
                    chosen = bindings_for_strategy(parse_strategy, bindings)
                except ValueError as exc:
                    require(False, str(exc))
                profile = self.parse_profile.for_approval(ParseStrategy(parse_strategy),
                    **{role: ParseModelBinding(**binding) for role, binding in chosen.items()})
            job_id = await uow.knowledge_ingestion.restart_parse(scope, expected_job_id,
                document_version_id=version_id, profile=profile, initiated_by=access.subject_ref, now=self.ingestion.clock())
            await uow.commit()
        return {"status": "RECEIVED", "job_id": str(job_id)}

    async def retry_index(self, access, library_id, version_id, *, expected_job_id):
        async with self.uow_factory() as uow:
            library = uow.library(access)
            row = await library.authorized(library_id, "MANAGE", lock=True)
            record = await library.version(library_id, version_id, operation="MANAGE")
            if record["job_id"] != expected_job_id:
                raise LibraryConflict("KNOWLEDGE_TASK_REVISION_CONFLICT")
            require(row.status == "ACTIVE" and record["governance"].get("review") == "APPROVED",
                    "仅可重试启用库内经审核通过的当前索引任务")
            job_id = await uow.knowledge_ingestion.restart_failed_index(
                replace(access, subject_ref=record["subject_ref"]), expected_job_id,
                document_version_id=version_id, now=self.ingestion.clock())
            await uow.commit()
        return {"status": "QUEUED", "job_id": str(job_id)}

    async def withdraw(self, access, library_id, version_id):
        async with self.uow_factory() as uow:
            library = uow.library(access)
            await library.authorized(library_id, "WITHDRAW", lock=True)
            record = await library.version(library_id, version_id, operation="WITHDRAW")
            require(record["status"] == "PUBLISHED" and record["governance"].get("review") == "APPROVED",
                    "只能撤回已发布版本")
            await uow.knowledge_retrieval.withdraw(access, version_id=version_id,
                expected_revision=record["revision"], now=self.ingestion.clock())
            await uow.commit()
        return {"status": "WITHDRAWN"}

    async def cancel(self, access, library_id, version_id):
        async with self.uow_factory() as uow:
            record = await uow.library(access).version(library_id, version_id, operation="UPLOAD")
        require(record["governance"].get("uploaded_by") == access.subject_ref,
                "只能取消当前主体上传的未发布任务")
        require(record["status"] not in {"PUBLISHED", "WITHDRAWN"}, "发布版本不能当作解析任务取消")
        await self.ingestion.cancel(replace(access, subject_ref=record["subject_ref"]), record["job_id"])
        return {"status": "CANCELLED"}

    async def source(self, access, library_id, version_id, *, published_only=False):
        async def authorized_record():
            async with self.uow_factory() as uow:
                record = await uow.library(access).version(library_id, version_id, operation="READ")
            if published_only:
                from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
                if record["status"] != "PUBLISHED":
                    raise AlgorithmError(ErrorCode.SNAPSHOT_STALE, "检索原文已经撤回或换版", stage="snapshot")
            return record

        record = await authorized_record()
        async def blocks():
            async with aclosing(self.ingestion.store.stream(record["source_handle"])) as source:
                async for block in source:
                    await authorized_record()
                    yield block
        return record, blocks()

    async def search(self, access, library_id, request):
        import base64
        from knowledge_algorithms.contracts.identity import AuthorizedScope
        from knowledge_algorithms.contracts.query import QueryInput, QueryMode, Predicate, OrderBy
        from knowledge_algorithms.contracts.wire import decode
        library = await self.get(access, library_id)
        scope = await self.retrieval.create_scope(access, resource_ref="library:" + str(library_id))
        handles = []
        for image in request.images:
            try:
                raw = base64.b64decode(image.image_base64, validate=True)
            except ValueError:
                require(False, "查询图片编码无效")
            handles.append(await self.retrieval.register_query_image(access, scope=scope,
                raw=raw, mime_type=image.mime_type))
        ordering = tuple(OrderBy(item.field, item.descending) for item in request.order_by)
        if ordering and ordering[-1].field != "stable_key":
            ordering += (OrderBy("stable_key", False),)
        query = QueryInput(QueryMode(request.mode), request.requested_count, tuple(request.output_types),
            text=request.text, image_handles=tuple(handles), combination=request.combination,
            predicates=tuple(Predicate(item.field, item.operator, item.value) for item in request.predicates),
            order_by=ordering)
        image_intent = (request.image_intent or "SOURCE") if len(handles) == 1 and not request.text else None
        receipt = await self.retrieval.retrieve(access, scope=scope, query=query,
            profile_ref=library["configuration"]["profile_ref"], direct=True, image_intent=image_intent)
        result = await self.present(access, receipt)
        if receipt.evidence_set.status == "EMPTY" or (
                receipt.evidence_set.status in {"COMPLETE", "PARTIAL"} and receipt.evidence_set.primary_hits):
            result["history_id"] = await self.search_history.save(access, library_id, request, receipt)
        return result

    async def history_detail(self, access, identity):
        record = await self.search_history.record(access, identity)
        entry = {key: value for key, value in record.items() if key != "manifest_ref"}
        entry["library_name"] = None
        response = dict(entry=entry, result=None, availability="UNAVAILABLE")
        try:
            name, evidence, partial = await self.search_history.replay(access, record)
            entry["library_name"] = name
            if partial and not evidence.primary_hits:
                return response

            async def reauthorize():
                entry["library_name"] = None
                entry["library_name"] = await self.search_history.reauthorize(access, record, evidence.evidence)

            result = await self.present(access, RetrievalReceipt(str(identity), evidence, None), reauthorize=reauthorize)
            result["history_id"] = identity
            return dict(entry=entry, result=result, availability="PARTIAL" if partial else "AVAILABLE")
        except FileNotFoundError:
            # 对象存储被管理员清理时，不能以当前召回填补原历史。
            return response
        except AlgorithmError as exc:
            if exc.code not in {ErrorCode.INVALID_CONTRACT, ErrorCode.SNAPSHOT_STALE, ErrorCode.SCOPE_CHANGED}:
                raise
            entry["library_name"] = None
            return response

    async def _history_hit(self, access, identity, stable_key):
        try:
            record = await self.search_history.record(access, identity)
            _, evidence, _ = await self.search_history.replay(access, record)
        except AlgorithmError as exc:
            if exc.code != ErrorCode.INVALID_CONTRACT:
                raise
            raise AlgorithmError(ErrorCode.SCOPE_CHANGED, "检索历史已过期、删除或不可读", stage="scope") from exc
        except FileNotFoundError as exc:
            raise AlgorithmError(ErrorCode.SNAPSHOT_STALE, "历史证据清单不存在", stage="snapshot") from exc
        hit = next((item for item in evidence.evidence if item.key.stable_key == stable_key), None)
        if hit is None:
            raise AlgorithmError(ErrorCode.SNAPSHOT_STALE, "历史证据不在首批结果中或已不可读", stage="snapshot")
        return record, evidence, hit

    async def history_content(self, access, identity, stable_key):
        record, _, hit = await self._history_hit(access, identity, stable_key)
        require(hit.byte_count <= self.retrieval.limits.max_payload_bytes, "内容读取超过预算")
        async def reauthorize():
            await self.search_history.reauthorize(access, record, (hit,))
        return hit, self.retrieval.content_blocks(hit, reauthorize)

    async def history_context(self, access, identity, stable_key, *, before, after):
        record, evidence, hit = await self._history_hit(access, identity, stable_key)
        # 同一段落使用首批中同来源的已授权命中进行定位与高亮。
        keys = {item.evidence_key for item in evidence.primary_hits} | {hit.key}
        items = tuple(item for item in evidence.evidence if item.key in keys
                      and item.key.document_version_ref == hit.key.document_version_ref)
        async def reauthorize():
            await self.search_history.reauthorize(access, record, items)
        return await self._passage_context(access, hit, items, record["conditions"].get("text"),
                                           before=before, after=after, reauthorize=reauthorize)

    async def history_pdf(self, access, identity, stable_key):
        record, _, hit = await self._history_hit(access, identity, stable_key)
        require(any(loc.page_no for loc in hit.locators), "当前检索证据没有原文页码")
        source, stream = await self.source(access, record["library_id"],
            UUID(hit.key.document_version_ref), published_only=True)
        require(source["mime_type"] == "application/pdf" and 0 < source["byte_count"] <= 67108864,
                "引用原文不支持 PDF 预览或超出大小上限")
        async def authorized_blocks():
            async with aclosing(stream):
                async for block in stream:
                    await self.search_history.reauthorize(access, record, (hit,))
                    yield block
        return source, authorized_blocks()

    async def present(self, access, receipt, *, relation_anchor=None, reauthorize=None):
        from knowledge_algorithms.contracts.identity import fingerprint
        from .knowledge_passages import PassageDocument, source_keys

        evidence = receipt.evidence_set
        primary_keys = {hit.evidence_key for hit in evidence.primary_hits}
        group_keys = primary_keys | ({relation_anchor} if relation_anchor is not None else set())
        groups = tuple(group for group in evidence.groups if group.anchor_key in group_keys)
        visible_keys = group_keys | {key for group in groups for key in group.linked_keys}
        visible = tuple(item for item in evidence.evidence if item.key in visible_keys)
        version_ids = {UUID(item.key.document_version_ref) for item in visible}
        anchors = {item.key: source_keys(item) for item in visible}
        async with self.uow_factory() as uow:
            versions = await uow.knowledge_retrieval.passage_versions(access, version_ids)
        items, documents, heading_cache = [], {}, {}

        async def open_document(version):
            ref = version["version_id"]
            if ref not in documents:
                documents[ref] = await PassageDocument.open(
                    self.retrieval.store, version, self.retrieval.limits.max_object_bytes)
            return documents[ref]

        displays, heading_origins = {}, {}
        for item in visible:
            if item.key not in primary_keys or not item.mime_type.startswith("text/"):
                continue
            origins = anchors[item.key]
            if len(origins) != 1:
                continue
            origin = origins[0]
            version = versions[origin.document_version_ref]
            if origin.parse_run_ref != version["parse_run_id"]:
                continue
            document = await open_document(version)
            display_key, display = await document.display(item, origins)
            displays[item.key] = (display_key, display)
            if display_key and display_key != origin.stable_key:
                origin = (await document.display_node(document.positions[display_key])).key
            heading_origins[item.key] = origin

        # 章节跟随完整段落的首节点；显示文字仍必须来自发布的允许视图。
        async with self.uow_factory() as uow:
            headings = await uow.knowledge_retrieval.passage_headings(access, set(heading_origins.values()))
        for item in visible:
            version = versions[item.key.document_version_ref]
            origins = anchors[item.key]
            chapter = None
            display_key, display = displays.get(item.key, (None, None))
            if (origin := heading_origins.get(item.key)) is not None:
                heading = headings.get((origin.parse_run_ref, origin.object_key))
                if heading:
                    heading_key = replace(origin, object_key=heading).stable_key
                    if heading_key not in heading_cache:
                        document = await open_document(version)
                        index = document.positions.get(heading_key)
                        heading_cache[heading_key] = ((await document.read(index)).text.strip()[:256] or None
                            if index is not None and document.view_ref == item.allowed_view_ref else None)
                    chapter = heading_cache[heading_key]
                    if display is not None and chapter:
                        display["chapter_key"] = heading_key
            items.append({"key": item.key.stable_key, "kind": item.kind.value, "version_id": item.key.document_version_ref,
                "source_title": version["title"], "document_id": version["document_id"],
                "passage_key": (display_key or fingerprint([key.stable_key for key in origins]))
                    if origins and item.mime_type.startswith("text/") else None,
                "passage_display": display,
                "chapter": chapter,
                "mime_type": item.mime_type, "byte_count": item.byte_count,
                "locators": encode(item.locators), "text": None,
                "content_ref": {"retrieval_ref": receipt.retrieval_ref, "key": item.key.stable_key}})
        preview_keys = primary_keys if relation_anchor is None else ()
        if reauthorize is None:
            previews = await self.retrieval.presentation_previews(access, receipt, preview_keys)
        else:
            previews = await self.retrieval.presentation_previews(access, receipt, preview_keys, reauthorize=reauthorize)
        previews = {key.stable_key: text for key, text in previews.items()}
        for value in items:
            value["text"] = previews.get(value["key"])
        return {"retrieval_ref": receipt.retrieval_ref, "status": evidence.status, "scope_exhausted": evidence.scope_exhausted,
            "reason": evidence.reason, "mode": evidence.mode.value,
            "continuation": evidence.continuation if evidence.mode.value == "EXHAUSTIVE" else None,
            "primary_hits": [{"key": hit.evidence_key.stable_key, "rank": hit.rank}
                for hit in evidence.primary_hits], "items": items,
            "linked": [{"anchor_key": group.anchor_key.stable_key,
                "linked_keys": [key.stable_key for key in group.linked_keys],
                "manifest_status": group.manifest_status, "continuation": group.continuation,
                "incomplete_reason": group.incomplete_reason,
                "relations": [{"relation_type": edge.relation_type.value,
                    "source_key": edge.source_key.stable_key, "target_key": edge.target_key.stable_key}
                    for edge in group.relations]} for group in groups],
            "channels": encode(evidence.channel_status),
            "model_delivery": "NOT_APPLICABLE"}

    async def passage_context(self, access, retrieval_ref, stable_key, *, before, after):
        row, scope, evidence = await self.retrieval.receipt(access, retrieval_ref)
        hit = next((item for item in evidence.evidence if item.key.stable_key == stable_key), None)
        require(hit is not None, "段落锚点不属于当前检索回执")
        primary_keys = {item.evidence_key for item in evidence.primary_hits} | {hit.key}
        async def reauthorize():
            await self.retrieval.validate(access, scope, expected_snapshot=evidence.snapshot)
        return await self._passage_context(access, hit, (item for item in evidence.evidence if item.key in primary_keys),
            row["plan"]["query"].get("text"), before=before, after=after, reauthorize=reauthorize)

    async def _passage_context(self, access, hit, items, text, *, before, after, reauthorize):
        from .knowledge_passages import PassageDocument, source_keys

        empty = {"blocks": [], "has_before": False, "has_after": False, "pdf_page": None}
        image_page = next((loc.page_no for loc in hit.locators if loc.page_no), None) \
            if hit.mime_type.startswith("image/") else None
        # 锚点固定在当前命中；相同段落的重复片段共用一份上下文。
        if image_page is None and not source_keys(hit):
            await reauthorize()
            return empty
        version_id = hit.key.document_version_ref
        async with self.uow_factory() as uow:
            versions = await uow.knowledge_retrieval.passage_versions(access, {UUID(version_id)})
        version = versions[version_id]
        if image_page is not None:
            # 图片命中只定位到原 PDF 页，不打开段落清单或猜测文字高亮。
            await reauthorize()
            return {**empty, "pdf_page": image_page if version["mime_type"] == "application/pdf" else None}
        document = await PassageDocument.open(self.retrieval.store, version, self.retrieval.limits.max_object_bytes)
        context = await document.context(hit, items, before=before, after=after, query_text=(text or "").strip())
        await reauthorize()
        return context or empty

    async def passage_pdf(self, access, retrieval_ref, stable_key):
        _, scope, evidence = await self.retrieval.receipt(access, retrieval_ref)
        hit = next((item for item in evidence.evidence if item.key.stable_key == stable_key), None)
        require(hit is not None and any(loc.page_no for loc in hit.locators), "当前检索证据没有原文页码")
        frozen, _ = await self.retrieval.validate(access, scope, expected_snapshot=evidence.snapshot)
        resource_ref = frozen["resource_ref"]
        require(resource_ref.startswith("library:"), "原文预览必须绑定可读的知识库")
        record, stream = await self.source(access, UUID(resource_ref.removeprefix("library:")),
            UUID(hit.key.document_version_ref), published_only=True)
        require(record["mime_type"] == "application/pdf", "当前检索原文不是 PDF")
        return record, stream

    async def historical_pdf(self, access, retrieval_ref, stable_key):
        hit, reauthorize = await self.retrieval.historical_evidence(
            access, retrieval_ref=retrieval_ref, stable_key=stable_key)
        resource_ref = await reauthorize()
        require(resource_ref.startswith("library:") and any(loc.page_no for loc in hit.locators),
                "此引用没有可读的资料库原文页码")
        record, stream = await self.source(access, UUID(resource_ref.removeprefix("library:")),
            UUID(hit.key.document_version_ref), published_only=True)
        require(record["mime_type"] == "application/pdf" and 0 < record["byte_count"] <= 67108864,
                "引用原文不支持 PDF 预览或超出大小上限")

        async def authorized_blocks():
            async with aclosing(stream):
                async for block in stream:
                    await reauthorize()
                    yield block

        return record, authorized_blocks()

    async def continue_search(self, access, retrieval_ref, cursor):
        return await self.present(access, await self.retrieval.continue_retrieval(
            access, retrieval_ref=retrieval_ref, cursor=cursor))

    async def continue_links(self, access, retrieval_ref, anchor_key, cursor):
        _, _, evidence = await self.retrieval.receipt(access, retrieval_ref)
        keys = [item.key for item in evidence.evidence if item.key.stable_key == anchor_key]
        require(len(keys) == 1, "关联锚点不属于当前检索回执")
        return await self.present(access, await self.retrieval.continue_relations(
            access, retrieval_ref=retrieval_ref, anchor_key=keys[0], cursor=cursor), relation_anchor=keys[0])
