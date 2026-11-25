"""运维业务与本 App 规范文档之间的用例端口；不模拟旧 KC/Bundle API。"""

from dataclasses import replace
from hashlib import sha256
import json
from uuid import UUID

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.parsing.contracts import ParseModelBinding, ParseStrategy, SourceDescriptor
from knowledge_algorithms.parsing.events import ParseEventType
from knowledge_algorithms.projections.contracts import ProjectionProfile
from knowledge_algorithms.projections.library_models import published_profile
from knowledge_algorithms.contracts.content import ContentNode
from platform_core.identity import uuid7
from platform_core.contracts.knowledge_parse_approval import available_parse_bindings, approval_parse_options, bindings_for_strategy
from ..domain.knowledge_ingestion import OWNER
from ..domain.knowledge_retrieval import RetrievalAccess
from .errors import AIOpsApplicationError


class OperationsDocuments:
    def __init__(self, *, uow_factory, ingestion, projection, retrieval, parse_profile, pipeline,
                 representation_bindings=None, resolve_parse_model=None, project_library=None):
        self.uow_factory, self.ingestion, self.projection = uow_factory, ingestion, projection
        self.retrieval, self.parse_profile, self.pipeline = retrieval, parse_profile, pipeline
        self._representation_bindings = dict(representation_bindings or {})
        self._resolve_parse_model = resolve_parse_model
        self._project_library = project_library

    async def parse_bindings(self, context, models):
        return await available_parse_bindings(models, (lambda role, model_id:
            self._resolve_parse_model(context, role, model_id)) if self._resolve_parse_model else None)

    async def parse_options(self, context, models):
        return approval_parse_options(await self.parse_bindings(context, models),
                                      executor_ready=self._resolve_parse_model is not None)

    async def approved_parse_profile(self, context, models, strategy):
        bindings = await self.parse_bindings(context, models) if strategy != "TEXT" else {}
        try:
            selected = bindings_for_strategy(strategy, bindings)
        except ValueError as exc:
            raise AIOpsApplicationError(code="AIOPS_PARSE_STRATEGY_UNAVAILABLE",
                                        message=str(exc), status_code=409) from exc
        return self.parse_profile.for_approval(ParseStrategy(strategy),
            **{role: ParseModelBinding(**binding) for role, binding in selected.items()})

    async def approve_parse(self, uow, access, indexes, version, profile):
        for index in indexes:
            await uow.knowledge_ingestion.approve_parse(access, index.ingestion_job_id,
                document_version_id=index.document_version_id, business_version_ref=version.asset_version_id,
                profile=profile, now=self.ingestion.clock())

    @staticmethod
    def access(context, *, actor_id=None, run_ref=None, spec_hash=None):
        require(context.app_id == OWNER, "运维知识只能由 AIOps 自身访问")
        tenant = UUID(str(context.tenant_id)) if context.tenant_id else None
        domain = UUID(str(context.domain_id)) if tenant else int(context.domain_id)
        return RetrievalAccess(domain, tenant, str(actor_id or getattr(context, "principal_id", None) or getattr(context, "asserted_user_id", None) or context.client_id),
                               run_ref=run_ref, spec_hash=spec_hash)

    def configuration(self, models):
        """模型不能只写配置而没有对应表示空间；禁止猜测向量维度或模型能力。"""
        from knowledge_algorithms.contracts.identity import fingerprint
        from knowledge_algorithms.contracts.wire import encode
        specs = self.pipeline.profile.representations
        requested = {str(value) for role, value in models.items()
                     if role in {"embedding", "visual_embedding"}}
        configured = {str(spec.model_ref) for spec in specs}
        if requested != configured:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_PROFILE_MISMATCH",
                message="所选向量模型与 AIOps 服务端表示空间不匹配，请由管理员配置模型、维度、版本和检索方向后重试",
                status_code=409,
            )
        return {"models": {key: str(value) for key, value in models.items()},
                "profile_ref": fingerprint(encode(self.pipeline.profile))}

    async def manual_processing_plan(self, context, parse_profile, models):
        require(self._project_library is not None, "运维资料库索引配置端口不可用")
        projection = await self._project_library(context, models)
        return self.processing_plan(parse_profile, projection=projection)

    def processing_plan(self, parse_profile=None, *, projection=None):
        """精确解析方案由已审批的资料版本冻结。"""
        retrieval = self.retrieval.profiles["default"]
        if projection is not None:
            retrieval = published_profile(retrieval, ({"generation_manifest": {"profile": encode(projection)}},))
        return {
            "parse": (parse_profile or self.parse_profile).profile_hash,
            **({"parse_strategy": parse_profile.strategy} if parse_profile else {}),
            "projection": fingerprint(encode(projection or self.pipeline.profile)),
            **({"projection_profile": encode(projection)} if projection is not None else {}),
            "retrieval": fingerprint(encode(retrieval)),
            "representations": fingerprint(encode(projection.representations) if projection is not None else self._representation_bindings),
        }

    def representation_model_references(self, model_id):
        return [
            {"service": "aiops-agent", "resource_type": "operations_knowledge_representation",
             "resource_id": spec.space_id, "display_name": "AIOps 已装配表示空间",
             "status": "CONFIGURED", "binding_role": "representation", "usage": "representation"}
            for spec in self.pipeline.profile.representations if str(spec.model_ref) == str(model_id)
        ]

    def assert_processing_plan(self, expected):
        projection = self.plan_projection(expected)
        live = self.processing_plan(projection=projection)
        if expected and expected.get("parse_strategy") in {strategy.value for strategy in ParseStrategy}:
            live = {**live, "parse": expected["parse"], "parse_strategy": expected["parse_strategy"]}
        if not expected or expected != live:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_PROCESSING_PLAN_MISMATCH",
                message="资料的处理方案缺失或已变化，需核验后再开始处理",
                status_code=409,
            )

    @staticmethod
    def plan_projection(plan):
        if plan and plan.get("projection_profile") is not None:
            return decode(ProjectionProfile, plan["projection_profile"])
        return None

    def capability_view(self, facts, *, parse_models=None, index_models=None):
        """仅用适用库的持久化证据声明已验证能力；装配本身只能证明已配置。"""
        profile = self.retrieval.profiles["default"]
        parse_models, index_models = parse_models or {}, index_models or {}
        semantic = {"TEXT_TO_TEXT": "PENDING_VERIFICATION"} if index_models.get("embedding") else {}
        return {
            "native_parse": (
                "AVAILABLE" if self.parse_profile.profile_hash in facts.get("verified_parse_profiles", ())
                else "PENDING_VERIFICATION"
            ),
            "ocr": "PENDING_VERIFICATION" if parse_models.get("ocr") else "NOT_ENABLED",
            "visual_enrichment": "PENDING_VERIFICATION" if parse_models.get("ocr") and parse_models.get("vlm") else "NOT_ENABLED",
            "lexical": (
                "PENDING_VERIFICATION" if any(channel.kind == "LEXICAL" for channel in profile.channels)
                else "NOT_ENABLED"
            ),
            "semantic_directions": [
                {"direction": direction, "state": state} for direction, state in sorted(semantic.items())
            ],
        }

    async def stage(self, *, file_name, media_type, body, expected_hash=None, document_id=None, version_id=None):
        require(type(body) is bytes and 0 < len(body) <= self.parse_profile.max_input_bytes,
                "运维知识文件大小超出解析预算")
        require(file_name and len(file_name) <= 512 and "/" not in file_name and "\\" not in file_name,
                "运维知识文件名无效")
        digest = sha256(body).hexdigest()
        require(expected_hash is None or expected_hash == digest, "运维知识上传正文哈希不一致")
        stored = await self.ingestion.store.put_bytes(body, area="sources",
                                                     max_bytes=self.parse_profile.max_input_bytes)
        document_id, version_id = document_id or uuid7(), version_id or uuid7()
        return {"document_id": document_id, "document_version_id": version_id,
            "source": SourceDescriptor(OWNER, str(version_id), digest, media_type, len(body), stored.handle_ref),
            "file_name": file_name}

    async def register(self, uow, access, *, staged, asset_id, asset_version_id, awaiting_approval=False):
        job_id = await uow.knowledge_ingestion.register(access, document_id=staged["document_id"],
            document_version_id=staged["document_version_id"], source=staged["source"],
            profile=self.parse_profile, business_asset_ref="ops-asset:" + str(asset_id),
            business_version_ref=str(asset_version_id), title=staged["file_name"], now=self.ingestion.clock(),
            awaiting_approval=awaiting_approval)
        return {"document_id": staged["document_id"], "document_version_id": staged["document_version_id"],
            "ingestion_job_id": job_id}

    async def retry(self, access, asset_version_id):
        async with self.uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(domain_id=access.domain_id,
                asset_version_id=asset_version_id, for_update=True)
            require(version is not None and version.status in {"PROCESSING", "TAGGING"}, "业务版本不能重试")
            relations = await uow.operations_knowledge.get_version_relations(asset_version_id=asset_version_id)
            await self.retry_documents(uow, access, relations["indexes"])
            await uow.commit()

    async def retry_documents(self, uow, access, indexes):
        """重置可重试的解析任务，事务由调用方管理。"""
        now = self.ingestion.clock()
        for index in indexes:
            await uow.knowledge_documents.retry_document(access, index.document_version_id, now=now)

    async def rebind_pending_parse(self, uow, access, document_version_id, profile):
        return await uow.knowledge_documents.retry_document(
            access, document_version_id, now=self.ingestion.clock(), pending_profile=profile,
        )

    async def inspect(self, access, version_id):
        async with self.uow_factory() as uow:
            return await uow.knowledge_documents.inspect(access, UUID(str(version_id)))

    async def process(self, access, version_id, *, expected_plan=None):
        if expected_plan is not None:
            self.assert_processing_plan(expected_plan)
        record = await self.inspect(access, version_id)
        require(record["status"] != "WITHDRAWN", "已撤回文档不能继续处理")
        if record["processing_status"] == "PARSED" and record["generation_id"]:
            if expected_plan is not None and (
                record["projection_profile_hash"] != expected_plan["projection"]
                or record["parse_profile_hash"] != expected_plan["parse"]
            ):
                raise AIOpsApplicationError(
                    code="AIOPS_KNOWLEDGE_PROCESSING_PLAN_MISMATCH",
                    message="已生成的索引与登记时的表示方案不一致",
                    status_code=409,
                )
            return record
        subject = replace(access, subject_ref=record["subject_ref"])
        lease = await self.ingestion.get_job(subject, UUID(record["ingestion_job_id"]))
        if expected_plan is not None and lease.profile.profile_hash != expected_plan["parse"]:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_PROCESSING_PLAN_MISMATCH",
                message="解析任务与登记时的解析方案不一致",
                status_code=409,
            )
        require(self.pipeline is not None, "运维知识流水线尚未装配")
        if (lease.status == "PARSING" and lease.lease_until > self.ingestion.clock()
            and lease.lease_owner != self.pipeline.worker_id):
            return record
        projection = self.plan_projection(expected_plan)
        kwargs = {"projection_profile": projection} if projection is not None else {}
        result = await self.pipeline.process(subject, UUID(record["ingestion_job_id"]), **kwargs)
        current = await self.inspect(access, version_id)
        if expected_plan is not None and (
            current["generation_id"] != result["generation_id"]
            or current["projection_profile_hash"] != expected_plan["projection"]
            or current["parse_profile_hash"] != expected_plan["parse"]
        ):
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_PROCESSING_PLAN_MISMATCH",
                message="索引代次或表示方案与本次处理不一致",
                status_code=409,
            )
        return {**current, "generation_id": result["generation_id"]}

    async def excerpts(self, access, version_ids, *, max_chars=100000):
        require(0 < max_chars <= 1000000 and len(version_ids) <= 128, "运维知识预览预算无效")
        excerpts, used = [], 0
        for version_id in version_ids:
            record = await self.inspect(access, version_id)
            require(record["processing_status"] == "PARSED", "文档解析尚未完成")
            subject = replace(access, subject_ref=record["subject_ref"])
            lease = await self.ingestion.get_job(subject, UUID(record["ingestion_job_id"]))
            validator = await self.ingestion._validator(lease)
            require(validator.completed, "文档没有通过解析终态验证")
            async for event, _ in self.projection._events(lease):
                if event.event_type != ParseEventType.NODE_BATCH:
                    continue
                for raw in json.loads(event.payload_json)["objects"]:
                    node = decode(ContentNode, {key: value for key, value in raw.items() if key != "local_key"})
                    if not node.text:
                        continue
                    if used + len(node.text) > max_chars:
                        return {"excerpts": excerpts, "warnings": ["正文预览达到预算，未展示全部内容"]}
                    used += len(node.text)
                    excerpts.append({"document_version_id": str(version_id), "file_name": record["title"],
                        "content": node.text, "locator": {"locators": raw["locators"], "key": node.key.stable_key}})
        return {"excerpts": excerpts, "warnings": []}

    async def assert_processed_documents(self, uow, access, indexes, processed, *, expected_plan):
        """接纳标签前，复核用于生成证据的文档、解析运行及投影没有被替换。"""
        self.assert_processing_plan(expected_plan)
        for index in indexes:
            current = await uow.knowledge_documents.inspect(access, index.document_version_id)
            original = processed[index.document_version_id]
            require(current["status"] != "WITHDRAWN" and current["processing_status"] == "PARSED"
                    and current["document_id"] == str(index.document_id)
                    and current["ingestion_job_id"] == str(index.ingestion_job_id)
                    and current["parse_profile_hash"] == expected_plan["parse"]
                    and current["projection_profile_hash"] == expected_plan["projection"]
                    and all(current[key] == original[key] for key in
                            ("document_id", "ingestion_job_id", "parse_run_id", "generation_id", "source_hash")),
                    "标签处理期间文档任务或索引代次已变化")

    async def publish(self, uow, access, indexes):
        require(indexes, "运维知识版本没有文档引用")
        for index in indexes:
            record = await uow.knowledge_documents.inspect(access, index.document_version_id)
            require(record["document_id"] == str(index.document_id)
                and record["ingestion_job_id"] == str(index.ingestion_job_id)
                and record["parse_run_id"] == str(index.parse_run_id)
                and record["generation_id"] == str(index.index_generation_id), "文档任务或索引代次已经变化")
            await uow.knowledge_retrieval.publish(access, version_id=index.document_version_id,
                generation_id=index.index_generation_id,
                expected_generation_id=UUID(record["active_generation_id"]) if record["active_generation_id"] else None,
                now=self.ingestion.clock())

    async def withdraw(self, uow, access, indexes, *, cancel=False):
        leases = []
        for index in indexes:
            record = await uow.knowledge_documents.inspect(access, index.document_version_id)
            if cancel and record["processing_status"] not in {"PARSED", "FAILED", "CANCELLED"}:
                lease = await uow.knowledge_ingestion.get_job(replace(access, subject_ref=record["subject_ref"]), index.ingestion_job_id)
                await uow.knowledge_ingestion.cancel(lease, now=self.ingestion.clock())
                leases.append(lease)
            if record["status"] != "WITHDRAWN":
                await uow.knowledge_retrieval.withdraw(access, version_id=index.document_version_id,
                    expected_revision=record["revision"], now=self.ingestion.clock())
        return leases

    async def source(self, access, version_id, *, authorize):
        record = await self.inspect(access, version_id)
        async def chunks():
            await authorize()
            digest, count = sha256(), 0
            async for block in self.ingestion.store.stream(record["source_handle"]):
                await authorize()
                count += len(block)
                require(count <= record["byte_count"], "原文件超过登记大小")
                digest.update(block)
                yield block
            require(count == record["byte_count"] and digest.hexdigest() == record["source_hash"],
                    "运维知识原文件摘要发生变化")
        return record, chunks()
