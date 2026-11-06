"""知识检索 App 的正式资料组织、上传与审核发布用例。"""

from dataclasses import replace
from hashlib import sha256
from uuid import UUID
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.contracts import SourceDescriptor
from platform_core.identity import uuid7
from ..domain.knowledge_ingestion import OWNER
from ..domain.knowledge_retrieval import KnowledgeGrant


class LibraryAuthorization:
    def __init__(self, uow_factory):
        self.uow_factory = uow_factory

    async def authorize(self, access, resource_ref):
        require(resource_ref.startswith("library:"), "知识范围必须是本 App 资料库")
        library_id = UUID(resource_ref[8:])
        async with self.uow_factory() as uow:
            revision, versions, permissions = await uow.library(access).grant(library_id,
                include_unpublished=True)
        operations = {"READ"} | (permissions & {"PUBLISH", "WITHDRAW"})
        return KnowledgeGrant(revision, versions, frozenset(operations))


class KnowledgeLibraryService:
    def __init__(self, *, uow_factory, ingestion, retrieval, parse_profile, configurations):
        self.uow_factory, self.ingestion, self.retrieval = uow_factory, ingestion, retrieval
        self.parse_profile, self.configurations = parse_profile, dict(configurations)

    async def list(self, access):
        async with self.uow_factory() as uow:
            values = await uow.library(access).listing()
        return [self.library_view(value) for value in values]

    async def create(self, access, *, display_name, description, profile_ref):
        require(profile_ref in self.configurations, "资料库配置未在服务端登记")
        async with self.uow_factory() as uow:
            value = await uow.library(access).create(library_id=uuid7(), name=display_name,
                description=description, configuration={"profile_ref": profile_ref})
            await uow.commit()
        return self.library_view(value)

    async def get(self, access, library_id):
        async with self.uow_factory() as uow:
            value = await uow.library(access).detail(library_id)
        return self.library_view(value)

    def library_view(self, value):
        profile = self.retrieval.profiles[value["configuration"]["profile_ref"]]
        return {**value, "search_capabilities": [
            {"kind": channel.kind, "direction": channel.direction, "required": channel.required}
            for channel in profile.channels]}

    async def update(self, access, library_id, *, expected_revision, changes):
        if "configuration" in changes:
            require(set(changes["configuration"]) == {"profile_ref"}
                and changes["configuration"]["profile_ref"] in self.configurations, "资料库配置无效")
        async with self.uow_factory() as uow:
            value = await uow.library(access).change(library_id,
                expected_revision=expected_revision, changes=changes)
            await uow.commit()
        return self.library_view(value)

    async def files(self, access, library_id, *, page, page_size):
        async with self.uow_factory() as uow:
            return await uow.library(access).files(library_id, page=page, page_size=page_size)

    async def upload(self, access, library_id, *, name, mime_type, raw, idempotency_key):
        require(name and len(name) <= 512 and "/" not in name and "\\" not in name
                and 0 < len(raw) <= self.parse_profile.max_input_bytes, "文件名或上传大小无效")
        require(0 < len(idempotency_key) <= 128, "上传缺少幂等键")
        digest, document_id, version_id = sha256(raw).hexdigest(), uuid7(), uuid7()
        async with self.uow_factory() as uow:
            await uow.library(access).authorized(library_id, "UPLOAD")
        stored = await self.ingestion.store.put_bytes(raw, area="sources", max_bytes=self.parse_profile.max_input_bytes)
        source = SourceDescriptor(OWNER, str(version_id), digest, mime_type, len(raw), stored.handle_ref)
        async with self.uow_factory() as uow:
            library = uow.library(access)
            await library.authorized(library_id, "UPLOAD", lock=True)
            existing = await library.upload_receipt(library_id, idempotency_key)
            if existing:
                require(existing["sha256"] == digest and existing["name"] == name and existing["mime_type"] == mime_type,
                        "相同上传幂等键对应不同文件")
                return existing
            job_id = await uow.knowledge_ingestion.register(access, document_id=document_id,
                document_version_id=version_id, source=source, profile=self.parse_profile,
                business_asset_ref="library:" + str(library_id) + ":" + str(document_id),
                business_version_ref=idempotency_key, title=name, now=self.ingestion.clock())
            await library.govern(library_id, document_id, version_id, operation="UPLOAD",
                changes={"review": "PENDING", "uploaded_by": access.subject_ref, "name": name})
            await uow.commit()
        return {"document_id": str(document_id), "version_id": str(version_id), "job_id": str(job_id),
                "sha256": digest, "name": name, "mime_type": mime_type, "status": "RECEIVED"}

    async def review(self, access, library_id, version_id, *, decision, comment):
        require(decision in {"APPROVE", "REJECT"}, "审核决定无效")
        async with self.uow_factory() as uow:
            library = uow.library(access)
            record = await library.version(library_id, version_id, operation="REVIEW")
            require(record["governance"].get("review") == "PENDING", "版本已审核，不能覆盖审核记录")
            await library.govern(library_id, record["document_id"], version_id, operation="REVIEW",
                changes={"review": "APPROVED" if decision == "APPROVE" else "REJECTED",
                    "reviewed_by": access.subject_ref, "comment": comment})
            await uow.commit()
        return {"status": "APPROVED" if decision == "APPROVE" else "REJECTED"}

    async def publish(self, access, library_id, version_id):
        async with self.uow_factory() as uow:
            record = await uow.library(access).version(library_id, version_id, operation="PUBLISH")
        require(record["governance"].get("review") == "APPROVED" and record["generation_id"] is not None,
                "版本必须先通过审核并完成索引构建")
        await self.retrieval.publish(access, resource_ref="library:" + str(library_id),
            version_id=version_id, generation_id=record["generation_id"])
        return {"status": "PUBLISHED"}

    async def withdraw(self, access, library_id, version_id):
        async with self.uow_factory() as uow:
            record = await uow.library(access).version(library_id, version_id, operation="WITHDRAW")
        await self.retrieval.withdraw(access, resource_ref="library:" + str(library_id),
            version_id=version_id, expected_revision=record["revision"])
        return {"status": "WITHDRAWN"}

    async def cancel(self, access, library_id, version_id):
        async with self.uow_factory() as uow:
            record = await uow.library(access).version(library_id, version_id, operation="UPLOAD")
        require(record["governance"].get("uploaded_by") == access.subject_ref,
                "只能取消当前主体上传的未发布任务")
        require(record["status"] not in {"PUBLISHED", "WITHDRAWN"}, "发布版本不能当作解析任务取消")
        await self.ingestion.cancel(replace(access, subject_ref=record["subject_ref"]), record["job_id"])
        return {"status": "CANCELLED"}

    async def source(self, access, library_id, version_id):
        async with self.uow_factory() as uow:
            record = await uow.library(access).version(library_id, version_id, operation="READ")
        async def blocks():
            async for block in self.ingestion.store.stream(record["source_handle"]):
                async with self.uow_factory() as uow:
                    await uow.library(access).version(library_id, version_id, operation="READ")
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
        receipt = await self.retrieval.retrieve(access, scope=scope, query=query,
            profile_ref=library["configuration"]["profile_ref"])
        return await self.present(access, receipt)

    async def present(self, access, receipt):
        evidence = receipt.evidence_set
        items = []
        for item in evidence.evidence:
            preview = None
            if item.mime_type.startswith("text/") and item.byte_count <= 65536:
                _, stream = await self.retrieval.stream_content(access,
                    retrieval_ref=receipt.retrieval_ref, stable_key=item.key.stable_key)
                raw = b"".join([chunk async for chunk in stream])
                preview = raw.decode("utf-8")[:4000]
            items.append({"key": item.key.stable_key, "kind": item.kind.value,
                "mime_type": item.mime_type, "byte_count": item.byte_count,
                "locators": encode(item.locators), "text": preview,
                "content_ref": {"retrieval_ref": receipt.retrieval_ref, "key": item.key.stable_key}})
        return {"retrieval_ref": receipt.retrieval_ref, "status": evidence.status,
            "reason": evidence.reason, "mode": evidence.mode.value, "continuation": evidence.continuation,
            "primary_hits": [{"key": hit.evidence_key.stable_key, "rank": hit.rank}
                for hit in evidence.primary_hits], "items": items,
            "linked": [{"anchor_key": group.anchor_key.stable_key,
                "linked_keys": [key.stable_key for key in group.linked_keys],
                "manifest_status": group.manifest_status, "continuation": group.continuation,
                "incomplete_reason": group.incomplete_reason,
                "relations": [{"relation_type": edge.relation_type.value,
                    "source_key": edge.source_key.stable_key, "target_key": edge.target_key.stable_key}
                    for edge in group.relations]} for group in evidence.groups],
            "channels": encode(evidence.channel_status),
            "model_delivery": "NOT_APPLICABLE"}

    async def continue_search(self, access, retrieval_ref, cursor):
        return await self.present(access, await self.retrieval.continue_retrieval(
            access, retrieval_ref=retrieval_ref, cursor=cursor))

    async def continue_links(self, access, retrieval_ref, anchor_key, cursor):
        _, _, evidence = await self.retrieval.receipt(access, retrieval_ref)
        keys = [item.key for item in evidence.evidence if item.key.stable_key == anchor_key]
        require(len(keys) == 1, "关联锚点不属于当前检索回执")
        return await self.present(access, await self.retrieval.continue_relations(
            access, retrieval_ref=retrieval_ref, anchor_key=keys[0], cursor=cursor))
