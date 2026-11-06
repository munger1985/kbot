"""本 App 召回、可续取回执与有界内容读取；不自行生成答案。"""

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from uuid import UUID

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.identity import AuthorizedScope, RetrievalSnapshot, fingerprint
from knowledge_algorithms.contracts.retrieval import EvidenceSet
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.retrieval.engine import RetrievalEngine
from .knowledge_hydration import hydrate_published
from ..adapters.knowledge_backend import PublishedKnowledgeBackend
from ..domain.knowledge_ingestion import BINDING, utcnow
from ..domain.knowledge_retrieval import RetrievalLimits


@dataclass(frozen=True, slots=True)
class RetrievalReceipt:
    retrieval_ref: str
    evidence_set: EvidenceSet


class KnowledgeRetrievalService:
    def __init__(self, *, uow_factory, object_store, authorization, profiles, tokenizer,
                 image_cost, cursor_codec, model_gateway_factory=None, text_scan_factory=None, limits=None, clock=utcnow):
        require(profiles and all(name for name in profiles), "必须在组合根登记检索 Profile")
        self.uow_factory, self.store, self.authorization = uow_factory, object_store, authorization
        self.profiles, self.tokenizer, self.image_cost = dict(profiles), tokenizer, image_cost
        self.cursor_codec, self.model_gateway_factory = cursor_codec, model_gateway_factory
        self.text_scan_factory = text_scan_factory
        self.limits, self.clock = limits or RetrievalLimits(), clock

    async def grant(self, access, resource_ref):
        grant = await self.authorization.authorize(access, resource_ref)
        require(len(grant.version_ids) <= self.limits.max_versions, "授权文档数量超过固定后端预算")
        return grant

    async def create_scope(self, access, *, resource_ref):
        require(type(resource_ref) is str and 0 < len(resource_ref) <= 512, "业务资源引用无效")
        grant = await self.grant(access, resource_ref)
        require("READ" in grant.operations, "当前业务资源未授权知识读取")
        async with self.uow_factory() as uow:
            scope = await uow.knowledge_retrieval.create_scope(access, resource_ref=resource_ref,
                grant=grant, now=self.clock(), ttl_seconds=self.limits.scope_ttl_seconds)
            await uow.commit()
        return scope

    async def validate(self, access, scope, *, expected_snapshot=None):
        BINDING.check_owner(scope.owner_binding_ref)
        try:
            UUID(scope.scope_ref)
        except (ValueError, TypeError, AttributeError):
            require(False, "授权范围标识无效")
        delegated_hash = getattr(access, "delegated_scope_hash", None)
        require(delegated_hash is None or delegated_hash == scope.scope_hash, "知识委托与当前范围不一致")
        async with self.uow_factory() as uow:
            frozen = await uow.knowledge_retrieval.load_scope(access, scope, now=self.clock(),
                                                            max_versions=self.limits.max_versions)
        grant = await self.grant(access, frozen["resource_ref"])
        if "READ" not in grant.operations or grant.revision != scope.authorization_revision or not {
            UUID(ref) for ref, _ in frozen["members"]} <= set(grant.version_ids):
            raise AlgorithmError(ErrorCode.SCOPE_CHANGED, "当前业务授权已经改变", stage="scope")
        async with self.uow_factory() as uow:
            records = await uow.knowledge_retrieval.version_records(
                access, tuple(UUID(ref) for ref, _ in frozen["members"]))
        views = dict(frozen["members"])
        if any(record["status"] != "PUBLISHED" or record["document_status"] != "PUBLISHED"
               or record["generation_status"] != "READY"
               or record["allowed_view_hash"] != views[record["version_id"]] for record in records):
            raise AlgorithmError(ErrorCode.SNAPSHOT_STALE, "已发布内容撤回、换版或允许视图改变", stage="snapshot")
        revision = fingerprint(records)
        snapshot = RetrievalSnapshot(BINDING.binding_ref, scope.scope_hash, revision,
            fingerprint([(record["version_id"], record["generation_id"]) for record in records]),
            fingerprint([(record["version_id"], record["version_revision"], record["document_revision"])
                         for record in records]))
        if expected_snapshot is not None and snapshot != expected_snapshot:
            raise AlgorithmError(ErrorCode.SNAPSHOT_STALE, "索引快照已经改变", stage="snapshot")
        return frozen, snapshot

    async def backend(self, access, scope, profile_ref, *, cursor=None, expected_snapshot=None):
        require(profile_ref in self.profiles, "检索 Profile 未在组合根登记")
        frozen, snapshot = await self.validate(access, scope, expected_snapshot=expected_snapshot)
        async with self.uow_factory() as uow:
            bundle = await uow.knowledge_retrieval.published_bundle(access, scope, now=self.clock(), limits=self.limits)
        # 防止读取对象前后发生代次切换。
        await self.validate(access, scope, expected_snapshot=snapshot)
        index, items, relations = await hydrate_published(bundle, store=self.store, tokenizer=self.tokenizer,
            image_cost=self.image_cost, limits=self.limits)
        async def validate():
            await self.validate(access, scope, expected_snapshot=snapshot)
        profile = self.profiles[profile_ref]
        backend = PublishedKnowledgeBackend(index=index, items=items, relations=relations, scope=scope,
            snapshot=snapshot, subject_ref=access.subject_ref, expires_at=frozen["expires_at"],
            profile_hash=fingerprint(encode(profile)), cursor_codec=self.cursor_codec,
            validate=validate, clock=self.clock, primary_cursor=cursor)
        if self.text_scan_factory is not None:
            backend.text_scan, backend.scan_policy = self.text_scan_factory(backend)
        return backend, profile

    async def retrieve(self, access, *, scope, query, profile_ref, targets=(), cursor=None, expected_snapshot=None):
        backend, profile = await self.backend(access, scope, profile_ref, cursor=cursor,
                                               expected_snapshot=expected_snapshot)
        fingerprints = {}
        async def image_reader(handle):
            raw, descriptor = await self.query_image(access, scope, handle)
            return raw, descriptor["mime_type"]
        for handle in query.image_handles:
            _, descriptor = await self.query_image(access, scope, handle)
            fingerprints[handle] = (descriptor["sha256"], None)
        backend.image_fingerprints = fingerprints
        gateway = self.model_gateway_factory(access, image_reader=image_reader) if self.model_gateway_factory else None
        evidence = await RetrievalEngine(backend, gateway).retrieve(query, scope, profile, targets=targets)
        await self.validate(access, scope, expected_snapshot=evidence.snapshot)
        raw = json.dumps(encode(evidence), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        stored = await self.store.put_bytes(raw, max_bytes=self.limits.max_object_bytes)
        async with self.uow_factory() as uow:
            identity = await uow.knowledge_retrieval.save_receipt(access, scope, evidence, manifest_ref=stored.handle_ref,
                query=query, profile_ref=profile_ref, now=self.clock(), limits=self.limits)
            await uow.commit()
        return RetrievalReceipt(str(identity), evidence)

    async def receipt(self, access, retrieval_ref):
        require(type(retrieval_ref) is str and len(retrieval_ref) <= 36, "检索回执标识无效")
        try:
            identity = UUID(retrieval_ref)
        except ValueError:
            require(False, "检索回执标识无效")
        async with self.uow_factory() as uow:
            row = await uow.knowledge_retrieval.receipt(access, identity, now=self.clock())
        scope = decode(AuthorizedScope, row["plan"]["scope"])
        snapshot = decode(RetrievalSnapshot, row["snapshot"])
        await self.validate(access, scope, expected_snapshot=snapshot)
        raw = await self.store.read(row["manifest_ref"], max_bytes=self.limits.max_object_bytes)
        evidence = decode(EvidenceSet, json.loads(raw))
        require(evidence.snapshot == snapshot and evidence.scope_hash == scope.scope_hash,
                "检索回执与冻结清单不一致")
        return row, scope, evidence

    async def continue_retrieval(self, access, *, retrieval_ref, cursor):
        from knowledge_algorithms.contracts.query import QueryInput, QueryMode
        row, scope, evidence = await self.receipt(access, retrieval_ref)
        require(evidence.mode != QueryMode.RELEVANCE and cursor == evidence.continuation
                and cursor is not None, "续取必须使用本回执已声明的排序游标")
        return await self.retrieve(access, scope=scope, query=decode(QueryInput, row["plan"]["query"]),
            profile_ref=row["plan"]["profile_ref"], targets=evidence.plan.targets, cursor=cursor,
            expected_snapshot=evidence.snapshot)

    async def read_manifest(self, access, *, retrieval_ref, keys):
        require(0 < len(keys) <= self.limits.max_read_items and len(set(keys)) == len(keys), "内容读取数量无效")
        _, _, evidence = await self.receipt(access, retrieval_ref)
        items = {item.key: item for item in evidence.evidence}
        for key in keys:
            BINDING.check_owner(key.owner_binding_ref)
        return tuple({"evidence_key": encode(key), "status": "AVAILABLE" if key in items else "NOT_IN_RECEIPT",
            "item": encode(items[key]) if key in items else None} for key in keys)

    async def stream_content(self, access, *, retrieval_ref, stable_key):
        _, scope, evidence = await self.receipt(access, retrieval_ref)
        matches = [item for item in evidence.evidence if item.key.stable_key == stable_key]
        require(len(matches) == 1, "内容对象不在当前检索回执")
        item = matches[0]
        require(item.byte_count <= self.limits.max_payload_bytes and item.expires_at > int(self.clock().timestamp()),
                "内容读取超过预算或已过期")
        async def chunks():
            count, digest = 0, sha256()
            iterator = self.store.stream(item.content_handle)
            try:
                async for block in iterator:
                    await self.validate(access, scope, expected_snapshot=evidence.snapshot)
                    count += len(block)
                    require(count <= item.byte_count, "内容超过冻结字节数")
                    digest.update(block)
                    yield block
                require(count == item.byte_count and digest.hexdigest() == item.content_hash, "内容摘要或大小发生变化")
            finally:
                await iterator.aclose()
        return item, chunks()


    async def stream_historical_content(self, access, *, retrieval_ref, stable_key):
        """重新授权读取原证据；不重搜、不续期原 Scope、不以新版本替换历史引用。"""
        access = replace(access, historical_read=True)
        identity = UUID(retrieval_ref)
        async with self.uow_factory() as uow:
            manifest = await uow.knowledge_retrieval.historical_manifest(access, identity)
        raw = await self.store.read(manifest, max_bytes=self.limits.max_object_bytes)
        evidence = decode(EvidenceSet, json.loads(raw))
        matches = [item for item in evidence.evidence if item.key.stable_key == stable_key]
        require(len(matches) == 1 and matches[0].byte_count <= self.limits.max_payload_bytes,
                "历史对象不在原检索清单或超过读取预算")
        item = matches[0]
        async def reauthorize():
            BINDING.check_owner(item.key.owner_binding_ref)
            async with self.uow_factory() as uow:
                record = await uow.knowledge_retrieval.historical_record(access, identity, item.key, now=self.clock())
            require(record["scope_hash"] == evidence.scope_hash, "历史清单身份发生变化")
            grant = await self.grant(access, record["resource_ref"])
            require("READ" in grant.operations and UUID(item.key.document_version_ref) in grant.version_ids,
                    "当前业务权限不允许读取历史证据")
        await reauthorize()
        async def chunks():
            count, digest = 0, sha256()
            iterator = self.store.stream(item.content_handle)
            try:
                async for block in iterator:
                    await reauthorize()
                    count += len(block)
                    require(count <= item.byte_count, "历史对象超过冻结大小")
                    digest.update(block)
                    yield block
                require(count == item.byte_count and digest.hexdigest() == item.content_hash,
                        "历史证据内容摘要失效")
            finally:
                await iterator.aclose()
        return item, chunks()

    async def register_query_image(self, access, *, scope, raw, mime_type):
        from io import BytesIO
        import warnings
        from PIL import Image
        await self.validate(access, scope)
        require(type(raw) is bytes and 0 < len(raw) <= self.limits.max_object_bytes,
                "查询图片超过有界输入预算")
        formats = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
        require(mime_type in formats, "查询图片类型未登记")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(raw)) as image:
                    require(image.format == formats[mime_type] and image.width * image.height <= 25000000
                            and getattr(image, "n_frames", 1) == 1, "查询图片格式、像素或帧数无效")
                    image.verify()
        except (OSError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise AlgorithmError(ErrorCode.UNSUPPORTED_INPUT, "查询图片无效或解码超过预算", stage="query_image")
        copied = await self.store.put_bytes(raw, area="parsed", max_bytes=self.limits.max_object_bytes)
        descriptor = {"handle": copied.handle_ref, "sha256": sha256(raw).hexdigest(),
                      "mime_type": mime_type, "byte_count": len(raw)}
        async with self.uow_factory() as uow:
            identity = await uow.knowledge_retrieval.save_query_image(access, scope, descriptor=descriptor,
                now=self.clock(), limits=self.limits)
            await uow.commit()
        return "query-image-" + identity

    async def query_image(self, access, scope, image_handle):
        require(type(image_handle) is str and image_handle.startswith("query-image-"), "查询图片句柄无效")
        try:
            UUID(image_handle[12:])
        except ValueError:
            require(False, "查询图片句柄无效")
        async with self.uow_factory() as uow:
            descriptor = await uow.knowledge_retrieval.query_image(access, image_handle[12:], scope, now=self.clock())
        raw = await self.store.read(descriptor["handle"], max_bytes=self.limits.max_object_bytes)
        require(len(raw) == descriptor["byte_count"] and sha256(raw).hexdigest() == descriptor["sha256"],
                "查询图片不可变内容摘要失效")
        return raw, descriptor



    async def continue_relations(self, access, *, retrieval_ref, anchor_key, cursor):
        from dataclasses import replace
        from knowledge_algorithms.contracts.query import QueryInput
        from knowledge_algorithms.contracts.retrieval import RelationRequest
        from knowledge_algorithms.references.relations import direct_figure_manifest, LinkedManifest, RelationType
        from knowledge_algorithms.retrieval.selection import EvidenceGroup, PayloadUnit, select_evidence_groups
        row, scope, evidence = await self.receipt(access, retrieval_ref)
        group = next((group for group in evidence.groups if group.anchor_key == anchor_key), None)
        require(group is not None and cursor is not None and cursor == group.continuation,
                "关系续取必须使用本回执锚点的精确游标")
        backend, profile = await self.backend(access, scope, row["plan"]["profile_ref"],
                                                expected_snapshot=evidence.snapshot)
        items = {item.key: item for item in evidence.evidence}
        anchor = items[anchor_key]
        page = await backend.follow_relations(RelationRequest(anchor.source_keys, profile.relation_page_size, cursor),
                                               scope, evidence.snapshot)
        edges = {edge.relation_key: edge for edge in group.relations}
        for edge in page.relations:
            require(edge.relation_key not in edges or edges[edge.relation_key] == edge, "续取关系内容冲突")
            edges[edge.relation_key] = edge
        kind = ("FIGURE" if anchor.kind.value in {"FIGURE", "IMAGE", "PHOTO", "PAGE"} else
                "CAPTION" if anchor.kind.value == "CAPTION" else "REGION" if anchor.kind.value == "REGION" else "TEXT")
        linked, selected = [], {}
        for source_key in anchor.source_keys:
            eligible = tuple(edge for edge in edges.values() if not (
                kind == "TEXT" and edge.relation_type in {RelationType.REFERS_TO, RelationType.CITES_DOCUMENT} and edge.source_key == source_key
                and anchor.source_spans and not any(span.node_key == edge.source_span.node_key
                    and span.char_start <= edge.source_span.char_start < edge.source_span.char_end <= span.char_end
                    for span in anchor.source_spans)))
            manifest = direct_figure_manifest(source_key, anchor_kind=kind, relations=eligible,
                                               relation_pages_exhausted=page.exhausted)
            linked.extend(manifest.linked_keys)
            selected.update({edge.relation_key: edge for edge in manifest.relations})
        fetched = await backend.load_evidence(tuple(key for key in dict.fromkeys(linked) if key not in items),
                                                scope, evidence.snapshot)
        items.update({item.key: item for item in fetched.items})
        new_group = LinkedManifest(anchor_key, tuple(key for key in dict.fromkeys(linked) if key in items),
            tuple(selected.values()), "COMPLETE" if page.exhausted and not fetched.missing else "PARTIAL",
            continuation=page.continuation,
            incomplete_reason="EVIDENCE_UNAVAILABLE" if fetched.missing else None if page.exhausted else "RELATION_PAGE_BUDGET")
        groups = tuple(new_group if group.anchor_key == anchor_key else group for group in evidence.groups)
        selection = select_evidence_groups(tuple(EvidenceGroup(group.anchor_key.stable_key, group.anchor_key,
            tuple(PayloadUnit(key, items[key].estimated_cost) for key in (group.anchor_key,) + group.linked_keys),
            tuple(edge.relation_key for edge in group.relations)) for group in groups),
            targets=evidence.plan.targets, matches=evidence.matches,
            budget=profile.payload_budget, max_cover_states=profile.max_cover_states)
        updated = replace(evidence, request_id=fingerprint([evidence.request_id, "relations", anchor_key.stable_key, cursor]),
            evidence=tuple(items.values()), groups=groups, context_selection=selection)
        await self.validate(access, scope, expected_snapshot=updated.snapshot)
        raw = json.dumps(encode(updated), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        stored = await self.store.put_bytes(raw, max_bytes=self.limits.max_object_bytes)
        async with self.uow_factory() as uow:
            identity = await uow.knowledge_retrieval.save_receipt(access, scope, updated, manifest_ref=stored.handle_ref,
                query=decode(QueryInput, row["plan"]["query"]), profile_ref=row["plan"]["profile_ref"],
                now=self.clock(), limits=self.limits)
            await uow.commit()
        return RetrievalReceipt(str(identity), updated)

    async def retain_evidence(self, access, *, retrieval_ref, keys, expires_at):
        from datetime import datetime, timezone
        require(getattr(access, "run_ref", None) and getattr(access, "spec_hash", None),
                "保留证据必须绑定当前 Run/Spec")
        require(type(expires_at) is int and int(self.clock().timestamp()) < expires_at
                <= int(self.clock().timestamp()) + 2592000, "证据保留期限超出预算")
        require(0 < len(keys) <= self.limits.max_read_items and len(set(keys)) == len(keys), "保留证据数量无效")
        _, scope, evidence = await self.receipt(access, retrieval_ref)
        require(set(keys) <= {item.key for item in evidence.evidence}, "保留证据不在当前检索回执")
        async with self.uow_factory() as uow:
            await uow.knowledge_retrieval.load_scope(access, scope, now=self.clock(),
                max_versions=self.limits.max_versions, lock=True)
            identities = tuple([await uow.knowledge_ingestion.pin(access,
                owner_run_ref="run-spec:" + fingerprint([access.run_ref, access.spec_hash]),
                evidence_key=key, expires_at=datetime.fromtimestamp(expires_at, timezone.utc), now=self.clock()) for key in keys])
            await uow.commit()
        return tuple(str(identity) for identity in identities)

    async def release_evidence(self, access, *, pin_ids):
        require(getattr(access, "run_ref", None) and getattr(access, "spec_hash", None),
                "释放证据必须绑定当前 Run/Spec")
        require(0 < len(pin_ids) <= self.limits.max_read_items, "释放保留引用数量无效")
        async with self.uow_factory() as uow:
            for pin_id in pin_ids:
                await uow.knowledge_ingestion.release_pin(access, pin_id=UUID(pin_id),
                    owner_run_ref="run-spec:" + fingerprint([access.run_ref, access.spec_hash]), now=self.clock())
            await uow.commit()


    async def publish(self, access, *, resource_ref, version_id, generation_id, expected_generation_id=None):
        grant = await self.grant(access, resource_ref)
        require(version_id in grant.version_ids and "PUBLISH" in grant.operations, "当前业务授权不允许发布该版本")
        async with self.uow_factory() as uow:
            await uow.knowledge_retrieval.publish(access, version_id=version_id, generation_id=generation_id,
                expected_generation_id=expected_generation_id, now=self.clock())
            await uow.commit()

    async def withdraw(self, access, *, resource_ref, version_id, expected_revision):
        grant = await self.grant(access, resource_ref)
        require(version_id in grant.version_ids and "WITHDRAW" in grant.operations, "当前业务授权不允许撤回该版本")
        async with self.uow_factory() as uow:
            await uow.knowledge_retrieval.withdraw(access, version_id=version_id,
                                                  expected_revision=expected_revision, now=self.clock())
            await uow.commit()

    async def register_citation_target(self, access, *, resource_ref, target, identifiers):
        from knowledge_algorithms.references.citations import canonical_identifier
        grant = await self.grant(access, resource_ref)
        require("PUBLISH" in grant.operations and UUID(target.document_version_ref) in grant.version_ids,
                "文献标识登记需要目标文档发布权限")
        BINDING.check_owner(target.owner_binding_ref)
        require(type(identifiers) is tuple and 0 < len(identifiers) <= 32, "文献标识数量无效")
        try:
            normalized = tuple(dict.fromkeys(canonical_identifier(value) for value in identifiers))
        except ValueError:
            require(False, "文献标识必须为规范 DOI 或 URN")
        scope = await self.create_scope(access, resource_ref=resource_ref)
        backend, _ = await self.backend(access, scope, next(iter(self.profiles)))
        require(target in backend.items and target.projection_ref is None, "文献目标不在当前允许视图")
        async with self.uow_factory() as uow:
            await uow.knowledge_retrieval.register_citation_target(access,
                version_id=UUID(target.document_version_ref), target=target, identifiers=normalized)
            await uow.commit()

    async def release_run(self, access, *, scope):
        require(access.run_ref and access.spec_hash and access.delegated_scope_hash == scope.scope_hash,
                "释放运行缺少原始专用委托")
        async with self.uow_factory() as uow:
            await uow.knowledge_retrieval.revoke_run_scope(access, scope, now=self.clock())
            await uow.commit()
