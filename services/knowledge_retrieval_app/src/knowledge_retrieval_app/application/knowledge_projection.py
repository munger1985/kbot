"""App 允许视图与索引准备；所有文件和模型操作都在数据库事务外。"""

from dataclasses import replace
import asyncio
import json

from knowledge_algorithms.contracts.content import ContentKind, ContentNode, VisualAsset
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.parsing.events import ParseEventType
from knowledge_algorithms.parsing.wire import event_from_wire
from knowledge_algorithms.projections.contracts import AllowedContent, UnitSource
from knowledge_algorithms.projections.tables import markdown_table_spans
from knowledge_retrieval_app.domain.knowledge_ingestion import BINDING


class AppAllowedDocument:
    def __init__(self, lease, store, manifest):
        self.binding, self.store = BINDING, store
        self.document_version_ref = str(lease.document_version_id)
        self.parse_run_ref = lease.identity.parse_run_ref
        self.allowed_view_ref, self.allowed_view_hash = manifest["view_ref"], manifest["view_hash"]
        self.items = tuple(manifest["items"])

    async def contents(self):
        for item in self.items:
            raw = await self.store.read(item["handle"], max_bytes=item["max_bytes"])
            value = decode(AllowedContent, json.loads(raw))
            require(value.key.stable_key == item["key"], "允许视图对象身份发生变化")
            yield value


class KnowledgeProjectionService:
    def __init__(self, ingestion, *, max_manifest_bytes=8388608):
        require(max_manifest_bytes > 0, "允许视图清单预算无效")
        self.ingestion, self.max_manifest_bytes = ingestion, max_manifest_bytes

    async def _events(self, lease):
        after = 0
        while after < lease.sequence:
            async with self.ingestion.uow_factory() as uow:
                page = await uow.knowledge_ingestion.batches(lease, after_sequence=after,
                    limit=self.ingestion.page_limit)
                artifacts = await uow.knowledge_ingestion.batch_artifacts(lease, tuple(r.sequence for r in page))
            require(page, "规范解析结果缺页")
            for receipt in page:
                raw = await self.ingestion.store.read(receipt.payload_ref,
                    max_bytes=lease.profile.max_event_bytes * 2 + 8192)
                event = event_from_wire(json.loads(raw))
                require(event.event_digest == receipt.event_digest, "规范解析结果摘要失效")
                yield event, artifacts.get(receipt.sequence, {})
                after = receipt.sequence

    async def allowed_document(self, scope, job_id, *, policy, policy_revision):
        lease = await self.ingestion.get_job(scope, job_id)
        require(lease.status == "PARSED" and policy_revision, "只有完成解析才能生成允许视图")
        # 先验证整个终态清单，后构造允许视图；不把部分流当作完整文档。
        validator = await self.ingestion._validator(lease)
        require(validator.completed, "规范解析缺少完成清单")
        assets = {}
        async for event, copies in self._events(lease):
            if event.event_type == ParseEventType.ASSET_BATCH:
                for raw in json.loads(event.payload_json)["objects"]:
                    asset = decode(VisualAsset, {k: v for k, v in raw.items() if k != "local_key"})
                    require(asset.content_handle in copies, "原图没有已提交的 App 副本")
                    assets.setdefault(asset.figure_key.object_key, []).append(copies[asset.content_handle]["handle_ref"])
        items, manifest_size = [], 0
        headings = []
        async for event, _ in self._events(lease):
            if event.event_type != ParseEventType.NODE_BATCH:
                continue
            for raw in json.loads(event.payload_json)["objects"]:
                node = decode(ContentNode, {k: v for k, v in raw.items() if k != "local_key"})
                handles = assets.get(node.key.object_key, [])
                # 同一图形多个裁片不得无声选择其一；需 App 策略显式处理。
                require(len(handles) <= 1, "多原图对象需要明确的 App 允许视图策略")
                sources = (UnitSource(0, len(node.text), node.key, 0, len(node.text)),) if node.text else ()
                if node.kind == ContentKind.SECTION:
                    level = node.heading_level or 1
                    headings = [value for value in headings if value[0] < level]
                    headings.append((level, node.text))
                canonical = AllowedContent(node.key, node.kind, node.text, sources, node.locators,
                    node.provenance, heading_path=tuple(value[1] for value in headings),
                    content_handle=handles[0] if handles else None)
                content = await policy(canonical)
                if content is None:
                    continue
                # 脱敏后重新提取表格跨度，不能复用修改前的字符偏移。
                if content.kind == ContentKind.TABLE:
                    header, rows = markdown_table_spans(content.text)
                    content = replace(content, table_header=header, table_rows=rows)
                require(content.key == canonical.key and content.kind == canonical.kind
                        and content.provenance == canonical.provenance and content.locators == canonical.locators
                        and content.content_handle in {None, canonical.content_handle},
                        "允许视图不能伪造身份、定位、来源或原图句柄")
                position = 0
                for source in content.sources:
                    require(source.source_key == canonical.key and source.source_end <= len(node.text)
                            and content.text[source.unit_start:source.unit_end] ==
                                node.text[source.source_start:source.source_end], "允许视图原文映射不准确")
                    require(not content.text[position:source.unit_start].strip(), "生成文字不能混入原文")
                    position = source.unit_end
                require(not content.text[position:].strip(), "生成文字不能混入原文")
                encoded = json.dumps(encode(content), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                stored = await self.ingestion.store.put_bytes(encoded, max_bytes=lease.profile.max_event_bytes * 2 + 8192)
                item = {"key": content.key.stable_key, "handle": stored.handle_ref, "max_bytes": len(encoded)}
                manifest_size += len(json.dumps(item).encode()) + 1
                require(manifest_size <= self.max_manifest_bytes - 2048, "允许视图清单超过预算")
                items.append(item)
        require(items, "允许视图没有可用对象")
        view_hash = fingerprint({"run": encode(lease.identity), "policy_revision": policy_revision, "items": items})
        manifest = {"view_ref": "allowed-" + view_hash, "view_hash": view_hash,
                    "policy_revision": policy_revision, "items": items}
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        stored = await self.ingestion.store.put_bytes(encoded, max_bytes=self.max_manifest_bytes)
        async with self.ingestion.uow_factory() as uow:
            await uow.knowledge_ingestion.set_allowed_view(lease, manifest_ref=stored.handle_ref,
                view_hash=view_hash, now=self.ingestion.clock())
            await uow.commit()
        return AppAllowedDocument(lease, self.ingestion.store, manifest)

    async def prepare(self, scope, job_id, *, document, builder, profile):
        lease = await self.ingestion.get_job(scope, job_id)
        require(document.document_version_ref == str(lease.document_version_id)
                and document.parse_run_ref == lease.identity.parse_run_ref, "投影输入不属于当前任务")
        generation_id, count, digests = None, 0, []
        async for batch in builder.build(document, profile):
            wire = encode(batch)
            raw = json.dumps(wire, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            stored = await self.ingestion.store.put_bytes(raw, area="representations",
                max_bytes=lease.profile.max_event_bytes * 16)
            async with self.ingestion.uow_factory() as uow:
                if generation_id is None:
                    generation_id = await uow.knowledge_ingestion.prepare_generation(lease,
                        projection_ref=batch.projection_ref, manifest={"profile": encode(profile),
                            "allowed_view_hash": document.allowed_view_hash}, now=self.ingestion.clock())
                await uow.knowledge_ingestion.store_projection(lease, generation_id=generation_id,
                    batch=batch, payload_ref=stored.handle_ref, profile=profile, now=self.ingestion.clock())
                await uow.commit()
            count += len(batch.units)
            digests.append(batch.payload_hash)
            require(len(digests) <= profile.max_units, "投影批次超过预算")
        require(generation_id is not None, "投影未生成任何单元")
        async with self.ingestion.uow_factory() as uow:
            await uow.knowledge_ingestion.complete_generation(lease, generation_id=generation_id,
                unit_count=count, batch_digest=fingerprint(digests), batch_count=len(digests),
                now=self.ingestion.clock())
            await uow.commit()
        return generation_id

    async def process_outbox(self, scope, outbox_id, *, worker_id, lease_seconds, build):
        async with self.ingestion.uow_factory() as uow:
            claim = await uow.knowledge_ingestion.claim_outbox(scope, outbox_id, worker_id=worker_id,
                now=self.ingestion.clock(), lease_seconds=lease_seconds)
            await uow.commit()
        async def heartbeat():
            while True:
                await asyncio.sleep(lease_seconds / 3)
                async with self.ingestion.uow_factory() as uow:
                    await uow.knowledge_ingestion.renew_outbox(scope, outbox_id, worker_id=worker_id,
                        attempt=claim["attempt"], now=self.ingestion.clock(), lease_seconds=lease_seconds)
                    await uow.commit()
        # build 是 App 组合根固定的场景用例；续租失败立即取消计算，不写入终态。
        build_task = asyncio.create_task(build(claim["document_version_id"]))
        renew_task = asyncio.create_task(heartbeat())
        try:
            completed, _ = await asyncio.wait((build_task, renew_task), return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                task.result()
        except Exception:
            for task in (build_task, renew_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(build_task, renew_task, return_exceptions=True)
            async with self.ingestion.uow_factory() as uow:
                await uow.knowledge_ingestion.finish_outbox(scope, outbox_id, worker_id=worker_id,
                    attempt=claim["attempt"], now=self.ingestion.clock(), error_code="PROJECTION_FAILED")
                await uow.commit()
            raise
        finally:
            for task in (build_task, renew_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(build_task, renew_task, return_exceptions=True)
        async with self.ingestion.uow_factory() as uow:
            await uow.knowledge_ingestion.finish_outbox(scope, outbox_id, worker_id=worker_id,
                attempt=claim["attempt"], now=self.ingestion.clock())
            await uow.commit()
