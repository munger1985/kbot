"""App 允许视图与索引准备；所有文件和模型操作都在数据库事务外。"""

from contextlib import aclosing
from dataclasses import replace
import asyncio
import json

from knowledge_algorithms.contracts.content import ContentKind, ContentNode, VisualAsset
from knowledge_algorithms.contracts.errors import AlgorithmError, require
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.parsing.events import ParseEventType
from knowledge_algorithms.parsing.wire import event_from_wire
from knowledge_algorithms.execution import Deadline
from knowledge_algorithms.projections.builder import consume_batch_bytes, uses_representation
from knowledge_algorithms.projections.splitting import REVISION as SPLIT_REVISION
from knowledge_algorithms.projections.contracts import (AllowedContent, ProjectionBatch,
    ProjectionProfile, UnitSource)
from uuid import UUID
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
        async with self.ingestion.uow_factory() as uow:
            checkpoint = await uow.knowledge_ingestion.allowed_view_checkpoint(lease)
        if checkpoint is not None:
            return await self._read_allowed_document(lease, *checkpoint, policy_revision=policy_revision)
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

    async def _read_allowed_document(self, lease, ref, expected_hash, *, policy_revision=None):
        require(bool(ref) and bool(expected_hash), "索引来源缺少冻结允许视图")
        stored = await self.ingestion.store.read(ref, max_bytes=self.max_manifest_bytes)
        manifest = json.loads(stored)
        require(manifest["view_hash"] == expected_hash
                and manifest["view_ref"] == "allowed-" + expected_hash
                and (policy_revision is None or manifest["policy_revision"] == policy_revision),
                "索引来源与冻结允许视图不一致")
        return AppAllowedDocument(lease, self.ingestion.store, manifest)

    async def _read_committed_batch(self, saved, sequence, projection_ref, *, limit, deadline, stage):
        raw = await deadline.run(self.ingestion.store.read(saved["ref"], max_bytes=limit), stage=stage)
        batch = decode(ProjectionBatch, json.loads(raw))
        require(batch.sequence == sequence and (projection_ref is None or batch.projection_ref == projection_ref)
                and len(batch.units) == saved["count"] and batch.payload_hash == saved["hash"]
                and batch.payload_hash == fingerprint(encode(batch.units)), "已提交索引批次摘要、身份或数量变化")
        return batch

    async def prepare(self, scope, job_id, *, document, builder, profile, fence=None):
        lease = await self.ingestion.get_job(scope, job_id)
        require(document.document_version_ref == str(lease.document_version_id)
                and document.parse_run_ref == lease.identity.parse_run_ref, "投影输入不属于当前任务")
        projection_ref = builder.projection_ref(document, profile)
        manifest = {"profile": encode(profile), "allowed_view_hash": document.allowed_view_hash,
                    "split_revision": SPLIT_REVISION}
        limit = lease.profile.max_event_bytes * 16
        async with self.ingestion.uow_factory() as uow:
            checkpoint = await uow.knowledge_ingestion.initial_projection_checkpoint(lease, projection_ref)
        generation_id, committed, planned = None, {}, None
        if checkpoint is not None:
            generation_id, status, stored = checkpoint
            require(all(stored.get(key) == value for key, value in manifest.items()), "初始索引恢复点与当前 Profile 不一致")
            if status == "READY":
                return generation_id
            committed = stored.get("batches", {})
            planned = stored.get("planned_batch_count")
        if planned is None:
            planned = await builder.plan_batch_count(document, profile, max_batch_bytes=limit - 512)
            require(planned > 0, "投影未生成任何单元")
            async with self.ingestion.uow_factory() as uow:
                if fence is not None:
                    await uow.knowledge_ingestion.assert_outbox_claim(lease.scope, **fence, now=self.ingestion.clock())
                generation_id = await uow.knowledge_ingestion.prepare_generation(lease, projection_ref=projection_ref,
                    manifest=manifest, planned_batch_count=planned, now=self.ingestion.clock())
                await uow.commit()
        async def load_saved(sequence, deadline):
            item = committed.get(str(sequence))
            if item is not None:
                return await self._read_committed_batch(item, sequence, projection_ref,
                    limit=limit, deadline=deadline, stage="index_resume")
            return None

        batches = builder.build(document, profile, load_saved=load_saved, max_batch_bytes=limit - 512)
        return await self._write_batches(lease, batches, profile, manifest,
                                         generation_id=generation_id, fence=fence, committed=committed)

    async def rebuild(self, lease, candidate_id, *, builder, fence):
        async with self.ingestion.uow_factory() as uow:
            status, candidate = await uow.knowledge_ingestion.projection_manifest(lease, candidate_id)
            source_status, source = await uow.knowledge_ingestion.projection_manifest(
                lease, UUID(candidate["source_generation_id"]))
        require(source_status == "READY" and source["allowed_view_hash"] == candidate["allowed_view_hash"],
                "重建来源不再是完整的发布索引")
        if status in {"READY", "CANDIDATE"}:
            return candidate_id
        profile = decode(ProjectionProfile, candidate["profile"])
        limit = lease.profile.max_event_bytes * 16
        committed = candidate.get("batches", {})
        previous = decode(ProjectionProfile, source["profile"])
        if candidate.get("resplit"):
            require(candidate.get("split_revision") == SPLIT_REVISION, "索引候选切片规则已变化")
            async with self.ingestion.uow_factory() as uow:
                checkpoint = await uow.knowledge_ingestion.allowed_view_checkpoint(lease)
            require(checkpoint is not None and checkpoint[1] == candidate["allowed_view_hash"],
                    "重建来源的允许视图已变化")
            document = await self._read_allowed_document(lease, *checkpoint)
            projection_ref = candidate["projection_ref"]
            if candidate.get("planned_batch_count") is None:
                planned = await builder.plan_batch_count(document, profile, max_batch_bytes=limit - 512)
                require(planned > 0, "投影未生成任何单元")
                async with self.ingestion.uow_factory() as uow:
                    await uow.knowledge_ingestion.assert_outbox_claim(lease.scope, **fence, now=self.ingestion.clock())
                    prepared = await uow.knowledge_ingestion.prepare_generation(lease, projection_ref=projection_ref,
                        manifest=candidate, planned_batch_count=planned, now=self.ingestion.clock())
                    require(prepared == candidate_id, "重建候选身份已变化")
                    await uow.commit()

            async def load_saved(sequence, deadline):
                item = committed.get(str(sequence))
                if item is not None:
                    return await self._read_committed_batch(item, sequence, projection_ref, limit=limit,
                                                           deadline=deadline, stage="index_rebuild_resume")
                return None

            # 仅批次安排或部分模型变化时，对完全相同的模型输入与来源复用未改变的表示。
            may_reuse = (not candidate.get("file_candidate") and source.get("split_revision") == SPLIT_REVISION
                         and previous.max_tokens == profile.max_tokens
                         and bool(set(previous.representations) & set(profile.representations)))
            async def prior_units():
                if may_reuse:
                    for sequence in range(1, source["batch_count"] + 1):
                        yield source["batches"][str(sequence)], sequence

            async with aclosing(prior_units()) as records:
                pending = iter(())
                async def reuse(unit, deadline):
                    nonlocal pending
                    while True:
                        item = next(pending, None)
                        if item is None:
                            record = await anext(records, None)
                            if record is None:
                                return None
                            batch = await self._read_committed_batch(*record, None, limit=limit,
                                deadline=deadline, stage="index_rebuild_source")
                            pending = iter(batch.units)
                            continue
                        old = item.unit
                        if (old.source_key, old.sources, old.text, old.context, old.content_handle) == (
                            unit.source_key, unit.sources, unit.text, unit.context, unit.content_handle,
                        ):
                            return item

                batches = builder.build(document, profile, load_saved=load_saved, max_batch_bytes=limit - 512,
                                        projection_ref=projection_ref, reuse_representations=reuse if may_reuse else None)
                return await self._write_batches(lease, batches, profile, candidate,
                                                 generation_id=candidate_id, fence=fence, committed=committed)
        prior = {spec.space_id: spec for spec in previous.representations}
        async def batches():
            for sequence in range(1, source["batch_count"] + 1):
                deadline = Deadline(profile.timeout_seconds)
                item = source["batches"][str(sequence)]
                saved = committed.get(str(sequence))
                if saved is not None:
                    yield await self._read_committed_batch(saved, sequence, candidate["projection_ref"],
                        limit=limit, deadline=deadline, stage="index_rebuild_resume")
                    continue
                batch = await self._read_committed_batch(item, sequence, None,
                    limit=limit, deadline=deadline, stage="index_rebuild_read")
                units = []
                remaining = limit - 512
                for projected in batch.units:
                    await deadline.checkpoint(stage="index_rebuild")
                    unit = projected.unit
                    key = fingerprint({"source_unit": unit.unit_key, "projection": candidate["projection_ref"]})
                    reusable = None
                    if not candidate.get("file_candidate"):
                        available = {value.space_id for value in projected.representations}
                        available.update(projected.skipped_image_spaces or ())
                        require(all(spec.space_id in available for spec in profile.representations
                            if prior.get(spec.space_id) == spec and uses_representation(unit, spec)),
                            "已发布索引缺少可复用的表示或已记录的单图输入缺口")
                        # 复用只允许来源、输入和空间均一致的表示及缺口。
                        reusable = projected
                    unit = replace(unit, unit_key=key,
                        key=replace(unit.key, projection_ref=candidate["projection_ref"], object_key=key))
                    result = await builder.represent_unit(unit, profile, deadline, reusable)
                    remaining = consume_batch_bytes(result, remaining)
                    units.append(result)
                yield ProjectionBatch(candidate["projection_ref"], sequence, tuple(units),
                                      fingerprint(encode(tuple(units))))
        return await self._write_batches(lease, batches(), profile, candidate,
                                         generation_id=candidate_id, fence=fence, committed=committed)

    async def _write_batches(self, lease, batches, profile, manifest, *, generation_id, fence=None, committed=None):
        async def verify(uow):
            if fence is not None:
                await uow.knowledge_ingestion.assert_outbox_claim(lease.scope, **fence, now=self.ingestion.clock())
        async def write(batch):
            wire = encode(batch)
            raw = json.dumps(wire, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            stored = await self.ingestion.store.put_bytes(raw, area="representations",
                max_bytes=lease.profile.max_event_bytes * 16)
            async with self.ingestion.uow_factory() as uow:
                await verify(uow)
                await uow.knowledge_ingestion.store_projection(lease, generation_id=generation_id,
                    batch=batch, payload_ref=stored.handle_ref, profile=profile, now=self.ingestion.clock())
                await uow.commit()

        count, digests, has_images = 0, [], False
        committed = committed or {}
        async with aclosing(batches):
            async for batch in batches:
                if str(batch.sequence) not in committed:
                    await Deadline(profile.timeout_seconds).run(write(batch), stage="index_commit")
                count += len(batch.units)
                has_images = has_images or any(unit.unit.content_handle for unit in batch.units)
                digests.append(batch.payload_hash)
                require(len(digests) <= profile.max_units, "投影批次超过预算")
        async def complete():
            async with self.ingestion.uow_factory() as uow:
                await verify(uow)
                await uow.knowledge_ingestion.complete_generation(lease, generation_id=generation_id,
                    unit_count=count, batch_digest=fingerprint(digests), batch_count=len(digests),
                    now=self.ingestion.clock(), has_images=has_images, candidate_only=bool(manifest.get("file_candidate")))
                await uow.commit()
        await Deadline(profile.timeout_seconds).run(complete(), stage="index_complete")
        return generation_id

    async def process_outbox(self, scope, outbox_id, *, worker_id, lease_seconds, build, pass_claim=False, on_success=None):
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
        build_task = asyncio.create_task(build(claim if pass_claim else claim["document_version_id"]))
        renew_task = asyncio.create_task(heartbeat())
        try:
            completed, _ = await asyncio.wait((build_task, renew_task), return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                task.result()
            # 先锁库，索引消息交付与自动发布同一事务，失败一起回滚。
            async with self.ingestion.uow_factory() as uow:
                after_delivery = await on_success(uow, claim) if on_success is not None else None
                if after_delivery is False:
                    await uow.knowledge_ingestion.defer_outbox(scope, outbox_id, worker_id=worker_id,
                        attempt=claim["attempt"], now=self.ingestion.clock())
                    await uow.commit()
                    return
                await uow.knowledge_ingestion.finish_outbox(scope, outbox_id, worker_id=worker_id,
                    attempt=claim["attempt"], now=self.ingestion.clock())
                if after_delivery is not None:
                    await after_delivery()
                await uow.commit()
        except Exception as exc:
            for task in (build_task, renew_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(build_task, renew_task, return_exceptions=True)
            async with self.ingestion.uow_factory() as uow:
                await uow.knowledge_ingestion.finish_outbox(scope, outbox_id, worker_id=worker_id,
                    attempt=claim["attempt"], now=self.ingestion.clock(),
                    error_code=exc.code.value if isinstance(exc, AlgorithmError) else "PROJECTION_FAILED")
                await uow.commit()
            raise
        finally:
            for task in (build_task, renew_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(build_task, renew_task, return_exceptions=True)
