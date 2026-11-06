"""知识摄取聚合仓储；所有 SQL 和实体访问留在 App 持久化边界。"""

from datetime import timedelta
from uuid import UUID

from sqlalchemy import update

from knowledge_algorithms.contracts.content import ContentNode, Page, VisualAsset
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.parsing.contracts import ParseProfile, SourceDescriptor
from knowledge_algorithms.parsing.events import ParseEventType, ParseRunIdentity, local_key
from knowledge_algorithms.references.figures import FigureResolution
from knowledge_algorithms.references.relations import ResolvedRelation

from knowledge_retrieval_app.domain.knowledge_ingestion import BINDING, BatchReceipt, JobLease
from .knowledge_document import KnowledgeDocumentRepository
from .knowledge_document_version import KnowledgeDocumentVersionRepository
from .knowledge_parse_run import KnowledgeParseRunRepository
from .knowledge_parse_batch import KnowledgeParseBatchRepository
from .knowledge_node import KnowledgeNodeRepository
from .knowledge_visual_asset import KnowledgeVisualAssetRepository
from .knowledge_reference_occurrence import KnowledgeReferenceOccurrenceRepository
from .knowledge_relation import KnowledgeRelationRepository
from .knowledge_search_unit import KnowledgeSearchUnitRepository
from .knowledge_unit_source import KnowledgeUnitSourceRepository
from .knowledge_representation import KnowledgeRepresentationRepository
from .knowledge_index_generation import KnowledgeIndexGenerationRepository
from .knowledge_ingestion_job import KnowledgeIngestionJobRepository
from .knowledge_outbox import KnowledgeOutboxRepository
from .knowledge_retrieval_receipt import KnowledgeRetrievalReceiptRepository
from .knowledge_evidence_pin import KnowledgeEvidencePinRepository
from .knowledge_scope_snapshot import KnowledgeScopeSnapshotRepository
from .knowledge_scope_member import KnowledgeScopeMemberRepository


class KnowledgeIngestionRepository:
    def __init__(self, session, guard=lambda: None):
        self._session, self._guard = session, guard

    def _repos(self, scope):
        self._guard()
        require(type(scope.domain_id) is int and scope.tenant_id is None, "Oracle 摄取范围必须为可信 Domain")
        return {"knowledge_document": KnowledgeDocumentRepository(self._session, scope),
            "knowledge_document_version": KnowledgeDocumentVersionRepository(self._session, scope),
            "knowledge_parse_run": KnowledgeParseRunRepository(self._session, scope),
            "knowledge_parse_batch": KnowledgeParseBatchRepository(self._session, scope),
            "knowledge_node": KnowledgeNodeRepository(self._session, scope),
            "knowledge_visual_asset": KnowledgeVisualAssetRepository(self._session, scope),
            "knowledge_reference_occurrence": KnowledgeReferenceOccurrenceRepository(self._session, scope),
            "knowledge_relation": KnowledgeRelationRepository(self._session, scope),
            "knowledge_search_unit": KnowledgeSearchUnitRepository(self._session, scope),
            "knowledge_unit_source": KnowledgeUnitSourceRepository(self._session, scope),
            "knowledge_representation": KnowledgeRepresentationRepository(self._session, scope),
            "knowledge_index_generation": KnowledgeIndexGenerationRepository(self._session, scope),
            "knowledge_ingestion_job": KnowledgeIngestionJobRepository(self._session, scope),
            "knowledge_outbox": KnowledgeOutboxRepository(self._session, scope),
            "knowledge_retrieval_receipt": KnowledgeRetrievalReceiptRepository(self._session, scope),
            "knowledge_evidence_pin": KnowledgeEvidencePinRepository(self._session, scope),
            "knowledge_scope_snapshot": KnowledgeScopeSnapshotRepository(self._session, scope),
            "knowledge_scope_member": KnowledgeScopeMemberRepository(self._session, scope)}

    def _lease(self, scope, job, run, version):
        require(job.context_json.get("subject_ref") == scope.subject_ref, "摄取委托不属于当前主体")
        return JobLease(scope, version.document_version_id,
            ParseRunIdentity(BINDING.binding_ref, str(job.ingestion_job_id), str(run.parse_run_id),
                int(job.attempt), int(job.fence), version.source_sha256, run.profile_hash),
            decode(ParseProfile, run.profile_json), decode(SourceDescriptor, version.source_descriptor_json),
            job.lease_owner or "", job.lease_until, int(job.recovery_sequence or 0), job.last_event_digest,
            job.status, job.compute_ref)

    async def register(self, scope, *, document_id, document_version_id, source, profile, business_asset_ref,
                       business_version_ref, title, now):
        repositories = self._repos(scope)
        BINDING.check_owner(source.owner_binding_ref)
        require(source.document_version_ref == str(document_version_id), "文档版本与源描述不符")
        document = await repositories["knowledge_document"].get(document_id, lock=True)
        if document is None:
            document = await repositories["knowledge_document"].add(dict(document_id=document_id,
                business_asset_ref=business_asset_ref, title=title, status="DRAFT", metadata_json={}))
        require(document.business_asset_ref == business_asset_ref, "文档身份被其他业务资产占用")
        version = await repositories["knowledge_document_version"].get(document_version_id, lock=True)
        if version is not None:
            require(version.document_id == document_id and version.source_descriptor_json == encode(source)
                    and version.business_version_ref == business_version_ref,
                    "同版本不得绑定其他源内容")
            rows = (await self._session.scalars(repositories["knowledge_ingestion_job"].statement().where(
                KnowledgeIngestionJobRepository.entity.document_version_id == document_version_id))).all()
            require(len(rows) == 1, "文档版本的摄取任务身份不唯一")
            require(rows[0].context_json.get("subject_ref") == scope.subject_ref, "摄取任务不属于当前主体")
            run = await repositories["knowledge_parse_run"].get(rows[0].parse_run_id)
            require(run.profile_hash == profile.profile_hash, "同文档版本不得隐式切换解析 Profile")
            return rows[0].ingestion_job_id
        version = await repositories["knowledge_document_version"].add(dict(
            document_version_id=document_version_id, document_id=document_id, source_sha256=source.sha256,
            mime_type=source.mime_type, byte_count=source.byte_count, source_object_ref=source.source_handle,
            business_version_ref=business_version_ref, source_descriptor_json=encode(source), status="RECEIVED"))
        run = await repositories["knowledge_parse_run"].add(dict(document_version_id=document_version_id,
            source_sha256=source.sha256, profile_hash=profile.profile_hash, parser_revision="PENDING",
            model_snapshot_json={}, attempt=1, fence=1, status="RECEIVED", profile_json=encode(profile),
            checkpoint_json={}))
        job = await repositories["knowledge_ingestion_job"].add(dict(document_version_id=document_version_id,
            parse_run_id=run.parse_run_id, attempt=1, fence=1, lease_owner=None, lease_until=now,
            retry_at=now, recovery_sequence=0, status="RECEIVED", context_json={"subject_ref": scope.subject_ref}))
        return job.ingestion_job_id

    async def get_job(self, scope, job_id):
        repositories = self._repos(scope)
        job = await repositories["knowledge_ingestion_job"].get(job_id)
        require(job is not None, "摄取任务不存在于当前范围")
        version = await repositories["knowledge_document_version"].get(job.document_version_id)
        run = await repositories["knowledge_parse_run"].get(job.parse_run_id)
        require(version is not None and run is not None, "摄取任务来源断裂")
        return self._lease(scope, job, run, version)

    async def _locked(self, lease, now, *, require_live=True):
        repositories = self._repos(lease.scope)
        version = await repositories["knowledge_document_version"].get(lease.document_version_id, lock=True)
        require(version is not None, "文档版本不属于当前范围")
        job = await repositories["knowledge_ingestion_job"].get(UUID(lease.identity.job_ref), lock=True)
        require(job is not None and job.parse_run_id == UUID(lease.identity.parse_run_ref)
                and int(job.fence) == lease.identity.fence and int(job.attempt) == lease.identity.attempt
                and job.lease_owner == lease.lease_owner, "任务代次、领取者或栅栏已经失效")
        run = await repositories["knowledge_parse_run"].get(job.parse_run_id, lock=True)
        current = self._lease(lease.scope, job, run, version)
        if require_live:
            current.require_live(now)
        return repositories, version, job, run, current

    async def claim(self, scope, job_id, *, worker_id, now, lease_seconds):
        require(worker_id and 1 <= lease_seconds <= 3600, "领取参数无效")
        repositories = self._repos(scope)
        initial = await repositories["knowledge_ingestion_job"].get(job_id)
        require(initial is not None, "任务不属于当前范围")
        version = await repositories["knowledge_document_version"].get(initial.document_version_id, lock=True)
        job = await repositories["knowledge_ingestion_job"].get(job_id, lock=True)
        require(job.status in {"RECEIVED", "PARSING"} and job.lease_until <= now, "任务不可重复领取")
        run = await repositories["knowledge_parse_run"].get(job.parse_run_id, lock=True)
        if job.status == "PARSING":
            # 租约失效后产生新解析代次，绝不把旧非确定性结果改绑到新 fence。
            run.status = "SUPERSEDED"
            run = await repositories["knowledge_parse_run"].add(dict(
                document_version_id=version.document_version_id, source_sha256=version.source_sha256,
                profile_hash=run.profile_hash, parser_revision="PENDING", model_snapshot_json={},
                attempt=int(job.attempt) + 1, fence=int(job.fence) + 1, status="PARSING",
                profile_json=run.profile_json, checkpoint_json={}))
            job.parse_run_id = run.parse_run_id
            job.attempt = int(job.attempt) + 1
            job.fence = int(job.fence) + 1
            job.recovery_sequence, job.last_event_digest, job.compute_ref = 0, None, None
        job.lease_owner, job.lease_until, job.status = worker_id, now + timedelta(seconds=lease_seconds), "PARSING"
        run.status, version.status = "PARSING", "PARSING"
        await self._session.flush()
        return self._lease(scope, job, run, version)

    async def renew(self, lease, *, now, lease_seconds):
        require(1 <= lease_seconds <= 3600, "续期参数无效")
        repositories, version, job, run, current = await self._locked(lease, now)
        job.lease_until = now + timedelta(seconds=lease_seconds)
        await self._session.flush()
        return self._lease(lease.scope, job, run, version)

    async def bind_compute(self, lease, compute_ref, *, now):
        _, _, job, _, _ = await self._locked(lease, now)
        require(job.compute_ref in {None, compute_ref}, "计算回执不能被覆盖")
        job.compute_ref = compute_ref

    async def batches(self, lease, *, after_sequence=0, limit=100):
        require(1 <= limit <= 1000, "批次读取预算无效")
        repositories = self._repos(lease.scope)
        entity = KnowledgeParseBatchRepository.entity
        rows = (await self._session.scalars(repositories["knowledge_parse_batch"].statement().where(
            entity.parse_run_id == UUID(lease.identity.parse_run_ref), entity.sequence > after_sequence
        ).order_by(entity.sequence).limit(limit))).all()
        return tuple(BatchReceipt(int(row.sequence), row.event_digest, row.payload_ref, row.batch_key) for row in rows)

    async def receive(self, lease, *, expected_sequence, events, payload_refs, artifact_maps, now):
        repositories, version, job, run, current = await self._locked(lease, now, require_live=False)
        require(bool(events) and len(events) == len(payload_refs) == len(artifact_maps), "事件页与持久副本数量不一致")
        require(current.sequence == expected_sequence, "接收恢复点已被并发推进，请重读后重试")
        for event in events:
            require(event.run == lease.identity, "事件归属或代次不一致")
            if event.sequence <= current.sequence:
                repo = repositories["knowledge_parse_batch"]
                rows = (await self._session.scalars(repo.statement().where(
                    repo.entity.parse_run_id == run.parse_run_id, repo.entity.sequence == event.sequence))).all()
                require(len(rows) == 1 and rows[0].event_digest == event.event_digest
                        and rows[0].batch_key == event.batch_key, "重放事件与已持久内容冲突")
        if all(event.sequence <= current.sequence for event in events):
            return current.sequence
        current.require_live(now)
        for event, payload_ref, artifact_map in zip(events, payload_refs, artifact_maps):
            require(event.run == lease.identity, "事件归属或代次不一致")
            if event.sequence <= current.sequence:
                continue
            require(event.sequence == int(job.recovery_sequence) + 1, "接收序号必须连续")
            require(job.status == "PARSING", "终态后不得继续接收")
            repo = repositories["knowledge_parse_batch"]
            duplicate = (await self._session.scalars(repo.statement().where(
                repo.entity.parse_run_id == run.parse_run_id, repo.entity.batch_key == event.batch_key))).first()
            require(duplicate is None, "批次键不能绑定不同序号")
            await repositories["knowledge_parse_batch"].add(dict(
                parse_run_id=run.parse_run_id, batch_key=event.batch_key, sequence=event.sequence,
                payload_hash=event.payload_hash, payload_ref=payload_ref, dependency_keys_json=list(event.dependency_keys),
                artifact_manifest_json=artifact_map, event_digest=event.event_digest, event_type=event.event_type.value))
            await self._materialize(repositories, run, version, event, payload_ref, artifact_map)
            job.recovery_sequence, job.last_event_digest = event.sequence, event.event_digest
            if event.event_type == ParseEventType.RUN_FAILED:
                import json
                failure = json.loads(event.payload_json)
                job.error_code = failure.get("error_code", "CHANNEL_FAILED")
                run.quality_report_json = {"status":"FAILED", "error_code":job.error_code,
                                           "stage":failure.get("stage", "parse")}
                run.status = job.status = version.status = "FAILED"
            elif event.event_type == ParseEventType.RUN_COMPLETED:
                import json
                payload = json.loads(event.payload_json)
                run.quality_report_json, run.manifest_hash = payload["quality_report"], event.payload_hash
                run.status = job.status = version.status = (
                    "FAILED" if payload["quality_report"]["status"] == "FAILED" else "PARSED")
                if run.status == "PARSED":
                    await repositories["knowledge_outbox"].add(dict(
                        document_version_id=version.document_version_id, event_type="BUILD_PROJECTION",
                        idempotency_key=str(run.parse_run_id) + ":projection", payload_ref=payload_ref,
                        status="PENDING", retry_at=now, attempt=0))
        await self._session.flush()
        return int(job.recovery_sequence)

    async def mark_acknowledged(self, lease, *, sequence, event_digest, now):
        repositories, _, job, run, _ = await self._locked(lease, now, require_live=False)
        require(0 < sequence <= int(job.recovery_sequence), "ACK 不能超过 App 已提交恢复点")
        repo = repositories["knowledge_parse_batch"]
        receipt = (await self._session.scalars(repo.statement().where(
            repo.entity.parse_run_id == run.parse_run_id, repo.entity.sequence == sequence))).one()
        require(receipt.event_digest == event_digest, "ACK 记录摘要冲突")
        statement = update(repo.entity).where(repo.entity.parse_run_id == run.parse_run_id,
            repo.entity.domain_id == lease.scope.domain_id, repo.entity.sequence <= sequence,
            repo.entity.acknowledged_at.is_(None))
        if hasattr(repo.entity, "tenant_id"):
            statement = statement.where(repo.entity.tenant_id == lease.scope.tenant_id)
        await self._session.execute(statement.values(acknowledged_at=now))

    async def _materialize(self, repositories, run, version, event, payload_ref, artifact_map):
        import json
        payload = json.loads(event.payload_json)
        async def nodes():
            repo = repositories["knowledge_node"]
            return (await self._session.scalars(repo.statement().where(repo.entity.parse_run_id == run.parse_run_id))).all()
        if event.event_type == ParseEventType.RUN_STARTED:
            require(payload["profile_hash"] == run.profile_hash, "解析配置摘要不一致")
            run.parser_revision = payload["parser_revision"]
        elif event.event_type == ParseEventType.PAGE_BATCH:
            for raw in payload["objects"]:
                page = decode(Page, raw)
                await repositories["knowledge_node"].add(dict(parse_run_id=run.parse_run_id,
                    local_key=page.local_key, kind="PAGE", reading_order=page.page_no,
                    locator_json={"page_no": page.page_no}, provenance_json={"kind": "NATIVE"},
                    metadata_json=encode(page)))
        elif event.event_type == ParseEventType.NODE_BATCH:
            for raw in payload["objects"]:
                node = decode(ContentNode, {k: v for k, v in raw.items() if k != "local_key"})
                BINDING.check_owner(node.key.owner_binding_ref)
                require(node.key.parse_run_ref == str(run.parse_run_id)
                        and node.key.document_version_ref == str(version.document_version_id)
                        and node.key.object_key == raw["local_key"], "规范节点版本或身份不一致")
                require(node.key.object_key == local_key(algorithm_version=run.profile_json["algorithm_version"],
                    native_ref=node.native_ref, kind=node.kind.value), "规范节点本地键不一致")
                metadata = encode(node)
                metadata.pop("text")
                await repositories["knowledge_node"].add(dict(parse_run_id=run.parse_run_id,
                    local_key=node.key.object_key, kind=node.kind.value, reading_order=node.reading_order,
                    original_text_ref=payload_ref + "#" + node.key.object_key,
                    locator_json=encode(node.locators), provenance_json=encode(node.provenance), metadata_json=metadata))
        elif event.event_type == ParseEventType.ASSET_BATCH:
            catalog = {n.local_key: n for n in await nodes()}
            for raw in payload["objects"]:
                asset = decode(VisualAsset, {k: v for k, v in raw.items() if k != "local_key"})
                BINDING.check_owner(asset.key.owner_binding_ref)
                require(asset.key.parse_run_ref == str(run.parse_run_id)
                        and asset.key.document_version_ref == str(version.document_version_id)
                        and asset.figure_key.owner_binding_ref == BINDING.binding_ref
                        and asset.figure_key.parse_run_ref == str(run.parse_run_id)
                        and asset.figure_key.document_version_ref == str(version.document_version_id),
                        "图像版本或解析代次不一致")
                require(asset.content_handle in artifact_map,
                        "图像缺少逻辑节点或已校验的 App 副本")
                copied = artifact_map[asset.content_handle]
                require(copied["sha256"] == asset.sha256 and copied["byte_count"] == asset.byte_count,
                        "图像副本与规范元数据不一致")
                await repositories["knowledge_visual_asset"].add(dict(parse_run_id=run.parse_run_id,
                    node_id=catalog[asset.figure_key.object_key].node_id if asset.figure_key.object_key in catalog else None,
                    local_key=asset.key.object_key,
                    content_ref=copied["handle_ref"], content_sha256=asset.sha256, mime_type=asset.mime_type,
                    byte_count=asset.byte_count, dimensions_json={}, crop_transform_json=encode(asset.crop_transform),
                    locator_json=encode(asset.locator), provenance_json=encode(asset.provenance), metadata_json=encode(asset)))
        elif event.event_type == ParseEventType.REFERENCE_BATCH:
            catalog = {n.local_key: n for n in await nodes()}
            for raw in payload["objects"]:
                resolution = decode(FigureResolution, {k: v for k, v in raw.items() if k != "local_key"})
                ref = resolution.occurrence
                require(ref.source_key.owner_binding_ref == BINDING.binding_ref,
                        "引用缺少本 App 精确源节点")
                require(ref.source_key.parse_run_ref == str(run.parse_run_id)
                        and ref.source_key.document_version_ref == str(version.document_version_id),
                        "引用源越出当前文档或解析代次")
                await repositories["knowledge_reference_occurrence"].add(dict(parse_run_id=run.parse_run_id,
                    source_node_id=catalog[ref.source_key.object_key].node_id if ref.source_key.object_key in catalog else None,
                    local_key=ref.occurrence_key,
                    source_span_json=encode(ref.span), raw_reference=ref.raw_label, normalized_label=ref.raw_label,
                    numbering_scope=ref.numbering_scope, resolution_status=resolution.status.value,
                    candidate_keys_json=encode(resolution)))
        elif event.event_type == ParseEventType.STRUCTURE_FINALIZED:
            catalog = {n.local_key: n for n in await nodes()}
            parents = {}
            for key, node in catalog.items():
                parent = (node.metadata_json or {}).get("parent_key")
                require(parent is None or parent in catalog, "结构父节点未解析")
                parents[key] = parent
                node.parent_node_id = catalog[parent].node_id if parent else None
            for key in parents:
                seen, current = set(), key
                while current is not None:
                    require(current not in seen, "结构父节点存在环")
                    seen.add(current)
                    current = parents[current]
            assets_repo = repositories["knowledge_visual_asset"]
            assets = {a.local_key: a for a in (await self._session.scalars(assets_repo.statement().where(
                assets_repo.entity.parse_run_id == run.parse_run_id))).all()}
            refs_repo = repositories["knowledge_reference_occurrence"]
            refs = {r.local_key: r for r in (await self._session.scalars(refs_repo.statement().where(
                refs_repo.entity.parse_run_id == run.parse_run_id))).all()}
            for asset in assets.values():
                descriptor = decode(VisualAsset, asset.metadata_json)
                require(descriptor.figure_key.object_key in catalog, "图像前向节点依赖未解析")
                asset.node_id = catalog[descriptor.figure_key.object_key].node_id
            for reference in refs.values():
                resolution = decode(FigureResolution, reference.candidate_keys_json)
                require(resolution.occurrence.source_key.object_key in catalog, "引用前向源依赖未解析")
                reference.source_node_id = catalog[resolution.occurrence.source_key.object_key].node_id
                for target_key in (() if resolution.target_key is None else (resolution.target_key,)):
                    require(target_key.owner_binding_ref == BINDING.binding_ref
                            and target_key.document_version_ref == str(version.document_version_id)
                            and target_key.parse_run_ref == str(run.parse_run_id)
                            and target_key.object_key in catalog, "引用解析目标不在当前文档代次")
            for raw in payload["relations"]:
                relation = decode(ResolvedRelation, raw)
                require(relation.source_key.owner_binding_ref == BINDING.binding_ref
                    and relation.source_key.parse_run_ref == str(run.parse_run_id)
                    and relation.source_key.document_version_ref == str(version.document_version_id),
                    "关系越出当前解析代次")
                require(relation.target_key.owner_binding_ref == BINDING.binding_ref
                    and relation.target_key.parse_run_ref == str(run.parse_run_id)
                    and relation.target_key.document_version_ref == str(version.document_version_id),
                    "关系目标越出当前解析代次")
                source = catalog.get(relation.source_key.object_key)
                target = catalog.get(relation.target_key.object_key)
                asset = assets.get(relation.target_key.object_key)
                require(source is not None and (target is not None) != (asset is not None), "关系目标未解析或不唯一")
                occurrence = refs.get(relation.occurrence_key) if relation.occurrence_key else None
                await repositories["knowledge_relation"].add(dict(parse_run_id=run.parse_run_id,
                    source_node_id=source.node_id, target_node_id=target.node_id if target else None,
                    target_asset_id=asset.visual_asset_id if asset else None,
                    target_document_version_id=version.document_version_id, target_parse_run_id=run.parse_run_id,
                    relation_type=relation.relation_type.value, occurrence_id=occurrence.reference_occurrence_id if occurrence else None,
                    basis_json=encode(relation)))
            run.checkpoint_json = {"structure_ref": payload_ref}

    async def cancel(self, lease, *, now):
        _, version, job, run, _ = await self._locked(lease, now, require_live=False)
        if job.status == "CANCELLED":
            return
        require(job.status not in {"PARSED", "FAILED"}, "已终止任务不能被改写")
        job.status = run.status = version.status = "CANCELLED"
        job.fence = int(job.fence) + 1
        job.cancelled_at = now

    async def prepare_generation(self, lease, *, projection_ref, manifest, now):
        repositories, version, job, run, _ = await self._locked(lease, now, require_live=False)
        require(run.status == job.status == "PARSED", "未完成解析不能准备索引代次")
        repo = repositories["knowledge_index_generation"]
        rows = (await self._session.scalars(repo.statement().where(repo.entity.parse_run_id == run.parse_run_id,
            repo.entity.projection_ref == projection_ref))).all()
        if rows:
            require(len(rows) == 1 and all(rows[0].manifest_json.get(key) == value
                    for key, value in manifest.items()), "索引代次幂等身份冲突")
            return rows[0].index_generation_id
        row = await repo.add(dict(document_version_id=version.document_version_id, parse_run_id=run.parse_run_id,
            projection_ref=projection_ref, manifest_json=manifest, channel_indexes_json={}, status="PREPARING", active_revision=0))
        return row.index_generation_id

    async def claim_outbox(self, scope, outbox_id, *, worker_id, now, lease_seconds):
        require(worker_id and 1 <= lease_seconds <= 3600, "索引任务租约无效")
        repo = self._repos(scope)["knowledge_outbox"]
        row = await repo.get(outbox_id, lock=True)
        require(row is not None and row.status in {"PENDING", "RUNNING"}
                and (row.retry_at is None or row.retry_at <= now)
                and (row.lease_until is None or row.lease_until <= now), "索引任务尚不可领取")
        row.status, row.lease_owner, row.lease_until = "RUNNING", worker_id, now + timedelta(seconds=lease_seconds)
        row.attempt = int(row.attempt or 0) + 1
        return {"outbox_id": row.outbox_id, "attempt": int(row.attempt), "payload_ref": row.payload_ref,
                "document_version_id": row.document_version_id}

    async def renew_outbox(self, scope, outbox_id, *, worker_id, attempt, now, lease_seconds):
        require(1 <= lease_seconds <= 3600, "索引续租参数无效")
        row = await self._repos(scope)["knowledge_outbox"].get(outbox_id, lock=True)
        require(row is not None and row.status == "RUNNING" and row.lease_owner == worker_id
                and int(row.attempt) == attempt and row.lease_until > now, "索引任务续租栅栏失效")
        row.lease_until = now + timedelta(seconds=lease_seconds)

    async def finish_outbox(self, scope, outbox_id, *, worker_id, attempt, now, error_code=None):
        repo = self._repos(scope)["knowledge_outbox"]
        row = await repo.get(outbox_id, lock=True)
        require(row is not None and row.status == "RUNNING" and row.lease_owner == worker_id
                and int(row.attempt) == attempt and row.lease_until > now, "索引任务栅栏失效")
        row.status = "PENDING" if error_code else "DELIVERED"
        row.error_code, row.delivered_at = error_code, None if error_code else now
        row.lease_until = now
        row.retry_at = now + timedelta(seconds=min(300, 2 ** min(attempt, 8))) if error_code else None


    async def batch_artifacts(self, lease, sequences):
        require(len(sequences) <= 1000, "制品清单页超过预算")
        repo = self._repos(lease.scope)["knowledge_parse_batch"]
        rows = (await self._session.scalars(repo.statement().where(
            repo.entity.parse_run_id == UUID(lease.identity.parse_run_ref),
            repo.entity.sequence.in_(sequences)))).all()
        return {int(row.sequence): row.artifact_manifest_json or {} for row in rows}

    async def set_allowed_view(self, lease, *, manifest_ref, view_hash, now):
        _, version, job, run, _ = await self._locked(lease, now, require_live=False)
        require(run.status == job.status == "PARSED", "未完成解析不能准备允许视图")
        require(version.allowed_view_hash in {None, view_hash}
                and version.allowed_view_manifest_ref in {None, manifest_ref}, "同一解析代次允许视图冲突")
        version.allowed_view_hash, version.allowed_view_manifest_ref = view_hash, manifest_ref

    async def store_projection(self, lease, *, generation_id, batch, payload_ref, profile, now):
        from knowledge_algorithms.contracts.identity import fingerprint
        repositories, version, job, run, _ = await self._locked(lease, now, require_live=False)
        generation = await repositories["knowledge_index_generation"].get(generation_id, lock=True)
        require(run.status == job.status == "PARSED" and generation is not None
                and generation.parse_run_id == run.parse_run_id
                and generation.projection_ref == batch.projection_ref
                and generation.status in {"PREPARING", "READY"}
                and generation.manifest_json["profile"] == encode(profile), "投影不属于当前准备代次")
        require(batch.sequence > 0 and batch.units
                and batch.payload_hash == fingerprint(encode(batch.units)), "投影批次摘要或序号无效")
        manifest = dict(generation.manifest_json)
        batches = dict(manifest.get("batches", {}))
        key = str(batch.sequence)
        if key in batches:
            require(batches[key] == {"hash": batch.payload_hash, "count": len(batch.units), "ref": payload_ref},
                    "投影重放内容冲突")
            return
        require(generation.status == "PREPARING" and batch.sequence == len(batches) + 1, "投影批次乱序或已封闭")
        node_repo = repositories["knowledge_node"]
        catalog = {n.local_key: n for n in (await self._session.scalars(node_repo.statement().where(
            node_repo.entity.parse_run_id == run.parse_run_id))).all()}
        unit_repo = repositories["knowledge_search_unit"]
        for projected in batch.units:
            unit = projected.unit
            require(unit.key.owner_binding_ref == BINDING.binding_ref
                    and unit.key.document_version_ref == str(version.document_version_id)
                    and unit.key.parse_run_ref == str(run.parse_run_id)
                    and unit.key.projection_ref == batch.projection_ref and unit.key.object_key == unit.unit_key
                    and unit.source_key.owner_binding_ref == BINDING.binding_ref
                    and unit.source_key.document_version_ref == str(version.document_version_id)
                    and unit.source_key.parse_run_ref == str(run.parse_run_id)
                    and unit.source_key.projection_ref is None
                    and unit.source_key.object_key in catalog
                    and unit.allowed_view_hash == version.allowed_view_hash, "检索单元来源或允许视图不一致")
            duplicate = (await self._session.scalars(unit_repo.statement().where(
                unit_repo.entity.parse_run_id == run.parse_run_id,
                unit_repo.entity.projection_ref == batch.projection_ref,
                unit_repo.entity.unit_key == unit.unit_key))).first()
            require(duplicate is None, "投影单元不能跨批次重复")
            row = await unit_repo.add(dict(parse_run_id=run.parse_run_id, projection_ref=batch.projection_ref,
                unit_key=unit.unit_key, source_node_id=catalog[unit.source_key.object_key].node_id, kind=unit.kind.value,
                allowed_text_ref=payload_ref + "#" + unit.unit_key, context_ref=payload_ref + "#" + unit.unit_key,
                allowed_view_ref=unit.allowed_view_ref, allowed_view_hash=unit.allowed_view_hash,
                input_hash=fingerprint(encode(unit)), profile_hash=fingerprint(encode(profile))))
            for source in unit.sources:
                require(source.source_key.owner_binding_ref == BINDING.binding_ref
                        and source.source_key.document_version_ref == str(version.document_version_id)
                        and source.source_key.parse_run_ref == str(run.parse_run_id)
                        and source.source_key.object_key in catalog, "投影源跨度越出当前文档")
                await repositories["knowledge_unit_source"].add(dict(search_unit_id=row.search_unit_id,
                    source_node_id=catalog[source.source_key.object_key].node_id,
                    unit_start=source.unit_start, unit_end=source.unit_end,
                    source_start=source.source_start, source_end=source.source_end, locator_json=encode(unit.locators)))
            if unit.content_handle:
                asset_repo = repositories["knowledge_visual_asset"]
                assets = (await self._session.scalars(asset_repo.statement().where(
                    asset_repo.entity.parse_run_id == run.parse_run_id,
                    asset_repo.entity.node_id == catalog[unit.source_key.object_key].node_id,
                    asset_repo.entity.content_ref == unit.content_handle))).all()
                require(len(assets) == 1, "投影图像必须绑定精确 App 原图副本")
                await repositories["knowledge_unit_source"].add(dict(search_unit_id=row.search_unit_id,
                    source_asset_id=assets[0].visual_asset_id, locator_json=encode(unit.locators)))
            specs = {spec.space_id: spec for spec in profile.representations}
            require(len({r.space_id for r in projected.representations}) == len(projected.representations),
                    "投影模型空间重复")
            for representation in projected.representations:
                require(representation.space_id in specs, "未声明的模型空间")
                specification = specs[representation.space_id]
                representation.validate(specification, profile.max_representation_elements)
                await repositories["knowledge_representation"].add(dict(search_unit_id=row.search_unit_id,
                    space_id=representation.space_id, model_snapshot_json=encode(specification),
                    input_hash=fingerprint(encode(unit)), shape_json={
                        "kind": specification.kind, "dimension": specification.dimension},
                    vector_ref=payload_ref if representation.vector is not None else None,
                    tensor_ref=payload_ref if representation.matrix is not None else None,
                    mask_ref=payload_ref if representation.matrix is not None else None))
        batches[key] = {"hash": batch.payload_hash, "count": len(batch.units), "ref": payload_ref}
        manifest["batches"] = batches
        generation.manifest_json = manifest

    async def complete_generation(self, lease, *, generation_id, unit_count, batch_digest, batch_count, now):
        from knowledge_algorithms.contracts.identity import fingerprint
        repositories, _, job, run, _ = await self._locked(lease, now, require_live=False)
        generation = await repositories["knowledge_index_generation"].get(generation_id, lock=True)
        require(run.status == job.status == "PARSED" and generation is not None
                and generation.parse_run_id == run.parse_run_id and generation.status in {"PREPARING", "READY"},
                "索引代次不可封闭")
        manifest = dict(generation.manifest_json)
        batches = manifest.get("batches", {})
        require(batch_count == len(batches) and batch_count > 0
                and unit_count == sum(item["count"] for item in batches.values())
                and batch_digest == fingerprint([batches[str(i)]["hash"] for i in range(1, batch_count + 1)]),
                "投影最终清单不匹配")
        manifest.update(unit_count=unit_count, batch_digest=batch_digest, batch_count=batch_count)
        generation.manifest_json, generation.status = manifest, "READY"
        # READY 只表示投影完整，不切换 published_parse_run/current_generation。

    async def pin(self, scope, *, owner_run_ref, evidence_key, expires_at, now):
        repositories = self._repos(scope)
        BINDING.check_owner(evidence_key.owner_binding_ref)
        version = await repositories["knowledge_document_version"].get(UUID(evidence_key.document_version_ref), lock=True)
        run = await repositories["knowledge_parse_run"].get(UUID(evidence_key.parse_run_ref))
        require(version is not None and run is not None and run.document_version_id == version.document_version_id
                and owner_run_ref and expires_at > now, "保留引用范围或期限无效")
        generation_id = None
        if evidence_key.projection_ref is not None:
            repo = repositories["knowledge_search_unit"]
            exists = (await self._session.scalars(repo.statement().where(
                repo.entity.parse_run_id == run.parse_run_id,
                repo.entity.projection_ref == evidence_key.projection_ref,
                repo.entity.unit_key == evidence_key.object_key))).all()
            generation_repo = repositories["knowledge_index_generation"]
            generations = (await self._session.scalars(generation_repo.statement().where(
                generation_repo.entity.parse_run_id == run.parse_run_id,
                generation_repo.entity.projection_ref == evidence_key.projection_ref,
                generation_repo.entity.status == "READY"))).all()
            require(len(exists) == len(generations) == 1, "证据单元或完整索引代次不存在")
            generation_id = generations[0].index_generation_id
        else:
            matches = []
            for kind in ("knowledge_node", "knowledge_visual_asset"):
                repo = repositories[kind]
                matches.extend((await self._session.scalars(repo.statement().where(
                    repo.entity.parse_run_id == run.parse_run_id,
                    repo.entity.local_key == evidence_key.object_key))).all())
            require(len(matches) == 1 and run.status == "PARSED", "精确规范证据对象不存在")
        pin_repo = repositories["knowledge_evidence_pin"]
        pins = (await self._session.scalars(pin_repo.statement().where(
            pin_repo.entity.parse_run_id == run.parse_run_id, pin_repo.entity.owner_run_ref == owner_run_ref))).all()
        same = [row for row in pins if row.evidence_key_json == encode(evidence_key) and row.released_at is None]
        require(len(same) <= 1, "保留引用身份重复")
        if same:
            same[0].expires_at = max(same[0].expires_at, expires_at)
            return same[0].evidence_pin_id
        row = await repositories["knowledge_evidence_pin"].add(dict(owner_run_ref=owner_run_ref,
            evidence_key_json=encode(evidence_key), parse_run_id=run.parse_run_id,
            projection_ref=evidence_key.projection_ref, index_generation_id=generation_id, expires_at=expires_at))
        return row.evidence_pin_id

    async def release_pin(self, scope, *, pin_id, owner_run_ref, now):
        repositories = self._repos(scope)
        row = await repositories["knowledge_evidence_pin"].get(pin_id, lock=True)
        require(row is not None and row.owner_run_ref == owner_run_ref, "保留引用不属于当前运行")
        if row.released_at is None:
            row.released_at = now

    async def retained(self, scope, *, parse_run_id, now):
        repo = self._repos(scope)["knowledge_evidence_pin"]
        return (await self._session.scalars(repo.statement().where(
            repo.entity.parse_run_id == parse_run_id, repo.entity.released_at.is_(None),
            repo.entity.expires_at > now).limit(1))).first() is not None
