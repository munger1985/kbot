"""已发布代次、授权范围、检索回执与精确跨文档引用的 App 仓储。"""

from datetime import timedelta
from uuid import UUID

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import AuthorizedScope, fingerprint
from knowledge_algorithms.contracts.wire import encode
from platform_core.identity import uuid7

from .knowledge_ingestion import KnowledgeIngestionRepository
from ..domain.knowledge_ingestion import BINDING


class KnowledgeRetrievalRepository(KnowledgeIngestionRepository):
    async def version_records(self, access, version_ids, *, lock=False):
        repositories = self._repos(access)
        result = []
        for version_id in sorted(set(version_ids), key=str):
            version = await repositories["knowledge_document_version"].get(version_id, lock=lock)
            require(version is not None, "授权文档版本不在当前 App/Domain/Tenant")
            document = await repositories["knowledge_document"].get(version.document_id)
            require(document is not None, "文档来源断裂")
            generation = (await repositories["knowledge_index_generation"].get(version.active_generation_id)
                          if version.active_generation_id else None)
            result.append({"version_id": str(version_id), "document_id": str(version.document_id),
                "title": document.title, "business_asset_ref": document.business_asset_ref,
                "citation_target": document.metadata_json.get("citation_target"),
                "created_at": version.created_at.isoformat(), "status": version.status,
                "document_status": document.status, "version_revision": int(version.row_version),
                "document_revision": int(document.row_version), "source_sha256": version.source_sha256,
                "allowed_view_hash": version.allowed_view_hash,
                "allowed_view_manifest_ref": version.allowed_view_manifest_ref,
                "parse_run_id": str(version.published_parse_run_id) if version.published_parse_run_id else None,
                "generation_id": str(version.active_generation_id) if version.active_generation_id else None,
                "generation_status": generation.status if generation else None,
                "projection_ref": generation.projection_ref if generation else None,
                "generation_manifest": generation.manifest_json if generation else None})
        return tuple(result)

    async def create_scope(self, access, *, resource_ref, grant, now, ttl_seconds):
        repositories = self._repos(access)
        records = await self.version_records(access, grant.version_ids)
        records = tuple(record for record in records if record["status"] == "PUBLISHED"
                        and record["document_status"] == "PUBLISHED"
                        and record["generation_status"] == "READY")
        members = tuple((record["version_id"], record["allowed_view_hash"]) for record in records)
        identity = uuid7()
        scope_hash = fingerprint({"owner": BINDING.binding_ref, "subject": access.subject_ref,
            "domain": str(access.domain_id), "tenant": str(access.tenant_id),
            "scope_ref": str(identity), "resource": resource_ref, "revision": grant.revision,
            "run_ref": getattr(access, "run_ref", None), "spec_hash": getattr(access, "spec_hash", None),
            "members": members})
        row = await repositories["knowledge_scope_snapshot"].add(dict(scope_snapshot_id=identity,
            subject_ref=access.subject_ref, scope_hash=scope_hash, authorization_revision=grant.revision,
            policy_json={"resource_ref": resource_ref, "run_ref": getattr(access, "run_ref", None),
                         "spec_hash": getattr(access, "spec_hash", None)}, allowed_versions_hash=fingerprint(members),
            status="ACTIVE", expires_at=now + timedelta(seconds=ttl_seconds)))
        for version_ref, view_hash in members:
            await repositories["knowledge_scope_member"].add(dict(scope_snapshot_id=identity,
                document_version_id=UUID(version_ref), allowed_view_ref=view_hash))
        return AuthorizedScope(BINDING.binding_ref, str(row.scope_snapshot_id), scope_hash, grant.revision)

    async def load_scope(self, access, authorized, *, now, max_versions, lock=False):
        BINDING.check_owner(authorized.owner_binding_ref)
        repositories = self._repos(access)
        row = await repositories["knowledge_scope_snapshot"].get(UUID(authorized.scope_ref), lock=lock)
        require(row is not None and row.subject_ref == access.subject_ref
                and row.scope_hash == authorized.scope_hash and row.authorization_revision == authorized.authorization_revision
                and row.status == "ACTIVE" and row.expires_at is not None and row.expires_at > now,
                "授权范围不属于当前主体、已经过期或撤销")
        require(row.policy_json.get("run_ref") == getattr(access, "run_ref", None)
                and row.policy_json.get("spec_hash") == getattr(access, "spec_hash", None),
                "授权范围的 Run/Spec 委托不一致")
        repo = repositories["knowledge_scope_member"]
        members = (await self._session.scalars(repo.statement().where(
            repo.entity.scope_snapshot_id == row.scope_snapshot_id).limit(max_versions + 1))).all()
        require(len(members) <= max_versions, "授权范围超出预算")
        values = tuple(sorted(((str(member.document_version_id), member.allowed_view_ref) for member in members)))
        require(row.allowed_versions_hash == fingerprint(values), "授权范围成员摘要不一致")
        return {"resource_ref": row.policy_json["resource_ref"], "members": values,
                "expires_at": int(row.expires_at.timestamp())}

    async def published_bundle(self, access, authorized, *, now, limits):
        scope_record = await self.load_scope(access, authorized, now=now, max_versions=limits.max_versions)
        records = await self.version_records(access, tuple(UUID(ref) for ref, _ in scope_record["members"]))
        views = dict(scope_record["members"])
        repositories = self._repos(access)
        relations, assets = [], []
        for record in records:
            require(record["status"] == record["document_status"] == "PUBLISHED"
                    and record["generation_status"] == "READY"
                    and record["allowed_view_hash"] == views[record["version_id"]]
                    and record["parse_run_id"] is not None, "冻结范围中的文档已撤回、换版或允许视图改变")
            for logical, result in (("knowledge_relation", relations), ("knowledge_visual_asset", assets)):
                repo = repositories[logical]
                rows = (await self._session.scalars(repo.statement().where(
                    repo.entity.parse_run_id == UUID(record["parse_run_id"])).limit(limits.max_relations + 1))).all()
                require(len(rows) + len(result) <= limits.max_relations, "关联或图像清单超过预算")
                if logical == "knowledge_relation":
                    result.extend(row.basis_json for row in rows if row.basis_json)
                else:
                    result.extend({"descriptor": row.metadata_json, "handle": row.content_ref} for row in rows)
        return {"scope": scope_record, "versions": records, "relations": tuple(relations), "assets": tuple(assets)}

    async def publish(self, access, *, version_id, generation_id, expected_generation_id, now):
        repositories = self._repos(access)
        version = await repositories["knowledge_document_version"].get(version_id, lock=True)
        require(version is not None and version.status != "WITHDRAWN", "发布版本不在当前 App 范围或已永久撤回")
        generation = await repositories["knowledge_index_generation"].get(generation_id, lock=True)
        require(generation is not None and generation.document_version_id == version_id
                and generation.status == "READY" and generation.manifest_json.get("unit_count", 0) > 0,
                "索引代次未完整准备")
        document = await repositories["knowledge_document"].get(version.document_id, lock=True)
        require(document is not None and document.status not in {"DELETED", "WITHDRAWN"}, "文档不可发布")
        if version.active_generation_id == generation_id and version.status == document.status == "PUBLISHED":
            return
        require(version.active_generation_id == expected_generation_id, "发布指针已被其他任务修改")
        run = await repositories["knowledge_parse_run"].get(generation.parse_run_id)
        jobs = repositories["knowledge_ingestion_job"]
        job_rows = (await self._session.scalars(jobs.statement().where(
            jobs.entity.document_version_id == version_id))).all()
        require(run is not None and run.status == "PARSED" and run.source_sha256 == version.source_sha256
                and len(job_rows) == 1 and job_rows[0].parse_run_id == run.parse_run_id
                and job_rows[0].status == "PARSED", "发布代次已经失去任务栅栏")
        require(version.allowed_view_hash == generation.manifest_json.get("allowed_view_hash")
                and version.allowed_view_manifest_ref, "允许视图和索引代次不一致")
        document = await repositories["knowledge_document"].get(version.document_id, lock=True)
        require(document is not None and document.status not in {"DELETED", "WITHDRAWN"}, "文档不可发布")
        generation.active_revision = int(generation.active_revision or 0) + 1
        version.published_parse_run_id, version.active_generation_id = run.parse_run_id, generation_id
        version.status, version.updated_at = "PUBLISHED", now
        document.status, document.updated_at = "PUBLISHED", now

    async def withdraw(self, access, *, version_id, expected_revision, now):
        version = await self._repos(access)["knowledge_document_version"].get(version_id, lock=True)
        require(version is not None and int(version.row_version) == expected_revision, "撤回版本已经改变")
        version.status, version.updated_at = "WITHDRAWN", now

    async def register_citation_target(self, access, *, version_id, target, identifiers):
        repositories = self._repos(access)
        version = await repositories["knowledge_document_version"].get(version_id, lock=True)
        require(version is not None and str(version_id) == target.document_version_ref
                and str(version.published_parse_run_id) == target.parse_run_ref and target.projection_ref is None,
                "文献标识目标必须为当前已发布规范节点")
        document = await repositories["knowledge_document"].get(version.document_id, lock=True)
        document.metadata_json = {**document.metadata_json, "citation_target":
            {"key": encode(target), "identifiers": list(identifiers)}}

    async def save_receipt(self, access, authorized, evidence, *, manifest_ref, query, profile_ref, now, limits):
        repositories = self._repos(access)
        scope = await self.load_scope(access, authorized, now=now, max_versions=limits.max_versions, lock=True)
        row = await repositories["knowledge_retrieval_receipt"].add(dict(
            request_id=evidence.request_id, subject_ref=access.subject_ref,
            scope_snapshot_id=UUID(authorized.scope_ref), snapshot_json=encode(evidence.snapshot),
            query_hash=query.query_hash, profile_hash=evidence.plan.profile_hash,
            plan_json={"query": encode(query), "profile_ref": profile_ref, "scope": encode(authorized)},
            evidence_manifest_ref=manifest_ref, coverage_json={"status": evidence.status},
            expires_at=now + timedelta(seconds=max(1, scope["expires_at"] - int(now.timestamp())))))
        return row.retrieval_receipt_id

    async def receipt(self, access, receipt_id, *, now):
        row = await self._repos(access)["knowledge_retrieval_receipt"].get(receipt_id)
        require(row is not None and row.plan_json.get("kind", "RETRIEVAL") == "RETRIEVAL"
                and row.subject_ref == access.subject_ref
                and row.expires_at is not None and row.expires_at > now, "检索回执不存在于当前主体或已过期")
        return {"plan": row.plan_json, "snapshot": row.snapshot_json,
                "manifest_ref": row.evidence_manifest_ref, "expires_at": int(row.expires_at.timestamp())}

    async def historical_record(self, access, receipt_id, key, *, now):
        """历史读取独立于短期 Scope；精确 Run/Spec pin 与当前版本状态仍须有效。"""
        require(getattr(access, "run_ref", None) and getattr(access, "spec_hash", None),
                "历史引用缺少当前 Run/Spec 授权")
        repositories = self._repos(access)
        receipt = await repositories["knowledge_retrieval_receipt"].get(receipt_id)
        require(receipt is not None and receipt.subject_ref == access.subject_ref
                and receipt.plan_json.get("kind", "RETRIEVAL") == "RETRIEVAL",
                "历史检索回执不属于当前主体")
        scope = await repositories["knowledge_scope_snapshot"].get(receipt.scope_snapshot_id)
        require(scope is not None and scope.status == "ACTIVE" and scope.subject_ref == access.subject_ref
                and scope.policy_json.get("run_ref") == access.run_ref
                and scope.policy_json.get("spec_hash") == access.spec_hash
                and scope.scope_hash == access.delegated_scope_hash, "历史引用 Run/Spec/Scope 不一致")
        pins = repositories["knowledge_evidence_pin"]
        rows = (await self._session.scalars(pins.statement().where(
            pins.entity.owner_run_ref == "run-spec:" + fingerprint([access.run_ref, access.spec_hash]),
            pins.entity.parse_run_id == UUID(key.parse_run_ref),
            pins.entity.expires_at > now, pins.entity.released_at.is_(None)).limit(501))).all()
        require(len(rows) <= 500 and any(pin.evidence_key_json == encode(key) for pin in rows),
                "历史引用未保留、已经释放或超过期限")
        members = repositories["knowledge_scope_member"]
        member = (await self._session.scalars(members.statement().where(
            members.entity.scope_snapshot_id == scope.scope_snapshot_id,
            members.entity.document_version_id == UUID(key.document_version_ref)))).one_or_none()
        version = await repositories["knowledge_document_version"].get(UUID(key.document_version_ref))
        document = await repositories["knowledge_document"].get(version.document_id) if version else None
        require(member is not None and version is not None and document is not None
                and version.status == document.status == "PUBLISHED"
                and version.allowed_view_hash == member.allowed_view_ref,
                "历史引用已撤回或允许视图发生变化")
        return {"resource_ref": scope.policy_json["resource_ref"],
                "manifest_ref": receipt.evidence_manifest_ref, "scope_hash": scope.scope_hash}

    async def historical_manifest(self, access, receipt_id):
        """仅定位冻结清单；在返回任何内容前还需逐对象授权与 pin 校验。"""
        receipt = await self._repos(access)["knowledge_retrieval_receipt"].get(receipt_id)
        require(receipt is not None and receipt.subject_ref == access.subject_ref
                and receipt.plan_json.get("kind", "RETRIEVAL") == "RETRIEVAL", "历史回执不可读取")
        return receipt.evidence_manifest_ref

    async def save_query_image(self, access, authorized, *, descriptor, now, limits):
        scope = await self.load_scope(access, authorized, now=now, max_versions=limits.max_versions)
        row = await self._repos(access)["knowledge_retrieval_receipt"].add(dict(
            request_id=fingerprint({"kind": "QUERY_IMAGE", "scope": authorized.scope_hash, "image": descriptor}),
            subject_ref=access.subject_ref, scope_snapshot_id=UUID(authorized.scope_ref),
            profile_hash=fingerprint({"kind": "QUERY_IMAGE", "revision": "native-image/v1"}),
            plan_json={"kind": "QUERY_IMAGE", "scope": encode(authorized),
                       "run_ref": getattr(access, "run_ref", None), "spec_hash": getattr(access, "spec_hash", None),
                       "descriptor": descriptor},
            evidence_manifest_ref=descriptor["handle"],
            expires_at=now + timedelta(seconds=max(1, scope["expires_at"] - int(now.timestamp())))))
        return str(row.retrieval_receipt_id)

    async def query_image(self, access, image_ref, authorized, *, now):
        row = await self._repos(access)["knowledge_retrieval_receipt"].get(UUID(image_ref))
        require(row is not None and row.subject_ref == access.subject_ref
                and row.expires_at is not None and row.expires_at > now
                and row.plan_json.get("kind") == "QUERY_IMAGE"
                and row.plan_json["scope"] == encode(authorized)
                and row.plan_json.get("run_ref") == getattr(access, "run_ref", None)
                and row.plan_json.get("spec_hash") == getattr(access, "spec_hash", None),
                "查询图片不属于当前主体、运行、范围或已过期")
        return row.plan_json["descriptor"]


    async def revoke_run_scope(self, access, authorized, *, now):
        """过期范围也能释放；只接受原主体、原 Run/Spec 和精确 Scope。"""
        repositories = self._repos(access)
        row = await repositories["knowledge_scope_snapshot"].get(UUID(authorized.scope_ref), lock=True)
        require(row is not None and row.subject_ref == access.subject_ref
                and row.scope_hash == authorized.scope_hash
                and row.policy_json.get("run_ref") == access.run_ref
                and row.policy_json.get("spec_hash") == access.spec_hash,
                "释放范围不属于当前 Run/Spec")
        row.status = "REVOKED"
        repo = repositories["knowledge_evidence_pin"]
        owner = "run-spec:" + fingerprint([access.run_ref, access.spec_hash])
        pins = (await self._session.scalars(repo.statement().where(
            repo.entity.owner_run_ref == owner, repo.entity.released_at.is_(None)).with_for_update())).all()
        for pin in pins:
            pin.released_at = now
