"""KC 独占创建 Collection；以父空间行锁串行化系统资源初始化。"""

from uuid import UUID

from sqlalchemy import or_, select, text

from knowledge_core.domain.case_resource import CaseResourceConflict
from knowledge_core.entities.collection import KcCollectionEntity
from platform_core.app_resources.repository import CaseResourceRepository
from platform_core.identity import uuid7


class SystemCaseResourceRepository:
    def __init__(self, session):
        self._session = session
        self._scopes = CaseResourceRepository(session)

    async def ensure(self, request):
        if (request.tenant_id is not None or not request.domain_id.isdecimal()
            or int(request.domain_id) <= 0 or str(int(request.domain_id)) != request.domain_id):
            raise CaseResourceConflict("CASE_RESOURCE_SCOPE_INACTIVE")
        scope = await self._scopes.scope(domain_id=request.domain_id, tenant_id=request.tenant_id, lock=True)
        if scope is None or not scope["eligible"]:
            raise CaseResourceConflict("CASE_RESOURCE_SCOPE_INACTIVE")
        all_rows = (await self._session.scalars(select(KcCollectionEntity).where(
            KcCollectionEntity.domain_id == int(request.domain_id)))).all()
        rows = [row for row in all_rows if row.collection_id == request.expected_collection_id
            or row.display_name == "diagnosis-cases"
            or (isinstance(row.metadata_json, dict)
                and row.metadata_json.get("content_kind") == "diagnosis_case")]
        if len(rows) > 1:
            raise CaseResourceConflict("CASE_RESOURCE_CONFLICT")
        existing = rows[0] if rows else None
        if request.expected_collection_id is not None:
            if existing is None:
                raise CaseResourceConflict("CASE_RESOURCE_KNOWN_COLLECTION_MISSING")
            if existing.collection_id != request.expected_collection_id:
                raise CaseResourceConflict("CASE_RESOURCE_CONFLICT")
        if existing is None and not request.allow_create:
            raise CaseResourceConflict("CASE_RESOURCE_KNOWN_COLLECTION_MISSING")
        if existing is None:
            existing = KcCollectionEntity(collection_id=uuid7(),
                domain_id=int(request.domain_id),
                display_name="diagnosis-cases", description="AIOps 固定诊断案例索引资源",
                models_json={}, parse_policy_json={"parse_strategy": "TEXT"},
                status="ACTIVE", default_security_level=1,
                metadata_json={"owner_app_id": "aiops", "fixed_resource": True,
                    "content_kind": "diagnosis_case", "domain_id": str(request.domain_id)},
                row_version=1, created_by=request.actor_id, updated_by=request.actor_id)
            self._session.add(existing)
            await self._session.flush()
        metadata = existing.metadata_json or {}
        if not isinstance(metadata, dict):
            raise CaseResourceConflict("CASE_RESOURCE_CONFLICT")
        if (existing.status != "ACTIVE" or metadata.get("owner_app_id") != "aiops"
            or metadata.get("fixed_resource") is not True
            or metadata.get("content_kind") != "diagnosis_case"
            or metadata.get("domain_id", request.domain_id) != request.domain_id
            ):
            raise CaseResourceConflict("CASE_RESOURCE_CONFLICT")

        return existing
