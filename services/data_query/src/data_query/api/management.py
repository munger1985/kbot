"""受服务身份保护的 Data Query 管理面。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status

from data_query.api.dependencies import (
    actor_id_from_context,
    domain_id_from_context,
    get_auth_context,
    require_scope,
)
from data_query.application import DataQueryManagementService
from data_query.contracts import (
    AgentBindingCreate,
    AgentBindingPage,
    AgentBindingStatusChange,
    AgentBindingView,
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceConnectionTest,
    DataSourceConnectionTestResult,
    DataSourceDetail,
    DataSourcePage,
    DataSourceStatusChange,
    DataSourceView,
    DataQueryAuditPage,
    PublishSemanticModelCommand,
    ReturnSemanticModelForRevisionCommand,
    RetireSemanticModelVersionCommand,
    DeleteSemanticModelCommand,
    SubmitSemanticModelReviewCommand,
    PolicyBindingCreate,
    PolicyBindingDetail,
    PolicyBindingPage,
    PolicyBindingStatusChange,
    PolicyBindingView,
    SchemaSnapshotReceipt,
    SchemaSnapshotDetail,
    SchemaSnapshotPage,
    SchemaObjectSelection,
    ManualSchemaDefinition,
    SemanticModelCandidateRequest,
    SemanticModelGenerationReceipt,
    SemanticModelGenerationView,
    SemanticModelDraftUpdate,
    SemanticModelValidationRequest,
    SemanticModelValidationReceipt,
    SemanticModelValidationResult,
    SemanticModelDraftCreate,
    SemanticModelDetail,
    SemanticModelDraftView,
    SemanticModelPage,
    SemanticModelSearch,
    VerifiedQueryPage,
    VerifiedQueryView,
    PromoteVerifiedQueryCommand,
)
from platform_core.contracts import AuthContext


router = APIRouter(prefix="/internal/v1/data-query/management", tags=["Data Query Management"])


def get_service(request: Request) -> DataQueryManagementService:
    return request.app.state.management_service


Service = Annotated[DataQueryManagementService, Depends(get_service)]
Auth = Annotated[AuthContext, Depends(get_auth_context)]


def _require_management(request: Request, context: AuthContext) -> tuple[int, str]:
    require_scope(request, "data_query.manage")
    return domain_id_from_context(context), actor_id_from_context(context)


@router.get("/connector-capabilities")
async def connector_capabilities(request: Request, context: Auth) -> dict[str, object]:
    _require_management(request, context)
    return {"items": [
        {"source_type": "POSTGRESQL", "display_name": "PostgreSQL", "snapshot": True, "execute": True},
        {"source_type": "MYSQL", "display_name": "MySQL", "snapshot": True, "execute": True},
        {"source_type": "ORACLE", "display_name": "Oracle", "snapshot": True, "execute": True},
    ]}


@router.post("/data-sources", response_model=DataSourceView, status_code=status.HTTP_201_CREATED)
async def create_source(body: DataSourceCreate, request: Request, service: Service, context: Auth) -> DataSourceView:
    domain_id, actor_id = _require_management(request, context)
    return await service.create_source(domain_id=domain_id, actor_id=actor_id, command=body)


@router.post("/data-sources/test-connection", response_model=DataSourceConnectionTestResult)
async def test_source_connection(
    body: DataSourceConnectionTest, request: Request, service: Service, context: Auth,
) -> DataSourceConnectionTestResult:
    domain_id, actor_id = _require_management(request, context)
    return await service.test_source_connection(domain_id=domain_id, command=body)


@router.get("/data-sources", response_model=DataSourcePage)
async def list_sources(
    request: Request, service: Service, context: Auth,
    cursor: UUID | None = None, limit: int = Query(default=50, ge=1, le=200),
) -> DataSourcePage:
    domain_id, actor_id = _require_management(request, context)
    return await service.list_sources(domain_id=domain_id, after_id=cursor, limit=limit)


@router.get("/data-sources/{data_source_id}", response_model=DataSourceDetail)
async def get_source(
    data_source_id: UUID, request: Request, service: Service, context: Auth,
) -> DataSourceDetail:
    domain_id, _actor_id = _require_management(request, context)
    return await service.get_source(domain_id=domain_id, data_source_id=data_source_id)


@router.put("/data-sources/{data_source_id}", response_model=DataSourceView)
async def update_source(
    data_source_id: UUID, body: DataSourceUpdate, request: Request,
    service: Service, context: Auth,
) -> DataSourceView:
    domain_id, actor_id = _require_management(request, context)
    return await service.update_source(
        domain_id=domain_id, actor_id=actor_id,
        data_source_id=data_source_id, command=body,
    )


@router.patch("/data-sources/{data_source_id}/status", response_model=DataSourceView)
async def change_source_status(
    data_source_id: UUID, body: DataSourceStatusChange, request: Request,
    service: Service, context: Auth,
) -> DataSourceView:
    domain_id, actor_id = _require_management(request, context)
    return await service.change_source_status(
        domain_id=domain_id, actor_id=actor_id, data_source_id=data_source_id, command=body,
    )


@router.post("/data-sources/{data_source_id}/snapshots", response_model=SchemaSnapshotReceipt, status_code=status.HTTP_202_ACCEPTED)
async def create_snapshot(data_source_id: UUID, request: Request, service: Service, context: Auth) -> SchemaSnapshotReceipt:
    domain_id, actor_id = _require_management(request, context)
    return await service.request_snapshot(domain_id=domain_id, data_source_id=data_source_id, actor_id=actor_id)


@router.get("/data-sources/{data_source_id}/snapshots", response_model=SchemaSnapshotPage)
async def list_snapshots(data_source_id: UUID, request: Request, service: Service, context: Auth) -> SchemaSnapshotPage:
    domain_id, _actor_id = _require_management(request, context)
    return await service.list_snapshots(domain_id=domain_id, data_source_id=data_source_id)


@router.get("/snapshots/{snapshot_id}", response_model=SchemaSnapshotDetail)
async def get_snapshot(snapshot_id: UUID, request: Request, service: Service, context: Auth) -> SchemaSnapshotDetail:
    domain_id, _actor_id = _require_management(request, context)
    return await service.get_snapshot(domain_id=domain_id, snapshot_id=snapshot_id)


@router.post("/snapshots/{snapshot_id}/selection", response_model=SchemaSnapshotDetail)
async def select_snapshot_objects(snapshot_id: UUID, body: SchemaObjectSelection, request: Request, service: Service, context: Auth) -> SchemaSnapshotDetail:
    domain_id, actor_id = _require_management(request, context)
    return await service.select_snapshot_objects(domain_id=domain_id, actor_id=actor_id, snapshot_id=snapshot_id, command=body)


@router.post("/snapshots/{snapshot_id}/objects/{object_id}/retry", response_model=SchemaSnapshotDetail)
async def retry_snapshot_object(snapshot_id: UUID, object_id: UUID, request: Request, service: Service, context: Auth) -> SchemaSnapshotDetail:
    domain_id, actor_id = _require_management(request, context)
    return await service.retry_snapshot_object(domain_id=domain_id, actor_id=actor_id, snapshot_id=snapshot_id, object_id=object_id)


@router.post("/snapshots/{snapshot_id}/objects/{object_id}/manual-ddl", response_model=SchemaSnapshotDetail)
async def supply_manual_schema(snapshot_id: UUID, object_id: UUID, body: ManualSchemaDefinition, request: Request, service: Service, context: Auth) -> SchemaSnapshotDetail:
    domain_id, actor_id = _require_management(request, context)
    return await service.supply_manual_schema(domain_id=domain_id, actor_id=actor_id, snapshot_id=snapshot_id, object_id=object_id, command=body)


@router.post("/snapshots/{snapshot_id}/semantic-model-draft", response_model=SemanticModelGenerationReceipt, status_code=status.HTTP_202_ACCEPTED)
async def generate_semantic_model_draft(snapshot_id: UUID, body: SemanticModelCandidateRequest, request: Request, service: Service, context: Auth) -> SemanticModelGenerationReceipt:
    domain_id, actor_id = _require_management(request, context)
    return await service.generate_model_draft(
        domain_id=domain_id, actor_id=actor_id, snapshot_id=snapshot_id, command=body,
    )


@router.get("/semantic-model-generation-jobs/{generation_job_id}", response_model=SemanticModelGenerationView)
async def get_semantic_model_generation_job(generation_job_id: UUID, request: Request, service: Service, context: Auth) -> SemanticModelGenerationView:
    domain_id, _actor_id = _require_management(request, context)
    return await service.get_model_generation_job(domain_id=domain_id, generation_job_id=generation_job_id)


@router.post("/semantic-models", response_model=SemanticModelDraftView, status_code=status.HTTP_201_CREATED)
async def create_semantic_model(body: SemanticModelDraftCreate, request: Request, service: Service, context: Auth) -> SemanticModelDraftView:
    domain_id, actor_id = _require_management(request, context)
    return await service.create_model_draft(domain_id=domain_id, actor_id=actor_id, command=body)


@router.get("/semantic-models", response_model=SemanticModelPage)
async def list_semantic_models(request: Request, service: Service, context: Auth, cursor: UUID | None = None, limit: int = Query(default=50, ge=1, le=200)) -> SemanticModelPage:
    domain_id, _actor_id = _require_management(request, context)
    return await service.list_semantic_models(domain_id=domain_id, after_id=cursor, limit=limit)


@router.post("/semantic-models/search", response_model=SemanticModelPage)
async def search_semantic_models(
    body: SemanticModelSearch, request: Request, service: Service, context: Auth,
) -> SemanticModelPage:
    domain_id, _actor_id = _require_management(request, context)
    return await service.search_semantic_models(domain_id=domain_id, search=body)


@router.get("/semantic-models/{semantic_model_id}", response_model=SemanticModelDetail)
async def get_semantic_model(semantic_model_id: UUID, request: Request, service: Service, context: Auth) -> SemanticModelDetail:
    domain_id, _actor_id = _require_management(request, context)
    return await service.get_semantic_model(domain_id=domain_id, semantic_model_id=semantic_model_id)


@router.patch("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}", response_model=SemanticModelDraftView)
async def update_semantic_model_draft(semantic_model_id: UUID, semantic_model_version_id: UUID, body: SemanticModelDraftUpdate, request: Request, service: Service, context: Auth) -> SemanticModelDraftView:
    domain_id, _actor_id = _require_management(request, context)
    return await service.update_model_draft(
        domain_id=domain_id, semantic_model_id=semantic_model_id,
        semantic_model_version_id=semantic_model_version_id, command=body,
    )


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/validations", response_model=SemanticModelValidationReceipt, status_code=status.HTTP_202_ACCEPTED)
async def create_model_validation(semantic_model_id: UUID, semantic_model_version_id: UUID, body: SemanticModelValidationRequest, request: Request, service: Service, context: Auth) -> SemanticModelValidationReceipt:
    domain_id, actor_id = _require_management(request, context)
    return await service.create_model_validation(
        domain_id=domain_id, actor_id=actor_id,
        semantic_model_id=semantic_model_id,
        semantic_model_version_id=semantic_model_version_id, command=body,
    )


@router.get("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/validations/{run_id}", response_model=SemanticModelValidationResult)
async def get_model_validation(semantic_model_id: UUID, semantic_model_version_id: UUID, run_id: UUID, request: Request, service: Service, context: Auth) -> SemanticModelValidationResult:
    domain_id, _actor_id = _require_management(request, context)
    return await service.get_model_validation(
        domain_id=domain_id, semantic_model_id=semantic_model_id,
        semantic_model_version_id=semantic_model_version_id, run_id=run_id,
    )


@router.get("/verified-queries", response_model=VerifiedQueryPage)
async def list_verified_queries(request: Request, service: Service, context: Auth, cursor: UUID | None = None, limit: int = Query(default=50, ge=1, le=200)) -> VerifiedQueryPage:
    domain_id, _actor_id = _require_management(request, context)
    return await service.list_verified_queries(domain_id=domain_id, after_id=cursor, limit=limit)


@router.post("/verified-queries", response_model=VerifiedQueryView, status_code=status.HTTP_201_CREATED)
async def promote_verified_query(body: PromoteVerifiedQueryCommand, request: Request, service: Service, context: Auth) -> VerifiedQueryView:
    domain_id, actor_id = _require_management(request, context)
    return await service.promote_verified_query(domain_id=domain_id, actor_id=actor_id, command=body)


@router.get("/audits", response_model=DataQueryAuditPage)
async def list_audits(request: Request, service: Service, context: Auth, cursor: UUID | None = None, limit: int = Query(default=50, ge=1, le=200)) -> DataQueryAuditPage:
    domain_id, _actor_id = _require_management(request, context)
    return await service.list_audits(domain_id=domain_id, after_id=cursor, limit=limit)


@router.post("/policy-bindings", response_model=PolicyBindingView, status_code=status.HTTP_201_CREATED)
async def create_policy_binding(body: PolicyBindingCreate, request: Request, service: Service, context: Auth) -> PolicyBindingView:
    domain_id, actor_id = _require_management(request, context)
    return await service.create_policy(domain_id=domain_id, actor_id=actor_id, command=body)


@router.get("/policy-bindings", response_model=PolicyBindingPage)
async def list_policy_bindings(request: Request, service: Service, context: Auth, cursor: UUID | None = None, limit: int = Query(default=50, ge=1, le=200)) -> PolicyBindingPage:
    domain_id, _actor_id = _require_management(request, context)
    return await service.list_policies(domain_id=domain_id, after_id=cursor, limit=limit)


@router.get("/policy-bindings/{policy_binding_id}", response_model=PolicyBindingDetail)
async def get_policy_binding(policy_binding_id: UUID, request: Request, service: Service, context: Auth) -> PolicyBindingDetail:
    domain_id, _actor_id = _require_management(request, context)
    return await service.get_policy(domain_id=domain_id, policy_binding_id=policy_binding_id)


@router.patch("/policy-bindings/{policy_binding_id}/status", response_model=PolicyBindingView)
async def change_policy_binding_status(policy_binding_id: UUID, body: PolicyBindingStatusChange, request: Request, service: Service, context: Auth) -> PolicyBindingView:
    domain_id, actor_id = _require_management(request, context)
    return await service.change_policy_status(domain_id=domain_id, actor_id=actor_id, policy_binding_id=policy_binding_id, command=body)


@router.post("/agent-bindings", response_model=AgentBindingView, status_code=status.HTTP_201_CREATED)
async def create_agent_binding(body: AgentBindingCreate, request: Request, service: Service, context: Auth) -> AgentBindingView:
    domain_id, actor_id = _require_management(request, context)
    return await service.create_agent_binding(domain_id=domain_id, actor_id=actor_id, command=body)


@router.get("/agent-bindings", response_model=AgentBindingPage)
async def list_agent_bindings(request: Request, service: Service, context: Auth, cursor: UUID | None = None, limit: int = Query(default=50, ge=1, le=200)) -> AgentBindingPage:
    domain_id, _actor_id = _require_management(request, context)
    return await service.list_agent_bindings(domain_id=domain_id, after_id=cursor, limit=limit)


@router.patch("/agent-bindings/{agent_binding_id}/status", response_model=AgentBindingView)
async def change_agent_binding_status(agent_binding_id: UUID, body: AgentBindingStatusChange, request: Request, service: Service, context: Auth) -> AgentBindingView:
    domain_id, actor_id = _require_management(request, context)
    return await service.change_agent_binding_status(domain_id=domain_id, actor_id=actor_id, agent_binding_id=agent_binding_id, command=body)


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/publish", status_code=status.HTTP_204_NO_CONTENT)
async def publish_semantic_model(
    semantic_model_id: UUID,
    semantic_model_version_id: UUID,
    body: PublishSemanticModelCommand,
    request: Request,
    service: Service,
    context: Auth,
) -> Response:
    domain_id, actor_id = _require_management(request, context)
    if body.semantic_model_id != semantic_model_id or body.semantic_model_version_id != semantic_model_version_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail={"code": "RESOURCE_ID_MISMATCH"})
    await service.publish_model_version(domain_id=domain_id, actor_id=actor_id, command=body)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/submit-review", status_code=status.HTTP_204_NO_CONTENT)
async def submit_semantic_model_review(
    semantic_model_id: UUID, semantic_model_version_id: UUID,
    body: SubmitSemanticModelReviewCommand, request: Request, service: Service, context: Auth,
) -> Response:
    domain_id, actor_id = _require_management(request, context)
    await service.submit_model_review(
        domain_id=domain_id, actor_id=actor_id, semantic_model_id=semantic_model_id,
        semantic_model_version_id=semantic_model_version_id, command=body,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/return-for-revision", status_code=status.HTTP_204_NO_CONTENT)
async def return_semantic_model_for_revision(
    semantic_model_id: UUID, semantic_model_version_id: UUID,
    body: ReturnSemanticModelForRevisionCommand, request: Request, service: Service, context: Auth,
) -> Response:
    domain_id, actor_id = _require_management(request, context)
    await service.return_model_for_revision(
        domain_id=domain_id, actor_id=actor_id, semantic_model_id=semantic_model_id,
        semantic_model_version_id=semantic_model_version_id, command=body,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/retire", status_code=status.HTTP_204_NO_CONTENT)
async def retire_semantic_model_version_endpoint(
    semantic_model_id: UUID, semantic_model_version_id: UUID,
    body: RetireSemanticModelVersionCommand, request: Request, service: Service, context: Auth,
) -> Response:
    domain_id, actor_id = _require_management(request, context)
    await service.retire_model_version(
        domain_id=domain_id, actor_id=actor_id, semantic_model_id=semantic_model_id,
        semantic_model_version_id=semantic_model_version_id, command=body,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/semantic-models/{semantic_model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_semantic_model(
    semantic_model_id: UUID, body: DeleteSemanticModelCommand,
    request: Request, service: Service, context: Auth,
) -> Response:
    domain_id, _actor_id = _require_management(request, context)
    await service.delete_semantic_model(
        domain_id=domain_id, semantic_model_id=semantic_model_id, command=body,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
