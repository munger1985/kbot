"""问数治理的公开 BFF；浏览器不得直连 Data Query 内部服务。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from platform_clients import DataQueryClient
from platform_core.contracts import PUBLIC_API_V1
from platform_core.contracts.data_query import DataQueryPlanV1
from platform_core.security import get_auth_context


router = APIRouter(prefix=f"{PUBLIC_API_V1}/data-query", tags=["Data Query"])


class StatusChangeRequest(BaseModel):
    status: str = Field(pattern=r"^DISABLED$")


class DataSourceEndpointRequest(BaseModel):
    host: str = Field(min_length=1, max_length=253, pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
    port: int = Field(ge=1, le=65535)
    database: str = Field(min_length=1, max_length=128)
    allowed_schemas: tuple[str, ...] = Field(min_length=1, max_length=32)
    tls_enabled: bool = True


class DataSourceCredentialsRequest(BaseModel):
    username: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=1024)


class DataSourceConnectionTestRequest(BaseModel):
    source_type: str = Field(pattern=r"^(POSTGRESQL|MYSQL|ORACLE)$")
    endpoint: DataSourceEndpointRequest
    credentials: DataSourceCredentialsRequest


class DataSourceCreateRequest(DataSourceConnectionTestRequest):
    display_name: str = Field(min_length=1, max_length=256)


class DataSourceUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=256)
    endpoint: DataSourceEndpointRequest
    credentials: DataSourceCredentialsRequest | None = None
    expected_row_version: int = Field(ge=1)


class SemanticModelDraftCreateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    data_source_id: UUID
    schema_snapshot_id: UUID
    definition: dict[str, Any]


class SchemaObjectSelectionRequest(BaseModel):
    object_ids: tuple[UUID, ...] = Field(min_length=1, max_length=5000)


class ManualSchemaDefinitionRequest(BaseModel):
    ddl: str = Field(min_length=10, max_length=100_000)


class SemanticModelCandidateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    business_context: str | None = Field(default=None, max_length=4000)
    object_ids: tuple[UUID, ...] = Field(default=(), max_length=64)
    ai_model_id: UUID | None = None
    allow_ai_metadata: bool = False


class SemanticModelDraftUpdateRequest(BaseModel):
    definition: dict[str, Any]
    expected_row_version: int = Field(ge=1)


class SemanticModelValidationRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    ai_model_id: UUID
    allow_ai_metadata: bool = False


class PromoteVerifiedQueryRequest(BaseModel):
    data_query_run_id: UUID
    assertion: dict[str, object] = Field(default_factory=dict)


class QueryBudgetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rows: int = Field(default=1000, ge=1, le=10000)
    max_result_bytes: int = Field(default=1_048_576, ge=1024, le=16_777_216)
    statement_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_concurrent_runs: int = Field(default=4, ge=1, le=64)


class PolicyBindingCreateRequest(BaseModel):
    semantic_model_ids: tuple[UUID, ...] = Field(min_length=1, max_length=64)
    budget: QueryBudgetRequest = Field(default_factory=QueryBudgetRequest)


class AgentBindingCreateRequest(BaseModel):
    agent_id: UUID
    semantic_model_id: UUID
    policy_binding_id: UUID


class PublishSemanticModelRequest(BaseModel):
    schema_snapshot_id: UUID
    expected_row_version: int = Field(ge=1)


class SubmitSemanticModelReviewRequest(BaseModel):
    expected_row_version: int = Field(ge=1)


class ReturnSemanticModelForRevisionRequest(BaseModel):
    review_comment: str = Field(min_length=2, max_length=2000)
    expected_row_version: int = Field(ge=1)

    @field_validator("review_comment")
    @classmethod
    def normalize_review_comment(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("审核意见至少需要 2 个字符")
        return normalized


class RetireSemanticModelVersionRequest(BaseModel):
    expected_row_version: int = Field(ge=1)


class DataQueryRunCreateRequest(BaseModel):
    original_question: str = Field(min_length=1, max_length=8000)
    standalone_query: str = Field(min_length=1, max_length=8000)
    plan: DataQueryPlanV1
    agent_id: UUID
    parent_agent_run_id: UUID | None = None
    parent_agent_task_id: UUID | None = None
    deadline_at: datetime | None = None


def _client(request: Request) -> DataQueryClient:
    client = getattr(request.app.state, "data_query_client", None)
    if client is None:
        raise RuntimeError("Data Query Client 尚未初始化")
    return cast(DataQueryClient, client)


def _internal_context(request: Request):
    return get_auth_context(request)


def _version(if_match: str) -> int:
    value = if_match.strip()
    if not (value.startswith('"rv-') and value.endswith('"')):
        raise HTTPException(status_code=422, detail={"code": "ETAG_INVALID", "message": 'If-Match 必须使用 "rv-N" 格式'})
    try:
        version = int(value[4:-1])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "ETAG_INVALID", "message": 'If-Match 必须使用 "rv-N" 格式'}) from exc
    if version < 1:
        raise HTTPException(status_code=422, detail={"code": "ETAG_INVALID", "message": "row_version 必须大于零"})
    return version


async def _list(resource: str, cursor: UUID | None, limit: int, request: Request) -> dict[str, Any]:
    return await _client(request).management_list(
        resource=resource, cursor=cursor, limit=limit, auth_context=_internal_context(request)
    )


@router.get("/connector-capabilities")
async def list_connector_capabilities(request: Request) -> dict[str, Any]:
    return await _client(request).management_capabilities(auth_context=_internal_context(request))


@router.get("/data-sources")
async def list_data_sources(request: Request, cursor: UUID | None = None, limit: int = 50) -> dict[str, Any]:
    return await _list("data-sources", cursor, limit, request)


@router.post("/data-sources/test-connection")
async def test_data_source_connection(
    body: DataSourceConnectionTestRequest, request: Request,
) -> dict[str, Any]:
    return await _client(request).management_test_connection(
        payload=body.model_dump(), auth_context=_internal_context(request)
    )


@router.post("/data-sources", status_code=201)
async def create_data_source(
    body: DataSourceCreateRequest, request: Request,
) -> dict[str, Any]:
    return await _client(request).management_create(
        resource="data-sources", payload=body.model_dump(),
        auth_context=_internal_context(request),
    )


@router.post("/data-sources/{data_source_id}/snapshots", status_code=202)
async def request_data_source_snapshot(
    data_source_id: UUID, request: Request,
) -> dict[str, Any]:
    return await _client(request).management_request_snapshot(
        data_source_id=data_source_id, auth_context=_internal_context(request)
    )


@router.get("/data-sources/{data_source_id}/snapshots")
async def list_data_source_snapshots(data_source_id: UUID, request: Request) -> dict[str, Any]:
    return await _client(request).management_action(
        method="GET", path=f"data-sources/{data_source_id}/snapshots",
        payload=None, auth_context=_internal_context(request),
    )


@router.get("/snapshots/{snapshot_id}")
async def get_schema_snapshot(snapshot_id: UUID, request: Request) -> dict[str, Any]:
    return await _client(request).management_action(
        method="GET", path=f"snapshots/{snapshot_id}", payload=None,
        auth_context=_internal_context(request),
    )


@router.post("/snapshots/{snapshot_id}/selection")
async def select_schema_objects(snapshot_id: UUID, body: SchemaObjectSelectionRequest, request: Request) -> dict[str, Any]:
    return await _client(request).management_action(
        method="POST", path=f"snapshots/{snapshot_id}/selection",
        payload=body.model_dump(mode="json"), auth_context=_internal_context(request),
    )


@router.post("/snapshots/{snapshot_id}/objects/{object_id}/retry")
async def retry_schema_object(snapshot_id: UUID, object_id: UUID, request: Request) -> dict[str, Any]:
    return await _client(request).management_action(
        method="POST", path=f"snapshots/{snapshot_id}/objects/{object_id}/retry",
        payload={}, auth_context=_internal_context(request),
    )


@router.post("/snapshots/{snapshot_id}/objects/{object_id}/manual-ddl")
async def supply_manual_schema(snapshot_id: UUID, object_id: UUID, body: ManualSchemaDefinitionRequest, request: Request) -> dict[str, Any]:
    return await _client(request).management_action(
        method="POST", path=f"snapshots/{snapshot_id}/objects/{object_id}/manual-ddl",
        payload=body.model_dump(), auth_context=_internal_context(request),
    )


@router.post("/snapshots/{snapshot_id}/semantic-model-draft", status_code=202)
async def generate_semantic_model_draft(snapshot_id: UUID, body: SemanticModelCandidateRequest, request: Request) -> dict[str, Any]:
    return await _client(request).management_action(
        method="POST", path=f"snapshots/{snapshot_id}/semantic-model-draft",
        payload=body.model_dump(mode="json"), auth_context=_internal_context(request),
    )


@router.get("/semantic-model-generation-jobs/{generation_job_id}")
async def get_semantic_model_generation_job(generation_job_id: UUID, request: Request) -> dict[str, Any]:
    return await _client(request).management_get(
        resource="semantic-model-generation-jobs", resource_id=generation_job_id,
        auth_context=_internal_context(request),
    )


@router.get("/data-sources/{data_source_id}")
async def get_data_source(data_source_id: UUID, request: Request) -> dict[str, Any]:
    return await _client(request).management_get(resource="data-sources", resource_id=data_source_id, auth_context=_internal_context(request))


@router.put("/data-sources/{data_source_id}")
async def update_data_source(
    data_source_id: UUID,
    body: DataSourceUpdateRequest,
    request: Request,
) -> dict[str, Any]:
    return await _client(request).management_update(
        resource="data-sources",
        resource_id=data_source_id,
        payload=body.model_dump(mode="json"),
        auth_context=_internal_context(request),
    )


@router.patch("/data-sources/{data_source_id}/status")
async def change_data_source_status(data_source_id: UUID, body: StatusChangeRequest, request: Request, if_match: str = Header(alias="If-Match")) -> dict[str, Any]:
    return await _client(request).management_change_status(resource="data-sources", resource_id=data_source_id, payload={"status": body.status, "expected_row_version": _version(if_match)}, auth_context=_internal_context(request))


@router.get("/semantic-models")
async def list_semantic_models(request: Request, cursor: UUID | None = None, limit: int = 50) -> dict[str, Any]:
    return await _list("semantic-models", cursor, limit, request)


@router.post("/semantic-models", status_code=201)
async def create_semantic_model(body: SemanticModelDraftCreateRequest, request: Request) -> dict[str, Any]:
    return await _client(request).management_create(
        resource="semantic-models", payload=body.model_dump(mode="json"),
        auth_context=_internal_context(request),
    )


@router.get("/semantic-models/{semantic_model_id}")
async def get_semantic_model(semantic_model_id: UUID, request: Request) -> dict[str, Any]:
    return await _client(request).management_get(resource="semantic-models", resource_id=semantic_model_id, auth_context=_internal_context(request))


@router.patch("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}")
async def update_semantic_model_draft(semantic_model_id: UUID, semantic_model_version_id: UUID, body: SemanticModelDraftUpdateRequest, request: Request) -> dict[str, Any]:
    return await _client(request).management_action(
        method="PATCH",
        path=f"semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}",
        payload=body.model_dump(mode="json"), auth_context=_internal_context(request),
    )


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/validations", status_code=202)
async def create_semantic_model_validation(semantic_model_id: UUID, semantic_model_version_id: UUID, body: SemanticModelValidationRequest, request: Request, idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128)) -> dict[str, Any]:
    return await _client(request).management_action(
        method="POST",
        path=f"semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/validations",
        payload={**body.model_dump(mode="json"), "idempotency_key": idempotency_key},
        auth_context=_internal_context(request),
    )


@router.get("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/validations/{run_id}")
async def get_semantic_model_validation(semantic_model_id: UUID, semantic_model_version_id: UUID, run_id: UUID, request: Request) -> dict[str, Any]:
    return await _client(request).management_action(
        method="GET",
        path=f"semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/validations/{run_id}",
        payload=None, auth_context=_internal_context(request),
    )


@router.get("/policy-bindings")
async def list_policy_bindings(request: Request, cursor: UUID | None = None, limit: int = 50) -> dict[str, Any]:
    return await _list("policy-bindings", cursor, limit, request)


@router.get("/policy-bindings/{policy_binding_id}")
async def get_policy_binding(policy_binding_id: UUID, request: Request) -> dict[str, Any]:
    return await _client(request).management_get(resource="policy-bindings", resource_id=policy_binding_id, auth_context=_internal_context(request))


@router.patch("/policy-bindings/{policy_binding_id}/status")
async def change_policy_binding_status(policy_binding_id: UUID, body: StatusChangeRequest, request: Request, if_match: str = Header(alias="If-Match")) -> dict[str, Any]:
    return await _client(request).management_change_status(resource="policy-bindings", resource_id=policy_binding_id, payload={"status": body.status, "expected_row_version": _version(if_match)}, auth_context=_internal_context(request))


@router.post("/policy-bindings", status_code=201)
async def create_policy_binding(body: PolicyBindingCreateRequest, request: Request) -> dict[str, Any]:
    return await _client(request).management_create(
        resource="policy-bindings",
        payload={
            "semantic_model_ids": [str(item) for item in body.semantic_model_ids],
            "budget": body.budget.model_dump(),
        }, auth_context=_internal_context(request),
    )


@router.get("/agent-bindings")
async def list_agent_bindings(request: Request, cursor: UUID | None = None, limit: int = 50) -> dict[str, Any]:
    return await _list("agent-bindings", cursor, limit, request)


@router.patch("/agent-bindings/{agent_binding_id}/status")
async def change_agent_binding_status(agent_binding_id: UUID, body: StatusChangeRequest, request: Request, if_match: str = Header(alias="If-Match")) -> dict[str, Any]:
    return await _client(request).management_change_status(resource="agent-bindings", resource_id=agent_binding_id, payload={"status": body.status, "expected_row_version": _version(if_match)}, auth_context=_internal_context(request))


@router.post("/agent-bindings", status_code=201)
async def create_agent_binding(body: AgentBindingCreateRequest, request: Request) -> dict[str, Any]:
    return await _client(request).management_create(
        resource="agent-bindings", payload=body.model_dump(mode="json"), auth_context=_internal_context(request)
    )


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/publish", status_code=204)
async def publish_semantic_model(semantic_model_id: UUID, semantic_model_version_id: UUID, body: PublishSemanticModelRequest, request: Request) -> None:
    await _client(request).management_publish_model(
        semantic_model_id=semantic_model_id, semantic_model_version_id=semantic_model_version_id,
        payload={"semantic_model_id": str(semantic_model_id), "semantic_model_version_id": str(semantic_model_version_id), **body.model_dump(mode="json")}, auth_context=_internal_context(request),
    )


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/submit-review", status_code=204)
async def submit_semantic_model_review(semantic_model_id: UUID, semantic_model_version_id: UUID, body: SubmitSemanticModelReviewRequest, request: Request) -> None:
    await _client(request).management_submit_model_review(
        semantic_model_id=semantic_model_id, semantic_model_version_id=semantic_model_version_id,
        payload={"expected_row_version": body.expected_row_version}, auth_context=_internal_context(request),
    )


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/return-for-revision", status_code=204)
async def return_semantic_model_for_revision(semantic_model_id: UUID, semantic_model_version_id: UUID, body: ReturnSemanticModelForRevisionRequest, request: Request) -> None:
    await _client(request).management_return_model_for_revision(
        semantic_model_id=semantic_model_id, semantic_model_version_id=semantic_model_version_id,
        payload=body.model_dump(mode="json"), auth_context=_internal_context(request),
    )


@router.post("/semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/retire", status_code=204)
async def retire_semantic_model_version(
    semantic_model_id: UUID, semantic_model_version_id: UUID,
    body: RetireSemanticModelVersionRequest, request: Request,
) -> None:
    await _client(request).management_action(
        method="POST",
        path=f"semantic-models/{semantic_model_id}/versions/{semantic_model_version_id}/retire",
        payload=body.model_dump(mode="json"), auth_context=_internal_context(request),
    )


@router.delete("/semantic-models/{semantic_model_id}", status_code=204)
async def delete_semantic_model(
    semantic_model_id: UUID, request: Request,
    if_match: str = Header(alias="If-Match"),
) -> None:
    await _client(request).management_action(
        method="DELETE", path=f"semantic-models/{semantic_model_id}",
        payload={"expected_row_version": _version(if_match)},
        auth_context=_internal_context(request),
    )


@router.get("/verified-queries")
async def list_verified_queries(request: Request, cursor: UUID | None = None, limit: int = 50) -> dict[str, Any]:
    return await _list("verified-queries", cursor, limit, request)


@router.post("/verified-queries", status_code=201)
async def promote_verified_query(body: PromoteVerifiedQueryRequest, request: Request) -> dict[str, Any]:
    return await _client(request).management_create(resource="verified-queries", payload=body.model_dump(mode="json"), auth_context=_internal_context(request))


@router.get("/audits")
async def list_audits(request: Request, cursor: UUID | None = None, limit: int = 50) -> dict[str, Any]:
    return await _list("audits", cursor, limit, request)


@router.post("/runs", status_code=202)
async def create_data_query_run(
    body: DataQueryRunCreateRequest,
    request: Request,
    idempotency_key: str = Header(
        alias="Idempotency-Key", min_length=1, max_length=128
    ),
) -> dict[str, Any]:
    return await _client(request).create_run(
        payload={
            **body.model_dump(mode="json"),
            "idempotency_key": idempotency_key,
        },
        auth_context=_internal_context(request),
    )


@router.get("/runs/{data_query_run_id}")
async def get_data_query_run(
    data_query_run_id: UUID, request: Request
) -> dict[str, Any]:
    return await _client(request).get_run(
        data_query_run_id=data_query_run_id,
        auth_context=_internal_context(request),
    )


@router.get("/runs/{data_query_run_id}/result")
async def get_data_query_result(
    data_query_run_id: UUID, request: Request
) -> dict[str, Any]:
    return await _client(request).get_result(
        data_query_run_id=data_query_run_id,
        auth_context=_internal_context(request),
    )


@router.post("/runs/{data_query_run_id}/cancel", status_code=202)
async def cancel_data_query_run(
    data_query_run_id: UUID, request: Request
) -> dict[str, Any]:
    return await _client(request).cancel_run(
        data_query_run_id=data_query_run_id,
        auth_context=_internal_context(request),
    )
