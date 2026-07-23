"""Collection 与 Agent 绑定管理 API。"""

from uuid import UUID
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import get_actor_id, require_domain_match
from knowledge_core.application.collections import (
    BindAgentCollectionCommand, ChangeCollectionStatusCommand, CollectionAlreadyExistsError,
    CollectionDeletionStateError, CollectionInUseError, CollectionNotFoundError, CreateCollectionCommand,
)

router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/domains/{{domain_id}}",
    tags=["Knowledge Core Collections"],
)


class CreateCollectionRequest(BaseModel):
    collection_key: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    display_name: str = Field(min_length=1, max_length=256)
    embedding_model_id: UUID
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int = Field(default=1, ge=0, le=999)
    metadata: dict = Field(default_factory=dict)


class BindingRequest(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


class CollectionStatusRequest(BaseModel):
    status: str = Field(pattern=r"^(ACTIVE|DISABLED)$")


def _collection(entity) -> dict:
    return {
        "collection_id": entity.collection_id, "app_id": int(entity.app_id),
        "domain_id": int(entity.domain_id), "collection_key": entity.collection_key,
        "display_name": entity.display_name, "description": entity.description,
        "embedding_model_id": entity.embedding_model_id, "status": entity.status,
        "default_security_level": int(entity.default_security_level),
        "metadata": entity.metadata_json or {},
    }


@router.post("/collections", status_code=status.HTTP_201_CREATED)
async def create_collection(domain_id: int, payload: CreateCollectionRequest, request: Request):
    require_domain_match(request, domain_id)
    actor = get_actor_id(request)
    try:
        entity = await request.app.state.kc_collection_service.create(CreateCollectionCommand(
            domain_id=domain_id, collection_key=payload.collection_key,
            display_name=payload.display_name, embedding_model_id=payload.embedding_model_id,
            description=payload.description, default_security_level=payload.default_security_level,
            metadata=payload.metadata, actor_id=actor,
        ))
    except CollectionAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail={"code": "COLLECTION_EXISTS", "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_COLLECTION", "message": str(exc)}) from exc
    return _collection(entity)


@router.get("/collections")
async def list_collections(domain_id: int, request: Request):
    require_domain_match(request, domain_id)
    entities = await request.app.state.kc_collection_service.list(domain_id=domain_id)
    return {"collections": [_collection(entity) for entity in entities]}


@router.get("/collections/{collection_key}")
async def get_collection(domain_id: int, collection_key: str, request: Request):
    require_domain_match(request, domain_id)
    try:
        entity = await request.app.state.kc_collection_service.get(domain_id=domain_id, collection_key=collection_key)
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
    return _collection(entity)


@router.patch("/collections/{collection_key}")
async def change_collection_status(domain_id: int, collection_key: str, payload: CollectionStatusRequest, request: Request):
    require_domain_match(request, domain_id)
    try:
        entity = await request.app.state.kc_collection_service.change_status(ChangeCollectionStatusCommand(
            domain_id=domain_id, collection_key=collection_key, status=payload.status,
            actor_id=get_actor_id(request),
        ))
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
    except CollectionDeletionStateError as exc:
        raise HTTPException(status_code=409, detail={"code": "COLLECTION_DELETING", "message": str(exc)}) from exc
    return _collection(entity)


@router.delete("/collections/{collection_key}", status_code=status.HTTP_202_ACCEPTED)
async def delete_collection(domain_id: int, collection_key: str, request: Request):
    require_domain_match(request, domain_id)
    try:
        job_id = await request.app.state.kc_collection_service.request_delete(
            domain_id=domain_id, collection_key=collection_key,
            actor_id=get_actor_id(request),
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
    except CollectionInUseError as exc:
        raise HTTPException(status_code=409, detail={"code": "COLLECTION_IN_USE", "message": str(exc)}) from exc
    except CollectionDeletionStateError as exc:
        raise HTTPException(status_code=409, detail={"code": "COLLECTION_DELETING", "message": str(exc)}) from exc
    return {"status": "DELETING", "purge_job_id": job_id}


@router.put("/agents/{agent_id}/collections/{collection_key}/binding")
async def bind_collection(domain_id: int, agent_id: UUID, collection_key: str, request: Request, payload: BindingRequest | None = None):
    require_domain_match(request, domain_id)
    body = payload or BindingRequest()
    try:
        entity = await request.app.state.kc_binding_service.bind_agent(BindAgentCollectionCommand(
            domain_id=domain_id, collection_key=collection_key, agent_id=agent_id,
            actor_id=get_actor_id(request), note=body.note,
        ))
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
    return {
        "binding_id": entity.binding_id,
        "collection_id": entity.collection_id,
        "agent_id": agent_id,
        "status": entity.status,
    }


@router.delete("/agents/{agent_id}/collections/{collection_key}/binding", status_code=status.HTTP_204_NO_CONTENT)
async def unbind_collection(domain_id: int, agent_id: UUID, collection_key: str, request: Request):
    require_domain_match(request, domain_id)
    try:
        await request.app.state.kc_binding_service.unbind_agent(BindAgentCollectionCommand(
            domain_id=domain_id, collection_key=collection_key, agent_id=agent_id,
            actor_id=get_actor_id(request),
        ))
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc


@router.get("/agents/{agent_id}/collection-bindings")
async def list_agent_bindings(domain_id: int, agent_id: UUID, request: Request):
    require_domain_match(request, domain_id)
    bindings = await request.app.state.kc_binding_service.list_agent(
        domain_id=domain_id,
        agent_id=agent_id,
    )
    return {"bindings": [{
        "binding_id": binding.binding_id, "collection_id": binding.collection_id,
        "consumer_type": binding.consumer_type, "consumer_id": binding.consumer_id,
        "status": binding.status, "note": binding.note,
    } for binding in bindings]}
