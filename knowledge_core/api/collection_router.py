"""Collection and Agent binding management APIs."""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from knowledge_core.application.collections import (
    BindAgentCollectionCommand, ChangeCollectionStatusCommand, CollectionAlreadyExistsError,
    CollectionDeletionStateError, CollectionInUseError, CollectionNotFoundError, CreateCollectionCommand,
)

router = APIRouter(prefix="/api/v2/knowledge/domains/{domain_id}", tags=["Knowledge Core Collections V2"])


class CreateCollectionRequest(BaseModel):
    collection_key: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    display_name: str = Field(min_length=1, max_length=256)
    embedding_model_id: int = Field(gt=0)
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int = Field(default=1, ge=0, le=999)
    metadata: dict = Field(default_factory=dict)


class BindingRequest(BaseModel):
    agent_id: int = Field(gt=0)
    note: str | None = Field(default=None, max_length=1000)


class CollectionStatusRequest(BaseModel):
    status: str = Field(pattern=r"^(ACTIVE|DISABLED)$")


def _collection(entity) -> dict:
    return {
        "collection_id": int(entity.collection_id), "app_id": int(entity.app_id),
        "domain_id": int(entity.domain_id), "collection_key": entity.collection_key,
        "display_name": entity.display_name, "description": entity.description,
        "embedding_model_id": int(entity.embedding_model_id), "status": entity.status,
        "default_security_level": int(entity.default_security_level),
        "metadata": entity.metadata_json or {},
    }


@router.post("/collections", status_code=status.HTTP_201_CREATED)
async def create_collection(domain_id: int, payload: CreateCollectionRequest, request: Request):
    actor = request.headers.get("X-KBot-Actor-Id", "svc:apex")
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
    entities = await request.app.state.kc_collection_service.list(domain_id=domain_id)
    return {"collections": [_collection(entity) for entity in entities]}


@router.get("/collections/{collection_key}")
async def get_collection(domain_id: int, collection_key: str, request: Request):
    try:
        entity = await request.app.state.kc_collection_service.get(domain_id=domain_id, collection_key=collection_key)
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
    return _collection(entity)


@router.patch("/collections/{collection_key}")
async def change_collection_status(domain_id: int, collection_key: str, payload: CollectionStatusRequest, request: Request):
    try:
        entity = await request.app.state.kc_collection_service.change_status(ChangeCollectionStatusCommand(
            domain_id=domain_id, collection_key=collection_key, status=payload.status,
            actor_id=request.headers.get("X-KBot-Actor-Id", "svc:apex"),
        ))
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
    except CollectionDeletionStateError as exc:
        raise HTTPException(status_code=409, detail={"code": "COLLECTION_DELETING", "message": str(exc)}) from exc
    return _collection(entity)


@router.delete("/collections/{collection_key}", status_code=status.HTTP_202_ACCEPTED)
async def delete_collection(domain_id: int, collection_key: str, request: Request):
    try:
        job_id = await request.app.state.kc_collection_service.request_delete(
            domain_id=domain_id, collection_key=collection_key,
            actor_id=request.headers.get("X-KBot-Actor-Id", "svc:apex"),
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
    except CollectionInUseError as exc:
        raise HTTPException(status_code=409, detail={"code": "COLLECTION_IN_USE", "message": str(exc)}) from exc
    except CollectionDeletionStateError as exc:
        raise HTTPException(status_code=409, detail={"code": "COLLECTION_DELETING", "message": str(exc)}) from exc
    return {"status": "DELETING", "purge_job_id": job_id}


@router.put("/agents/{agent_id}/collections/{collection_key}/binding")
async def bind_collection(domain_id: int, agent_id: int, collection_key: str, request: Request, payload: BindingRequest | None = None):
    body = payload or BindingRequest(agent_id=agent_id)
    if body.agent_id != agent_id:
        raise HTTPException(status_code=422, detail={"code": "AGENT_ID_MISMATCH", "message": "path and body agent_id differ"})
    try:
        entity = await request.app.state.kc_binding_service.bind_agent(BindAgentCollectionCommand(
            domain_id=domain_id, collection_key=collection_key, agent_id=str(agent_id),
            actor_id=request.headers.get("X-KBot-Actor-Id", "svc:agent-config"), note=body.note,
        ))
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
    return {"binding_id": int(entity.binding_id), "collection_id": int(entity.collection_id), "agent_id": agent_id, "status": entity.status}


@router.delete("/agents/{agent_id}/collections/{collection_key}/binding", status_code=status.HTTP_204_NO_CONTENT)
async def unbind_collection(domain_id: int, agent_id: int, collection_key: str, request: Request):
    try:
        await request.app.state.kc_binding_service.unbind_agent(BindAgentCollectionCommand(
            domain_id=domain_id, collection_key=collection_key, agent_id=str(agent_id),
            actor_id=request.headers.get("X-KBot-Actor-Id", "svc:agent-config"),
        ))
    except CollectionNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc


@router.get("/agents/{agent_id}/collection-bindings")
async def list_agent_bindings(domain_id: int, agent_id: int, request: Request):
    bindings = await request.app.state.kc_binding_service.list_agent(domain_id=domain_id, agent_id=str(agent_id))
    return {"bindings": [{
        "binding_id": int(binding.binding_id), "collection_id": int(binding.collection_id),
        "consumer_type": binding.consumer_type, "consumer_id": binding.consumer_id,
        "status": binding.status, "note": binding.note,
    } for binding in bindings]}
