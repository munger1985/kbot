"""AIOps 运维知识库内部 API。"""

import hashlib
import json
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from aiops_agent.api.dependencies import get_aiops_auth_context, require_service_scope
from aiops_agent.contracts.knowledge import (
    DiagnosisClosureCommand,
    DiagnosisCaseDraft,
    KnowledgeBaseWrite,
    KnowledgeReviewCommand,
    KnowledgeSearchRequest,
    KnowledgeTagStatusCommand,
    KnowledgeTagWrite,
    ManualUploadMetadata,
)
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops.knowledge import KnowledgeReuseFeedback


router = APIRouter(
    prefix="/internal/v1/aiops/operations-knowledge",
    tags=["AIOps Operations Knowledge"],
)

_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
_MAX_UPLOAD_FILES = 20


class CollectionModelsPayload(BaseModel):
    models: dict[str, UUID]


class KnowledgeBaseDeletePayload(BaseModel):
    expected_row_version: int


class KnowledgeAssetDeletePayload(BaseModel):
    expected_row_version: int
    purge_source_objects: bool = False


@router.get("/knowledge-bases")
async def list_knowledge_bases(
    request: Request,
    include_archived: bool = False,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return {"items": await request.app.state.operations_knowledge_service.list_knowledge_bases(
        domain_id=domain_id, include_archived=include_archived,
    )}


@router.post("/knowledge-bases", status_code=201)
async def create_knowledge_base(
    payload: KnowledgeBaseWrite,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.create_knowledge_base(
        domain_id=domain_id, command=payload, context=context,
    )

@router.get("/knowledge-bases/{knowledge_base_id}")
async def get_knowledge_base(
    knowledge_base_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.get_knowledge_base(
        domain_id=domain_id, knowledge_base_id=knowledge_base_id,
    )


@router.put("/knowledge-bases/{knowledge_base_id}")
async def update_knowledge_base(
    knowledge_base_id: UUID,
    payload: KnowledgeBaseWrite,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.update_knowledge_base(
        domain_id=domain_id,
        knowledge_base_id=knowledge_base_id,
        command=payload,
        context=context,
    )


@router.delete("/knowledge-bases/{knowledge_base_id}", status_code=204)
async def delete_knowledge_base(
    knowledge_base_id: UUID,
    payload: KnowledgeBaseDeletePayload,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    await request.app.state.operations_knowledge_service.delete_knowledge_base(
        domain_id=domain_id,
        knowledge_base_id=knowledge_base_id,
        expected_row_version=payload.expected_row_version,
        context=context,
    )


def _scope(request: Request, context: AuthContext) -> tuple[int, str]:
    require_service_scope(request, "aiops.manage")
    if context.domain_id is None or int(context.domain_id) < 1:
        from fastapi import HTTPException
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"})
    return int(context.domain_id), context.asserted_user_id or context.client_id


def _use_scope(request: Request, context: AuthContext) -> tuple[int, str]:
    require_service_scope(request, "aiops.run")
    if context.domain_id is None or int(context.domain_id) < 1:
        from fastapi import HTTPException
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"})
    return int(context.domain_id), context.asserted_user_id or context.client_id


@router.get("/overview")
async def overview(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.overview(
        domain_id=domain_id
    )


@router.post("/cases/{asset_id}/feedback", status_code=201)
async def record_case_reuse_feedback(
    asset_id: UUID, payload: KnowledgeReuseFeedback, request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    _, domain_id, _ = _use_scope(request, context)
    return await request.app.state.operations_knowledge_service.record_reuse_feedback(
        domain_id=domain_id, asset_id=asset_id, command=payload, context=context,
    )

@router.get("/cases/overview")
async def cases_overview(request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, _ = _use_scope(request, context)
    return await request.app.state.operations_knowledge_service.cases_overview(domain_id=domain_id)


@router.get("/cases")
async def list_cases(
    request: Request, status: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _use_scope(request, context)
    return {"items": await request.app.state.operations_knowledge_service.list_cases(
        domain_id=domain_id, status=status, limit=limit,
    )}


@router.get("/cases/{asset_id}")
async def get_case(asset_id: UUID, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, _ = _use_scope(request, context)
    return await request.app.state.operations_knowledge_service.get_case(
        domain_id=domain_id, asset_id=asset_id,
    )


@router.post("/cases/{asset_id}/versions", status_code=202)
async def create_case_version(
    asset_id: UUID,
    payload: DiagnosisCaseDraft,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.create_case_version(
        domain_id=domain_id,
        asset_id=asset_id,
        command=payload,
        context=context,
    )


@router.get("/knowledge-bases/{knowledge_base_id}/models")
async def get_collection_models(
    knowledge_base_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.get_collection_models(
        domain_id=domain_id, knowledge_base_id=knowledge_base_id, context=context,
    )


@router.put("/knowledge-bases/{knowledge_base_id}/models")
async def configure_collection_models(
    knowledge_base_id: UUID,
    payload: CollectionModelsPayload,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.configure_collections(
        domain_id=domain_id, knowledge_base_id=knowledge_base_id,
        models=payload.models, context=context,
    )


@router.get("/tags")
async def list_tags(request: Request, include_disabled: bool = True,
                    context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, _ = _scope(request, context)
    return {"items": await request.app.state.operations_knowledge_service.list_tags(
        domain_id=domain_id, include_disabled=include_disabled
    )}


@router.post("/tags", status_code=201)
async def create_tag(payload: KnowledgeTagWrite, request: Request,
                     context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor = _scope(request, context)
    return await request.app.state.operations_knowledge_service.create_tag(
        domain_id=domain_id, actor_id=actor, command=payload
    )


@router.put("/tags/{tag_id}")
async def update_tag(tag_id: UUID, payload: KnowledgeTagWrite, request: Request,
                     context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor = _scope(request, context)
    return await request.app.state.operations_knowledge_service.update_tag(
        domain_id=domain_id, tag_id=tag_id, actor_id=actor, command=payload
    )


@router.post("/tags/{tag_id}:enable")
async def enable_tag(tag_id: UUID, payload: KnowledgeTagStatusCommand, request: Request,
                     context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor = _scope(request, context)
    return await request.app.state.operations_knowledge_service.set_tag_status(
        domain_id=domain_id, tag_id=tag_id, actor_id=actor,
        expected_row_version=payload.expected_row_version, status="ACTIVE"
    )


@router.post("/tags/{tag_id}:disable")
async def disable_tag(tag_id: UUID, payload: KnowledgeTagStatusCommand, request: Request,
                      context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, actor = _scope(request, context)
    return await request.app.state.operations_knowledge_service.set_tag_status(
        domain_id=domain_id, tag_id=tag_id, actor_id=actor,
        expected_row_version=payload.expected_row_version, status="DISABLED"
    )


@router.get("/assets")
async def list_assets(
    request: Request,
    asset_kind: str | None = None,
    status: str | None = None,
    knowledge_base_id: UUID | None = None,
    limit: int = Query(100, ge=1, le=500),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return {"items": await request.app.state.operations_knowledge_service.list_assets(
        domain_id=domain_id, asset_kind=asset_kind, status=status,
        knowledge_base_id=knowledge_base_id, limit=limit,
    )}


@router.get("/assets/{asset_id}")
async def get_asset(
    asset_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.get_asset(
        domain_id=domain_id, asset_id=asset_id
    )


@router.delete("/assets/{asset_id}", status_code=204)
async def delete_asset(
    asset_id: UUID,
    payload: KnowledgeAssetDeletePayload,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    await request.app.state.operations_knowledge_service.delete_asset(
        domain_id=domain_id,
        asset_id=asset_id,
        expected_row_version=payload.expected_row_version,
        purge_source_objects=payload.purge_source_objects,
        context=context,
    )


@router.get("/versions/{asset_version_id}")
async def get_version(
    asset_version_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.get_version(
        domain_id=domain_id, asset_version_id=asset_version_id
    )


@router.get("/versions/{asset_version_id}/source")
async def stream_source(
    asset_version_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    result = await request.app.state.operations_knowledge_service.stream_source(
        domain_id=domain_id, asset_version_id=asset_version_id,
        context=context, range_header=request.headers.get("Range"),
    )
    headers = {
        key: value for key, value in result.headers.items()
        if key in {"content-length", "content-range", "content-disposition", "accept-ranges"}
    }
    return StreamingResponse(
        result.body,
        status_code=result.status_code,
        media_type=result.headers.get("content-type", "application/octet-stream"),
        headers=headers,
    )


@router.post("/uploads:ingest", status_code=202)
async def ingest_upload(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    content_type = request.headers.get("Content-Type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise HTTPException(
            415,
            {"code": "UNSUPPORTED_MEDIA_TYPE", "message": "资料上传只接受 multipart/form-data"},
        )
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if not idempotency_key:
        raise HTTPException(
            428,
            {"code": "IDEMPOTENCY_KEY_REQUIRED", "message": "缺少 Idempotency-Key Header"},
        )
    form = await request.form()
    try:
        knowledge_base_id = UUID(str(form.get("knowledge_base_id", "")))
        grouping_mode = str(form.get("grouping_mode", "EACH_FILE")).upper()
        if grouping_mode not in {"EACH_FILE", "SINGLE_BUNDLE"}:
            raise ValueError("grouping_mode must be EACH_FILE or SINGLE_BUNDLE")
        raw_declarations = json.loads(str(form.get("files", "[]")))
        raw_metadata = json.loads(str(form.get("metadata", "{}")))
        if not isinstance(raw_declarations, list) or not raw_declarations:
            raise ValueError("files is required")
        if len(raw_declarations) > _MAX_UPLOAD_FILES:
            raise ValueError(f"每次最多上传 {_MAX_UPLOAD_FILES} 个文件")
        metadata = ManualUploadMetadata.model_validate(raw_metadata)
        bundle_title = metadata.display_name
        if grouping_mode == "SINGLE_BUNDLE":
            raw_bundle = json.loads(str(form.get("bundle", "{}")))
            bundle_title = str(raw_bundle.get("title", "")).strip()
            if not bundle_title or len(bundle_title) > 256:
                raise ValueError("Bundle 名称不能为空且不能超过 256 个字符")
        documents: list[dict] = []
        total_bytes = 0
        part_names: set[str] = set()
        client_file_ids: set[str] = set()
        for ordinal, declaration in enumerate(raw_declarations):
            if not isinstance(declaration, dict):
                raise ValueError("文件声明格式无效")
            part_name = str(declaration.get("part_name", "")).strip()
            if not part_name or part_name in part_names:
                raise ValueError("文件 part_name 不能为空或重复")
            part_names.add(part_name)
            client_file_id = str(declaration.get("client_file_id", "")).strip()
            if not client_file_id or client_file_id in client_file_ids:
                raise ValueError("文件 client_file_id 不能为空或重复")
            client_file_ids.add(client_file_id)
            upload = form.get(part_name)
            if not isinstance(upload, UploadFile):
                raise ValueError(f"缺少文件分片 {part_name}")
            body = await upload.read()
            total_bytes += len(body)
            if not body or total_bytes > _MAX_UPLOAD_BYTES:
                raise ValueError("上传文件为空或总大小超过 64 MiB")
            digest = hashlib.sha256(body).hexdigest()
            declared_digest = str(declaration.get("content_sha256", "")).lower()
            if len(declared_digest) != 64 or declared_digest != digest:
                raise ValueError(f"文件 {upload.filename or part_name} 的摘要不一致")
            declared_size = declaration.get("byte_size")
            if not isinstance(declared_size, int) or declared_size != len(body):
                raise ValueError(f"文件 {upload.filename or part_name} 的大小声明不一致")
            role = str(declaration.get("role", "CONTENT")).upper()
            if role not in {"CONTENT", "SUPPLEMENT"}:
                raise ValueError("文件角色必须是 CONTENT 或 SUPPLEMENT")
            display_name = str(
                declaration.get("display_name") or upload.filename or part_name
            ).strip()
            if not display_name or len(display_name) > 256:
                raise ValueError("文件显示名称不能为空且不能超过 256 个字符")
            documents.append({
                "client_file_id": client_file_id,
                "file_name": str(upload.filename or display_name),
                "display_name": display_name,
                "media_type": str(
                    upload.content_type
                    or declaration.get("declared_mime_type")
                    or "application/octet-stream"
                ),
                "body": body,
                "content_sha256": digest,
                "ordinal": ordinal,
                "role": role,
                "required_flag": bool(declaration.get("required_flag", False)),
            })
        if grouping_mode == "SINGLE_BUNDLE" and not any(
            document["role"] == "CONTENT" for document in documents
        ):
            raise ValueError("Bundle 至少需要一个正文文件")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(
            422,
            {"code": "AIOPS_KNOWLEDGE_UPLOAD_INVALID", "message": str(exc)},
        ) from exc
    finally:
        await form.close()
    return await request.app.state.operations_knowledge_service.ingest_upload(
        domain_id=domain_id,
        knowledge_base_id=knowledge_base_id,
        context=context,
        grouping_mode=grouping_mode,
        bundle_title=bundle_title,
        documents=documents,
        metadata=metadata,
        idempotency_key=idempotency_key,
    )


@router.post("/assets/{asset_id}/versions:ingest", status_code=202)
async def ingest_manual_version(
    asset_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    raw_metadata = unquote(request.headers.get("X-Upload-Metadata", ""))
    metadata = ManualUploadMetadata.model_validate(json.loads(raw_metadata))
    try:
        expected_asset_row_version = int(
            request.headers.get("X-Expected-Asset-Row-Version", "")
        )
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(
            422,
            {"code": "AIOPS_KNOWLEDGE_VERSION_CONFLICT", "message": "缺少有效的资产版本号"},
        ) from exc
    body = await request.body()
    return await request.app.state.operations_knowledge_service.ingest_manual_version(
        domain_id=domain_id,
        asset_id=asset_id,
        expected_asset_row_version=expected_asset_row_version,
        context=context,
        file_name=unquote(request.headers.get("X-File-Name", "manual.bin")),
        media_type=request.headers.get("Content-Type", "application/octet-stream"),
        body=body,
        metadata=metadata,
        idempotency_key=request.headers.get("Idempotency-Key", ""),
        declared_sha256=request.headers.get("X-Content-SHA256"),
    )


@router.post("/reports/{report_id}:extract-case", status_code=202)
async def extract_case(
    report_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.extract_case(
        domain_id=domain_id, report_id=report_id, context=context
    )


@router.get("/conversations/{conversation_id}/case-candidate")
async def conversation_case_candidate(
    conversation_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _use_scope(request, context)
    return await request.app.state.operations_knowledge_service.conversation_case_preview(
        domain_id=domain_id, conversation_id=conversation_id, context=context,
    )


@router.post("/conversations/{conversation_id}:close-diagnosis", status_code=202)
async def close_conversation_diagnosis(
    conversation_id: UUID,
    payload: DiagnosisClosureCommand,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _use_scope(request, context)
    return await request.app.state.operations_knowledge_service.close_diagnosis(
        domain_id=domain_id,
        conversation_id=conversation_id,
        command=payload,
        context=context,
    )


@router.get("/reports/{report_id}/case-eligibility")
async def case_eligibility(report_id: UUID, request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, _ = _use_scope(request, context)
    return await request.app.state.operations_knowledge_service.case_eligibility(
        domain_id=domain_id, report_id=report_id,
    )


@router.post("/versions/{asset_version_id}:reconcile")
async def reconcile_version(
    asset_version_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.reconcile_version(
        domain_id=domain_id, asset_version_id=asset_version_id, context=context
    )


async def _review(
    *, decision: str, asset_version_id: UUID, payload: KnowledgeReviewCommand,
    request: Request, context: AuthContext,
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.review(
        domain_id=domain_id,
        asset_version_id=asset_version_id,
        decision=decision,
        expected_row_version=payload.expected_row_version,
        comment=payload.comment,
        context=context,
    )


@router.post("/versions/{asset_version_id}:publish")
async def publish_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                          request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="PUBLISH", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.post("/versions/{asset_version_id}:submit-review")
async def submit_case_review(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                             request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="SUBMIT_REVIEW", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.post("/versions/{asset_version_id}:approve")
async def approve_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                          request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.approve_manual(
        domain_id=domain_id,
        asset_version_id=asset_version_id,
        expected_row_version=payload.expected_row_version,
        comment=payload.comment,
        context=context,
    )


@router.post("/versions/{asset_version_id}:reject")
async def reject_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                         request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="REJECT", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.post("/versions/{asset_version_id}:retire")
async def retire_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                         request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="RETIRE", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.post("/versions/{asset_version_id}:retry")
async def retry_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                        request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="RETRY", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.get("/reviews")
async def list_reviews(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return {"items": await request.app.state.operations_knowledge_service.list_reviews(
        domain_id=domain_id, limit=limit
    )}


@router.post(":search")
async def search(
    payload: KnowledgeSearchRequest,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor = _scope(request, context)
    return await request.app.state.operations_knowledge_service.search(
        domain_id=domain_id, agent_id=actor, request=payload, context=context
    )
