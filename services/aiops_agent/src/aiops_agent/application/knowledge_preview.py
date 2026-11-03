"""有权限的精确知识版本预览；不调用 Agent Discovery，也不放宽正式检索发布门禁。"""

from uuid import UUID

from aiops_agent.application.case_knowledge_projection import case_projection_is_current
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from platform_clients.knowledge_core import KnowledgeCoreClientError


def _unavailable(message):
    return AIOpsApplicationError(
        code="AIOPS_KNOWLEDGE_PREVIEW_UNAVAILABLE", message=message, status_code=409,
    )


def _identity(detail):
    return (
        detail["asset"]["asset_id"], detail["asset"].get("row_version"),
        detail["version"]["asset_version_id"], detail["version"].get("row_version"),
        detail["version"]["status"],
        tuple((row["collection_id"], row["bundle_id"], row["bundle_revision_id"],
            row["status"]) for row in detail["indexes"]),
    )


async def preview_knowledge_version(service, *, domain_id, asset_version_id, context, kc_scope):
    """HTTP 管理边界鉴权后，只预览 Registry 已登记的原件，不接受任意 KC 文档标识。"""
    detail = await service.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
    if detail["version"]["status"] not in {"DRAFT", "REVIEW_REQUIRED", "PUBLISHED"}:
        raise _unavailable("该知识版本尚未形成可预览正文")
    indexes = [row for row in detail["indexes"] if row["status"] == "READY"]
    if len(indexes) != 1:
        raise _unavailable("该知识版本的精确索引尚不可预览")
    index = indexes[0]
    is_case = detail["asset"]["asset_kind"] == "DIAGNOSIS_CASE"
    expected_hash = None
    if is_case:
        try:
            profile = DiagnosisCaseProfile.model_validate(detail["version"]["profile"])
            if profile.knowledge_projection is None or not case_projection_is_current(
                title=profile.knowledge_projection.title, profile=profile,
            ):
                raise ValueError("投影不一致")
            expected_hash = profile.knowledge_projection.body_sha256
        except (TypeError, ValueError) as exc:
            raise _unavailable("案例知识投影需要重新核验") from exc
    try:
        revision = await service._kc.get_revision_status(
            **kc_scope, bundle_id=UUID(index["bundle_id"]),
            bundle_revision_id=UUID(index["bundle_revision_id"]),
            include_members=True, auth_context=context,
        )
        if (
            str(revision.get("bundle_revision_id") or "") != index["bundle_revision_id"]
            or revision.get("status") != "READY"
            or revision.get("approval_status") != "APPROVED"
            or revision.get("withdrawn_at") is not None
        ):
            raise _unavailable("该知识版本的外部索引未就绪或已撤回")
        members = [
            row for row in revision.get("members") or ()
            if row.get("document_version_id") and row.get("document_role") in {"CONTENT", "SUPPLEMENT"}
            and row.get("member_status") in {"READY", "COMPLETED"}
            and (not is_case or (row.get("document_role") == "CONTENT"
                and row.get("content_hash_sha256") == expected_hash))
        ]
        if not members or (is_case and len(members) != 1) or len(members) > 20:
            raise _unavailable("可预览原文与知识版本登记不一致")
        documents = []
        for member in members:
            document_id = str(member["document_version_id"])
            preview = await service._kc.get_document_preview(
                **kc_scope, collection_id=UUID(index["collection_id"]),
                document_version_id=UUID(document_id), auth_context=context,
            )
            if str(preview.get("document_version_id") or "") != document_id:
                raise _unavailable("预览正文不属于登记的确切文档版本")
            documents.append({
                "document_version_id": document_id, "document_role": member["document_role"],
                "content_hash_sha256": member.get("content_hash_sha256"),
                "fragments": preview.get("fragments") or [],
            })
    except KnowledgeCoreClientError as exc:
        raise AIOpsApplicationError(
            code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE", message="知识正文预览暂时不可用",
            status_code=503, retryable=True,
        ) from exc
    # 外部读取不持有业务事务；返回前防止撤回、替换索引或并发编辑后串用正文。
    current = await service.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
    if _identity(current) != _identity(detail):
        raise _unavailable("预览期间知识版本已变化，请刷新")
    return {
        "usage": "MANAGEMENT_PREVIEW_NOT_AGENT_EVIDENCE",
        "asset_id": detail["asset"]["asset_id"], "asset_version_id": str(asset_version_id),
        "status": detail["version"]["status"], "bundle_revision_id": index["bundle_revision_id"],
        "documents": documents,
    }
