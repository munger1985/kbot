"""本 App 精确版本管理预览，不产生检索授权或模型证据。"""

from uuid import UUID
from knowledge_algorithms.contracts.errors import require
from .case_knowledge_projection import case_projection_is_current
from ..contracts.knowledge import DiagnosisCaseProfile


def _identity(detail):
    return (detail["asset"]["row_version"], detail["version"]["row_version"],
        tuple((row["document_version_id"], row["parse_run_id"], row["index_generation_id"], row["status"])
              for row in detail["indexes"]))


async def preview_knowledge_version(service, *, domain_id, asset_version_id, context, kc_scope):
    detail = await service.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
    require(detail["version"]["status"] in {"DRAFT", "REVIEW_REQUIRED", "PUBLISHED"},
            "该知识版本尚未形成可预览正文")
    indexes = detail["indexes"]
    require(indexes and len(indexes) <= 20 and all(row["status"] == "READY" for row in indexes),
            "精确索引尚不可预览")
    access = service._documents.access(context)
    documents = []
    for index in indexes:
        identity = UUID(index["document_version_id"])
        record = await service._documents.inspect(access, identity)
        require(record["parse_run_id"] == index["parse_run_id"] and record["generation_id"] == index["index_generation_id"]
                and record["status"] != "WITHDRAWN", "文档代次已经变化或撤回")
        if detail["asset"]["asset_kind"] == "DIAGNOSIS_CASE":
            profile = DiagnosisCaseProfile.model_validate(detail["version"]["profile"])
            require(profile.knowledge_projection is not None
                and case_projection_is_current(title=profile.knowledge_projection.title, profile=profile)
                and profile.knowledge_projection.body_sha256 == record["source_hash"],
                "案例规范正文与审阅投影不一致")
        projection = await service._documents.excerpts(access, (identity,))
        documents.append({"document_version_id": str(identity), "document_role": "CONTENT",
            "content_hash_sha256": record["source_hash"],
            "fragments": [{"content_text": row["content"], "locator": row["locator"]}
                          for row in projection["excerpts"]], "warnings": projection["warnings"]})
    current = await service.get_version(domain_id=domain_id, asset_version_id=asset_version_id)
    require(_identity(current) == _identity(detail), "预览期间知识版本已变化")
    return {"usage": "MANAGEMENT_PREVIEW_NOT_AGENT_EVIDENCE", "asset_id": detail["asset"]["asset_id"],
        "asset_version_id": str(asset_version_id), "status": detail["version"]["status"], "documents": documents}
