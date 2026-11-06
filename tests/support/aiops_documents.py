"""AIOps 本 App 文档端口的离线替身；不模拟 KC/Bundle 或访问外部环境。"""

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

from aiops_agent.application.operations_documents import OperationsDocuments
from aiops_agent.domain.knowledge_retrieval import RetrievalAccess
from platform_core.identity import uuid7


def install_documents(f):
    tenant = getattr(f.run, "tenant_id", None)
    access = RetrievalAccess(f.domain, tenant, str(f.actor))
    records, payloads = {}, {}
    documents = SimpleNamespace(records=records, payloads=payloads,
        access=lambda context, actor_id=None, **kwargs: replace(access, subject_ref=str(actor_id or f.actor), **kwargs),
        retrieval=SimpleNamespace(retrieve=AsyncMock(), authorization=SimpleNamespace(service=None)))
    async def stage(**kwargs):
        document, version = kwargs.get("document_id") or uuid7(), kwargs.get("version_id") or uuid7()
        payloads[version] = kwargs["body"]
        return {"document_id": document, "document_version_id": version,
            "file_name": kwargs["file_name"], "body": kwargs["body"]}
    async def register(uow, current_access, *, staged, asset_id, asset_version_id):
        document, version, job, parse = staged["document_id"], staged["document_version_id"], uuid7(), uuid7()
        record = dict(document_id=str(document), document_version_id=str(version),
            ingestion_job_id=str(job), parse_run_id=str(parse), subject_ref=current_access.subject_ref,
            source_hash=sha256(staged["body"]).hexdigest(), source_handle="synthetic-source",
            mime_type="text/markdown", byte_count=len(staged["body"]), title=staged["file_name"],
            status="RECEIVED", processing_status="RECEIVED", revision=1,
            generation_id=None, active_generation_id=None, allowed_view_hash=None, error_code=None)
        records[version] = record
        return dict(document_id=document, document_version_id=version, ingestion_job_id=job)
    async def inspect(current_access, version):
        return dict(records[UUID(str(version))])
    async def withdraw(uow, current_access, indexes, **kwargs):
        for index in indexes:
            records[index.document_version_id]["status"] = "WITHDRAWN"
        return []
    async def process(current_access, version):
        record = records[UUID(str(version))]
        if getattr(f, "phase", "READY") == "PROCESSING":
            return dict(record)
        from knowledge_algorithms.contracts.errors import require
        require(record["status"] != "WITHDRAWN", "合成文档已撤回")
        record.update(processing_status="PARSED", generation_id=record["generation_id"] or str(uuid7()), status="PARSED")
        return record
    documents.stage = AsyncMock(side_effect=stage)
    documents.register = AsyncMock(side_effect=register)
    documents.inspect = AsyncMock(side_effect=inspect)
    documents.withdraw = AsyncMock(side_effect=withdraw)
    documents.process = AsyncMock(side_effect=process)
    documents.excerpts = AsyncMock(return_value={"excerpts": [], "warnings": []})
    documents.publish = AsyncMock()
    documents.retry = AsyncMock()
    documents.configuration = lambda models: {"models": {k: str(v) for k,v in models.items()}, "profile_ref": "synthetic-profile"}
    f.service._documents = documents
    f.service._case_configuration = AsyncMock(return_value={"profile_ref": "synthetic-profile",
        "models": {"parser_llm": str(uuid7())}})
    async def find_revision(**kwargs):
        for version_id, relations in f.state.relations.items():
            if any(index.document_version_id == kwargs["revision_id"] for index in relations["indexes"]):
                version = f.state.versions[version_id]
                return f.state.assets[version.asset_id], version
        return None
    f.uow.operations_knowledge.find_case_by_intake_revision = AsyncMock(side_effect=find_revision)
    f.uow.knowledge_documents = SimpleNamespace(inspect=documents.inspect)
    f.uow.knowledge_retrieval = SimpleNamespace(withdraw=AsyncMock())
    f.documents = documents
    return documents
