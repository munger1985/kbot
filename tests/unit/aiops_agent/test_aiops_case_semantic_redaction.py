"""语义脱敏必须实际位于案例索引之前，失败不降级为规则扫描成功。"""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.case_semantic_redaction import (
    SemanticRedactionOutput, apply_semantic_redaction, redact_case_for_index, shared_content_hash,
)
from aiops_agent.application.case_redaction import CaseRedactor
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.contracts.knowledge import DiagnosisCaseDraft, DiagnosisCaseProfile
from platform_core.identity import uuid7


def profile():
    return DiagnosisCaseProfile(
        case_kind="DIAGNOSTIC_REFERENCE",
        problem_signature={"normalized_symptoms": ["华星精密客户的订单 MES-99882 超时"]},
        environment={"database_type": "POSTGRESQL", "database_version": "16"},
        root_cause={"summary": "华星精密订单 MES-99882 等待事务提交", "evidence_refs": ["artifact:raw#c1"]},
        verification={"result": "NOT_VERIFIED"},
        source={"run_ids": ["private-run"]},
    )


def test_semantic_spans_are_exact_consistent_and_do_not_rewrite_lineage():
    original = CaseRedactor().profile(profile())
    output = SemanticRedactionOutput(spans=(
        {"field_id": "title", "text": "华星精密", "category": "CUSTOMER"},
        {"field_id": "title", "text": "MES-99882", "category": "BUSINESS_ID"},
    ))
    title, sanitized = apply_semantic_redaction(
        title="华星精密订单 MES-99882", profile=original, output=output,
        model_ref={"model_id": "approved"}, prompt_ref={"prompt_version": "1"},
    )
    shared = sanitized.model_dump_json(exclude={"source"})
    assert "华星精密" not in title + shared and "MES-99882" not in title + shared
    assert sanitized.source == original.source
    assert sanitized.root_cause["evidence_refs"] == ["artifact:raw#c1"]
    assert sanitized.root_cause["summary"].endswith("等待事务提交")
    assert sanitized.redaction.semantic_provenance["output_hash"] == shared_content_hash(title, sanitized)
    assert sanitized.redaction.status == "REVIEW_REQUIRED"
    assert "华星精密" in original.root_cause["summary"]


def test_fabricated_span_is_rejected():
    with pytest.raises(ValueError, match="逐字"):
        apply_semantic_redaction(title="订单", profile=profile(), output=SemanticRedactionOutput(spans=({"field_id": "title", "text": "不存在的秘密", "category": "SECRET"},)), model_ref={}, prompt_ref={})


def test_semantic_scan_uses_approved_model_and_bounded_call_without_private_source():
    async def run():
        model_id = uuid7()
        model = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(output={"spans": []})))
        catalog = SimpleNamespace(get_model=AsyncMock(return_value={"model_name": "approved-model"}))
        registry = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="受控识别", ref=lambda: {"prompt_id": "case_semantic_redaction", "prompt_sha256": "hash"})))
        title, sanitized = await redact_case_for_index(title="锁等待调查", profile=profile(),
            collection={"models": {"parser_llm": str(model_id)}}, structured_model=model,
            prompt_registry=registry, model_catalog=catalog, model_scope={"tenant_id": str(uuid7())})
        kwargs = model.generate_structured.await_args.kwargs
        assert kwargs["model_snapshot"]["model_id"] == str(model_id)
        assert kwargs["deadline"] is not None
        assert "private-run" not in str(kwargs["input_payload"])
        assert "artifact:raw" not in str(kwargs["input_payload"])
        assert sanitized.redaction.semantic_provenance["status"] == "SCANNED"
        registry.resolve.assert_awaited_once_with("case_semantic_redaction")
    asyncio.run(run())


def test_case_revision_cannot_send_body_to_index_when_semantic_scan_unavailable():
    async def run():
        asset_id, version_id = uuid7(), uuid7()
        source = profile()
        asset = SimpleNamespace(asset_kind="DIAGNOSIS_CASE", row_version=1, current_version_id=None)
        version = SimpleNamespace(asset_version_id=version_id, status="DRAFT", profile_json=source.model_dump(mode="json"), version_no=1, source_hash="old")
        repo = SimpleNamespace(get_asset=AsyncMock(return_value=asset), list_versions=AsyncMock(return_value=[version]), get_version_relations=AsyncMock(return_value={"sources": [], "scopes": []}))
        class Uow:
            async def __aenter__(self): return SimpleNamespace(operations_knowledge=repo)
            async def __aexit__(self, *args): pass
        kc = SimpleNamespace(ingest_user_file=AsyncMock())
        service = OperationsKnowledgeService(uow_factory=Uow, knowledge_core=kc, model_catalogs={})
        service._actor = lambda context: uuid7()
        service._collection = AsyncMock(return_value={"models": {}})
        kwargs = {"domain_id": uuid7(), "asset_id": asset_id, "context": SimpleNamespace(),
            "command": DiagnosisCaseDraft(expected_asset_row_version=1, base_version_id=version_id, display_name="订单调查", profile=source)}
        if "tenant_id" in inspect.signature(service.create_case_version).parameters:
            kwargs["tenant_id"] = uuid7()
        with pytest.raises(Exception, match="语义脱敏服务未配置"):
            await service.create_case_version(**kwargs)
        kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())
