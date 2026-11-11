"""运维知识提炼合同与原文命令边界测试。"""

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import UUID
from types import SimpleNamespace
from tests.support.aiops_knowledge_scope import knowledge_agent_repository, KNOWLEDGE_BASE_ID, active_knowledge_bases_fixture
import unittest
from unittest.mock import ANY, AsyncMock

from pydantic import ValidationError

from aiops_agent.application.operations_knowledge import (
    OperationsKnowledgeService,
    _diagnosis_case_eligibility,
    _freeze_case_version_identity,
    _prepare_case_revision,
    _validate_closure_evidence,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.knowledge import (
    DiagnosisClosureCommand,
    DiagnosisCaseDraft,
    DiagnosisCaseProfile,
    KnowledgeBaseWrite,
    KnowledgeSearchRequest,
    ManualUploadMetadata,
)
from aiops_agent.repositories.knowledge import OperationsKnowledgeRepository
from platform_core.identity import uuid7


async def _body_chunks(body: bytes):
    yield body


class _OrderedFlushSession:
    def __init__(self) -> None:
        self.pending: list[object] = []
        self.flushes: list[list[object]] = []

    def add(self, entity: object) -> None:
        self.pending.append(entity)

    def add_all(self, entities: list[object]) -> None:
        self.pending.extend(entities)

    async def flush(self) -> None:
        self.flushes.append(list(self.pending))
        self.pending.clear()


class OperationsKnowledgeContractTest(unittest.TestCase):
    def test_case_identity_is_frozen_before_orm_state_expires(self):
        asset_id = uuid7()
        active = True

        class Asset:
            @property
            def asset_id(self):
                if not active:
                    raise RuntimeError("expired ORM state")
                return asset_id

        class Version:
            @property
            def version_no(self):
                if not active:
                    raise RuntimeError("expired ORM state")
                return 7

        frozen = _freeze_case_version_identity((Asset(), Version()))
        active = False
        self.assertEqual((asset_id, 7), frozen)

    def test_diagnosis_closure_command_rejects_uncontracted_fields(self):
        payload = {
            "included_turn_ids": [uuid7()], "source_fingerprint": "a" * 64,
            "display_name": "连接池故障",
            "symptoms": ["连接池耗尽"],
            "root_cause_summary": "连接未释放",
        }
        command = DiagnosisClosureCommand.model_validate(payload)
        self.assertEqual("NOT_VERIFIED", command.verification_result)
        with self.assertRaises(ValidationError):
            DiagnosisClosureCommand.model_validate({**payload, "publish": True})

    def test_diagnosis_closure_evidence_stays_in_run_and_separates_samples(self):
        action_id, before_id, after_id, outsider_id = (
            uuid7(), uuid7(), uuid7(), uuid7()
        )
        base = {
            "included_turn_ids": [uuid7()], "source_fingerprint": "a" * 64,
            "display_name": "连接池故障",
            "symptoms": ["连接池耗尽"],
            "root_cause_summary": "连接未释放",
            "action_evidence_ids": [action_id],
            "before_evidence_ids": [before_id],
            "after_evidence_ids": [after_id],
        }
        action_refs, before_refs, after_refs = _validate_closure_evidence(
            command=DiagnosisClosureCommand.model_validate(base),
            allowed_ids={action_id, before_id, after_id},
            allowed_action_ids={action_id},
        )
        self.assertEqual([str(action_id)], action_refs)
        self.assertEqual({str(before_id)}, before_refs)
        self.assertEqual({str(after_id)}, after_refs)

        with self.assertRaises(AIOpsApplicationError):
            _validate_closure_evidence(
                command=DiagnosisClosureCommand.model_validate({
                    **base, "action_evidence_ids": [outsider_id],
                }),
                allowed_ids={action_id, before_id, after_id},
                allowed_action_ids={action_id},
            )
        with self.assertRaises(AIOpsApplicationError):
            _validate_closure_evidence(
                command=DiagnosisClosureCommand.model_validate({
                    **base, "action_evidence_ids": [after_id],
                }),
                allowed_ids={action_id, before_id, after_id},
                allowed_action_ids={action_id},
            )
        with self.assertRaises(AIOpsApplicationError):
            _validate_closure_evidence(
                command=DiagnosisClosureCommand.model_validate({
                    **base, "after_evidence_ids": [before_id],
                }),
                allowed_ids={action_id, before_id, after_id},
                allowed_action_ids={action_id},
            )

    def test_traceable_inspection_can_be_diagnostic_reference(self):
        report = SimpleNamespace(
            is_current=True, status="READY", report_type="INCIDENT"
        )
        content = {"facts": [{"record_id": "record-1", "claim_id": "c1", "level": "OBSERVED", "evidence_refs": ["e-1"]}], "scope": {
            "source_kind": "CHAT",
            "root_cause_grade": "CONFIRMED",
            "diagnosis_rationale": "连接耗尽由泄漏引起",
        }}
        self.assertTrue(_diagnosis_case_eligibility(
            report=report, report_content=content, evidence_refs=["e-1"]
        )["eligible"])
        report.report_type = "INSPECTION"
        result = _diagnosis_case_eligibility(
            report=report, report_content=content, evidence_refs=["e-1"]
        )
        self.assertTrue(result["eligible"])
        self.assertEqual([], result["reasons"])

    def test_case_revision_inherits_evidence_and_recalculates_case_kind(self):
        base = DiagnosisCaseProfile(
            case_kind="VERIFIED_RESOLUTION",
            problem_signature={"normalized_symptoms": ["连接失败"]},
            environment={"database_type": "Oracle"},
            root_cause={"summary": "连接泄漏", "confidence": "HIGH", "evidence_refs": ["root-1"]},
            actions=({"summary": "重启连接池", "evidence_refs": ["action-1"]},),
            verification={"result": "RESOLVED", "before_refs": ["before-1"], "after_refs": ["after-1"],
                "controlled_verification_ref": "comparison-1", "comparability": "COMPARABLE",
                "guardrails_passed": True, "measured_effect": "IMPROVED", "symptom_recovered": True},
            source={"report_id": "report-1"},
        )
        submitted = base.model_copy(deep=True)
        submitted.root_cause["summary"] = "修订后的连接泄漏结论"
        submitted.root_cause["evidence_refs"] = ["forged"]
        submitted.source["report_id"] = "forged-report"
        submitted.actions = (*submitted.actions, {"summary": "新增动作"})
        submitted.case_kind = "DIAGNOSTIC_REFERENCE"
        revised = _prepare_case_revision(
            command=DiagnosisCaseDraft(
                expected_asset_row_version=2,
                base_version_id=uuid7(),
                display_name="连接池故障案例",
                profile=submitted,
            ),
            base_profile=base,
        )
        self.assertEqual(["root-1"], revised.root_cause["evidence_refs"])
        self.assertEqual("report-1", revised.source["report_id"])
        self.assertEqual("DIAGNOSTIC_REFERENCE", revised.case_kind)
        self.assertEqual([], revised.actions[1]["evidence_refs"])

    def test_knowledge_base_write_requires_models(self):
        with self.assertRaises(ValidationError):
            KnowledgeBaseWrite(display_name="Oracle 运维库")

    def test_upload_metadata_rejects_internal_collection_fields(self):
        with self.assertRaises(ValidationError):
            ManualUploadMetadata.model_validate({
                "display_name": "Oracle 手册",
                "collection_id": str(uuid7()),
            })

    def test_upload_metadata_rejects_user_selected_scopes(self):
        with self.assertRaises(ValidationError):
            ManualUploadMetadata.model_validate({
                "display_name": "Oracle 手册",
                "scopes": [{"kind": "DATABASE_TYPE", "value": "ORACLE"}],
            })

    def test_manual_profile_only_emits_commands_present_in_evidence(self):
        asset = SimpleNamespace(asset_id=uuid7(), display_name="归档配置手册")
        version = SimpleNamespace(asset_version_id=uuid7())
        source = SimpleNamespace(source_locator_json={"metadata": {
            "publisher": "DBA Team", "document_version": "1.0",
            "published_date": "2026-09-30",
        }})
        scope = SimpleNamespace(
            scope_kind="DATABASE_TYPE", scope_value="ORACLE"
        )
        profile = OperationsKnowledgeService._manual_profile(
            asset=asset,
            version=version,
            scopes=[scope],
            sources=[source],
            revision={"extraction_evidence": [{
                "evidence_id": str(uuid7()),
                "section_key": "检查归档模式",
                "content_text": (
                    "先确认当前数据库角色。\n"
                    "SELECT LOG_MODE FROM V$DATABASE;\n"
                    "不得根据经验补写其他命令。"
                ),
                "locator": {"page": 3},
            }]},
        )
        self.assertEqual("OPS_MANUAL_PROFILE.v1", profile.schema_version)
        self.assertEqual(("ORACLE",), profile.scope["database_type"])
        self.assertEqual(
            "SELECT LOG_MODE FROM V$DATABASE;",
            profile.procedures[0].steps[0].command_text,
        )
        self.assertEqual(1, len(profile.procedures[0].steps))

    def test_manual_profile_does_not_infer_tags_from_keywords(self):
        profile = OperationsKnowledgeService._manual_profile(
            asset=SimpleNamespace(
                asset_id=uuid7(), display_name="数据库运维手册"
            ),
            version=SimpleNamespace(asset_version_id=uuid7()),
            scopes=[SimpleNamespace(
                scope_kind="DATABASE_TYPE", scope_value="ORACLE"
            )],
            sources=[SimpleNamespace(source_locator_json={"metadata": {}})],
            revision={"extraction_evidence": [{
                "evidence_id": str(uuid7()),
                "content_text": "使用 psql 连接 PostgreSQL 后执行检查。",
                "locator": {"page": 1},
            }]},
        )
        self.assertEqual((), profile.warnings)
        self.assertEqual(("ORACLE",), profile.scope["database_type"])

    def test_manual_profile_preserves_multiline_fenced_command(self):
        command = "ALTER SYSTEM SET LOG_ARCHIVE_CONFIG=\n  'DG_CONFIG=(DB1,DB2)'\n  SCOPE=BOTH;"
        profile = OperationsKnowledgeService._manual_profile(
            asset=SimpleNamespace(asset_id=uuid7(), display_name="DG 手册"),
            version=SimpleNamespace(asset_version_id=uuid7()),
            scopes=[],
            sources=[SimpleNamespace(source_locator_json={"metadata": {}})],
            revision={"extraction_evidence": [{
                "evidence_id": str(uuid7()),
                "content_text": f"```sql\n{command}\n```",
                "locator": {"page": 2},
            }]},
        )
        self.assertEqual(command, profile.procedures[0].steps[0].command_text)


class OperationsKnowledgeRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_usage_reports_only_assets_and_current_agent_versions(self):
        session = SimpleNamespace(scalar=AsyncMock(side_effect=[3, 2]))
        repository = OperationsKnowledgeRepository(session)  # type: ignore[arg-type]

        usage = await repository.knowledge_base_usage(
            knowledge_base_id=uuid7()
        )

        self.assertEqual({"assets": 3, "agent_versions": 2}, usage)
        self.assertNotIn("knowledge_assistants", usage)

    async def test_asset_version_flushes_parent_rows_before_dependents(self):
        session = _OrderedFlushSession()
        repository = OperationsKnowledgeRepository(session)  # type: ignore[arg-type]
        asset = object()
        version = object()
        scope = object()
        source = object()
        index = object()

        await repository.add_asset_version(
            asset=asset,  # type: ignore[arg-type]
            version=version,  # type: ignore[arg-type]
            scopes=[scope],  # type: ignore[list-item]
            sources=[source],  # type: ignore[list-item]
            indexes=[index],  # type: ignore[list-item]
        )

        self.assertEqual(
            [[asset], [version], [scope, source, index]],
            session.flushes,
        )

    async def test_new_version_flushes_version_before_dependents(self):
        session = _OrderedFlushSession()
        repository = OperationsKnowledgeRepository(session)  # type: ignore[arg-type]
        version = object()
        scope = object()

        await repository.add_version(
            version=version,  # type: ignore[arg-type]
            scopes=[scope],  # type: ignore[list-item]
            sources=[],
            indexes=[],
        )

        self.assertEqual([[version], [scope]], session.flushes)


class OperationsKnowledgeUploadTest(unittest.IsolatedAsyncioTestCase):
    async def test_single_file_upload_preserves_business_display_name(self):
        service = OperationsKnowledgeService(
            uow_factory=lambda: None,
            documents=SimpleNamespace(retrieval=SimpleNamespace(authorization=SimpleNamespace(service=None))),
        )
        service.ingest_manual = AsyncMock(return_value={
            "asset": {"asset_id": str(uuid7())},
            "version": {"asset_version_id": str(uuid7())},
            "replayed": False,
        })
        metadata = ManualUploadMetadata(display_name="Data Guard 运维手册")
        document = {
            "client_file_id": str(uuid7()),
            "file_name": "dg-guide.pdf",
            "display_name": "dg-guide.pdf",
            "media_type": "application/pdf",
            "body": b"pdf",
            "content_sha256": "a" * 64,
            "ordinal": 0,
            "role": "CONTENT",
            "required_flag": False,
        }

        receipt = await service.ingest_upload(
            domain_id=42,
            knowledge_base_id=uuid7(),
            context=SimpleNamespace(request_id="single-upload-test"),
            grouping_mode="EACH_FILE",
            documents=[document],
            metadata=metadata,
            idempotency_key="single-upload-test",
        )

        self.assertEqual("ACCEPTED", receipt["items"][0]["status"])
        forwarded = service.ingest_manual.await_args.kwargs
        self.assertEqual("Data Guard 运维手册", forwarded["metadata"].display_name)
        self.assertEqual("dg-guide.pdf", forwarded["file_name"])


# 旧公共 KC/Bundle 生命周期已退出，以下只覆盖本 App 原子发布协议。
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import inspect
import pytest
from unittest.mock import MagicMock


def lifecycle_fixture():
    from aiops_agent.entities import OperationsKnowledgeAssetEntity
    oracle = OperationsKnowledgeAssetEntity.__tablename__.startswith("KBOT_")
    domain, tenant = (42, None) if oracle else (uuid7(), uuid7())
    asset_id, version_id, doc, job, parse, generation, ref = (uuid7() for _ in range(7))
    asset = SimpleNamespace(asset_id=asset_id, asset_kind="MANUAL", knowledge_base_id=uuid7(),
        row_version=1, current_version_id=None, status="PROCESSING", updated_by=None)
    version = SimpleNamespace(asset_version_id=version_id, asset_id=asset_id, status="PROCESSING", row_version=1,
        tag_analysis_json={"status":"READY","approval":{"status":"APPROVED"}})
    index = SimpleNamespace(index_ref_id=ref, document_id=uuid7(), document_version_id=doc,
        ingestion_job_id=job, parse_run_id=None, index_generation_id=None, index_status="PROCESSING")
    record = {"document_version_id":str(doc), "ingestion_job_id":str(job), "parse_run_id":str(parse),
        "generation_id":str(generation), "processing_status":"PARSED"}
    detail = {"asset":{"asset_kind":"MANUAL","row_version":1}, "version":{"status":"PROCESSING","row_version":1},
        "indexes":[{"index_ref_id":str(ref),"document_version_id":str(doc),"status":"PENDING_APPROVAL"}]}
    repository = SimpleNamespace(get_version=AsyncMock(return_value=version), get_asset=AsyncMock(return_value=asset),
        get_version_relations=AsyncMock(return_value={"indexes":[index],"sources":[],"scopes":[]}), add_review=AsyncMock())
    uow = SimpleNamespace(operations_knowledge=repository, commit=AsyncMock())
    @asynccontextmanager
    async def factory():
        yield uow
    documents = SimpleNamespace(access=lambda _: SimpleNamespace(domain_id=domain, tenant_id=tenant),
        process=AsyncMock(side_effect=lambda *args: dict(record)), excerpts=AsyncMock(return_value={"excerpts":[],"warnings":[]}),
        publish=AsyncMock(), withdraw=AsyncMock())
    service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
    service._uow_factory, service._documents = factory, documents
    service.get_version = AsyncMock(side_effect=lambda **_: deepcopy(detail))
    service._actor = lambda _: "fixture-user"
    service._manual_profile = MagicMock(return_value=SimpleNamespace(model_dump=lambda **_: {}, schema_version="test"))
    kwargs = dict(domain_id=domain, asset_version_id=version_id, context=SimpleNamespace())
    if "tenant_id" in inspect.signature(service.reconcile_version).parameters:
        kwargs["tenant_id"] = tenant
    return SimpleNamespace(**locals())


def test_ready_manual_publishes_app_document_and_business_pointer_in_one_transaction():
    async def run():
        f = lifecycle_fixture()
        await f.service.reconcile_version(**f.kwargs)
        f.documents.publish.assert_awaited_once()
        assert f.index.parse_run_id == f.parse and f.index.index_generation_id == f.generation
        assert f.version.status == f.asset.status == "PUBLISHED"
        assert f.asset.current_version_id == f.version_id
        f.uow.commit.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize("invalid", ["version_changed","asset_changed","document_changed","job_changed","not_approved"])
def test_manual_reconciliation_rechecks_identity_before_publication(invalid):
    async def run():
        f = lifecycle_fixture()
        if invalid == "version_changed": f.version.row_version += 1
        elif invalid == "asset_changed": f.asset.row_version += 1
        elif invalid == "document_changed": f.index.document_version_id = uuid7()
        elif invalid == "job_changed": f.index.ingestion_job_id = uuid7()
        else: f.version.tag_analysis_json = {}
        with pytest.raises(AIOpsApplicationError):
            await f.service.reconcile_version(**f.kwargs)
        f.documents.publish.assert_not_awaited()
        f.uow.commit.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("pending", ["RECEIVED","PARSING","PARSED"])
def test_pending_parse_or_projection_does_not_publish_partial_manual(pending):
    async def run():
        f = lifecycle_fixture()
        f.record.update(processing_status=pending, generation_id=None)
        result = await f.service.reconcile_version(**f.kwargs)
        assert result["version"]["status"] == "PROCESSING"
        f.documents.excerpts.assert_not_awaited()
        f.documents.publish.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("stale", [False, True])
def test_approval_only_advances_to_tagging_and_never_calls_parser_or_model(stale):
    async def run():
        f = lifecycle_fixture()
        f.detail["version"]["status"] = f.version.status = "REVIEW_REQUIRED"
        f.asset.status = "REVIEW_REQUIRED"
        kwargs = {**f.kwargs, "expected_row_version":2 if stale else 1,"comment":"已核对原文"}
        if stale:
            with pytest.raises(AIOpsApplicationError):
                await f.service.approve_manual(**kwargs)
            f.uow.commit.assert_not_awaited()
        else:
            await f.service.approve_manual(**kwargs)
            assert f.version.status == "TAGGING"
            assert f.version.tag_analysis_json["approval"]["status"] == "APPROVED"
            f.repository.add_review.assert_awaited_once()
            f.uow.commit.assert_awaited_once()
        f.documents.process.assert_not_awaited()
        f.documents.publish.assert_not_awaited()
    asyncio.run(run())
