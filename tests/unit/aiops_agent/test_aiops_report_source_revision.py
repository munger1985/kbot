"""展示修订保留知识，实质更正及失效来源不能借新标题绕过治理。"""

import asyncio
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
import inspect

import pytest

from aiops_agent.application.case_source_validation import validate_case_sources
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.report_source_revision import report_source_is_usable, same_report_diagnosis
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from platform_core.contracts.aiops import ReportSectionEdit
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_report_case_extraction import case_fixture, extract


async def revision_fixture():
    f = await case_fixture()
    f.state.reports = {f.report.report_id: f.report}
    f.state.report_sources = {f.report.report_id: f.uow.inspections.add_report_sources.await_args.args[0]}
    f.report.created_at = f.now
    f.uow.inspections.get_report_scoped.side_effect = lambda **kw: f.state.reports.get(kw["report_id"])
    f.uow.inspections.list_report_sources.side_effect = lambda **kw: f.state.report_sources.get(kw["report_id"], [])
    f.uow.inspections.list_report_successors_scoped.side_effect = lambda **kw: [
        item for item in f.state.reports.values() if item.supersedes_report_id == kw["report_id"]]
    async def publish(report):
        f.state.current.is_current = False
        report.created_at, report.is_current = f.now, True
        f.state.current = report
        f.state.reports[report.report_id] = report
        return report
    async def register(rows):
        for row in rows:
            f.state.report_sources.setdefault(row.report_id, []).append(row)
    f.uow.inspections.publish_report.side_effect = publish
    f.uow.inspections.add_report_sources.side_effect = register
    f.uow.operations_knowledge.mark_cases_for_report_review = AsyncMock(return_value=[])
    f.runtime = AIOpsRuntimeService(uow_factory=lambda: f.manager,
        blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
    return f


async def edit(f, *, sections=()):
    report = f.state.current
    return await f.runtime.edit_report(report_id=report.report_id, domain_id=f.domain, actor_id=f.actor,
        expected_report_version=report.report_version, title="修正报告标题", sections=sections, trace_id="revision-test")


def revision(f, change=None):
    """构造明确后继以覆盖模板重发与异常来源；人工编辑另走真实服务。"""
    before = f.state.current
    fields = ("ops_run_id", "target_id", "source_conversation_id", "report_key", "report_type", "status",
        "period_start", "period_end", "baseline_start", "baseline_end", "after_start", "after_end", "result",
        "security_level", "title", "template_id", "template_version", "generated_by_task_id")
    report = SimpleNamespace(**{name: getattr(before, name) for name in fields}, report_id=uuid7(),
        supersedes_report_id=before.report_id, report_version=before.report_version + 1, is_current=True)
    payload = deepcopy(f.state.artifacts[before.content_artifact_id].payload_json)
    if change:
        change(payload)
    payload["title"] = "新展示标题"
    payload["provenance"]["template"] = {"version": "new-layout"}
    payload["provenance"]["report_input_hash"] = "new-template-input"
    artifact = SimpleNamespace(artifact_id=uuid7(), ops_run_id=before.ops_run_id, schema_version="REPORT_CONTENT.v1",
        payload_json=payload, content_hash=sha256_json(payload), security_level=before.security_level)
    report.content_artifact_id, report.content_hash = artifact.artifact_id, artifact.content_hash
    f.state.artifacts[artifact.artifact_id] = artifact
    before.is_current = False
    f.state.reports[report.report_id] = report
    f.state.report_sources[report.report_id] = list(f.state.report_sources[before.report_id])
    f.state.current = report
    return report


async def valid(f, version_id=None):
    if version_id is None:
        return await report_source_is_usable(uow=f.uow, domain_id=f.domain, report=f.report)
    return await validate_case_sources(uow=f.uow, domain_id=f.domain, version_id=version_id)


def test_real_title_edit_preserves_case_and_conversation_lineage():
    async def run():
        f = await revision_fixture()
        f.report.source_conversation_id = f.conversation_id
        first = await extract(f)
        version_id = next(iter(f.state.versions))
        await edit(f)
        assert f.state.current.source_conversation_id == f.conversation_id
        assert await valid(f, version_id)
        f.uow.operations_knowledge.mark_cases_for_report_review.assert_not_awaited()
        assert not any(call.args[0].event_type == "OPS_CASE_WITHDRAWAL_REQUESTED" for call in f.uow.outbox.add.await_args_list)
        event = f.uow.runs.append_event.await_args.kwargs["payload_json"]
        assert event["correction_kind"] == "PRESENTATION_ONLY"
        pair = (next(iter(f.state.assets.values())), next(iter(f.state.versions.values())))
        origin_id = f.report.report_id
        f.uow.operations_knowledge.find_case_by_report_id.side_effect = lambda **kw: pair if kw["report_id"] == origin_id else None
        f.report = f.state.current
        replayed = await extract(f)
        assert replayed["replayed"] and replayed["version"]["asset_version_id"] == first["version"]["asset_version_id"]
        assert f.service._kc.ingest_user_file.await_count == 1
    asyncio.run(run())


def test_substantive_human_section_edit_withdraws_cases_from_all_ancestors():
    async def run():
        f = await revision_fixture()
        await extract(f)
        asset, version = next(iter(f.state.assets.values())), next(iter(f.state.versions.values()))
        asset.status = version.status = "PUBLISHED"
        asset.current_version_id = version.asset_version_id
        await edit(f)
        intermediate = f.state.current.report_id
        f.uow.operations_knowledge.mark_cases_for_report_review.side_effect = lambda **kw: (
            [(asset, version)] if kw["report_id"] == f.report.report_id else [])
        await edit(f, sections=(ReportSectionEdit(kind="ROOT_CAUSE", items=("人工修改原因，须重新审核",)),))
        calls = [item.kwargs["report_id"] for item in f.uow.operations_knowledge.mark_cases_for_report_review.await_args_list]
        assert calls == [intermediate, f.report.report_id]
        assert not await valid(f, version.asset_version_id)
        assert any(call.args[0].event_type == "OPS_CASE_WITHDRAWAL_REQUESTED" for call in f.uow.outbox.add.await_args_list)
        assert f.uow.runs.append_event.await_args.kwargs["payload_json"]["correction_kind"] == "SUBSTANTIVE"
    asyncio.run(run())


@pytest.mark.parametrize("field", ["summary", "facts", "gaps", "scope", "recommendations", "evidence_refs",
    "presentation_overrides", "status", "source_provenance", "unknown"])
def test_technical_changes_are_not_presentation_even_when_original_hash_is_valid(field):
    async def run():
        f = await revision_fixture()
        def change(payload):
            if field == "source_provenance":
                payload["provenance"]["source_artifact_hash"] = "changed"
            elif field == "unknown":
                payload["unknown_extension"] = {"technical": "unknown"}
            else:
                payload[field] = {"human": "changed"}
        revision(f, change)
        assert not await valid(f)
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["missing", "fork", "hash", "payload", "registered_source", "security",
    "period", "version", "target", "wrong_run", "not_ready", "wrong_parent", "artifact_security"])
def test_title_changes_do_not_exempt_originals_registry_scope_or_chain(failure):
    async def run():
        f = await revision_fixture()
        after = revision(f)
        assert await valid(f)
        if failure == "missing":
            f.state.reports.pop(after.report_id)
        elif failure == "fork":
            other = SimpleNamespace(**vars(after))
            other.report_id = uuid7()
            f.state.reports[other.report_id] = other
        elif failure == "hash":
            after.content_hash = "f" * 64
        elif failure == "payload":
            f.state.artifacts[after.content_artifact_id].payload_json["title"] = "未更新哈希"
        elif failure == "registered_source":
            f.state.report_sources[after.report_id] = []
        elif failure == "security":
            after.security_level += 1
        elif failure == "artifact_security":
            f.state.artifacts[after.content_artifact_id].security_level += 1
        elif failure == "period":
            after.period_end += timedelta(seconds=1)
        elif failure == "version":
            after.report_version += 1
        elif failure == "target":
            after.target_id = uuid7()
        elif failure == "wrong_run":
            after.ops_run_id = uuid7()
        elif failure == "not_ready":
            after.status = "FAILED"
        else:
            after.supersedes_report_id = uuid7()
        assert not await valid(f)
    asyncio.run(run())


def test_substantive_change_then_revert_does_not_restore_known_wrong_knowledge():
    async def run():
        f = await revision_fixture()
        original = deepcopy(f.content_artifact.payload_json)
        revision(f, lambda payload: payload.update(summary="明确技术更正"))
        revision(f, lambda payload: payload.update(original))
        assert same_report_diagnosis(original, f.state.artifacts[f.state.current.content_artifact_id].payload_json)
        assert not await valid(f)
    asyncio.run(run())


@pytest.mark.parametrize("payload", [[], "not-an-object", {"provenance": "invalid-provenance"}])
def test_malformed_revision_fails_closed_instead_of_breaking_search(payload):
    async def run():
        f = await revision_fixture()
        after = revision(f)
        artifact = f.state.artifacts[after.content_artifact_id]
        artifact.payload_json = payload
        after.content_hash = artifact.content_hash = sha256_json(payload)
        assert not await valid(f)
    asyncio.run(run())


def test_long_revision_chain_fails_closed_without_unbounded_reads():
    async def run():
        f = await revision_fixture()
        for _ in range(65):
            revision(f)
        assert not await valid(f)
        assert f.uow.inspections.list_report_successors_scoped.await_count == 64
    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "revoked", "source_hash"])
def test_real_publish_rechecks_original_source_after_presentation_update(failure):
    async def run():
        f = await revision_fixture()
        await extract(f)
        revision(f)
        asset, version = next(iter(f.state.assets.values())), next(iter(f.state.versions.values()))
        relations = f.state.relations[version.asset_version_id]
        relations.update(scopes=[], reviews=[])
        index = relations["indexes"][0]
        index.index_status, version.status = "READY", "REVIEW_REQUIRED"
        f.service._kc.get_bundle_status = AsyncMock(return_value={
            "current_revision_id": str(index.bundle_revision_id), "availability_status": "READY"})
        if failure == "revoked":
            f.uow.agents.version_has_target.return_value = False
        elif failure == "source_hash":
            f.artifact.content_hash = "f" * 64
        kwargs = dict(domain_id=f.domain, asset_version_id=version.asset_version_id,
            decision="PUBLISH", expected_row_version=version.row_version, comment="人工审核", context=f.context)
        if "tenant_id" in inspect.signature(f.service.review).parameters:
            kwargs["tenant_id"] = uuid7()
        if failure:
            with pytest.raises(AIOpsApplicationError):
                await f.service.review(**kwargs)
            assert asset.current_version_id is None
        else:
            published = await f.service.review(**kwargs)
            assert published["version"]["status"] == "PUBLISHED"
    asyncio.run(run())


@pytest.mark.parametrize("substantive", [False, True])
def test_background_revalidation_preserves_layout_but_pauses_technical_corrections(substantive):
    async def run():
        f = await revision_fixture()
        await extract(f)
        revision(f, (lambda payload: payload.update(summary="根因判断已更正")) if substantive else None)
        asset, version = next(iter(f.state.assets.values())), next(iter(f.state.versions.values()))
        index = f.state.relations[version.asset_version_id]["indexes"][0]
        asset.current_version_id, asset.status, version.status = version.asset_version_id, "PUBLISHED", "PUBLISHED"
        f.uow.operations_knowledge.next_case_index_for_validation = AsyncMock(return_value=(index, f.domain))
        f.uow.platform_notifications = SimpleNamespace(emit_case_review_required=AsyncMock())
        assert await f.service.revalidate_next_case()
        assert (version.status == "REVIEW_REQUIRED") is substantive
        assert (asset.current_version_id is None) is substantive
        assert f.uow.outbox.add.await_count == int(substantive)
        assert f.uow.platform_notifications.emit_case_review_required.await_count == int(substantive)
    asyncio.run(run())


def test_substantive_correction_is_filtered_before_any_remote_knowledge_read():
    from aiops_agent.contracts.knowledge import KnowledgeSearchRequest
    async def run():
        f = await revision_fixture()
        await extract(f)
        revision(f, lambda payload: payload.update(summary="更正结论"))
        asset, version = next(iter(f.state.assets.values())), next(iter(f.state.versions.values()))
        index = f.state.relations[version.asset_version_id]["indexes"][0]
        asset.current_version_id, asset.status, version.status = version.asset_version_id, "PUBLISHED", "PUBLISHED"
        f.uow.operations_knowledge.list_published_candidates = AsyncMock(return_value=[(asset, version, index)])
        f.service._kc.discover = AsyncMock()
        kwargs = dict(domain_id=f.domain, agent_id=str(f.run.agent_id),
            request=KnowledgeSearchRequest(query="问题检查", knowledge_base_ids=(uuid7(),)), context=f.context)
        if "tenant_id" in inspect.signature(f.service.search).parameters:
            kwargs["tenant_id"] = uuid7()
        result = await f.service.search(**kwargs)
        assert not result["results"]
        f.service._kc.discover.assert_not_awaited()
    asyncio.run(run())


def test_title_only_edit_does_not_erase_existing_human_sections_or_revoke_again():
    async def run():
        f = await revision_fixture()
        await edit(f, sections=(ReportSectionEdit(kind="ROOT_CAUSE", items=("人工复核说明",)),))
        previous = deepcopy(f.state.artifacts[f.state.current.content_artifact_id].payload_json)
        f.uow.operations_knowledge.mark_cases_for_report_review.reset_mock()
        await edit(f)
        current = f.state.artifacts[f.state.current.content_artifact_id].payload_json
        assert current["presentation_overrides"] == previous["presentation_overrides"]
        assert same_report_diagnosis(previous, current)
        f.uow.operations_knowledge.mark_cases_for_report_review.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("substantive", [False, True])
def test_revision_during_semantic_scan_is_checked_before_index_intake(substantive):
    async def run():
        f = await revision_fixture()
        response = f.service._structured_model.generate_structured.return_value
        async def scan(**kw):
            revision(f, (lambda payload: payload.update(summary="技术判断变化")) if substantive else None)
            return response
        f.service._structured_model.generate_structured.side_effect = scan
        if substantive:
            with pytest.raises(AIOpsApplicationError):
                await extract(f)
            f.service._kc.ingest_user_file.assert_not_awaited()
        else:
            result = await extract(f)
            assert result["version"]["status"] == "PROCESSING"
    asyncio.run(run())


def test_successor_repository_is_domain_scoped_and_detects_forks_without_unbounded_results():
    from aiops_agent.entities import ReportEntity, TargetEntity
    from aiops_agent.repositories.inspection import InspectionRepository
    async def run():
        reports = [SimpleNamespace(report_id=uuid7()), SimpleNamespace(report_id=uuid7())]
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: iter(reports))))
        repo = InspectionRepository(session)
        report_id, domain_id = uuid7(), uuid7()
        result = await repo.list_report_successors_scoped(report_id=report_id, domain_id=domain_id)
        statement = session.execute.await_args.args[0]
        sql = str(statement).lower()
        assert str(ReportEntity.supersedes_report_id.property.columns[0].name).lower() in sql
        assert str(TargetEntity.domain_id.property.columns[0].name).lower() in sql
        assert " join " in sql and report_id in statement.compile().params.values()
        assert domain_id in statement.compile().params.values() and statement._limit_clause.value == 2
        assert result == reports
    asyncio.run(run())
