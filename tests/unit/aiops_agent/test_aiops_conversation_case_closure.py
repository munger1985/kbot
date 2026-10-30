"""多轮直接收口的生产服务回归：选择、逐判断当前性、精确索引版本和撤权。"""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.case_source_validation import validate_case_sources
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.knowledge import DiagnosisClosureCommand
from aiops_agent.playbooks import canonical_hash
from platform_core.identity import uuid7
from tests.support.aiops_case_closure import install_closure
from tests.unit.aiops_agent.test_aiops_report_case_extraction import case_fixture
from tests.unit.aiops_agent.test_aiops_report_lifecycle import fixture


async def setup(*, second=True):
    f = await case_fixture()
    install_closure(f)
    f.state.artifacts.pop(f.state.closure.artifact_id)
    f.state.closure = None
    f.state.turns = [f.turn]
    f.state.runs = {f.run_id: f.run}
    f.linked = {}
    if second:
        g = fixture()
        g.turn.turn_no = 2
        g.turn.conversation_id = f.conversation_id
        g.run.target_id, g.run.domain_id, g.run.actor_id, g.run.agent_id = f.target, f.domain, f.actor, f.run.agent_id
        g.artifact.created_at = g.now
        g.artifact.payload_json["investigation_record"]["target_id"] = str(f.target)
        g.artifact.content_hash = canonical_hash(g.artifact.payload_json)
        f.state.artifacts[g.artifact.artifact_id] = g.artifact
        f.state.turns.append(g.turn)
        f.state.runs[g.run_id] = g.run
        f.later = g
    f.uow.turns.list_all_turns.side_effect = lambda **_: f.state.turns
    f.uow.turns.get_turn.side_effect = lambda **kw: next((turn for turn in f.state.turns if turn.turn_id == kw["turn_id"]), None)
    def link_for(turn):
        owner = f.run if turn.turn_id == f.turn_id else f.later.run
        return SimpleNamespace(ops_run_id=owner.ops_run_id, turn_id=turn.turn_id, purpose="PRIMARY")
    f.uow.turns.get_run_link.side_effect = lambda **kw: link_for(next(turn for turn in f.state.turns if turn.turn_id == kw["turn_id"]))
    f.uow.turns.get_run_link_by_ops_run_id.side_effect = lambda **kw: next((link_for(turn) for turn in f.state.turns if link_for(turn).ops_run_id == kw["ops_run_id"]), None)
    f.uow.turns.list_run_links.side_effect = lambda **kw: [
        link_for(next(turn for turn in f.state.turns if turn.turn_id == kw["turn_id"])), *f.linked.get(kw["turn_id"], [])]
    f.uow.turns.list_turns.side_effect = lambda **kw: [turn for turn in f.state.turns if turn.turn_no > kw["after_turn_no"]]
    f.uow.runs.get_run_scoped.side_effect = lambda **kw: f.state.runs.get(kw["ops_run_id"])
    async def latest(**kw):
        options = [f.state.versions[row_id] for row_id, relations in f.state.relations.items()
            if any(row.source_external_id == kw["external_id"] for row in relations["sources"])]
        version = max(options, key=lambda row: row.version_no) if options else None
        return (f.state.assets[version.asset_id], version) if version else None
    f.uow.operations_knowledge.find_case_by_external_id = AsyncMock(side_effect=latest)
    f.uow.operations_knowledge.list_versions = AsyncMock(side_effect=lambda **kw: [
        row for row in f.state.versions.values() if row.asset_id == kw["asset_id"]])
    async def add_version(**kw):
        kw["version"].row_version = 1
        f.state.versions[kw["version"].asset_version_id] = kw["version"]
        f.state.relations[kw["version"].asset_version_id] = {key: kw[key] for key in ("sources", "indexes")}
        for index in kw["indexes"]: index.index_ref_id = index.index_ref_id or uuid7()
    f.uow.operations_knowledge.add_version = AsyncMock(side_effect=add_version)
    return f


async def preview(f, ids=None):
    return await f.service.conversation_case_preview(domain_id=f.domain, conversation_id=f.conversation_id,
        context=f.context, included_turn_ids=ids)


async def close(f, candidate=None, **changes):
    candidate = candidate or await preview(f)
    data = dict(included_turn_ids=candidate["included_turn_ids"], source_fingerprint=candidate["source_fingerprint"],
        display_name="业务等待排查", symptoms=["短时业务等待"], root_cause_summary="当前证据未确认性能故障")
    data.update(changes)
    kwargs = dict(domain_id=f.domain, conversation_id=f.conversation_id, context=f.context,
        command=DiagnosisClosureCommand.model_validate(data))
    if "tenant_id" in inspect.signature(f.service.close_diagnosis).parameters:
        kwargs["tenant_id"] = uuid7()
    return await f.service.close_diagnosis(**kwargs)


def test_multi_turn_default_and_explicit_selection_keep_exact_claims():
    async def run():
        f = await setup()
        candidate = await preview(f)
        assert candidate["included_turn_ids"] == [str(f.turn_id), str(f.later.turn_id)]
        assert len(candidate["turn_candidates"]) == 2
        one = await preview(f, (f.turn_id,))
        assert one["included_turn_ids"] == [str(f.turn_id)]
        assert all(row["turn_id"] == str(f.turn_id) for row in one["evidence_candidates"])
        result = await close(f, one)
        source = result["version"]["profile"]["source"]
        assert source["turn_ids"] == [str(f.turn_id)]
        assert source["claim_refs"][0]["claim_id"] == "c1"
        assert result["version"]["status"] == "PROCESSING"
        assert result["asset"]["current_version_id"] is None
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["duplicate", "unknown", "cross_target", "stale"])
def test_invalid_selection_or_stale_fingerprint_never_reaches_model(mode):
    async def run():
        f = await setup()
        candidate = await preview(f)
        if mode == "duplicate": candidate["included_turn_ids"] = [str(f.turn_id)] * 2
        if mode == "unknown": candidate["included_turn_ids"] = [str(uuid7())]
        if mode == "cross_target": f.later.run.target_id = uuid7()
        if mode == "stale": candidate["source_fingerprint"] = "f" * 64
        with pytest.raises(AIOpsApplicationError): await close(f, candidate)
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.service._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


def test_other_target_is_not_silently_merged_and_explicit_selection_can_switch():
    async def run():
        f = await setup()
        f.later.run.target_id = uuid7()
        f.later.artifact.payload_json["investigation_record"]["target_id"] = str(f.later.run.target_id)
        f.later.artifact.content_hash = canonical_hash(f.later.artifact.payload_json)
        assert (await preview(f))["included_turn_ids"] == [str(f.later.turn_id)]
        assert (await preview(f, (f.turn_id,)))["included_turn_ids"] == [str(f.turn_id)]
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["model", "intake", "approval"])
def test_midflight_revocation_prevents_upload_or_persists_exact_withdrawal(phase):
    async def run():
        f = await setup()
        if phase == "model":
            async def revoked_model(**_):
                f.uow.agents.version_has_target.return_value = False
                return SimpleNamespace(output={"spans": []})
            f.service._structured_model.generate_structured.side_effect = revoked_model
        else:
            method = f.service._kc.ingest_user_file if phase == "intake" else f.service._kc.review_user_intake
            value = method.return_value
            async def revoked(**_):
                f.uow.agents.version_has_target.return_value = False
                return value
            method.side_effect = revoked
        with pytest.raises(AIOpsApplicationError): await close(f)
        if phase == "model":
            f.service._kc.ingest_user_file.assert_not_awaited()
        else:
            version = next(iter(f.state.versions.values()))
            assert version.status == "REVIEW_REQUIRED"
            index = f.state.relations[version.asset_version_id]["indexes"][0]
            accepted = f.service._kc.ingest_user_file.return_value.payload["items"][0]
            assert str(index.bundle_revision_id) == accepted["bundle_revision_id"]
            assert f.uow.outbox.add.await_count >= 1
        if phase != "approval": f.service._kc.review_user_intake.assert_not_awaited()
    asyncio.run(run())


def test_manual_recovery_and_high_security_receipt_do_not_grant_verified():
    async def run():
        f = await setup()
        result = await close(f, actions=["人工调整调用频率"], action_evidence_ids=[f.action_artifact.artifact_id],
            verification_result="RESOLVED")
        assert result["asset"]["security_level"] == 3
        profile = result["version"]["profile"]
        assert profile["case_kind"] == "DIAGNOSTIC_REFERENCE"
        assert profile["actions"][0]["source_kind"] == "USER_PROVIDED"
        assert profile["verification"]["method"] == "USER_REPORTED"
        closure = next(row for row in f.state.artifacts.values() if row.schema_version == "DIAGNOSIS_CLOSURE.v1")
        assert closure.security_level == 3
        assert closure.payload_json["source_refs"][-1]["artifact_id"] == str(f.action_artifact.artifact_id)
        assert await validate_case_sources(uow=f.uow, domain_id=f.domain, version_id=next(iter(f.state.versions)))
        f.action_artifact.content_hash = "f" * 64
        assert not await validate_case_sources(uow=f.uow, domain_id=f.domain, version_id=next(iter(f.state.versions)))
    asyncio.run(run())


def test_plan_or_observation_cannot_become_execution_evidence():
    async def run():
        f = await setup()
        with pytest.raises(AIOpsApplicationError):
            await close(f, actions=["声称已实施"], action_evidence_ids=[f.artifact.artifact_id])
        f.service._kc.ingest_user_file.assert_not_awaited()
    asyncio.run(run())


def refute_first_claim(f):
    import copy
    old = f.artifact.payload_json["investigation_record"]
    second = copy.deepcopy(old["claims"][0])
    second["claim_id"] = "c2"
    second["statement"] = "保留仍成立的活动采样计数。"
    old["claims"].append(second)
    f.artifact.content_hash = canonical_hash(f.artifact.payload_json)
    later = f.later.artifact.payload_json["investigation_record"]
    later.update(record_id=old["record_id"], revision=2, previous_revision=str(f.artifact.artifact_id),
        parent={"record_id": old["record_id"], "revision": 1, "target_id": old["target_id"],
            "run_id": str(f.run_id), "artifact_id": str(f.artifact.artifact_id),
            "content_hash": f.artifact.content_hash, "claims": copy.deepcopy(old["claims"])},
        claim_revisions=[{"prior_claim_id": "c1", "current_claim_id": "c1", "relation": "REFUTES",
            "reason": "补充证据推翻原先解释，仅保留其他仍有效判断"}])
    f.later.artifact.content_hash = canonical_hash(f.later.artifact.payload_json)


def test_refuted_claim_is_removed_without_discarding_other_valid_claims():
    async def run():
        f = await setup()
        refute_first_claim(f)
        candidate = await preview(f, (f.turn_id,))
        assert candidate["eligible"]
        result = await close(f, candidate)
        assert {row["claim_id"] for row in result["version"]["profile"]["source"]["claim_refs"]} == {"c2"}
        assert await validate_case_sources(uow=f.uow, domain_id=f.domain, version_id=next(iter(f.state.versions)))
        later = f.later.artifact.payload_json["investigation_record"]
        later["claim_revisions"][0]["prior_claim_id"] = "c2"
        f.later.artifact.content_hash = canonical_hash(f.later.artifact.payload_json)
        assert not await validate_case_sources(uow=f.uow, domain_id=f.domain, version_id=next(iter(f.state.versions)))
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["hash", "target_version", "turn_scope", "missing", "future_active"])
def test_every_frozen_source_is_replayed_for_later_use(mutation):
    async def run():
        f = await setup()
        await close(f)
        if mutation == "hash": f.artifact.content_hash = "f" * 64
        if mutation == "target_version": f.uow.targets.get_scoped.return_value.row_version += 1
        if mutation == "turn_scope": f.turn.conversation_id = uuid7()
        if mutation == "missing": f.state.artifacts.pop(f.later.artifact.artifact_id)
        if mutation == "future_active": f.later.turn.status = "RUNNING"
        assert not await validate_case_sources(uow=f.uow, domain_id=f.domain, version_id=next(iter(f.state.versions)))
    asyncio.run(run())


def test_idempotency_revalidates_sources_and_approval_failure_is_recoverable():
    async def run():
        f = await setup()
        candidate = await preview(f)
        f.service._kc.review_user_intake.side_effect = RuntimeError("索引暂不可用")
        with pytest.raises(RuntimeError): await close(f, candidate)
        version = next(iter(f.state.versions.values()))
        assert f.state.relations[version.asset_version_id]["indexes"][0].index_status == "PENDING_APPROVAL"
        result = await close(f, candidate)
        assert result["replayed"]
        assert f.service._kc.ingest_user_file.await_count == 1
        f.uow.agents.version_has_target.return_value = False
        with pytest.raises(AIOpsApplicationError): await close(f, candidate)
    asyncio.run(run())


def test_multi_turn_closure_report_retains_exact_source_runs_and_omits_revoked_appendix():
    async def run():
        from aiops_agent.application.runtime.service import AIOpsRuntimeService
        from aiops_agent.application.reporting import resolve_system_template
        f = await setup()
        await close(f, actions=["人工处理"], action_evidence_ids=[f.action_artifact.artifact_id])
        closure = next(row for row in f.state.artifacts.values() if row.schema_version == "DIAGNOSIS_CLOSURE.v1")
        f.uow.runs.get_latest_artifact_by_prefix.side_effect = lambda **kw: closure if kw["ops_run_id"] == closure.ops_run_id else None
        f.uow.runs.list_tasks.return_value = [f.task, f.later.task]
        runtime = AIOpsRuntimeService(uow_factory=lambda: f.manager, blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
        receipt = await runtime.generate_conversation_report(domain_id=f.domain, conversation_id=f.conversation_id,
            actor_id=f.actor, template=resolve_system_template("system:diagnosis.standard"), trace_id="closure")
        rows = f.uow.inspections.add_report_sources.await_args.args[0]
        assert (f.run_id, f.action_artifact.artifact_id) in {(row.ops_run_id, row.source_artifact_id) for row in rows}
        assert (f.later.run_id, closure.artifact_id) in {(row.ops_run_id, row.source_artifact_id) for row in rows}
        assert receipt.status == "READY"
        f.uow.targets.get_scoped.return_value.row_version += 1
        await runtime.generate_conversation_report(domain_id=f.domain, conversation_id=f.conversation_id,
            actor_id=f.actor, template=resolve_system_template("system:diagnosis.standard"), trace_id="closure-invalid")
        report = f.state.artifacts[f.state.current.content_artifact_id].payload_json
        assert not report["scope"].get("closure_artifact_id")
        assert f.state.current.status == "PARTIAL"
    asyncio.run(run())


def test_concurrent_accepted_different_revision_is_registered_and_withdrawn_not_lost():
    async def run():
        f = await setup()
        candidate = await preview(f)
        first_response = f.service._kc.ingest_user_file.return_value
        other_response = SimpleNamespace(payload={"items": [{"status": "ACCEPTED",
            "bundle_id": str(uuid7()), "bundle_revision_id": str(uuid7())}]})
        async def overlap(**_):
            f.service._kc.ingest_user_file.side_effect = None
            f.service._kc.ingest_user_file.return_value = other_response
            await close(f, candidate)
            return first_response
        f.service._kc.ingest_user_file.side_effect = overlap
        result = await close(f, candidate)
        assert result["replayed"]
        revisions = {str(index.bundle_revision_id) for relations in f.state.relations.values()
            for index in relations["indexes"]}
        assert revisions == {first_response.payload["items"][0]["bundle_revision_id"],
            other_response.payload["items"][0]["bundle_revision_id"]}
        assert len(f.state.versions) == 2
        assert f.uow.outbox.add.await_count >= 1
        # 多余版本不会遮蔽用户已获得的有效候选，重复提交仍返回该版本。
        assert (await close(f, candidate))["version"]["asset_version_id"] == result["version"]["asset_version_id"]
    asyncio.run(run())


@pytest.mark.parametrize("stale_fact", [False, True])
def test_report_case_keeps_valid_claim_without_reviving_refuted_claim(stale_fact):
    async def run():
        from aiops_agent.application.runtime.service import AIOpsRuntimeService
        from aiops_agent.application.reporting import resolve_system_template
        from aiops_agent.application.diagnosis.investigation_record import investigation_report_source
        from aiops_agent.contracts.investigation import InvestigationRecord
        from tests.unit.aiops_agent.test_aiops_report_case_extraction import extract
        f = await setup()
        refute_first_claim(f)
        f.uow.runs.list_tasks.return_value = [f.task, f.later.task]
        runtime = AIOpsRuntimeService(uow_factory=lambda: f.manager,
            blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
        await runtime.generate_conversation_report(domain_id=f.domain, conversation_id=f.conversation_id,
            actor_id=f.actor, template=resolve_system_template("system:diagnosis.standard"), trace_id="partial-claim")
        f.report = f.state.current
        f.report.is_current = True
        content = f.state.artifacts[f.report.content_artifact_id]
        assert {fact["claim_id"] for fact in content.payload_json["facts"]
            if fact.get("source_run_id") == str(f.run_id)} == {"c2"}
        if stale_fact:
            original = investigation_report_source(InvestigationRecord.model_validate(
                f.artifact.payload_json["investigation_record"]))["facts"][0]
            content.payload_json["facts"].append(original)
            content.content_hash = canonical_hash(content.payload_json)
            f.report.content_hash = content.content_hash
            with pytest.raises(AIOpsApplicationError): await extract(f)
            f.service._kc.ingest_user_file.assert_not_awaited()
        else:
            result = await extract(f)
            assert {ref["claim_id"] for ref in result["version"]["profile"]["source"]["claim_refs"]
                if ref["source_run_id"] == str(f.run_id)} == {"c2"}
            assert await validate_case_sources(uow=f.uow, domain_id=f.domain,
                version_id=next(iter(f.state.versions)))
    asyncio.run(run())


@pytest.mark.parametrize("mismatch", [None, "run", "task", "target", "result_hash", "executor_request_id",
    "executor_instance_id", "status_version", "grant_jti_hash"])
def test_formal_receipt_requires_exact_authoritative_execution(mismatch):
    async def run():
        from aiops_agent.application.case_closure import closure_action_is_valid
        from aiops_agent.contracts.change.models import ExecutionResultArtifact
        f = await setup(second=False)
        receipt = ExecutionResultArtifact(execution_id=str(uuid7()), proposal_id=str(uuid7()), executor_request_id=str(uuid7()),
            executor_instance_id="test-executor", status="SUCCEEDED", status_version=4, occurred_at=f.now,
            result_hash="a" * 64, proposal_hash="b" * 64, command_hash="c" * 64, grant_jti_hash="d" * 64)
        artifact = SimpleNamespace(artifact_id=uuid7(), ops_run_id=f.run_id, ops_task_id=f.task_id,
            schema_version=receipt.schema_version, trust_level="SOURCE_VERIFIED", payload_json=receipt.model_dump(mode="json"))
        execution = SimpleNamespace(**receipt.model_dump(), result_artifact_id=artifact.artifact_id,
            ops_run_id=f.run_id, ops_task_id=f.task_id, target_id=f.target)
        if mismatch:
            field = {"run": "ops_run_id", "task": "ops_task_id", "target": "target_id"}.get(mismatch, mismatch)
            setattr(execution, field, 5 if field == "status_version" else
                (uuid7() if field.endswith("_id") and field != "executor_instance_id" else "different"))
        f.uow.changes = SimpleNamespace(get_execution=AsyncMock(return_value=execution))
        assert await closure_action_is_valid(uow=f.uow, run=f.run, artifact=artifact) == (mismatch is None)
    asyncio.run(run())


@pytest.mark.parametrize("invalidate_latest", [False, True])
def test_report_uses_latest_closure_across_selected_turns_and_never_revives_older_one(invalidate_latest):
    async def run():
        from datetime import timedelta
        from aiops_agent.application.runtime.service import AIOpsRuntimeService
        from aiops_agent.application.reporting import resolve_system_template
        f = await setup()
        await close(f, await preview(f, (f.later.turn_id,)))
        old = next(row for row in f.state.artifacts.values() if row.schema_version == "DIAGNOSIS_CLOSURE.v1")
        f.service._kc.ingest_user_file.return_value = SimpleNamespace(payload={"items": [{"status": "ACCEPTED",
            "bundle_id": str(uuid7()), "bundle_revision_id": str(uuid7())}]})
        await close(f, await preview(f, (f.turn_id,)), root_cause_summary="后来确认的第一轮收口，不能被较旧人工归纳覆盖")
        newest = next(row for row in f.state.artifacts.values()
            if row.schema_version == "DIAGNOSIS_CLOSURE.v1" and row.artifact_id != old.artifact_id)
        newest.created_at = old.created_at + timedelta(seconds=1)
        if invalidate_latest: newest.content_hash = "f" * 64
        async def latest(**kw):
            return newest if kw["ops_run_id"] == f.run_id else old
        f.uow.runs.get_latest_artifact_by_prefix.side_effect = latest
        f.uow.runs.list_tasks.return_value = [f.task, f.later.task]
        runtime = AIOpsRuntimeService(uow_factory=lambda: f.manager,
            blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
        await runtime.generate_conversation_report(domain_id=f.domain, conversation_id=f.conversation_id,
            actor_id=f.actor, template=resolve_system_template("system:diagnosis.standard"), trace_id="latest-closure")
        content = f.state.artifacts[f.state.current.content_artifact_id].payload_json
        assert content["scope"].get("closure_artifact_id") == (None if invalidate_latest else str(newest.artifact_id))
        assert f.state.current.status == ("PARTIAL" if invalidate_latest else "READY")
    asyncio.run(run())
