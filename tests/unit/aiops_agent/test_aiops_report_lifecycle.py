"""正式报告按问题完成度、确切来源和处置验证状态冻结，不重新诊断。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
from aiops_agent.application.reporting import normalize_report_source, report_presentation, resolve_system_template, render_pdf
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
from aiops_agent.contracts.turn_answer import AIOpsTurnResult, TurnEvidenceFact
from platform_core.identity import uuid7


def test_report_source_identity_is_per_artifact_not_per_run():
    from aiops_agent.entities import ReportSourceEntity
    from tests.support.aiops_report_sources import assert_report_sources_persist
    columns = [column.name for column in ReportSourceEntity.__table__.primary_key]
    assert columns == ["report_id", "ops_run_id", "source_artifact_id"]
    report_id, run_id = uuid7(), uuid7()
    sources = [ReportSourceEntity(report_id=report_id, ops_run_id=run_id,
        source_artifact_id=uuid7(), source_kind="COMPARISON", content_hash="a" * 64,
        observed_at=datetime.now(UTC)) for _ in range(4)]
    assert_report_sources_persist(sources)


@pytest.mark.parametrize("state", ["old", "current", "unknown"])
def test_deployment_preserves_report_sources_and_rejects_unknown_key(state, monkeypatch):
    from pathlib import Path
    from aiops_agent.entities import ReportSourceEntity
    root = Path(__file__).resolve().parents[3]
    if ReportSourceEntity.__tablename__.startswith("KBOT_"):
        canonical = (root / "database/oracle/aiops_agent/004_ops_inspection.sql").read_text()
        upgrade = (root / "database/oracle/operations/apply_aiops_schema_53.sql").read_text()
        assert "PRIMARY KEY (REPORT_ID, OPS_RUN_ID, SOURCE_ARTIFACT_ID)" in canonical
        assert "PRIMARY KEY (REPORT_ID, OPS_RUN_ID, SOURCE_ARTIFACT_ID)" in upgrade
        assert "l_before <> l_after" in upgrade
        assert "DELETE " not in upgrade and "TRUNCATE " not in upgrade and "CASCADE" not in upgrade
        assert "l_name IS NULL" in upgrade
        return
    from scripts.deployment import init_postgresql as deployment
    columns = {"old": ["report_id", "ops_run_id"],
        "current": ["report_id", "ops_run_id", "source_artifact_id"], "unknown": ["report_id"]}[state]
    inspector = SimpleNamespace(get_pk_constraint=lambda *_, **__: {
        "name": 'report"source_pk', "constrained_columns": columns})
    monkeypatch.setattr(deployment, "inspect", lambda _: inspector)
    async def run():
        connection = SimpleNamespace(run_sync=AsyncMock(side_effect=lambda callback: callback(None)), execute=AsyncMock())
        if state == "unknown":
            with pytest.raises(RuntimeError, match="拒绝"):
                await deployment._align_aiops_report_source_key(connection, schema="report_test")
            connection.execute.assert_not_awaited()
        else:
            await deployment._align_aiops_report_source_key(connection, schema="report_test")
            if state == "current":
                connection.execute.assert_not_awaited()
            else:
                sql = str(connection.execute.await_args.args[0])
                assert '"report_test".aiops_report_source' in sql
                assert 'DROP CONSTRAINT "report""source_pk"' in sql
                assert "PRIMARY KEY (report_id, ops_run_id, source_artifact_id)" in sql
                assert "DELETE " not in sql and "TRUNCATE " not in sql and "CASCADE" not in sql
    asyncio.run(run())

def fixture(*, completed=True):
    now = datetime(2026, 10, 30, 3, tzinfo=UTC)
    actor, domain, target, run_id, turn_id, task_id, conversation_id = (uuid7() for _ in range(7))
    fact = TurnEvidenceFact(
        evidence_ref="artifact:source#observation", artifact_id=str(uuid7()),
        source_id="database", step_id="a1", tool_id="db.fixture_observation",
        measurement_semantics="CURRENT_ACTIVITY", presentation_kind="TABLE",
        captured_at=now.isoformat(), columns=({"name": "sessions"},), rows=((3,),), row_count=1,
    )
    record = freeze_investigation(
        draft=DiagnosisAnswerDraft(claims=({
            "claim_id": "c1", "kind": "FACT", "subject": "当前数据库",
            "statement": "本次活动采样返回三条会话记录。",
            "level": "OBSERVED", "applicable_time": now.isoformat(),
            "evidence_fields": ({"evidence_ref": fact.evidence_ref, "row_index": 0, "field": "sessions"},),
            "reasoning": "数据库返回的本时点计数，不外推业务健康或根因。",
        },), goal_satisfied=completed),
        evidence=(fact,),
        context=SimpleNamespace(
            run_id=run_id, task_id=task_id, target_id=target,
            plan_snapshot={"answer_context": {"question": "列出当前会话", "task_frame": {
                "task_kind": "DIRECT_QUERY", "success_criteria": ["列出当前会话"],
            }}},
        ),
    )
    payload = AIOpsTurnResult(
        status="COMPLETED" if completed else "PARTIAL",
        sufficiency_status="ANSWERABLE" if completed else "PARTIAL",
        evidence=(fact,), investigation_record=record,
        blocks=({"block_type": "ANALYSIS_MARKDOWN", "schema_version": "AIOPS_ANALYSIS_BLOCK.v1",
            "payload": {"markdown": "本次活动采样返回三条会话记录。"}, "evidence_refs": (fact.evidence_ref,)},),
    ).model_dump(mode="json")
    artifact = SimpleNamespace(artifact_id=uuid7(), ops_run_id=run_id, ops_task_id=task_id,
        schema_version="AIOPS_TURN_RESULT.v1", payload_json=payload,
        content_hash=sha256_json(payload), provenance_json={}, security_level=1, trust_level="SOURCE_VERIFIED")
    run = SimpleNamespace(ops_run_id=run_id, target_id=target, domain_id=domain, actor_id=actor,
        status="COMPLETED" if completed else "PARTIAL", trigger_type="CHAT",
        created_at=now - timedelta(minutes=2), completed_at=now - timedelta(minutes=1),
        started_at=None, final_artifact_id=artifact.artifact_id,
        plan_snapshot_json={"diagnosis": {"question_summary": "列出当前会话"}})
    task = SimpleNamespace(ops_task_id=task_id, output_artifact_id=artifact.artifact_id)
    turn = SimpleNamespace(turn_id=turn_id, turn_no=1, status=run.status)
    conversation = SimpleNamespace(conversation_id=conversation_id, title="生产库调查", created_by=actor)
    state = SimpleNamespace(current=None, closure=None, artifacts={artifact.artifact_id: artifact})

    async def add_artifact(entity):
        entity.artifact_id = entity.artifact_id or uuid7()
        state.artifacts[entity.artifact_id] = entity
        return entity

    async def publish_report(entity):
        state.current = entity
        return entity

    uow = SimpleNamespace(
        conversations=SimpleNamespace(get_conversation=AsyncMock(return_value=conversation)),
        turns=SimpleNamespace(list_all_turns=AsyncMock(return_value=[turn]),
            get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=run_id))),
        runs=SimpleNamespace(get_run_scoped=AsyncMock(return_value=run),
            get_artifact=AsyncMock(side_effect=lambda *, artifact_id: state.artifacts.get(artifact_id)),
            list_tasks=AsyncMock(return_value=[task]), database_now=AsyncMock(return_value=now),
            get_artifact_by_key=AsyncMock(side_effect=lambda **_: state.closure),
            add_artifact=AsyncMock(side_effect=add_artifact), append_event=AsyncMock()),
        inspections=SimpleNamespace(
            get_current_report_for_run_template=AsyncMock(side_effect=lambda **_: state.current),
            get_current_report_for_conversation_template=AsyncMock(side_effect=lambda **_: state.current),
            publish_report=AsyncMock(side_effect=publish_report), add_report_sources=AsyncMock()),
        outbox=SimpleNamespace(add=AsyncMock()), commit=AsyncMock(),
        platform_notifications=SimpleNamespace(emit_report_ready=AsyncMock()),
    )
    manager = AsyncMock()
    manager.__aenter__.return_value = uow
    service = AIOpsRuntimeService(uow_factory=lambda: manager, blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
    return SimpleNamespace(**locals())


async def generate(f, *, conversation=False):
    kwargs = dict(domain_id=f.domain, actor_id=f.actor,
        template=resolve_system_template("system:diagnosis.standard"), trace_id="report-regression")
    if conversation:
        return await f.service.generate_conversation_report(conversation_id=f.conversation_id, **kwargs)
    return await f.service.generate_user_report(ops_run_id=f.run_id, period_kind="AD_HOC", **kwargs)


@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("completed", [False, True])
def test_report_completion_is_independent_of_root_cause(conversation, completed):
    async def run():
        f = fixture(completed=completed)
        receipt = await generate(f, conversation=conversation)
        assert receipt.status == ("READY" if completed else "PARTIAL")
        content = f.state.artifacts[f.state.current.content_artifact_id].payload_json
        assert content["scope"]["root_cause_grade"] == "INCONCLUSIVE"
        assert not content["summary"].startswith("根因等级：")
        goals = content["scope"]["investigation_goals"]
        assert goals[0]["record_id"] == f.record.record_id
        assert goals[0]["goals"][0]["status"] == ("SATISFIED" if completed else "PARTIAL")
        assert content["provenance"]["report_input_hash"]
    asyncio.run(run())


@pytest.mark.parametrize("conversation", [False, True])
def test_exact_same_report_inputs_are_idempotent_but_changed_result_is_not(conversation):
    async def run():
        f = fixture()
        first = await generate(f, conversation=conversation)
        replay = await generate(f, conversation=conversation)
        assert replay.report_id == first.report_id
        assert f.uow.inspections.publish_report.await_count == 1
        changed = SimpleNamespace(**vars(f.artifact))
        changed.artifact_id = uuid7()
        f.state.artifacts[changed.artifact_id] = changed
        f.run.final_artifact_id = changed.artifact_id
        f.task.output_artifact_id = changed.artifact_id
        second = await generate(f, conversation=conversation)
        assert second.report_id != first.report_id
        assert second.report_version == 2
        assert f.uow.inspections.publish_report.await_count == 2
    asyncio.run(run())


def test_closure_after_first_report_creates_new_version_without_adding_turn():
    async def run():
        f = fixture()
        first = await generate(f, conversation=True)
        closure_payload = {
            "conversation_id": str(f.conversation_id), "source_turn_ids": [str(f.turn_id)],
            "profile": {"problem_signature": {"normalized_symptoms": ["业务曾有等待"]},
                "root_cause": {"summary": "人工复盘，不代替数据库证据"},
                "actions": [{"summary": "业务方已调整调用频率"}],
                "verification": {"method": "USER_REPORTED", "result": "RESOLVED"}},
        }
        closure = SimpleNamespace(artifact_id=uuid7(), ops_run_id=f.run_id, ops_task_id=f.task_id,
            schema_version="DIAGNOSIS_CLOSURE.v1", payload_json=closure_payload,
            content_hash=sha256_json(closure_payload), security_level=1)
        f.state.closure = closure
        f.state.artifacts[closure.artifact_id] = closure
        second = await generate(f, conversation=True)
        assert second.report_id != first.report_id
        assert second.report_version == 2
        content = f.state.artifacts[f.state.current.content_artifact_id].payload_json
        assert content["scope"]["verification"]["method"] == "USER_REPORTED"
        sources = f.uow.inspections.add_report_sources.await_args.args[0]
        assert {row.source_artifact_id for row in sources} == {f.artifact.artifact_id, closure.artifact_id}
        from tests.support.aiops_report_sources import assert_report_sources_persist
        assert_report_sources_persist(sources)
        actions = next(section["items"] for section in report_presentation(
            payload=content, template=resolve_system_template("system:diagnosis.standard"))["sections"]
            if section["kind"] == "ACTIONS")
        assert any("业务方已调整调用频率" in item and "人工" in item for item in actions)
        assert any("人工" in item and "恢复" in item for item in actions)
        assert not any("没有已记录" in item for item in actions)
        replay = await generate(f, conversation=True)
        assert replay.report_id == second.report_id
    asyncio.run(run())


def test_manual_actions_are_not_recommendations_or_platform_verified_results():
    source = normalize_report_source(schema_version="DIAGNOSIS_CLOSURE.v1", source_kind="CHAT", payload={
        "profile": {"actions": [{"summary": "调整调用频率"}],
            "verification": {"method": "USER_REPORTED", "result": "RESOLVED"}},
    })
    assert not source["solution"]
    content = {"scope": {"executed_actions": source["executed_actions"],
        "verification": source["verification"]}}
    presentation = report_presentation(payload=content, template=resolve_system_template("system:diagnosis.standard"))
    actions = next(section["items"] for section in presentation["sections"] if section["kind"] == "ACTIONS")
    assert any("人工" in item and "调整调用频率" in item for item in actions)
    assert not any("已验证恢复" in item for item in actions)
    assert render_pdf(presentation).startswith(b"%PDF-")


@pytest.mark.parametrize("method,result", [("ACTION_POSTCONDITION", "INCONCLUSIVE"), ("MATCHED_CONTROL", "IMPROVED")])
def test_report_keeps_direct_action_and_measured_business_effect_separate(method, result):
    presentation = report_presentation(template=resolve_system_template("system:diagnosis.standard"), payload={
        "scope": {"verification": {"method": method, "result": result, "direct_effect": "ACHIEVED",
            "measured_effect": "INCONCLUSIVE" if method == "ACTION_POSTCONDITION" else "IMPROVED",
            "symptom_recovered": None, "guardrails_passed": None, "attribution_strength": "UNKNOWN"}},
    })
    actions = next(section["items"] for section in presentation["sections"] if section["kind"] == "ACTIONS")
    assert any("动作" in item and "达到" in item for item in actions)
    assert any("业务效果" in item for item in actions)
    assert not any("没有已记录" in item for item in actions)
    if method == "ACTION_POSTCONDITION":
        assert any("业务效果" in item and "无法判断" in item for item in actions)
        assert not any("业务已恢复" in item for item in actions)

@pytest.mark.parametrize("conversation", [False, True])
def test_same_source_different_template_content_creates_new_report(conversation):
    from dataclasses import replace

    async def run():
        f = fixture()
        first = await generate(f, conversation=conversation)
        original = resolve_system_template("system:diagnosis.standard")
        changed = replace(original, definition={**original.definition, "title": "新增验收说明"})
        kwargs = dict(domain_id=f.domain, actor_id=f.actor, template=changed, trace_id="template-correction")
        if conversation:
            second = await f.service.generate_conversation_report(conversation_id=f.conversation_id, **kwargs)
        else:
            second = await f.service.generate_user_report(ops_run_id=f.run_id, period_kind="AD_HOC", **kwargs)
        assert second.report_id != first.report_id
        assert second.report_version == 2
    asyncio.run(run())


@pytest.mark.parametrize("conversation", [False, True])
def test_current_report_reuse_rechecks_raw_source_authorization(conversation, monkeypatch):
    from aiops_agent.application.errors import AIOpsApplicationError
    from aiops_agent.application.diagnosis import source_validation

    async def run():
        f = fixture()
        await generate(f, conversation=conversation)
        monkeypatch.setattr(source_validation, "validate_investigation_sources", AsyncMock(return_value=False))
        with pytest.raises(AIOpsApplicationError):
            await generate(f, conversation=conversation)
        assert f.uow.inspections.publish_report.await_count == 1
    asyncio.run(run())

@pytest.mark.parametrize("conversation", [False, True])
def test_optional_source_gap_is_visible_but_does_not_override_satisfied_goal(conversation):
    async def run():
        f = fixture()
        f.artifact.payload_json["evidence_gaps"] = [{"source_id": "monitoring", "code": "SOURCE_NO_DATA",
            "detail": "本任务所需的数据库计数已取得；独立监控背景暂缺采样。"}]
        f.artifact.content_hash = sha256_json(f.artifact.payload_json)
        receipt = await generate(f, conversation=conversation)
        assert receipt.status == "READY"
        content = f.state.artifacts[f.state.current.content_artifact_id].payload_json
        assert any(gap["code"] == "SOURCE_NO_DATA" for gap in content["gaps"])
        assert content["scope"]["investigation_goals"][0]["goals"][0]["status"] == "SATISFIED"
    asyncio.run(run())


@pytest.mark.parametrize("conversation", [False, True])
def test_unversioned_old_report_inputs_are_not_silently_reused(conversation):
    async def run():
        f = fixture()
        first = await generate(f, conversation=conversation)
        previous = f.state.artifacts[f.state.current.content_artifact_id]
        previous.payload_json["provenance"].pop("report_input_hash")
        second = await generate(f, conversation=conversation)
        assert second.report_id != first.report_id
        assert second.report_version == 2
    asyncio.run(run())


def test_frozen_report_reexport_preserves_goals_manual_actions_and_diagnosis():
    source = normalize_report_source(schema_version="REPORT_CONTENT.v1", source_kind="CHAT", payload={
        "status": "READY", "summary": "摘要不能替代诊断正文", "facts": [{"summary": "已核验事实"}],
        "scope": {"diagnosis_rationale": "完整诊断与证据边界", "root_cause_grade": "INCONCLUSIVE",
            "investigation_goals": [{"record_id": "record", "goals": [{"goal_id": "g1", "status": "SATISFIED"}]}],
            "executed_actions": ["业务方调整调用"], "action_source": "USER_PROVIDED",
            "verification": {"method": "USER_REPORTED", "result": "IMPROVED"}},
    })
    assert source["diagnosis_rationale"] == "完整诊断与证据边界"
    assert source["investigation_goals"][0]["goals"][0]["status"] == "SATISFIED"
    assert source["executed_actions"] == ("业务方调整调用",)
    assert source["action_source"] == "USER_PROVIDED"
    assert source["verification"] == {"method": "USER_REPORTED", "result": "IMPROVED"}

@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("subject_type,prefix", [("POSTGRESQL_QUERY_ID", "postgresql"), ("MYSQL_DIGEST", "mysql")])
def test_sql_specialized_template_is_resolved_before_idempotency_check(conversation, subject_type, prefix):
    async def run():
        f = fixture()
        f.run.plan_snapshot_json["answer_context"] = {"task_frame": {"subject_ref": {"subject_type": subject_type}}}
        first = await generate(f, conversation=conversation)
        second = await generate(f, conversation=conversation)
        assert first.template_id == f"system:{prefix}.sql_healthcheck"
        assert f.state.artifacts[f.state.current.content_artifact_id].payload_json["report_type"] == f"{prefix.upper()}_SQL_HEALTHCHECK"
        assert second.report_id == first.report_id
        assert f.uow.inspections.publish_report.await_count == 1
    asyncio.run(run())
