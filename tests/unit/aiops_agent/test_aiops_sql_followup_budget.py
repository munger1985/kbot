"""选行先取证、非最终阶段让出预算，以及截止来源保真的跨引擎回归。"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.application.diagnosis.deadline_result import deadline_result
from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
from aiops_agent.application.investigation import TurnPlanningService
from aiops_agent.application.investigation.selected_sql import selected_sql_investigation
from aiops_agent.config import AIOpsDiagnosisConfig
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.tools import InvestigationTaskCompiler, ToolExecutionSnapshotBuilder
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.identity import uuid7
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from tests.unit.aiops_agent import test_aiops_dba_playbook_planning as planning
from tests.unit.aiops_agent import test_aiops_dynamic_query as dynamic
from tests.unit.aiops_agent.test_aiops_turn_answer import _context, _tool_artifact, _TestPrompts
from tests.unit.aiops_agent.test_aiops_sql_diagnosis import task_frame, observation


SUBJECTS = (
    ("ORACLE", {"subject_type": "ORACLE_SQL_ID", "sql_id": "36apy6dncjggq"}),
    ("MYSQL", {"subject_type": "MYSQL_DIGEST", "digest": "A" * 64, "schema_name": None}),
    ("POSTGRESQL", {"subject_type": "POSTGRESQL_QUERY_ID", "query_id": "1285382359565720997", "database_oid": 16510, "user_oid": 16384, "top_level": True}),
)


@pytest.mark.parametrize("engine,subject", SUBJECTS)
def test_selected_sql_is_compiled_before_any_model_call_and_keeps_open_capabilities(engine, subject):
    async def exercise():
        uow = planning._PlanningUow()
        uow.target.db_type = engine
        uow.target.version_code = {"ORACLE": "19c", "MYSQL": "8.0.36", "POSTGRESQL": "16.15"}[engine]
        question = "继续深入分析所选 SQL，沿用上一轮数据库资源和证据"
        uow.message.payload_json = {"text": question, "content": [{"content_type": "TEXT", "text": question}], "sql_selection": {"subject_ref": subject}}
        diagnostics = DiagnosticRegistry.load()
        registry = PlaybookRegistry.load(allowed_tools=frozenset((item.definition.tool_id, item.definition.version) for item in diagnostics.tools))
        reasoner = SimpleNamespace(freeze_prompts=AsyncMock(return_value={}), plan=AsyncMock(side_effect=AssertionError("选行首轮不应重新规划")), plan_compact=AsyncMock(side_effect=AssertionError("选行不应重新分类")))
        service = TurnPlanningService(
            uow_factory=lambda: uow, investigation_reasoner=reasoner, playbook_registry=registry,
            task_compiler=InvestigationTaskCompiler(registry),
            tool_snapshot_builder=ToolExecutionSnapshotBuilder(playbook_registry=registry, diagnostic_registry=diagnostics),
            agent_catalog=planning._AgentCatalog(),
        )
        service._load_prior_sql_evidence = AsyncMock(return_value=None)
        await service.execute({"domain_id": str(uow.run.domain_id), "turn_id": str(uow.turn.turn_id)})
        snapshot = uow.run.plan_snapshot_json
        assert snapshot["planning_route"]["mode"] == "SELECTED_SQL_EVIDENCE_FIRST"
        assert snapshot["answer_context"]["task_frame"]["subject_ref"] == subject
        actions = snapshot["answer_context"]["investigation_plan"]["actions"]
        assert actions[0]["tool_id"] == "db.instance.identity"
        assert len(actions) == 2
        assert all("readonly_query" not in action["tool_id"] for action in actions)
        assert uow.tasks and uow.revisions
        assert snapshot["answer_context"]["task_frame"]["success_criteria"]
        reasoner.plan.assert_not_awaited()
        reasoner.plan_compact.assert_not_awaited()
        # 后续重规划仍从真实完整能力目录发现，而不是保存一个封闭的工具白名单。
        capabilities = DbaCapabilitySnapshot.model_validate(snapshot["capability_snapshot"])
        assert "db.sql.object_candidates" in {tool["tool_id"] for tool in service._tool_snapshot_builder.discover_tools(capabilities)}
    asyncio.run(exercise())


def test_selected_oracle_cached_plan_requires_complete_cursor_identity():
    subject = {**SUBJECTS[0][1], "dbid": 123, "inst_id": 1, "container_id": 3, "child_number": 0, "plan_hash_value": 456}
    result = selected_sql_investigation(question="分析这个游标", target_context={"db_type": "ORACLE", "sql_selection": {"subject_ref": subject}}, tools=({"tool_id": "db.sql.cursor_details"}, {"tool_id": "db.oracle.statement.plan"}))
    assert [action.tool_id for action in result.plan.actions] == ["db.sql.cursor_details", "db.oracle.statement.plan"]
    assert result.plan.actions[1].input == {key: value for key, value in subject.items() if key != "subject_type"}
    with pytest.raises(ValueError):
        selected_sql_investigation(question="分析", target_context={"db_type": "MYSQL", "sql_selection": {"subject_ref": subject}}, tools=())


def test_policy_failure_does_not_replan_independent_legal_actions_or_their_dependencies():
    async def exercise():
        output = dynamic.DynamicQueryPlanningTest()._investigation(sql="SELECT snap_id FROM dba_hist_snapshot")
        valid = output.plan.actions[0].model_copy(update={"action_id": "a2", "input": {"sql": "SELECT sid FROM v$session", "parameters": {}}})
        dependent = valid.model_copy(update={"action_id": "a3", "depends_on": ("a1",)})
        output = output.model_copy(update={"plan": output.plan.model_copy(update={"actions": (*output.plan.actions, valid, dependent)})})
        reasoner = SimpleNamespace(repair_policy_invalid_plan=AsyncMock(side_effect=AssertionError("不应阻塞独立动作")))
        service = object.__new__(TurnPlanningService)
        service._investigation_reasoner = reasoner
        _, result, frozen, _ = await service._prepare_queries_with_repair(
            context=dynamic.DynamicQueryPlanningRepairTest._context(), planned=StructuredModelResult(output=output, receipt=None),
            available_tools=({"tool_id": "db.oracle.readonly_query"},), available_playbooks=(), model_snapshot={}, revision_no=1,
        )
        assert [action.action_id for action in result.plan.actions] == ["a2"]
        assert [action["action_id"] for action in frozen] == ["a2"]
        assert "ORACLE_DIAGNOSTICS_PACK_REQUIRED" in str(result.task_frame.constraints)
        assert "a3" in str(result.task_frame.constraints)
        reasoner.repair_policy_invalid_plan.assert_not_awaited()
    asyncio.run(exercise())


def test_required_policy_repair_receives_work_deadline_not_full_run_deadline():
    async def exercise():
        rejected = dynamic.DynamicQueryPlanningTest()._investigation(sql="SELECT custom_function(sid) AS result FROM v$session")
        repaired = dynamic.DynamicQueryPlanningTest()._investigation(sql="SELECT sid FROM v$session WHERE status = :status")
        reasoner = SimpleNamespace(repair_policy_invalid_plan=AsyncMock(return_value=StructuredModelResult(output=repaired, receipt=None)))
        service = object.__new__(TurnPlanningService)
        service._investigation_reasoner = reasoner
        service._diagnosis_config = AIOpsDiagnosisConfig(answer_reserve_seconds=30)
        context = dynamic.DynamicQueryPlanningRepairTest._context()
        context.deadline = datetime.now(UTC) + timedelta(seconds=50)
        await service._prepare_queries_with_repair(
            context=context, planned=StructuredModelResult(output=rejected, receipt=None),
            available_tools=({"tool_id": "db.oracle.readonly_query"},), available_playbooks=(), model_snapshot={}, revision_no=1,
        )
        assert reasoner.repair_policy_invalid_plan.await_args.kwargs["deadline"] == context.deadline - timedelta(seconds=30)
    asyncio.run(exercise())


@pytest.mark.parametrize("remaining,reserve,called", [(50, 30, True), (29, 30, False), (19, 5, True)])
def test_assessment_respects_configured_answer_reserve_even_on_model_failure(remaining, reserve, called):
    async def exercise():
        context = replace(_context(artifacts=(_tool_artifact(semantics="CUMULATIVE_SINCE_LOAD"),)), deadline_at=(datetime.now(UTC) + timedelta(seconds=remaining)).isoformat())
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=AIOpsModelError("MODEL_DEADLINE_EXCEEDED")))
        handler = DbaEvidenceAssessmentHandler(model_client=model, prompts=_TestPrompts(), diagnosis_config=AIOpsDiagnosisConfig(answer_reserve_seconds=reserve))
        result = await handler.execute(context)
        assert result.evidence
        assert bool(model.generate_structured.await_count) == called
        if called:
            assert model.generate_structured.await_args.kwargs["deadline"] == datetime.fromisoformat(context.deadline_at) - timedelta(seconds=reserve)
    asyncio.run(exercise())


def test_deadline_does_not_trust_sufficiency_copy_and_preserves_formal_raw_projection():
    raw = _tool_artifact(semantics="CUMULATIVE_SINCE_LOAD")
    artifact = SimpleNamespace(artifact_id=uuid7(), schema_version=raw["schema_version"], payload_json=raw["payload"], trust_level="SOURCE_VERIFIED")
    source = DbaToolResult.model_validate(raw["payload"])
    expected = database_evidence_fact(result=source, outcome=source.tool_outcomes[0], artifact_id=artifact.artifact_id)
    result = deadline_result(artifacts=[artifact], task_frame={}, database_type="ORACLE", projected_ids={str(artifact.artifact_id)})
    assert result.evidence == (expected,)
    assert deadline_result(artifacts=[artifact], task_frame={}, database_type="ORACLE", projected_ids=set()) is None
    copy = SimpleNamespace(artifact_id=uuid7(), schema_version="DBA_SUFFICIENCY.v1", trust_level="SOURCE_VERIFIED", payload_json=DbaSufficiencyAssessment(status="ANSWERABLE", evidence=(expected,)).model_dump(mode="json"))
    assert deadline_result(artifacts=[copy], task_frame={}, database_type="ORACLE", projected_ids={str(copy.artifact_id), str(expected.artifact_id)}) is None


def test_deadline_prior_requires_current_source_revalidation_and_stays_historical():
    fact = observation()
    artifact = SimpleNamespace(artifact_id=uuid7(), schema_version="PRIOR_TURN_SQL_EVIDENCE.v1", trust_level="SOURCE_VERIFIED", payload_json={"facts": [fact.model_dump(mode="json")]})
    args = dict(artifacts=[artifact], task_frame=task_frame(), database_type="POSTGRESQL", projected_ids={str(artifact.artifact_id)})
    assert deadline_result(**args) is None
    result = deadline_result(**args, validated_prior_ids=frozenset({str(artifact.artifact_id)}))
    assert result.status == "PARTIAL"
    assert result.evidence[0].source_id == "prior_turn.sql_observation"
    assert result.evidence[0].captured_at == fact.captured_at
    assert "历史观测" in str(result.evidence[0].warnings)

def test_deadline_keeps_raw_tool_failure_in_body_and_evidence_gaps():
    raw = _tool_artifact(semantics="CUMULATIVE_SINCE_LOAD")
    source = DbaToolResult.model_validate(raw["payload"])
    from aiops_agent.contracts.tool_execution import ToolOutcome
    from aiops_agent.contracts.artifacts.database import EvidenceGap
    failed = ToolOutcome(step_id="details", tool_id="db.sql.cursor_details", tool_version="1.0.0", status="GAP",
        gap=EvidenceGap(code="QUERY_COLUMN_AMBIGUOUS", tool_id="db.sql.cursor_details", detail="查询列名存在歧义"))
    source = source.model_copy(update={"source_id": "db.sql.cursor_details", "status": "PARTIAL",
                                      "tool_outcomes": (*source.tool_outcomes, failed)})
    artifact = SimpleNamespace(artifact_id=uuid7(), schema_version="DBA_TOOL_RESULT.v1",
        payload_json=source.model_dump(mode="json"), trust_level="SOURCE_VERIFIED")
    result = deadline_result(artifacts=[artifact], task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE",
        "subject_ref": {"subject_type": "ORACLE_SQL_ID", "sql_id": "abc123"}}, database_type="ORACLE",
        projected_ids={str(artifact.artifact_id)})
    assert result.status == "PARTIAL"
    assert {gap.code for gap in result.evidence_gaps} == {"QUERY_COLUMN_AMBIGUOUS", "RUN_DEADLINE_EXCEEDED"}
    assert "QUERY_COLUMN_AMBIGUOUS" in result.blocks[0].payload["markdown"]
    assert "这不表示该 SQL 不存在" in result.blocks[0].payload["markdown"]
