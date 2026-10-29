"""Oracle 取证、模型期限和终态详情 HTTP 序列化的跨项目回归。"""

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from aiops_agent.adapters.model_serving import AIOpsModelError, AIOpsStructuredModelClient
from aiops_agent.api import conversations
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.config import AIOpsDiagnosisConfig
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink, AIOpsOutboxDispatcher
from platform_core.contracts.aiops.investigation import EvidenceSourceStrategy
from platform_core.identity import uuid7
from tests.unit.aiops_agent import test_aiops_turn_application as support
from tests.unit.aiops_agent.test_aiops_turn_retry import _RetryOutbox


@pytest.mark.parametrize("strategy", list(EvidenceSourceStrategy))
@pytest.mark.parametrize("status", ["COMPLETED", "PARTIAL", "FAILED", "CANCELLED"])
def test_terminal_turn_http_details_accepts_every_domain_strategy(strategy, status):
    uow = support._Uow()
    service, receipt = asyncio.run(support.ConversationTurnApplicationTest._start(None, uow))
    turn = uow.turns.turns[0]
    turn.status = status
    turn.completed_at = datetime.now(UTC)
    turn.current_plan_artifact_id = uuid7()
    turn.task_frame_artifact_id = uuid7()
    uow.runs.artifacts.extend((
        SimpleNamespace(artifact_id=turn.current_plan_artifact_id, payload_json={"revision_no": 1, "actions": []}),
        SimpleNamespace(artifact_id=turn.task_frame_artifact_id, payload_json={"evidence_source_strategy": strategy.value}),
    ))
    app = FastAPI()
    app.include_router(conversations.router)
    app.state.conversation_turn_service = service
    app.dependency_overrides[conversations.get_aiops_auth_context] = lambda: object()
    with patch.object(conversations, "_scope", return_value=(turn.domain_id, turn.created_by)):
        with TestClient(app) as client:
            response = client.get(f"/internal/v1/aiops/conversations/{receipt['conversation_id']}/turns/{receipt['turn_id']}")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == status and result["completed_at"]
    assert result["investigation_plan"]["task_frame"]["evidence_source_strategy"] == strategy.value


def test_oracle_cursor_template_qualifies_shared_column_names():
    resolved = DiagnosticRegistry.load().resolve(
        tool_id="db.sql.cursor_details", tool_version="1.0.0", db_type="ORACLE", db_version="19.3",
        capabilities={"dynamic_performance_views"}, entitlements=set(),
    )
    assert "FROM gv$sql cursor_stats" in resolved.sql
    assert "cursor_stats.con_id AS container_id" in resolved.sql
    assert "database_identity.dbid" in resolved.sql
    assert "WHERE cursor_stats.sql_id = :sql_id" in resolved.sql
    assert hashlib.sha256(resolved.sql.encode()).hexdigest() == resolved.definition.template_sha256
    # 对联表的每个游标字段都必须限定来源，而非仅修复当前的 con_id。
    import re
    expressions = resolved.sql.split("SELECT database_identity.dbid,", 1)[1].split("FROM gv$sql", 1)[0]
    assert all(expression.strip().startswith("cursor_stats.") for expression in expressions.split(","))
    assert not re.search(r"(?<![.:\w])(?:con_id|inst_id|child_number|last_active_time)\b", resolved.sql)


class _Output(BaseModel):
    answer: str


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("expired", [False, True])
def test_http_timeout_is_deadline_only_when_budget_really_expires(streaming, expired):
    now = datetime.now(UTC)
    deadline = now + timedelta(seconds=30)
    clock = iter((now, deadline if expired else now + timedelta(seconds=1)))
    session = SimpleNamespace(post=lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))
    client = AIOpsStructuredModelClient(
        base_url="http://model.invalid", audience="model-serving", caller_service="aiops-agent",
        timeout_seconds=90, session=session,
    )
    prompt = "仅依据证据"
    kwargs = dict(purpose="diagnosis.test", model_snapshot={
        "technical_name": "test", "served_model_name": "test", "model_id": str(uuid7()), "tenant_id": str(uuid7()),
    }, prompt_ref={"content": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
        input_payload={}, deadline=deadline, idempotency_key="test-deadline")
    async def invoke():
        if streaming:
            return [item async for item in client.stream_text(**kwargs)]
        return await client.generate_structured(output_model=_Output, **kwargs)
    with patch("aiops_agent.adapters.model_serving.build_internal_auth_headers", return_value={}), patch(
        "aiops_agent.adapters.model_serving.datetime", SimpleNamespace(now=lambda _: next(clock)),
    ):
        with pytest.raises(AIOpsModelError) as error:
            asyncio.run(invoke())
    assert error.value.code == ("MODEL_DEADLINE_EXCEEDED" if expired else "MODEL_SERVICE_UNAVAILABLE")
    assert error.value.retryable is not expired


@pytest.mark.parametrize("transient_first", [False, True])
def test_replan_deadline_falls_back_without_outbox_backoff(transient_first):
    async def invoke():
        uow = support._Uow()
        now = datetime.now(UTC)
        outbox = _RetryOutbox({"revision_no": 2}, now)
        outbox.message.event_type = "aiops.turn.replanning_requested"
        uow.outbox = outbox
        uow.runs.database_now = AsyncMock(return_value=now)
        errors = [AIOpsModelError("MODEL_DEADLINE_EXCEEDED")]
        if transient_first:
            errors.insert(0, AIOpsModelError("MODEL_SERVICE_UNAVAILABLE"))
        planning = SimpleNamespace(
            execute_replan=AsyncMock(side_effect=errors), fall_back_from_replan=AsyncMock(),
        )
        sink = AIOpsDomainOutboxSink(runtime_service=object(), fallback=support._NoopSink(), turn_planning_service=planning)
        dispatcher = AIOpsOutboxDispatcher(uow_factory=lambda: uow, sink=sink, dispatcher_id="test", lease_seconds=30, interval_seconds=1)
        await dispatcher.run_once()
        if transient_first:
            assert outbox.message.status == "RETRY_WAIT"
            planning.fall_back_from_replan.assert_not_awaited()
            uow.runs.database_now.return_value = outbox.message.available_at + timedelta(milliseconds=1)
            await dispatcher.run_once()
        assert outbox.message.status == "PUBLISHED"
        assert len(outbox.releases) == int(transient_first)
        planning.fall_back_from_replan.assert_awaited_once_with(
            {"revision_no": 2}, error_code="MODEL_DEADLINE_EXCEEDED",
        )
    asyncio.run(invoke())

@pytest.mark.parametrize("expired", [False, True])
def test_service_checks_work_budget_again_after_transient_model_failure(expired):
    async def invoke():
        now = datetime.now(UTC)
        clock = iter((now, now + timedelta(seconds=11 if expired else 1)))
        service = object.__new__(TurnPlanningService)
        service._diagnosis_config = AIOpsDiagnosisConfig(answer_reserve_seconds=20)
        service._prepare = AsyncMock(return_value=SimpleNamespace(deadline=now + timedelta(seconds=30)))
        service._execute_replan_once = AsyncMock(side_effect=AIOpsModelError("MODEL_SERVICE_UNAVAILABLE"))
        service.fall_back_from_replan = AsyncMock(return_value={"status": "ANSWERING"})
        payload = {"revision_no": 2}
        with patch("aiops_agent.application.investigation.service.datetime", SimpleNamespace(now=lambda _: next(clock))):
            if expired:
                assert await service.execute_replan(payload) == {"status": "ANSWERING"}
                service.fall_back_from_replan.assert_awaited_once_with(payload, error_code="MODEL_DEADLINE_EXCEEDED")
            else:
                with pytest.raises(AIOpsModelError, match="MODEL_SERVICE_UNAVAILABLE"):
                    await service.execute_replan(payload)
                service.fall_back_from_replan.assert_not_awaited()
    asyncio.run(invoke())



def test_replan_retry_never_uses_answer_reserve_to_start_another_model_call():
    async def invoke():
        service = object.__new__(TurnPlanningService)
        service._diagnosis_config = AIOpsDiagnosisConfig(answer_reserve_seconds=20)
        service._prepare = AsyncMock(return_value=SimpleNamespace(deadline=datetime.now(UTC) + timedelta(seconds=19)))
        service.fall_back_from_replan = AsyncMock(return_value={"status": "ANSWERING"})
        service._load_replan_inputs = AsyncMock(side_effect=AssertionError("不应继续规划"))
        payload = {"revision_no": 2}
        assert await service.execute_replan(payload) == {"status": "ANSWERING"}
        service.fall_back_from_replan.assert_awaited_once_with(payload, error_code="MODEL_DEADLINE_EXCEEDED")
        service._load_replan_inputs.assert_not_awaited()
    asyncio.run(invoke())
