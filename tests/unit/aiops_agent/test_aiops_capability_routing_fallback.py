"""自然语言功能路由不得把模型猜测当作用户明确选择。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.conversation_starters import ConversationStarterCatalog
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.turn_planner import TurnPlannerService
from platform_core.contracts.aiops import AIOpsCapability, ScopeType, TurnPlanStepV1, TurnPlanV1
from platform_core.identity import uuid7


class _Prompt:
    content = "test"

    def ref(self):
        return {"prompt_id": "test", "prompt_version": "1.0.0", "prompt_sha256": "0" * 64}


class _Model:
    def __init__(self, capability_id, parameters):
        self.capability_id = capability_id
        self.parameters = parameters
        self.calls = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        payload = kwargs["input_payload"]
        receipt = SimpleNamespace(model_dump=lambda **_: {"purpose": kwargs["purpose"]})
        if kwargs["purpose"] == "aiops_capability_resolver":
            return SimpleNamespace(output={"status": "MATCHED", "capability_id": self.capability_id,
                                           "parameters": self.parameters, "confidence": 0.99, "reason": "test"},
                                   receipt=receipt)
        return SimpleNamespace(output=TurnPlanV1(
            turn_id=payload["turn_id"], revision_no=payload["revision_no"],
            requested_capabilities=(AIOpsCapability.ACTIVE_DIAGNOSIS,), requires_active_diagnosis=True,
            steps=(TurnPlanStepV1(step_id="s1", capability=AIOpsCapability.ACTIVE_DIAGNOSIS,
                                  objective="继续只读诊断", scope_type=ScopeType.RESOURCE),),
            policy_snapshot_hash=payload["policy_snapshot_hash"],
        ), receipt=receipt)


def _system(name, db_type):
    return {"business_system_id": str(uuid7()), "system_code": name, "display_name": name,
            "environment": "PROD", "resources": [{"resource_id": str(uuid7()), "resource_code": name + "_db",
                "display_name": name + "_db", "resource_type_key": "DATABASE", "topology_roles": [],
                "attributes": {}, "database": {"db_type": db_type}}]}


def _context(systems, *, question="继续深入分析上一轮 SQL", hint=None):
    return {"turn_id": str(uuid7()), "tenant_id": str(uuid7()), "planner_model_id": str(uuid7()),
            "plan_revision": 1, "policy_snapshot_hash": "a" * 64, "question": question,
            "capabilities": ["ACTIVE_DIAGNOSIS"], "systems": systems, "context_hint": hint or {},
            "recent_conversation_context": [], "conversation_starter": {}}


def _planner(model):
    return TurnPlannerService(uow_factory=None, structured_model=model,
                              prompt_registry=SimpleNamespace(resolve=AsyncMock(return_value=_Prompt())),
                              model_resolver=AsyncMock(return_value={"technical_name": "test"}))


@pytest.mark.parametrize("db_type", ["POSTGRESQL", "MYSQL"])
def test_wrong_database_capability_falls_back_and_never_executes_oracle_entry(db_type):
    context = _context([_system("MES", db_type), _system("ERP", "ORACLE")])
    model = _Model("oracle.sql.performance", {"sql_id": "1285382359565720997"})
    plan, _ = asyncio.run(_planner(model)._plan(context))
    assert plan.requires_active_diagnosis
    assert [call["purpose"] for call in model.calls] == ["aiops_capability_resolver", "aiops_turn_router"]
    assert model.calls[1]["input_payload"]["question"] == context["question"]
    assert "oracle.sql.performance" not in {item["capability_id"] for item in model.calls[0]["input_payload"]["eligible_capabilities"]}
    assert not context["conversation_starter"]


def test_invalid_model_parameters_fall_back_but_explicit_ui_remains_strict():
    context = _context([_system("OracleDB", "ORACLE")])
    model = _Model("oracle.sql.performance", {"sql_id": "1285382359565720997"})
    plan, _ = asyncio.run(_planner(model)._plan(context))
    assert plan.requires_active_diagnosis and len(model.calls) == 2
    target = SimpleNamespace(db_type="ORACLE", status="ENABLED", readonly_connection_enabled=True,
                             connectivity_status="CONNECTED")
    catalog = ConversationStarterCatalog()
    selection = SimpleNamespace(starter_id="oracle.sql.performance", catalog_version=catalog.version,
                                parameters={"sql_id": "1285382359565720997"})
    with pytest.raises(AIOpsApplicationError) as caught:
        catalog.freeze(selection=selection, target=target, capabilities={"ACTIVE_DIAGNOSIS"})
    assert caught.value.code == "OPS_VALIDATION_FAILED"


def test_valid_oracle_and_cross_database_top_sql_keep_deterministic_entry():
    oracle = _context([_system("OracleDB", "ORACLE")])
    model = _Model("oracle.sql.performance", {"sql_id": "6tjx7su0q5ttj"})
    plan, receipt = asyncio.run(_planner(model)._plan(oracle))
    assert plan.requires_active_diagnosis and len(model.calls) == 1
    assert oracle["conversation_starter"]["starter_id"] == "oracle.sql.performance"
    assert receipt.starter_id == "oracle.sql.performance"
    mixed = _context([_system("MES", "POSTGRESQL"), _system("ERP", "MYSQL")])
    model = _Model("database.sql.top-current", {"limit": 10})
    asyncio.run(_planner(model)._plan(mixed))
    assert mixed["conversation_starter"]["starter_id"] == "database.sql.top-current"
    assert len(model.calls) == 1


def test_resolver_candidates_follow_explicit_system_before_trusted_hint():
    pg = _system("MES", "POSTGRESQL")
    oracle = _system("ERP", "ORACLE")
    context = _context([pg, oracle], question="检查 MES 数据库", hint={"business_system_id": oracle["business_system_id"]})
    model = _Model("database.sql.top-current", {"limit": 10})
    asyncio.run(_planner(model)._plan(context))
    ids = {item["capability_id"] for item in model.calls[0]["input_payload"]["eligible_capabilities"]}
    assert "oracle.sql.performance" not in ids


def test_resolver_candidates_follow_trusted_postgresql_context_for_followup():
    pg = _system("MES", "POSTGRESQL")
    oracle = _system("ERP", "ORACLE")
    question = "继续深入分析 SQL 1285382359565720997，沿用上一轮数据库资源和证据。"
    context = _context([pg, oracle], question=question, hint={"business_system_id": pg["business_system_id"]})
    model = _Model("oracle.sql.performance", {"sql_id": "1285382359565720997"})
    asyncio.run(_planner(model)._plan(context))
    ids = {item["capability_id"] for item in model.calls[0]["input_payload"]["eligible_capabilities"]}
    assert "oracle.sql.performance" not in ids
    assert model.calls[1]["input_payload"]["question"] == question
