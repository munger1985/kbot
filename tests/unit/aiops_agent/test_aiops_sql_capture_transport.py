"""控制面真实路由和专用scope；替身AuthContext不冒充跨服务JWT验收。"""

from datetime import timedelta
import ast
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aiops_agent.api.executions.routes import event_router
from platform_core.contracts import ServiceIdentity
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureExchange
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_capture_execution import execution_store, claimed, report
from tests.unit.aiops_agent.test_aiops_sql_plan_material import SECRET


def http_fixture():
    store = execution_store()
    claimed(store)
    app = FastAPI()
    app.include_router(event_router)
    app.state.sql_plan_capture_execution_service = store.execution
    app.state.scopes = ()

    @app.middleware("http")
    async def scoped(request, call_next):
        request.state.service_identity = ServiceIdentity(issuer="test", subject="test-executor", audience="test-aiops",
            scopes=app.state.scopes, issued_at=store.now, expires_at=store.now + timedelta(minutes=5), token_id=uuid7())
        return await call_next(request)

    return SimpleNamespace(store=store, app=app, client=TestClient(app), base="/internal/v1/aiops/sql-plan-captures:")


@pytest.mark.parametrize("scopes", [("aiops.task",), ("aiops.credentials.issue",), ("aiops.run", "aiops.approve"),
    ("aiops.plan-capture.report",)])
def test_exchange_denies_every_non_exchange_scope_without_decrypting(scopes):
    f = http_fixture()
    f.app.state.scopes = scopes
    reply = f.client.post(f.base + "exchange", json={"grant": "s" * 64})
    assert reply.status_code == 403 and SECRET not in reply.text
    f.store.secret_reader.read.assert_not_awaited()
    assert not f.store.leases


def test_only_dedicated_exchange_returns_secret_once_and_report_is_independent_no_store():
    f = http_fixture()
    f.app.state.scopes = ("aiops.plan-capture.exchange",)
    reply = f.client.post(f.base + "exchange", json={"grant": "s" * 64})
    assert reply.status_code == 200 and reply.headers["cache-control"] == "no-store"
    exchanged = SqlPlanCaptureExchange.model_validate(reply.json())
    assert exchanged.password.get_secret_value() == SECRET and SECRET in exchanged.protected_document.get_secret_value()
    command = report(f.store, exchanged.resource_lease)
    denied = f.client.post(f.base + "report", json=command.model_dump(mode="json"))
    assert denied.status_code == 403 and SECRET not in denied.text
    duplicate = f.client.post(f.base + "exchange", json={"grant": "s" * 64})
    assert duplicate.status_code == 403 and duplicate.headers["cache-control"] == "no-store"
    assert duplicate.json()["detail"]["code"] == "PLAN_CAPTURE_ALREADY_CONSUMED" and SECRET not in duplicate.text
    f.app.state.scopes = ("aiops.plan-capture.report",)
    reported = f.client.post(f.base + "report", json=command.model_dump(mode="json"))
    assert reported.status_code == 200 and reported.headers["cache-control"] == "no-store"
    assert reported.json()["status"] == "SUCCEEDED" and SECRET not in reported.text
    invalid = command.model_dump(mode="json")
    invalid["resource_lease"]["release_token"] = "z" * 48
    reply = f.client.post(f.base + "report", json=invalid)
    assert reply.status_code == 403 and reply.headers["cache-control"] == "no-store"
    assert SECRET not in reply.text


def test_real_bootstrap_assigns_exchange_and_report_only_to_executor_and_routes_stay_internal():
    from aiops_agent.bootstrap.openapi import create_internal_contract_app
    from main_api.openapi_contracts import create_aiops_public_contract_app
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / "services/aiops_agent/src/aiops_agent/bootstrap/api.py").read_text())
    boundaries = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "create_scoped_internal_auth_middleware"
        for keyword in node.keywords if keyword.arg == "allowed_callers"]
    assert len(boundaries) == 1
    callers = boundaries[0]
    capture_scopes = {"aiops.plan-capture.exchange", "aiops.plan-capture.report"}
    seen = False
    for key, value in zip(callers.keys, callers.values, strict=True):
        scopes = set(ast.literal_eval(value.args[0]))
        if key.value.endswith("aiops-db-executor"):
            assert capture_scopes <= scopes
            seen = True
        else:
            assert not capture_scopes & scopes
    assert seen
    expected = {"/internal/v1/aiops/sql-plan-captures:exchange", "/internal/v1/aiops/sql-plan-captures:report"}
    assert expected <= set(create_internal_contract_app().openapi()["paths"])
    public = create_aiops_public_contract_app().openapi()["paths"]
    assert not any(path.endswith(("sql-plan-captures:exchange", "sql-plan-captures:report")) for path in public)
