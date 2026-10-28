"""独立采集审批的生产服务、真实密文、撤回失效、身份及HTTP门禁回归。"""

import asyncio
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.sql_plan_captures import SqlPlanCaptureService, REQUEST_SCHEMA, DECISION_SCHEMA, PREFIX
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_capture import SqlPlanCaptureRequest, SqlPlanCaptureDecision
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_sql_plan_material import Store, command, SECRET, NOW


def call(coro):
    return asyncio.run(coro)


def setup(engine="POSTGRESQL", *, distinct=True, approval_enabled=True, security_level=3):
    store = Store(engine)
    store.original.security_level = security_level
    policy_id = uuid7()
    store.binding.policy_id = policy_id
    store.run.plan_snapshot_json["binding"]["policy_id"] = str(policy_id)
    store.policy.rules_json["sql_plan_capture"] = {"enabled": True, "allowed_objects": ["mes.t"],
        "require_distinct_approver": distinct, "max_statement_timeout_seconds": 5}
    field, sql = {"POSTGRESQL": ("normalized_statement", "SELECT * FROM mes.t WHERE id=$1 OR n=$2"),
        "MYSQL": ("digest_text", "SELECT * FROM mes.t WHERE id=? OR n=?"),
        "ORACLE": ("sql_text", "SELECT * FROM MES.T WHERE ID=:1 OR N=:2")}[engine]
    def update(obs):
        index = next(i for i, column in enumerate(obs["columns"]) if column["name"] == field)
        obs["rows"][0][index] = sql
    store.update_observation(update)
    material = call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    assert material.capture_preflight.status == "AWAITING_DATABASE_VALIDATION"
    service = SqlPlanCaptureService(uow_factory=lambda: store, materials=store.service,
        approval_enabled=approval_enabled, clock=lambda: store.now)
    store.capture = service
    store.material = material
    store.reviewer = replace(store.scope, actor_id=type(store.scope.actor_id)(str(uuid7())))
    return store


def request(store, **updates):
    return call(store.capture.request(scope=store.scope, run_id=store.run.ops_run_id,
        command=SqlPlanCaptureRequest(request_id=uuid7(), material_id=store.material.material_id, **updates)))


def decide(store, view, *, decision="APPROVE", scope=None, **updates):
    payload = {"request_sha256": view.request_sha256, "decision": decision,
        "acknowledge_estimate_only": decision == "APPROVE",
        "acknowledge_oracle_plan_table_write": view.oracle_plan_table_write if decision == "APPROVE" else False,
        **updates}
    return call(store.capture.decide(scope=scope or store.reviewer, run_id=store.run.ops_run_id,
        request_id=view.request_id, command=SqlPlanCaptureDecision(**payload)))


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_three_engine_real_encryption_approval_lifecycle_never_schedules_or_grants_execution(engine):
    store = setup(engine)
    view = request(store)
    assert view.state == "PENDING" and not view.execution_authorized and not view.capture_scheduled
    assert view.allowed_objects == ("mes.t",) and view.oracle_plan_table_write == (engine == "ORACLE")
    assert view.expires_at == store.now + timedelta(minutes=5)
    approved = decide(store, view)
    assert approved.state == "APPROVED" and approved.decided_by == str(store.reviewer.actor_id)
    assert not approved.execution_authorized and not approved.capture_scheduled
    listed = call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id))
    assert listed.items == (approved,) and listed.can_request and not listed.can_review
    review = call(store.capture.list(scope=store.reviewer, run_id=store.run.ops_run_id, review=True))
    assert review.items == (approved,) and review.can_review and not review.can_request
    with pytest.raises(AIOpsApplicationError):
        decide(store, view)
    wire = approved.model_dump_json() + json.dumps(store.events, default=str) + json.dumps([item.payload_json for item in store.artifacts])
    for secret in (SECRET, "private_schema", "private_role", "12345678901234567890.123456789"):
        assert secret not in wire
    assert len(store.credentials) == 2 and len(store.credentials[-1].nonce) == 12
    revoked = call(store.capture.revoke(scope=store.scope, run_id=store.run.ops_run_id, request_id=view.request_id))
    assert revoked.state == "REVOKED" and store.credentials[-1].status == "REVOKED"
    assert call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "REVOKED"


def test_distinct_reviewer_and_explicit_acks_are_required_but_rejection_does_not_authorize():
    store = setup("ORACLE")
    view = request(store)
    for scope, extra in ((store.scope, {}), (store.reviewer, {"acknowledge_estimate_only": False}),
        (store.reviewer, {"acknowledge_oracle_plan_table_write": False})):
        with pytest.raises(AIOpsApplicationError):
            decide(store, view, scope=scope, **extra)
    assert not any(item.schema_version == DECISION_SCHEMA for item in store.artifacts)
    rejected = decide(store, view, decision="REJECT")
    assert rejected.state == "REJECTED" and not rejected.execution_authorized
    assert store.credentials[-1].status == "REVOKED"


def test_same_actor_requires_explicit_policy_and_policy_change_invalidates_prior_request():
    store = setup(distinct=False)
    view = request(store)
    assert decide(store, view, scope=store.scope).state == "APPROVED"
    store.policy.rules_json["sql_plan_capture"]["require_distinct_approver"] = True
    assert call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"


@pytest.mark.parametrize("change", [
    lambda store: setattr(store.target, "row_version", 2),
    lambda store: store.target.endpoint_json.update(database="other"),
    lambda store: setattr(store.target, "diagnostic_credential_id", uuid7()),
    lambda store: setattr(store.target, "status", "DISABLED"),
    lambda store: setattr(store.run, "cancel_requested_at", store.now),
    lambda store: setattr(store.binding, "binding_id", uuid7()),
    lambda store: setattr(store.policy, "status", "INACTIVE"),
    lambda store: store.policy.rules_json.update(readonly_database_enabled=False),
    lambda store: store.policy.rules_json["sql_plan_capture"].update(allowed_objects=["other.t"]),
    lambda store: setattr(store.capture, "_approval_enabled", False),
    lambda store: setattr(store.original, "content_hash", "0" * 64),
])
def test_current_state_changes_invalidate_approval_and_owner_can_still_revoke(change):
    store = setup()
    view = request(store)
    decide(store, view)
    change(store)
    listed = call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id))
    assert listed.items[0].state == "UNAVAILABLE"
    with pytest.raises(AIOpsApplicationError):
        decide(store, view)
    assert call(store.capture.revoke(scope=store.scope, run_id=store.run.ops_run_id, request_id=view.request_id)).state == "REVOKED"


@pytest.mark.parametrize("when", ["EXPIRE_REQUEST", "REVOKE_MATERIAL", "EXPIRE_MATERIAL"])
def test_request_cannot_survive_its_or_material_lifetime(when):
    store = setup()
    view = request(store)
    if when == "REVOKE_MATERIAL":
        call(store.service.revoke(scope=store.scope, run_id=store.run.ops_run_id, material_id=store.material.material_id))
    else:
        store.now += timedelta(seconds=300 if when == "EXPIRE_REQUEST" else 901)
    state = call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state
    assert state == ("UNAVAILABLE" if when == "REVOKE_MATERIAL" else "EXPIRED")
    with pytest.raises(AIOpsApplicationError):
        decide(store, view)


def test_rehashed_request_and_decision_cannot_self_attest_against_separate_ciphertext():
    store = setup()
    view = request(store)
    request_artifact = next(item for item in store.artifacts if item.schema_version == REQUEST_SCHEMA)
    original = json.loads(json.dumps(request_artifact.payload_json))
    request_artifact.payload_json["request_id"] = "not-a-valid-uuid"
    request_artifact.content_hash = canonical_hash(request_artifact.payload_json)
    assert call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"
    request_artifact.payload_json = json.loads(json.dumps(original))
    request_artifact.payload_json["statement_timeout_seconds"] = 10
    request_artifact.content_hash = canonical_hash(request_artifact.payload_json)
    assert call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"
    request_artifact.payload_json = original
    request_artifact.content_hash = canonical_hash(original)
    decide(store, view)
    recorded = next(item for item in store.artifacts if item.schema_version == DECISION_SCHEMA)
    recorded.payload_json["decided_by"] = str(store.scope.actor_id)
    recorded.content_hash = canonical_hash(recorded.payload_json)
    assert call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"


def test_duplicate_request_active_material_explicit_policy_and_bounds():
    store = setup()
    view = request(store)
    with pytest.raises(AIOpsApplicationError):
        request(store)
    with pytest.raises(AIOpsApplicationError):
        decide(store, view, request_sha256="0" * 64)
    call(store.capture.revoke(scope=store.scope, run_id=store.run.ops_run_id, request_id=view.request_id))
    with pytest.raises(AIOpsApplicationError):
        request(store, statement_timeout_seconds=6)
    store.policy.rules_json.pop("sql_plan_capture")
    with pytest.raises(AIOpsApplicationError):
        request(store)


def test_owner_isolation_review_metadata_scope_and_agent_filter_apply_to_every_operation():
    store = setup()
    view = request(store)
    for args in ({"scope": store.reviewer}, {"agent_ids": (uuid7(),)}):
        with pytest.raises(AIOpsApplicationError):
            call(store.capture.list(scope=args.get("scope", store.scope), run_id=store.run.ops_run_id,
                agent_ids=args.get("agent_ids", ())))
    for action in (lambda: store.capture.list(scope=store.reviewer, run_id=store.run.ops_run_id, review=True, agent_ids=(uuid7(),)),
        lambda: store.capture.decide(scope=store.reviewer, run_id=store.run.ops_run_id, request_id=view.request_id,
            command=SqlPlanCaptureDecision(request_sha256=view.request_sha256, decision="REJECT"), agent_ids=(uuid7(),)),
        lambda: store.capture.revoke(scope=store.scope, run_id=store.run.ops_run_id, request_id=view.request_id, agent_ids=(uuid7(),))):
        with pytest.raises(AIOpsApplicationError):
            call(action())


def test_contract_does_not_allow_sql_parameters_analyze_or_business_execution():
    for extra in ({"sql": "SELECT " + SECRET}, {"mode": "ANALYZE"}, {"statement_timeout_seconds": True}):
        with pytest.raises(ValidationError):
            SqlPlanCaptureRequest(request_id=uuid7(), material_id=uuid7(), **extra)


def test_public_and_internal_capture_endpoints_exist_and_main_route_permissions_are_explicit():
    from aiops_agent.bootstrap.openapi import create_internal_contract_app
    from main_api.openapi_contracts import create_aiops_public_contract_app
    from main_api.api import ops
    for app in (create_internal_contract_app(), create_aiops_public_contract_app()):
        schema = app.openapi()
        routes = {path: value for path, value in schema["paths"].items() if "/runs/{run_id}/sql-plan-capture" in path}
        assert len(routes) == 7
        assert sum("get" in value for value in routes.values()) == 3
        assert sum("post" in value for value in routes.values()) == 4
        assert sum("delete" in value for value in routes.values()) == 1
    if hasattr(ops, "AIOPS_ROUTE_PERMISSIONS"):
        for name in ("list_sql_plan_material_sources", "list_sql_plan_materials", "submit_sql_plan_material", "revoke_sql_plan_material",
            "list_sql_plan_captures", "request_sql_plan_capture", "revoke_sql_plan_capture"):
            assert ops.AIOPS_ROUTE_PERMISSIONS[name] == "aiops:use"
        for name in ("list_sql_plan_capture_reviews", "decide_sql_plan_capture"):
            assert ops.AIOPS_ROUTE_PERMISSIONS[name] == "aiops:proposal:approve"

@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_approval_artifacts_inherit_source_security_and_expose_policy_timeout_cap(engine):
    store = setup(engine, security_level=5)
    store.policy.rules_json["sql_plan_capture"]["max_statement_timeout_seconds"] = 2
    page = call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id))
    assert page.max_statement_timeout_seconds == 2
    view = request(store, statement_timeout_seconds=2)
    decide(store, view)
    artifacts = [item for item in store.artifacts if item.schema_version in {REQUEST_SCHEMA, DECISION_SCHEMA}]
    assert len(artifacts) == 2 and all(item.security_level == 5 for item in artifacts)
    request_artifact = next(item for item in artifacts if item.schema_version == REQUEST_SCHEMA)
    assert request_artifact.payload_json["security_level"] == 5
    request_artifact.security_level = 3
    assert call(store.capture.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"


def test_actual_http_capture_routes_separate_review_scopes_and_machine_agent_allowlist(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aiops_agent.api.runtime import routes
    from aiops_agent.api.dependencies import get_aiops_auth_context
    from aiops_agent.bootstrap.api import create_aiops_api
    from aiops_agent.config import AIOpsSettings
    from platform_core.contracts import ServiceIdentity, PrincipalKind
    store = setup()
    app = FastAPI()
    app.include_router(routes.router)
    app.exception_handlers.update(create_aiops_api(AIOpsSettings()).exception_handlers)
    app.state.sql_plan_capture_service = store.capture
    app.state.service_scopes = ("aiops.run",)
    app.state.caller_scope = store.scope
    context = SimpleNamespace(principal_kind="PORTAL", authorized_agent_ids=())
    app.dependency_overrides[get_aiops_auth_context] = lambda: context
    monkeypatch.setattr(routes, "_query_scope", lambda *_: app.state.caller_scope)

    @app.middleware("http")
    async def scope_context(http_request, call_next):
        http_request.state.service_identity = ServiceIdentity(issuer="test", subject="test-main-api",
            audience="test-aiops", scopes=app.state.service_scopes,
            issued_at=NOW, expires_at=NOW + timedelta(minutes=5), token_id=uuid7())
        return await call_next(http_request)

    client = TestClient(app)
    base = f"/internal/v1/aiops/runs/{store.run.ops_run_id}"
    payload = {"request_id": str(uuid7()), "material_id": str(store.material.material_id)}
    result = client.post(base + "/sql-plan-captures", json=payload)
    assert result.status_code == 201 and result.headers["cache-control"] == "no-store"
    view = result.json()
    assert not view["execution_authorized"] and not view["capture_scheduled"] and SECRET not in result.text
    decision = {"request_sha256": view["request_sha256"], "decision": "APPROVE", "acknowledge_estimate_only": True}
    review_path = "/sql-plan-capture-reviews"
    decision_path = f'/sql-plan-captures/{view["request_id"]}/decision'
    revoke_path = f'/sql-plan-captures/{view["request_id"]}'
    assert client.get(base + review_path).status_code == 403
    assert client.post(base + decision_path, json=decision).status_code == 403
    assert not any(item.schema_version == DECISION_SCHEMA for item in store.artifacts)
    if hasattr(routes, "_agent_filter"):
        app.state.service_scopes = ("aiops.run", "aiops.approve")
        context.principal_kind = PrincipalKind.APP_API_CLIENT
        for allowed in ((), (uuid7(),)):
            context.authorized_agent_ids = allowed
            for method, suffix, body in (("GET", "/sql-plan-captures", None),
                ("GET", review_path, None), ("POST", "/sql-plan-captures", payload),
                ("POST", decision_path, decision), ("DELETE", revoke_path, None)):
                denied = client.request(method, base + suffix, **({"json": body} if body else {}))
                assert denied.status_code == 404 and SECRET not in denied.text
        context.authorized_agent_ids = (store.run.agent_id,)
        assert client.get(base + "/sql-plan-captures").status_code == 200
        context.principal_kind = "PORTAL"
    app.state.service_scopes = ("aiops.approve",)
    app.state.caller_scope = store.reviewer
    review = client.get(base + review_path)
    assert review.status_code == 200 and review.headers["cache-control"] == "no-store"
    assert review.json()["can_review"] and not review.json()["can_request"]
    assert client.post(base + "/sql-plan-captures", json=payload).status_code == 403
    assert client.delete(base + revoke_path).status_code == 403
    approved = client.post(base + decision_path, json=decision)
    assert approved.status_code == 200 and approved.json()["state"] == "APPROVED"
    assert SECRET not in approved.text
    app.state.caller_scope = store.scope
    app.state.service_scopes = ("aiops.run",)
    revoked = client.delete(base + revoke_path)
    assert revoked.status_code == 200 and revoked.json()["state"] == "REVOKED"

def test_real_main_api_client_bootstrap_can_forward_independent_review_scope():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / "services/main_api/src/main_api/entrypoints/api.py").read_text())
    auth_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "AIOpsClientAuth"]
    assert len(auth_calls) == 1
    scopes = ast.literal_eval(next(item.value for item in auth_calls[0].keywords if item.arg == "scopes"))
    assert {"aiops.run", "aiops.approve"} <= set(scopes)
