"""参数材料的真实加密、生产用例、权限变化、撤回过期及安全 HTTP 传输回归。"""

import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.sql_plan_materials import SqlPlanMaterialService
from aiops_agent.application.sql_plan_material_vault import SqlPlanMaterialVault, NAMESPACE, KIND
from aiops_agent.application.managed_credentials import AIOpsManagedCredentialService, AIOpsManagedCredentialError
from aiops_agent.domain.evidence.sql_plan_material import parse_sql_plan_material, SqlPlanMaterialInvalid
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_material import SqlPlanMaterialSubmit
from platform_core.managed_credentials import ManagedCredentialCipher, ManagedCredentialPayload
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_target_facts import _scope
from tests.unit.aiops_agent.test_aiops_native_field_lineage import observation_for, tool_for, tool_artifact, NOW


SECRET = "private-representative-value-do-not-echo"


def document():
    return {"schema_version": "SQL_PLAN_PARAMETER_MATERIAL.v1",
        "parameters": [{"position": 1, "database_type_name": "pg_catalog.numeric", "value": SECRET},
            {"position": 2, "database_type_name": "pg_catalog.numeric", "value": "12345678901234567890.123456789"}],
        "execution_context": {"database_name": None, "schema_name": "private_schema", "role_name": "private_role",
            "search_path": ["private_schema", "pg_catalog"], "settings": {"TimeZone": "UTC"}}}


def command(artifact, **updates):
    return SqlPlanMaterialSubmit(submission_id=uuid7(), source_artifact_id=artifact.artifact_id,
        step_id="a1", row_index=0, protected_document=json.dumps(document()), **updates)


class Store:
    def __init__(self, engine="POSTGRESQL"):
        self.scope = _scope()
        self.cipher = ManagedCredentialCipher(key=b"m" * 32, key_version="test")
        self.vault = SqlPlanMaterialVault(cipher=self.cipher)
        self.run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7(), actor_id=self.scope.actor_id,
            **self.vault._scope(self.scope), status="COMPLETED", cancel_requested_at=None,
            agent_id=uuid7(), agent_version_id=uuid7(), plan_snapshot_json={"binding": {"policy_id": None}})
        self.target = SimpleNamespace(target_id=self.run.target_id, row_version=1, db_type=engine,
            endpoint_json={"host": "example.invalid", "database": "sample"}, status="ENABLED",
            readonly_connection_enabled=True, diagnostic_credential_id=uuid7())
        self.run.plan_snapshot_json["investigation_execution"] = {"database": {
            "target_row_version": self.target.row_version, "db_type": engine,
            "connection_profile": deepcopy(self.target.endpoint_json),
            "diagnostic_credential_id": str(self.target.diagnostic_credential_id), "automatic_access_enabled": True,
        }}
        self.binding = SimpleNamespace(binding_id=self.run.agent_version_id, policy_id=None)
        self.policy = SimpleNamespace(status="ACTIVE", policy_hash="a" * 64, rules_json={"readonly_database_enabled": True})
        self.original = tool_artifact(observation_for(tool_for(engine), target=self.run.target_id), self.run.ops_run_id)
        self.original.security_level = 3
        self.artifacts = [self.original]
        self.credentials = []
        self.events = []
        self.run_requests = []
        self.commits = 0
        async def get_run(**kwargs):
            self.run_requests.append(kwargs)
            return self.run if kwargs["ops_run_id"] == self.run.ops_run_id else None
        async def get_artifact(**kwargs):
            return next((item for item in self.artifacts if item.artifact_id == kwargs["artifact_id"]), None)
        async def by_key(**kwargs):
            return next((item for item in self.artifacts if item.ops_run_id == kwargs["ops_run_id"]
                and getattr(item, "artifact_key", None) == kwargs["artifact_key"]), None)
        async def add_artifact(item):
            self.artifacts.append(item)
            return item
        async def append_event(**kwargs):
            if not any(item["event_key"] == kwargs["event_key"] for item in self.events):
                self.events.append(kwargs)
        async def find(**kwargs):
            return next((item for item in self.credentials
                if all(str(getattr(item, key)) == str(value) for key, value in kwargs.items() if key != "lock")), None)
        async def add_credential(item):
            self.credentials.append(item)
        self.runs = SimpleNamespace(get_run_scoped=AsyncMock(side_effect=get_run), get_artifact=AsyncMock(side_effect=get_artifact),
            get_artifact_by_key=AsyncMock(side_effect=by_key), list_artifacts=AsyncMock(side_effect=lambda **_: list(self.artifacts)),
            add_artifact=AsyncMock(side_effect=add_artifact), append_event=AsyncMock(side_effect=append_event))
        self.targets = SimpleNamespace(get_scoped=AsyncMock(side_effect=lambda **_: self.target))
        self.agents = SimpleNamespace(get_active=AsyncMock(side_effect=lambda **_: self.binding))
        self.policies = SimpleNamespace(get_scoped=AsyncMock(side_effect=lambda **_: self.policy))
        self.managed_credentials = SimpleNamespace(find=AsyncMock(side_effect=find), add=AsyncMock(side_effect=add_credential),
            get=AsyncMock(side_effect=lambda **kwargs: next((item for item in self.credentials if item.credential_id == kwargs["credential_id"]), None)))
        self.now = NOW
        self.service = SqlPlanMaterialService(uow_factory=lambda: self, vault=self.vault, clock=lambda: self.now)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def commit(self):
        self.commits += 1

    def decrypt(self, index=-1):
        row = self.credentials[index]
        return self.cipher.decrypt(ManagedCredentialPayload(ciphertext=row.ciphertext, nonce=row.nonce, key_version=row.key_version),
            **self.vault._scope(self.scope), namespace=NAMESPACE, credential_kind=KIND, credential_id=row.credential_id)

    def update_observation(self, mutate):
        mutate(self.original.payload_json["tool_outcomes"][0]["observation"])
        self.original.content_hash = canonical_hash(self.original.payload_json)


def call(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("engine", ["POSTGRESQL", "MYSQL", "ORACLE"])
def test_three_engine_protected_submission_roundtrip_keeps_only_safe_receipts(engine):
    store = Store(engine)
    sources = call(store.service.sources(scope=store.scope, run_id=store.run.ops_run_id))
    assert len(sources) == 1 and sources[0].database_type == engine
    result = call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    assert result.state == "STORED" and result.parameter_count == 2 and result.execution_authorized is False
    assert result.representative_parameters == result.environment == "USER_DECLARED_UNVERIFIED"
    wire = result.model_dump_json() + json.dumps(store.events, default=str) + json.dumps(store.artifacts[-1].payload_json)
    for value in (SECRET, "private_schema", "private_role", "12345678901234567890.123456789"):
        assert value not in wire and value.encode() not in store.credentials[0].ciphertext
    assert store.decrypt()["document"] == document()
    assert store.credentials[0].namespace != "aiops"
    assert store.credentials[0].nonce and len(store.credentials[0].nonce) == 12
    assert store.artifacts[-1].trust_level == "USER_PROVIDED"
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items == (result,)
    assert any(request["lock"] for request in store.run_requests)


def test_secret_contract_repr_and_default_json_never_include_values_but_validate_bounds():
    store = Store()
    item = command(store.original)
    assert SECRET not in repr(item) and SECRET not in item.model_dump_json()
    for ttl in (True, 0, 1801, 1.5, "900"):
        with pytest.raises(ValidationError):
            SqlPlanMaterialSubmit.model_validate({**item.model_dump(), "ttl_seconds": ttl})


@pytest.mark.parametrize("value", [None, {}, [], 1, "", "{", '["' + SECRET + '"]', '{"a":1,"a":2}',
    '{"schema_version":"SQL_PLAN_PARAMETER_MATERIAL.v1","parameters":[],"execution_context":NaN}',
    "x" * 65537])
def test_malformed_documents_are_fixed_errors_without_input(value):
    with pytest.raises(SqlPlanMaterialInvalid) as caught:
        parse_sql_plan_material(value)
    assert str(caught.value) == "SQL_PLAN_MATERIAL_INVALID" and SECRET not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("change", [
    lambda value: value.update(sql="SELECT private"),
    lambda value: value.update(schema_version="unknown"),
    lambda value: value.update(parameters=[{"position": True, "database_type_name": "int8", "value": "1"}]),
    lambda value: value["parameters"][0].update(position=2),
    lambda value: value["parameters"][0].update(value=1.5),
    lambda value: value["parameters"][0].update(value=12345678901234567890),
    lambda value: value["parameters"][0].update(value={"secret": SECRET}),
    lambda value: value["parameters"][0].update(value="x" * 16385),
    lambda value: value["parameters"][0].update(value=[[[[[SECRET]]]]]),
    lambda value: value["parameters"][0].update(value=["1"] * 257),
    lambda value: value["parameters"][0].update(database_type_name=""),
    lambda value: value["parameters"][0].update(database_type_name="int8\n"),
    lambda value: value["parameters"][0].update(injected=SECRET),
    lambda value: value.update(parameters=value["parameters"] * 33),
    lambda value: value["execution_context"].pop("role_name"),
    lambda value: value["execution_context"].update(sql="SELECT private"),
    lambda value: value["execution_context"].update(search_path="mes"),
    lambda value: value["execution_context"].update(settings={"bad;key": SECRET}),
    lambda value: value["execution_context"].update(settings={"work_mem": 8}),
])
def test_document_schema_cannot_guess_sql_environment_or_lose_parameter_precision(change):
    value = document()
    change(value)
    with pytest.raises(SqlPlanMaterialInvalid):
        parse_sql_plan_material(json.dumps(value))


def test_null_boolean_arrays_and_empty_parameter_list_preserve_exact_material():
    value = document()
    value["parameters"] = [{"position": 1, "database_type_name": "declared.custom_type", "value": [None, True, ["1.000", "9007199254740993"]]}]
    assert parse_sql_plan_material(json.dumps(value)) == value
    value["parameters"] = []
    assert parse_sql_plan_material(json.dumps(value)) == value


@pytest.mark.parametrize("change", [
    lambda store: setattr(store.original, "trust_level", "USER_PROVIDED"),
    lambda store: setattr(store.original, "ops_run_id", uuid7()),
    lambda store: setattr(store.original, "content_hash", "0" * 64),
    lambda store: store.update_observation(lambda obs: obs.update(target_id=str(uuid7()))),
    lambda store: store.update_observation(lambda obs: obs.update(truncated=True)),
    lambda store: store.update_observation(lambda obs: obs.update(row_count=99)),
    lambda store: store.update_observation(lambda obs: obs.update(tool_id="db.postgresql.readonly_query")),
    lambda store: store.original.payload_json.update(status="FAILED"),
])
def test_invalid_native_sources_never_create_material(change):
    store = Store()
    change(store)
    with pytest.raises(AIOpsApplicationError):
        call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    assert not store.credentials


@pytest.mark.parametrize("field", ["database_oid", "user_oid", "query_id"])
def test_required_statistic_dimensions_cannot_be_borrowed_from_connection(field):
    store = Store()
    def mutate(obs):
        index = next(i for i, item in enumerate(obs["columns"]) if item["name"] == field)
        obs["columns"].pop(index)
        for row in obs["rows"]:
            row.pop(index)
    store.update_observation(mutate)
    assert not call(store.service.sources(scope=store.scope, run_id=store.run.ops_run_id))


def test_mysql_null_default_schema_is_preserved_not_filled_from_target():
    store = Store("MYSQL")
    def mutate(obs):
        index = next(i for i, item in enumerate(obs["columns"]) if item["name"] == "schema_name")
        obs["rows"][0][index] = None
    store.update_observation(mutate)
    source = call(store.service.sources(scope=store.scope, run_id=store.run.ops_run_id))[0]
    assert "schema_name" in source.subject and source.subject["schema_name"] is None


def test_duplicate_step_id_makes_source_ambiguous():
    store = Store()
    store.original.payload_json["tool_outcomes"] *= 2
    store.original.content_hash = canonical_hash(store.original.payload_json)
    assert not call(store.service.sources(scope=store.scope, run_id=store.run.ops_run_id))


def test_owner_and_domain_checks_apply_to_list_submit_sources_and_revoke():
    store = Store()
    item = call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    for scope in (replace(store.scope, actor_id=str(uuid7())), replace(store.scope, domain_id=999)):
        for coro in (
            store.service.sources(scope=scope, run_id=store.run.ops_run_id),
            store.service.list(scope=scope, run_id=store.run.ops_run_id),
            store.service.submit(scope=scope, run_id=store.run.ops_run_id, command=command(store.original)),
            store.service.revoke(scope=scope, run_id=store.run.ops_run_id, material_id=item.material_id),
        ):
            with pytest.raises(AIOpsApplicationError) as caught:
                call(coro)
            assert caught.value.status_code == 404


def test_agent_allowlist_is_checked_inside_owner_transaction_before_material_access():
    store = Store()
    item = call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    before = store.commits
    protected = store.credentials[0].ciphertext
    denied = (uuid7(),)
    for coro in (
        store.service.sources(scope=store.scope, run_id=store.run.ops_run_id, agent_ids=denied),
        store.service.list(scope=store.scope, run_id=store.run.ops_run_id, agent_ids=denied),
        store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original), agent_ids=denied),
        store.service.revoke(scope=store.scope, run_id=store.run.ops_run_id, material_id=item.material_id, agent_ids=denied),
    ):
        with pytest.raises(AIOpsApplicationError) as caught:
            call(coro)
        assert caught.value.status_code == 404
    assert store.credentials[0].ciphertext == protected and store.commits == before
    allowed = (store.run.agent_id,)
    assert call(store.service.sources(scope=store.scope, run_id=store.run.ops_run_id, agent_ids=allowed))
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id, agent_ids=allowed)).items[0].state == "STORED"


@pytest.mark.parametrize("change", [
    lambda store: setattr(store.target, "status", "DISABLED"),
    lambda store: setattr(store.target, "readonly_connection_enabled", False),
    lambda store: setattr(store.target, "diagnostic_credential_id", None),
    lambda store: setattr(store.binding, "binding_id", uuid7()),
    lambda store: setattr(store.run, "cancel_requested_at", NOW),
    lambda store: setattr(store.run, "status", "FAILED"),
])
def test_current_access_change_disables_use_but_cannot_prevent_owner_revoke(change):
    store = Store()
    item = call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    change(store)
    listed = call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id))
    assert listed.items[0].state == "UNAVAILABLE"
    with pytest.raises(AIOpsApplicationError):
        call(store.service.sources(scope=store.scope, run_id=store.run.ops_run_id))
    revoked = call(store.service.revoke(scope=store.scope, run_id=store.run.ops_run_id, material_id=item.material_id))
    assert revoked.state == "REVOKED"
    assert store.credentials[0].status == "REVOKED" and len(store.credentials[0].nonce) == 12
    assert set(store.decrypt(0)) == {"binding", "closure"} and store.decrypt(0)["closure"] == "REVOKED"


@pytest.mark.parametrize("change", [
    lambda store: setattr(store.target, "row_version", 2),
    lambda store: store.target.endpoint_json.update(database="changed"),
    lambda store: setattr(store.target, "diagnostic_credential_id", uuid7()),
    lambda store: setattr(store.original, "content_hash", "0" * 64),
])
def test_frozen_source_target_and_credentials_are_revalidated(change):
    store = Store()
    call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    change(store)
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"


def test_disabled_policy_and_binding_policy_changes_are_not_execution_authority():
    store = Store()
    policy_id = uuid7()
    store.binding.policy_id = policy_id
    store.run.plan_snapshot_json["binding"]["policy_id"] = str(policy_id)
    call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    store.policy.rules_json["readonly_database_enabled"] = False
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"
    store.policy.rules_json["readonly_database_enabled"] = True
    store.policy.policy_hash = "b" * 64
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"


def test_expiry_is_exact_and_scrubs_ciphertext_on_state_read_without_returning_values():
    store = Store()
    item = call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original, ttl_seconds=60)))
    store.now += timedelta(seconds=59)
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "STORED"
    store.now += timedelta(seconds=1)
    result = call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id))
    assert result.items[0].state == "EXPIRED" and result.items[0].expires_at == item.expires_at
    assert store.credentials[0].status == "REVOKED" and len(store.credentials[0].nonce) == 12
    assert set(store.decrypt(0)) == {"binding", "closure"} and store.decrypt(0)["closure"] == "EXPIRED"
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "EXPIRED"


def test_forged_receipt_cannot_extend_expiry_or_self_attest_binding():
    store = Store()
    call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    receipt = store.artifacts[-1]
    receipt.payload_json["expires_at"] = (NOW + timedelta(days=1)).isoformat()
    receipt.content_hash = canonical_hash(receipt.payload_json)
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"


def test_cross_material_ciphertext_substitution_and_bad_tag_are_never_accepted():
    store = Store()
    for _ in range(2):
        call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    first, second = store.credentials
    first.ciphertext, first.nonce = second.ciphertext, second.nonce
    assert call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items[0].state == "UNAVAILABLE"
    second.ciphertext = b"invalid"
    assert all(item.state == "UNAVAILABLE" for item in call(store.service.list(scope=store.scope, run_id=store.run.ops_run_id)).items)


def test_ordinary_managed_credential_service_cannot_resolve_protected_namespace():
    store = Store()
    call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    service = AIOpsManagedCredentialService(uow_factory=lambda: store, cipher=store.cipher)
    with pytest.raises(AIOpsManagedCredentialError):
        call(service.read(uow=store, **store.vault._scope(store.scope), credential_id=store.credentials[0].credential_id,
            credential_kind=KIND, external_key=store.run.ops_run_id))


def test_active_and_total_quota_apply_under_run_lock_and_revocation_is_idempotent():
    store = Store()
    items = [call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original))) for _ in range(4)]
    with pytest.raises(AIOpsApplicationError):
        call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    for item in items:
        call(store.service.revoke(scope=store.scope, run_id=store.run.ops_run_id, material_id=item.material_id))
    for _ in range(12):
        item = call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
        call(store.service.revoke(scope=store.scope, run_id=store.run.ops_run_id, material_id=item.material_id))
    assert len(store.credentials) == 16
    with pytest.raises(AIOpsApplicationError):
        call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    call(store.service.revoke(scope=store.scope, run_id=store.run.ops_run_id, material_id=items[0].material_id))
    assert len([event for event in store.events if event["event_type"] == "sql.plan_material.revoked"]) == 16


def test_submission_identifier_is_immutable_and_has_no_public_parameter_fingerprint():
    store = Store()
    item = command(store.original)
    call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=item))
    with pytest.raises(AIOpsApplicationError):
        call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=item))
    assert len(store.credentials) == 1
    assert not any("parameter" in key and "hash" in key for key in store.artifacts[-1].payload_json)


def test_client_explicitly_forwards_secret_only_on_protected_service_transport():
    from platform_clients.aiops import AIOpsManagementClient
    store = Store()
    item = command(store.original)
    client = object.__new__(AIOpsManagementClient)
    client._json = AsyncMock(return_value={})
    call(client.submit_sql_plan_material(store.run.ops_run_id, item, auth_context=object()))
    sent = client._json.call_args.kwargs["payload"]
    assert sent["protected_document"] == item.protected_document.get_secret_value()
    assert client._json.call_args.args[0] == "POST"
    assert "/sql-plan-materials" in client._json.call_args.args[1]


def test_material_endpoints_are_registered_at_both_boundaries_and_schema_is_write_only():
    from aiops_agent.bootstrap.openapi import create_internal_contract_app
    from main_api.openapi_contracts import create_aiops_public_contract_app
    for app, prefix in ((create_internal_contract_app(), "/internal/v1/aiops"),
                        (create_aiops_public_contract_app(), "/api/v1/ops")):
        schema = app.openapi()
        paths = [path for path in schema["paths"] if "/runs/{run_id}/sql-plan-material" in path]
        assert len(paths) == 3
        operations = [schema["paths"][path] for path in paths]
        assert sum("get" in item for item in operations) == 2
        assert sum("post" in item for item in operations) == 1
        assert sum("delete" in item for item in operations) == 1
        assert next(item["post"] for item in operations if "post" in item)["x-sensitive-input"] is True
        field = schema["components"]["schemas"]["SqlPlanMaterialSubmit"]["properties"]["protected_document"]
        assert field["writeOnly"] is True and field["format"] == "password"
    root = Path(__file__).resolve().parents[3]
    service = (root / "services/aiops_agent/src/aiops_agent/application/sql_plan_materials.py").read_text()
    assert "execute_dynamic" not in service and ".execute(" not in service and ".logger" not in service

def test_actual_http_routes_use_protected_service_and_sanitize_validation_errors(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aiops_agent.api.runtime import routes
    from aiops_agent.api.dependencies import get_aiops_auth_context
    from aiops_agent.bootstrap.api import create_aiops_api
    from aiops_agent.config import AIOpsSettings
    from platform_core.contracts import ServiceIdentity
    store = Store()
    app = FastAPI()
    app.include_router(routes.router)
    app.exception_handlers.update(create_aiops_api(AIOpsSettings()).exception_handlers)
    app.state.sql_plan_material_service = store.service
    app.state.scope_allowed = True
    context = SimpleNamespace(principal_kind="PORTAL", authorized_agent_ids=())
    app.dependency_overrides[get_aiops_auth_context] = lambda: context
    monkeypatch.setattr(routes, "_query_scope", lambda *_: store.scope)

    @app.middleware("http")
    async def scope_context(request, call_next):
        request.state.service_identity = ServiceIdentity(issuer="test", subject="test-main-api",
            audience="test-aiops", scopes=("aiops.run",) if app.state.scope_allowed else ("aiops.read",),
            issued_at=NOW, expires_at=NOW + timedelta(minutes=5), token_id=uuid7())
        return await call_next(request)

    client = TestClient(app)
    base = f"/internal/v1/aiops/runs/{store.run.ops_run_id}"
    result = client.get(base + "/sql-plan-material-sources")
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    payload = command(store.original).model_dump(mode="json")
    payload["protected_document"] = json.dumps(document())
    result = client.post(base + "/sql-plan-materials", json=payload)
    assert result.status_code == 201 and result.headers["cache-control"] == "no-store"
    assert result.json()["execution_authorized"] is False and SECRET not in result.text
    material_id = result.json()["material_id"]
    if hasattr(routes, "_agent_filter"):
        from platform_core.contracts import PrincipalKind
        context.principal_kind = PrincipalKind.APP_API_CLIENT
        for allowed in ((), (uuid7(),)):
            context.authorized_agent_ids = allowed
            for method, suffix, body in (("GET", "/sql-plan-material-sources", None),
                ("GET", "/sql-plan-materials", None), ("POST", "/sql-plan-materials", payload),
                ("DELETE", "/sql-plan-materials/" + material_id, None)):
                denied = client.request(method, base + suffix, **({"json": body} if body else {}))
                assert denied.status_code == 404 and SECRET not in denied.text
        context.authorized_agent_ids = (store.run.agent_id,)
        assert client.get(base + "/sql-plan-materials").status_code == 200
        context.principal_kind = "PORTAL"
    result = client.get(base + "/sql-plan-materials")
    assert result.status_code == 200 and len(result.json()["items"]) == 1 and SECRET not in result.text
    result = client.post(base + "/sql-plan-materials", json={**payload, "protected_document": {"secret": SECRET}})
    assert result.status_code == 422 and SECRET not in result.text
    result = client.post(base + "/sql-plan-materials", json={**payload, "submission_id": str(uuid7()), "protected_document": SECRET})
    assert result.status_code == 422 and SECRET not in result.text
    app.state.scope_allowed = False
    result = client.delete(base + "/sql-plan-materials/" + material_id)
    assert result.status_code == 403
    app.state.scope_allowed = True
    result = client.delete(base + "/sql-plan-materials/" + material_id)
    assert result.status_code == 200 and result.json()["state"] == "REVOKED"
    assert set(store.decrypt(0)) == {"binding", "closure"} and SECRET not in result.text

def test_main_api_sensitive_contract_suppresses_even_unknown_field_names_and_validator_messages():
    from fastapi.exceptions import RequestValidationError
    from starlette.requests import Request
    from main_api import app as main_app
    factory = next(value for name, value in vars(main_app).items()
        if name.startswith("create_") and name.endswith("main_api_app"))
    app = factory(enable_access_log=False)
    handler = app.exception_handlers[RequestValidationError]
    request = Request({"type": "http", "method": "POST", "path": "/protected-material-test", "headers": [],
        "route": SimpleNamespace(openapi_extra={"x-sensitive-input": True})})
    exc = RequestValidationError([{"loc": ("body", SECRET), "type": "extra_forbidden", "msg": SECRET, "input": SECRET}])
    response = call(handler(request, exc))
    assert response.status_code == 422 and SECRET.encode() not in response.body
    assert "field_errors" not in json.loads(response.body)

@pytest.mark.parametrize("change", [
    lambda store: store.run.plan_snapshot_json.pop("investigation_execution"),
    lambda store: store.run.plan_snapshot_json["investigation_execution"]["database"].update(automatic_access_enabled=False),
    lambda store: setattr(store.target, "row_version", 2),
    lambda store: store.target.endpoint_json.update(host="other.example.invalid"),
])
def test_cannot_attach_old_sql_material_to_a_new_or_unfrozen_target(change):
    store = Store()
    change(store)
    with pytest.raises(AIOpsApplicationError):
        call(store.service.submit(scope=store.scope, run_id=store.run.ops_run_id, command=command(store.original)))
    assert not store.credentials
