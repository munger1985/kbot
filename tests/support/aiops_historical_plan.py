"""离线重建旧版持久化格式，验证只读历史证据，不依赖已删除的估算执行服务。"""

import asyncio
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.sql_plan_materials import SqlPlanMaterialService, _source_rows, MATERIAL_SCHEMA, MATERIAL_PREFIX
from aiops_agent.application.sql_plan_material_vault import SqlPlanMaterialVault, NAMESPACE, KIND
from aiops_agent.application.sql_plan_capture_vault import SqlPlanCaptureVault, KIND as APPROVAL_KIND
from aiops_agent.application.sql_plan_results import SqlPlanCaptureResultReader, RESULT_KEY, RESULT_SCHEMA, task_binding
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureResult
from platform_core.contracts.aiops.sql_plan_verification import SqlPlanNativeVerification
from platform_core.managed_credentials import ManagedCredentialCipher
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_target_facts import _scope
from tests.unit.aiops_agent.test_aiops_native_field_lineage import observation_for, tool_for, tool_artifact, NOW


SECRET = "private-representative-value-do-not-echo"


def call(coro):
    return asyncio.run(coro)


def verification():
    def check(field, status="UNKNOWN"):
        return {"field": field, "status": status, "source": "UNAVAILABLE"}
    return SqlPlanNativeVerification(
        original_context=tuple(check(field) for field in ("DATABASE", "SCHEMA", "ROLE", "SEARCH_PATH", "OPTIMIZER_SETTINGS")),
        parameter_types=check("PARAMETER_TYPES"), parameter_values=check("PARAMETER_VALUES", "NOT_REQUESTED"),
        resolved_objects=({"node_id": 1, "object_kind": "TABLE", "schema_name": "mes", "object_name": "t",
            "native_object_id": "123", "resolution": "NATIVE_CATALOG"},),
        limitations=("PARTIAL_CONTEXT_ONLY", "POPULATION_NOT_OBSERVED"))


class HistoricalStore:
    def __init__(self, engine):
        self.scope, self.now = _scope(), NOW
        self.cipher = ManagedCredentialCipher(key=b"m" * 32, key_version="test")
        self.vault = SqlPlanMaterialVault(cipher=self.cipher)
        self.run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7(), actor_id=self.scope.actor_id,
            **self.vault._scope(self.scope), status="COMPLETED", cancel_requested_at=None,
            agent_id=uuid7(), agent_version_id=uuid7(), plan_snapshot_json={})
        self.target = SimpleNamespace(target_id=self.run.target_id, row_version=1, db_type=engine,
            endpoint_json={"host": "example.invalid", "database": "sample", "port": 5432, "service": "app", "tls_enabled": False},
            status="ENABLED", readonly_connection_enabled=True, diagnostic_credential_id=uuid7())
        policy_id = uuid7()
        self.binding = SimpleNamespace(binding_id=self.run.agent_version_id, policy_id=policy_id)
        self.policy = SimpleNamespace(status="ACTIVE", policy_hash="a" * 64, rules_json={"readonly_database_enabled": True})
        self.run.plan_snapshot_json = {"binding": {"policy_id": str(policy_id)}, "investigation_execution": {"database": {
            "target_row_version": 1, "db_type": engine, "connection_profile": deepcopy(self.target.endpoint_json),
            "diagnostic_credential_id": str(self.target.diagnostic_credential_id), "automatic_access_enabled": True}}}
        self.original = tool_artifact(observation_for(tool_for(engine), target=self.run.target_id), self.run.ops_run_id)
        self.original.security_level = 3
        field, sql = {"POSTGRESQL": ("normalized_statement", "SELECT * FROM mes.t WHERE id=$1 OR n=$2"),
            "MYSQL": ("digest_text", "SELECT * FROM mes.t WHERE id=? OR n=?"),
            "ORACLE": ("sql_text", "SELECT * FROM MES.T WHERE ID=:1 OR N=:2")}[engine]
        observation = self.original.payload_json["tool_outcomes"][0]["observation"]
        position = next(i for i, column in enumerate(observation["columns"]) if column["name"] == field)
        observation["rows"][0][position] = sql
        self.original.content_hash = canonical_hash(self.original.payload_json)
        self.artifacts, self.credentials, self.events = [self.original], [], []
        self.all_runs, self.all_tasks = [self.run], []
        async def find(**kwargs):
            return next((row for row in self.credentials if all(str(getattr(row, key)) == str(value)
                for key, value in kwargs.items() if key != "lock")), None)
        async def add_artifact(row):
            self.artifacts.append(row)
            return row
        self.runs = SimpleNamespace(
            get_run=AsyncMock(side_effect=lambda **args: next((row for row in self.all_runs if row.ops_run_id == args["ops_run_id"]), None)),
            get_run_scoped=AsyncMock(side_effect=lambda **args: next((row for row in self.all_runs if row.ops_run_id == args["ops_run_id"]), None)),
            get_task=AsyncMock(side_effect=lambda **args: next((row for row in self.all_tasks if row.ops_task_id == args["ops_task_id"]), None)),
            get_artifact=AsyncMock(side_effect=lambda **args: next((row for row in self.artifacts if row.artifact_id == args["artifact_id"]), None)),
            get_artifact_by_key=AsyncMock(side_effect=lambda **args: next((row for row in self.artifacts
                if row.ops_run_id == args["ops_run_id"] and getattr(row, "artifact_key", None) == args["artifact_key"]), None)),
            list_artifacts=AsyncMock(side_effect=lambda **_: list(self.artifacts)),
            add_artifact=AsyncMock(side_effect=add_artifact), append_event=AsyncMock(side_effect=lambda **args: self.events.append(args)))
        self.targets = SimpleNamespace(get_scoped=AsyncMock(side_effect=lambda **_: self.target))
        self.agents = SimpleNamespace(get_active=AsyncMock(side_effect=lambda **_: self.binding))
        self.policies = SimpleNamespace(get_scoped=AsyncMock(side_effect=lambda **_: self.policy))
        self.managed_credentials = SimpleNamespace(find=AsyncMock(side_effect=find))
        self.service = SqlPlanMaterialService(uow_factory=lambda: self, vault=self.vault, clock=lambda: self.now)
        self.reader = SqlPlanCaptureResultReader(materials=self.service, vault=SqlPlanCaptureVault(material_vault=self.vault))
        self.secret_reader = SimpleNamespace(read=AsyncMock())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def commit(self):
        pass

    def artifact(self, run, key, schema, payload, provenance=None):
        row = SimpleNamespace(artifact_id=uuid7(), ops_run_id=run.ops_run_id, artifact_key=key,
            schema_version=schema, payload_json=payload, content_hash=canonical_hash(payload),
            security_level=3, trust_level="SOURCE_VERIFIED", provenance_json=provenance or {})
        self.artifacts.append(row)
        return row

    def encrypted(self, external_id, kind, payload):
        credential_id = uuid7()
        encrypted = self.cipher.encrypt(payload, **self.vault._scope(self.scope), namespace=NAMESPACE,
            credential_kind=kind, credential_id=credential_id)
        self.credentials.append(SimpleNamespace(credential_id=credential_id, **self.vault._scope(self.scope),
            namespace=NAMESPACE, credential_kind=kind, external_key=str(external_id), status="ACTIVE",
            created_by=self.scope.actor_id, ciphertext=encrypted.ciphertext, nonce=encrypted.nonce, key_version=encrypted.key_version))


def historical_store(engine="POSTGRESQL", *, status="SUCCEEDED", native_verification=None):
    store = HistoricalStore(engine)
    material_id, request_id = uuid7(), uuid7()
    source = _source_rows(store.original, store.run, engine)[0].model_dump(mode="json")
    _, access = call(store.service._access(store, store.run))
    material = {"material_id": str(material_id), "ops_run_id": str(store.run.ops_run_id),
        "actor_id": str(store.scope.actor_id), "source": source, "source_hash": store.original.content_hash,
        "parameter_count": 2, "capture_preflight": None, "created_at": store.now.isoformat(),
        "expires_at": (store.now + timedelta(minutes=5)).isoformat(), **access}
    receipt = store.artifact(store.run, MATERIAL_PREFIX + str(material_id), MATERIAL_SCHEMA, material)
    store.encrypted(material_id, KIND, {"binding": material, "document": {"parameters": [SECRET]}})
    request = {"request_id": str(request_id), "ops_run_id": str(store.run.ops_run_id),
        "requested_by": str(store.scope.actor_id), "source": source, "material_id": str(material_id),
        "material_receipt_hash": receipt.content_hash, "security_level": 3, "frozen": access}
    decision = {"decision": "APPROVE", "decided_by": str(uuid7())}
    approval = canonical_hash({"request": request, "decision": decision})
    background = deepcopy(store.run)
    background.ops_run_id, background.workflow_kind = uuid7(), "SQL_PLAN_CAPTURE"
    background.deadline_at = store.now + timedelta(seconds=30)
    background.source_result_artifact_id = store.original.artifact_id
    info = {"source_run_id": str(store.run.ops_run_id), "request_id": str(request_id), "approval_sha256": approval}
    background.plan_snapshot_json = {"blueprint": {"version": "1"}, "sql_plan_capture": info}
    task = SimpleNamespace(ops_task_id=uuid7(), ops_run_id=background.ops_run_id, task_key="capture:1",
        task_type="DIAGNOSTIC", handler_id="sql.plan-capture", handler_version="1", input_schema_version="v1",
        output_schema_version=RESULT_SCHEMA, depends_on_json=[], input_artifacts_json=[], max_attempts=1, timeout_seconds=5)
    store.all_runs.append(background)
    store.all_tasks.append(task)
    execution = {**info, "capture_run_id": str(background.ops_run_id), "capture_task_id": str(task.ops_task_id),
        "plan_sha256": canonical_hash(background.plan_snapshot_json), "task_sha256": canonical_hash(task_binding(task)),
        "deadline_at": background.deadline_at.isoformat()}
    physical = store.original.payload_json["tool_outcomes"][0]["observation"]["provenance"]["instance_identity"]
    result = SqlPlanCaptureResult(capture_run_id=background.ops_run_id, request_id=request_id, database_type=engine,
        status=status, error_code=None if status == "SUCCEEDED" else "PLAN_CAPTURE_FAILED", captured_at=store.now,
        nodes=({"node_id": 0, "operation": "Aggregate", "metrics": {"estimated_rows": "1"}},
            {"node_id": 1, "parent_id": 0, "operation": "Seq Scan", "schema_name": "mes", "relation_name": "t",
                "metrics": {"estimated_rows": "10000", "optimizer_total_cost": "123.5"}}) if status == "SUCCEEDED" else (),
        physical_identity=physical, native_verification=native_verification,
        oracle_parameter_peeking="NOT_PROVEN" if engine == "ORACLE" else "NOT_APPLICABLE").model_dump(mode="json")
    provenance = {"producer": "aiops-db-executor", "verification": "EXECUTOR_REPORTED",
        "source_run_id": str(store.run.ops_run_id), "approval_sha256": approval}
    artifact = store.artifact(background, RESULT_KEY, RESULT_SCHEMA, result, provenance)
    store.artifact(store.run, RESULT_KEY + ":" + str(request_id), RESULT_SCHEMA, deepcopy(result), provenance)
    execution["result_sha256"] = artifact.content_hash
    store.encrypted(request_id, APPROVAL_KIND, {"request": request, "state": "APPROVED", "decision": decision, "execution": execution})
    store.material = SimpleNamespace(material_id=material_id)
    return store, background
