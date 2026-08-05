"""第 7 阶段组合编排的故障注入与幂等恢复测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest

from main_api.application.resource_composition import CompositionError, ResourceCompositionService, _hash
from platform_core.contracts import CollectionCompositionCreate


class _ReceiptRepository:
    def __init__(self):
        self.rows = {}

    async def get_by_idempotency(
        self, *, domain_id, actor_id, operation, idempotency_key, lock=False,
    ):
        del lock
        return self.rows.get((domain_id, actor_id, operation, idempotency_key))

    async def get(self, *, receipt_id, domain_id, actor_id):
        return next((row for row in self.rows.values() if row.receipt_id == receipt_id and row.domain_id == domain_id and row.actor_id == actor_id), None)

    async def add(self, entity):
        now = datetime.now(timezone.utc)
        entity.created_at = now
        entity.updated_at = now
        entity.attempt_count = 0
        entity.row_version = 1
        self.rows[(entity.domain_id, entity.actor_id, entity.operation, entity.idempotency_key)] = entity

    async def transition(self, entity, **values):
        entity.status = values["status"]
        if values.get("resource_id") is not None:
            entity.resource_id = values["resource_id"]
        if values.get("resource_version") is not None:
            entity.resource_version = values["resource_version"]
        if values.get("verification") is not None:
            entity.verification_json = values["verification"]
        entity.error_code = values.get("error_code")
        entity.attempt_count += 1
        entity.row_version += 1
        entity.updated_at = datetime.now(timezone.utc)


class _Uow:
    def __init__(self, repository):
        self.compositions = repository

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def commit(self):
        return None


class _Knowledge:
    def __init__(self):
        self.created = None
        self.command_count = 0
        self.raise_after_store = False
        self.raise_without_store = False

    async def create_collection(self, *, domain_id, payload, auth_context):
        del auth_context
        self.command_count += 1
        if not self.raise_without_store:
            self.created = {
                **payload, "domain_id": domain_id, "status": "ACTIVE",
                "row_version": 1,
            }
        if self.raise_after_store or self.raise_without_store:
            raise TimeoutError("注入下游超时")
        return self.created

    async def get_collection(self, *, domain_id, collection_id, auth_context):
        del domain_id, auth_context
        if self.created is None or str(self.created["collection_id"]) != str(collection_id):
            raise LookupError("知识库不存在")
        return self.created


class _ModelClient:
    def __init__(self, *, status="ACTIVE", unavailable=False):
        self.status = status
        self.unavailable = unavailable

    async def get_model(self, model_id):
        if self.unavailable:
            raise RuntimeError("注入目录不可用")
        return {"model_id": str(model_id), "status": self.status, "row_version": 1}


class _Notifications:
    async def resource_notifications(self, **kwargs):
        return []


def _body(model_id: UUID, *, name="组合知识库") -> CollectionCompositionCreate:
    return CollectionCompositionCreate.model_validate({
        "collection": {
            "display_name": name,
            "models": {"embedding": str(model_id)},
        }
    })


def _service(knowledge, model_client):
    repository = _ReceiptRepository()
    service = ResourceCompositionService(
        uow_factory=lambda: _Uow(repository),
        agent_client=SimpleNamespace(), knowledge_client=knowledge,
        data_query_client=SimpleNamespace(), model_clients=(model_client,),
        notification_center=_Notifications(),
    )
    return service, repository


@pytest.mark.anyio
async def test_timeout_after_commit_is_verified_without_duplicate_command():
    model_id = UUID("019f0000-0000-7000-8000-000000000001")
    knowledge = _Knowledge()
    knowledge.raise_after_store = True
    service, _ = _service(knowledge, _ModelClient())
    receipt = await service.create_collection(
        body=_body(model_id), domain_id=7, actor_id="operator",
        idempotency_key="s7-after-commit", context=None,
    )
    assert receipt.status == "SUCCEEDED"
    assert knowledge.command_count == 1

    replay = await service.create_collection(
        body=_body(model_id), domain_id=7, actor_id="operator",
        idempotency_key="s7-after-commit", context=None,
    )
    assert replay.idempotent_replay is True
    assert knowledge.command_count == 1


@pytest.mark.anyio
async def test_uncertain_command_requires_recovery_and_replay_only_verifies():
    model_id = UUID("019f0000-0000-7000-8000-000000000002")
    knowledge = _Knowledge()
    knowledge.raise_without_store = True
    service, repository = _service(knowledge, _ModelClient())
    with pytest.raises(CompositionError) as first:
        await service.create_collection(
            body=_body(model_id), domain_id=7, actor_id="operator",
            idempotency_key="s7-recovery", context=None,
        )
    assert first.value.code == "COMPOSITION_COMPENSATION_REQUIRED"
    row = next(iter(repository.rows.values()))
    assert row.status == "COMPENSATION_REQUIRED"
    assert knowledge.command_count == 1

    knowledge.raise_without_store = False
    knowledge.created = {
        "collection_id": row.resource_id, "domain_id": 7,
        "display_name": "组合知识库", "models": {"embedding": str(model_id)},
        "status": "ACTIVE", "row_version": 1,
    }
    recovered = await service.create_collection(
        body=_body(model_id), domain_id=7, actor_id="operator",
        idempotency_key="s7-recovery", context=None,
    )
    assert recovered.status == "SUCCEEDED"
    assert recovered.idempotent_replay is True
    assert knowledge.command_count == 1


@pytest.mark.anyio
async def test_precheck_failure_blocks_command_and_hash_conflict_is_rejected():
    model_id = UUID("019f0000-0000-7000-8000-000000000003")
    knowledge = _Knowledge()
    service, _ = _service(knowledge, _ModelClient(status="DISABLED"))
    with pytest.raises(CompositionError) as failure:
        await service.create_collection(
            body=_body(model_id), domain_id=7, actor_id="operator",
            idempotency_key="s7-precheck", context=None,
        )
    assert failure.value.code == "MODEL_NOT_ACTIVE"
    assert knowledge.command_count == 0

    with pytest.raises(CompositionError) as conflict:
        await service.create_collection(
            body=_body(model_id, name="另一个知识库"), domain_id=7,
            actor_id="operator", idempotency_key="s7-precheck", context=None,
        )
    assert conflict.value.code == "COMPOSITION_IDEMPOTENCY_CONFLICT"


@pytest.mark.anyio
async def test_precheck_500_and_command_500_are_persisted_without_silent_success():
    knowledge = _Knowledge()
    service, repository = _service(knowledge, _ModelClient())
    called = 0

    async def unavailable_precheck():
        raise RuntimeError("注入 500")

    async def command():
        nonlocal called
        called += 1

    async def verify():
        return None

    with pytest.raises(CompositionError) as precheck:
        await service._execute(
            domain_id=7, actor_id="operator", operation="FAULT_PRECHECK",
            idempotency_key="s7-precheck-500", request_payload={"x": 1},
            resource_type="test", resource_id="one",
            precheck=unavailable_precheck, command=command, verify=verify,
        )
    assert precheck.value.status_code == 503
    assert called == 0

    async def passed_precheck():
        return None

    async def failed_command():
        nonlocal called
        called += 1
        error = RuntimeError("注入下游 500")
        error.code = "DOWNSTREAM_500"
        raise error

    with pytest.raises(CompositionError) as command_failure:
        await service._execute(
            domain_id=7, actor_id="operator", operation="FAULT_COMMAND",
            idempotency_key="s7-command-500", request_payload={"x": 2},
            resource_type="test", resource_id="two",
            precheck=passed_precheck, command=failed_command, verify=verify,
        )
    assert command_failure.value.code == "COMPOSITION_COMPENSATION_REQUIRED"
    row = repository.rows[(7, "operator", "FAULT_COMMAND", "s7-command-500")]
    assert row.status == "COMPENSATION_REQUIRED"
    assert row.error_code == "DOWNSTREAM_500"


@pytest.mark.anyio
async def test_run_composition_traces_all_backend_resources_without_result_rows():
    run_id = UUID("019f0000-0000-7000-8000-000000000010")
    agent_id = UUID("019f0000-0000-7000-8000-000000000011")
    model_id = UUID("019f0000-0000-7000-8000-000000000012")
    collection_id = UUID("019f0000-0000-7000-8000-000000000013")
    semantic_id = UUID("019f0000-0000-7000-8000-000000000014")
    source_id = UUID("019f0000-0000-7000-8000-000000000015")
    dq_run_id = UUID("019f0000-0000-7000-8000-000000000016")
    evidence_id = UUID("019f0000-0000-7000-8000-000000000017")
    artifact_id = UUID("019f0000-0000-7000-8000-000000000018")

    class Agent:
        async def get_debug_run(self, **kwargs):
            return {
                "run": {
                    "run_id": str(run_id), "agent_id": str(agent_id),
                    "status": "SUCCEEDED", "row_version": 3,
                    "config_snapshot": {"collection_ids": [str(collection_id)]},
                },
                "tasks": [{"task_id": "task-1", "status": "SUCCEEDED"}],
                "artifacts": [{
                    "artifact_id": str(artifact_id), "artifact_type": "query_result",
                    "schema_version": "1", "producer": "data-query",
                    "content_hash": "abc", "payload": {"rows": [{"secret": "不得回传"}]},
                    "provenance": {
                        "data_query_run_id": str(dq_run_id),
                        "semantic_model_id": str(semantic_id),
                        "data_source_id": str(source_id),
                        "evidence_ids": [str(evidence_id)],
                    },
                }],
            }

        async def get_agent(self, **kwargs):
            return {
                "agent_id": str(agent_id), "status": "ACTIVE",
                "row_version": 5, "models": {"planner": str(model_id)},
            }

    class Knowledge(_Knowledge):
        async def get_collection(self, **kwargs):
            return {
                "collection_id": str(collection_id), "status": "ACTIVE",
                "row_version": 2, "models": {},
            }

    class DataQuery:
        async def management_get(self, *, resource, resource_id, auth_context):
            return {
                "status": "ACTIVE", "row_version": 4,
                "resource": resource, "resource_id": str(resource_id),
            }

        async def get_run(self, **kwargs):
            return {"status": "SUCCEEDED", "row_version": 2}

        async def get_result(self, **kwargs):
            return {
                "columns": [{"name": "count"}], "preview_rows": [{"count": 9}],
                "row_count": 1, "truncated": False,
                "provenance": {"semantic_model_id": str(semantic_id)},
            }

    class Notifications(_Notifications):
        async def resource_notifications(self, **kwargs):
            return [{"event_type": "agent.run.succeeded"}]

    repository = _ReceiptRepository()
    service = ResourceCompositionService(
        uow_factory=lambda: _Uow(repository), agent_client=Agent(),
        knowledge_client=Knowledge(), data_query_client=DataQuery(),
        model_clients=(_ModelClient(),), notification_center=Notifications(),
    )
    view = await service.run_composition(
        run_id=run_id, domain_id=7, actor_id="operator", context=None,
    )
    assert {node.resource_id for node in view.models} == {str(model_id)}
    assert {node.resource_id for node in view.collections} == {str(collection_id)}
    assert {node.resource_id for node in view.semantic_models} == {str(semantic_id)}
    assert {node.resource_id for node in view.data_sources} == {str(source_id)}
    assert {node.resource_id for node in view.data_query_runs} == {str(dq_run_id)}
    assert {node.resource_id for node in view.knowledge_evidence} == {str(evidence_id)}
    assert len(view.notifications) == 1 and len(view.tasks) == 1
    assert "preview_rows" not in view.data_query_runs[0].attributes
    assert "payload" not in view.artifacts[0].attributes


@pytest.mark.anyio
async def test_concurrent_prechecking_replay_never_sends_a_second_create():
    model_id = UUID("019f0000-0000-7000-8000-000000000020")
    knowledge = _Knowledge()
    service, _ = _service(knowledge, _ModelClient())
    body = _body(model_id)
    reserved_id = UUID("019f0000-0000-7000-8000-000000000021")
    await service._begin(
        domain_id=7, actor_id="operator", operation="COLLECTION_CREATE",
        idempotency_key="s7-concurrent",
        request_hash=_hash(body.model_dump(mode="json")),
        resource_type="collection", resource_id=str(reserved_id),
    )
    with pytest.raises(CompositionError) as conflict:
        await service.create_collection(
            body=body, domain_id=7, actor_id="operator",
            idempotency_key="s7-concurrent", context=None,
        )
    assert conflict.value.code == "COMPOSITION_IN_PROGRESS"
    assert knowledge.command_count == 0
