"""S6 通知目录、安全信封和幂等投影测试。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from main_api.application.notification_projection import NotificationProjectionService
from platform_core.identity import uuid7
from platform_core.notifications import EVENT_TYPES, NotificationEnvelope


class _Repository:
    def __init__(self):
        self.operations = {}
        self.inbox = []
        self.work_items = {}
        self.watchers = set()
        self.disabled = set()

    async def resource_work_item_closed(self, **scope):
        return any(item.status in {"COMPLETED", "CANCELLED"} and all(
            getattr(item, key) == value for key, value in scope.items()) for item in self.work_items.values())

    async def resolve_resource_work_items(self, *, outbox_id, occurred_at, status, **scope):
        for item in self.work_items.values():
            if item.status == "OPEN" and all(getattr(item, key) == value for key, value in scope.items()):
                item.status = status
                item.resolved_outbox_id = outbox_id
                item.last_occurred_at = occurred_at

    async def operation_for_update(self, *, producer, source_operation_id):
        return self.operations.get((producer, source_operation_id))

    async def add_operation(self, entity):
        entity.row_version = 1
        self.operations[(entity.producer_service, entity.source_operation_id)] = entity

    async def watcher_actor_ids(self, **kwargs):
        del kwargs
        return set(self.watchers)

    async def preference_enabled(self, *, actor_id, event_type, **kwargs):
        del kwargs
        return (actor_id, event_type) not in self.disabled

    async def inbox_exists(self, *, outbox_id, actor_id):
        return any(
            row.outbox_id == outbox_id and row.recipient_actor_id == actor_id
            for row in self.inbox
        )

    async def add_inbox(self, entity):
        entity.event_sequence = len(self.inbox) + 1
        self.inbox.append(entity)

    async def work_item_for_update(
        self, *, domain_id, actor_id, resource_type, resource_id, action_type,
    ):
        return self.work_items.get(
            (domain_id, actor_id, resource_type, resource_id, action_type)
        )

    async def add_work_item(self, entity):
        entity.row_version = 1
        if entity.status is None:
            entity.status = "OPEN"
        self.work_items[(
            entity.domain_id, entity.actor_id, entity.resource_type,
            entity.resource_id, entity.action_type,
        )] = entity


class _Uow:
    def __init__(self, repository):
        self.notifications = repository


def _outbox(
    *, event_type: str, occurred_at: datetime,
    recipients: list[str], event_key: str,
):
    producer = EVENT_TYPES[event_type].producer_service
    envelope = NotificationEnvelope(
        domain_id=10,
        event_type=event_type,
        resource_type="agent_run",
        resource_id="run-1",
        resource_name="运行 1",
        initiator_actor_id=recipients[0] if recipients else None,
        recipient_actor_ids=recipients,
        summary="安全摘要",
        occurred_at=occurred_at,
        correlation_id="trace-1",
        operation_id="run-1",
        safe_data={"status": "TEST"},
    )
    return SimpleNamespace(
        outbox_id=uuid7(), producer_service=producer,
        event_key=event_key, event_type=event_type,
        domain_id=10, payload_json=envelope.model_dump(mode="json"),
    )


class NotificationCenterS6Test(unittest.IsolatedAsyncioTestCase):
    def test_catalog_contains_only_kbot_business_events(self):
        serialized = " ".join(EVENT_TYPES).casefold()
        for forbidden in ("tenant", "permission", "role", "license", "api_key"):
            self.assertNotIn(forbidden, serialized)
        self.assertIn("aiops.report.ready", EVENT_TYPES)
        self.assertEqual(
            {("IN_APP",)},
            {definition.allowed_channels for definition in EVENT_TYPES.values()},
        )

    def test_envelope_rejects_sensitive_or_unknown_payload(self):
        with self.assertRaisesRegex(ValueError, "SENSITIVE"):
            NotificationEnvelope(
                domain_id=1, event_type="agent.run.completed",
                resource_type="agent_run", resource_id="run-1",
                summary="完成", correlation_id="trace-1",
                safe_data={"nested": {"password": "secret"}},
            )
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            NotificationEnvelope(
                domain_id=1, event_type="unknown.event",
                resource_type="run", resource_id="run-1",
                summary="完成", correlation_id="trace-1",
            )

    async def test_duplicate_delivery_creates_one_inbox_and_one_work_item(self):
        repository = _Repository()
        outbox = _outbox(
            event_type="agent.run.input_required",
            occurred_at=datetime.now(timezone.utc),
            recipients=["actor-1"], event_key="input-1",
        )
        projection = NotificationProjectionService()
        await projection.project(uow=_Uow(repository), outbox=outbox)
        await projection.project(uow=_Uow(repository), outbox=outbox)
        self.assertEqual(1, len(repository.inbox))
        self.assertEqual(1, len(repository.work_items))
        self.assertEqual("OPEN", next(iter(repository.work_items.values())).status)

    async def test_knowledge_ingestion_start_creates_running_workflow(self):
        repository = _Repository()
        await NotificationProjectionService().project(
            uow=_Uow(repository),
            outbox=_outbox(
                event_type="knowledge.ingestion.started",
                occurred_at=datetime.now(timezone.utc),
                recipients=["actor-1"],
                event_key="knowledge-start-1",
            ),
        )
        operation = repository.operations[("knowledge-core", "run-1")]
        self.assertEqual("RUNNING", operation.status)
        self.assertEqual(1, len(repository.inbox))

    async def test_out_of_order_event_does_not_regress_operation_or_work_item(self):
        repository = _Repository()
        now = datetime.now(timezone.utc)
        projection = NotificationProjectionService()
        await projection.project(
            uow=_Uow(repository),
            outbox=_outbox(
                event_type="agent.run.input_required",
                occurred_at=now, recipients=["actor-1"], event_key="input-1",
            ),
        )
        await projection.project(
            uow=_Uow(repository),
            outbox=_outbox(
                event_type="agent.run.completed",
                occurred_at=now + timedelta(seconds=5),
                recipients=["actor-1"], event_key="complete-1",
            ),
        )
        await projection.project(
            uow=_Uow(repository),
            outbox=_outbox(
                event_type="agent.run.input_required",
                occurred_at=now - timedelta(seconds=5),
                recipients=["actor-1"], event_key="input-old",
            ),
        )
        operation = repository.operations[("agent-runtime", "run-1")]
        work_item = next(iter(repository.work_items.values()))
        self.assertEqual("SUCCEEDED", operation.status)
        self.assertEqual("COMPLETED", work_item.status)

    async def test_actorless_system_event_updates_operation_without_inbox(self):
        repository = _Repository()
        await NotificationProjectionService().project(
            uow=_Uow(repository),
            outbox=_outbox(
                event_type="data_query.run.failed",
                occurred_at=datetime.now(timezone.utc),
                recipients=[], event_key="system-failure",
            ),
        )
        self.assertEqual(1, len(repository.operations))
        self.assertEqual([], repository.inbox)

    async def test_terminal_event_notifies_explicit_watcher_once(self):
        repository = _Repository()
        repository.watchers.add("watcher-1")
        outbox = _outbox(
            event_type="agent.run.completed",
            occurred_at=datetime.now(timezone.utc),
            recipients=["actor-1"], event_key="complete-watch",
        )
        await NotificationProjectionService().project(
            uow=_Uow(repository), outbox=outbox,
        )
        self.assertEqual(
            {"actor-1", "watcher-1"},
            {row.recipient_actor_id for row in repository.inbox},
        )


class AIOpsWorkflowProjectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_closes_every_reviewer_not_only_result_recipients(self):
        from unittest.mock import AsyncMock
        repository = _Repository()
        uow = _Uow(repository)
        uow.access = SimpleNamespace(notification_approvers=AsyncMock(return_value={"reviewer-a", "reviewer-b"}))
        proposal_id = str(uuid7())
        def event(kind, recipients):
            envelope = NotificationEnvelope(domain_id=10, event_type=kind, resource_type="aiops_proposal",
                resource_id=proposal_id, initiator_actor_id="applicant", recipient_actor_ids=recipients,
                summary="方案已更新", correlation_id="trace", safe_data={"run_id": str(uuid7())})
            return SimpleNamespace(outbox_id=uuid7(), producer_service="aiops", domain_id=10,
                payload_json=envelope.model_dump(mode="json"))
        projection = NotificationProjectionService()
        await projection.project(uow=uow, outbox=event("aiops.proposal.review_required", ["applicant"]))
        assert {item.actor_id for item in repository.work_items.values()} == {"reviewer-a", "reviewer-b"}
        await projection.project(uow=uow, outbox=event("aiops.proposal.approved", ["applicant"]))
        assert all(item.status == "COMPLETED" for item in repository.work_items.values())
        await projection.project(uow=uow, outbox=event("aiops.proposal.review_required", ["applicant"]))
        assert all(item.status == "COMPLETED" for item in repository.work_items.values())
        assert len(repository.inbox) == 3

    async def test_close_before_open_and_multiple_requests(self):
        repository = _Repository(); uow = _Uow(repository)
        run_id, first, second = map(str, (uuid7(), uuid7(), uuid7()))
        def event(kind, hitl, recipients):
            envelope = NotificationEnvelope(domain_id=10, event_type=kind, resource_type="aiops_run",
                resource_id=run_id, initiator_actor_id="applicant", recipient_actor_ids=recipients,
                summary="人工请求", correlation_id="trace",
                safe_data={"hitl_id": hitl, "source_status": "PENDING" if kind.endswith("required") else "ANSWERED"})
            return SimpleNamespace(outbox_id=uuid7(), producer_service="aiops", domain_id=10,
                payload_json=envelope.model_dump(mode="json"))
        projection = NotificationProjectionService()
        await projection.project(uow=uow, outbox=event("aiops.diagnostic.input_resolved", first, []))
        await projection.project(uow=uow, outbox=event("aiops.diagnostic.input_required", first, ["applicant"]))
        await projection.project(uow=uow, outbox=event("aiops.diagnostic.input_required", second, ["applicant"]))
        assert len(repository.work_items) == 2
        assert {item.resource_id: item.status for item in repository.work_items.values()} == {first: "COMPLETED", second: "OPEN"}
        assert len(repository.inbox) == 1

    async def test_missing_approver_is_not_replaced_by_applicant(self):
        from unittest.mock import AsyncMock
        repository = _Repository(); uow = _Uow(repository)
        uow.access = SimpleNamespace(notification_approvers=AsyncMock(return_value=set()))
        envelope = NotificationEnvelope(domain_id=10, event_type="aiops.proposal.review_required",
            resource_type="aiops_proposal", resource_id=str(uuid7()), initiator_actor_id="applicant",
            recipient_actor_ids=["applicant"], summary="待审批", correlation_id="trace")
        with self.assertRaisesRegex(ValueError, "NOTIFICATION_APPROVER_UNAVAILABLE"):
            await NotificationProjectionService().project(uow=uow, outbox=SimpleNamespace(
                outbox_id=uuid7(), producer_service="aiops", domain_id=10, payload_json=envelope.model_dump(mode="json")))
        assert not repository.inbox and not repository.work_items

    async def test_critical_situation_keeps_urgency(self):
        envelope = NotificationEnvelope(domain_id=10, event_type="aiops.situation.escalated",
            resource_type="aiops_situation", resource_id=str(uuid7()), recipient_actor_ids=["applicant"],
            summary="严重故障", correlation_id="trace", safe_data={"severity": "CRITICAL"})
        repository = _Repository()
        await NotificationProjectionService().project(uow=_Uow(repository), outbox=SimpleNamespace(
            outbox_id=uuid7(), producer_service="aiops", domain_id=10, payload_json=envelope.model_dump(mode="json")))
        assert repository.inbox[0].severity == "CRITICAL"

if __name__ == "__main__":
    unittest.main()
