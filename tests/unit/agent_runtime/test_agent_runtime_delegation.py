"""Agent Runtime 跨服务委派的恢复语义测试。"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest

from agent_runtime.application import AgentDelegationReconciler
from platform_clients import AIOpsClientError
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


class _UnavailableAIOpsClient:
    def __init__(self):
        self.idempotency_keys: list[str] = []

    async def create_delegation(
        self, request, *, idempotency_key, auth_context
    ):
        self.idempotency_keys.append(idempotency_key)
        raise AIOpsClientError(
            status_code=503,
            code="OPS_UPSTREAM_UNAVAILABLE",
            message="AIOps 服务暂时不可用",
            retryable=True,
        )


class _Delegations:
    def __init__(self, state):
        self.state = state

    async def claim_poll_candidate(self, *, now):
        row = self.state.delegation
        if row.lease_until is not None and row.lease_until > now:
            return None
        return row

    async def get(self, *, delegation_id, lock=False):
        if delegation_id == self.state.delegation.delegation_id:
            return self.state.delegation
        return None


class _Rows:
    def __init__(self, row, id_field):
        self.row = row
        self.id_field = id_field

    async def get(self, **kwargs):
        expected = kwargs[self.id_field]
        return self.row if getattr(self.row, self.id_field) == expected else None


class _Uow:
    def __init__(self, state):
        self.delegations = _Delegations(state)
        self.runs = _Rows(state.run, "run_id")
        self.tasks = _Rows(state.task, "task_id")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        return None


class AgentDelegationReconcilerTest(
    unittest.IsolatedAsyncioTestCase
):
    def test_interaction_event_points_to_authorized_public_resource(self):
        url = AgentDelegationReconciler._public_resource_url(
            event_type="interaction.required",
            payload={"hitl_id": str(uuid7())},
        )

        self.assertTrue(url.startswith("/api/v1/ops/hitl/"))

    async def test_submit_timeout_keeps_same_recoverable_delegation(self):
        now = datetime.now(UTC)
        run_id = uuid7()
        task_id = uuid7()
        delegation_id = uuid7()
        aiops_agent_id = uuid7()
        aiops_target_id = uuid7()
        context = AuthContext(
            principal_kind=PrincipalKind.PORTAL,
            client_id="km-portal",
            request_id="request-1",
            trace_id="trace-1",
            api_key_id="portal-key",
            domain_id="20",
            asserted_user_id="user-1",
        )
        state = SimpleNamespace(
            run=SimpleNamespace(
                run_id=run_id,
                domain_id=20,
                agent_id=aiops_agent_id,
                actor_id="user-1",
                original_input="分析数据库性能下降",
                deadline_at=now + timedelta(minutes=10),
                trace_id="trace-1",
                config_snapshot_json={
                    "agent": {
                        "config": {
                            "aiops_target_id": str(aiops_target_id),
                        }
                    }
                },
                policy_snapshot_json={
                    "auth_context": context.model_dump(mode="json")
                },
            ),
            task=SimpleNamespace(
                task_id=task_id,
                timeout_seconds=600,
                max_attempts=3,
                status="WAITING_EXTERNAL",
            ),
            delegation=SimpleNamespace(
                delegation_id=delegation_id,
                parent_run_id=run_id,
                parent_task_id=task_id,
                status="SUBMITTING",
                child_run_id=None,
                idempotency_key=f"task:{task_id}:delegation",
                attempt_count=0,
                max_attempts=3,
                lease_owner=None,
                lease_token=None,
                lease_until=None,
                row_version=1,
                last_child_event_sequence=0,
                error_code=None,
                error_message=None,
                next_poll_at=now,
            ),
        )
        client = _UnavailableAIOpsClient()
        reconciler = AgentDelegationReconciler(
            uow_factory=lambda: _Uow(state),
            aiops_client=client,
            reconciler_id="reconciler-1",
            lease_seconds=60,
            poll_interval_seconds=1,
        )

        worked = await reconciler.run_once()

        self.assertTrue(worked)
        self.assertEqual(state.delegation.status, "SUBMITTING")
        self.assertEqual(state.delegation.attempt_count, 1)
        self.assertEqual(
            client.idempotency_keys,
            [f"task:{task_id}:delegation"],
        )
        self.assertIsNone(state.delegation.lease_token)
        self.assertEqual(
            state.delegation.error_code, "OPS_UPSTREAM_UNAVAILABLE"
        )


if __name__ == "__main__":
    unittest.main()
